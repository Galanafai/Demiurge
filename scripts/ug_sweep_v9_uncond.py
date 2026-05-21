"""Phase E-fix: UG sweep for v9 step 170k using noise_prediction_fn + sampler.sample.

Matches the validated 3% baseline (v9_final_probe.json) exactly.
Applies UG via manual gradient loop wrapping sampler.sample.
No yaw_project.

Output: JSONL compatible with plot_ug_pareto.py.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from guidance.energy import pairwise_overlap_energy
from model.denoiser import N_MAX, DenoiserConfig, SceneDenoiser
from model.rotations import rot6d_to_quat_wxyz
from model.schedule import CosineSchedule, DDIMSampler
from scene.schema import SceneTensor, WorkspaceBounds
from validator.core import SceneValidator

_DATA_MEAN_XYZ   = torch.tensor([-0.049867, +0.378790, -0.751155])
_DATA_STD_XYZ    = torch.tensor([+0.419080, +0.457588, +0.127651])
_DATA_MEAN_SCALE = torch.tensor([-0.038706, -0.038706, -0.038706])
_DATA_STD_SCALE  = torch.tensor([+0.221619, +0.221619, +0.221619])


def load_model(ckpt_path: str, device: torch.device) -> SceneDenoiser:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    cfg = DenoiserConfig(
        n_layers=arch.get("n_layers", 16),
        d_model=arch.get("d_model", 384),
        n_heads=arch.get("n_heads", 8),
        ffn_mult=arch.get("ffn_mult", 4),
        dropout=0.0,
        use_slot_id_embed=arch.get("use_slot_id_embed", True),
    )
    model = SceneDenoiser(cfg).to(device)
    state = ckpt.get("ema_state", ckpt.get("model_state"))
    model.load_state_dict(state)
    model.eval()
    step = ckpt.get("step", "?")
    print(f"Loaded EMA step={step}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model


def decode_scene(
    x_cont: torch.Tensor,
    bounds: WorkspaceBounds,
    presence_threshold: float,
) -> tuple[SceneTensor, SceneTensor] | None:
    """Decode a single (N_MAX, 13) tensor. No yaw_project. Returns (norm, phys) or None."""
    xyz_norm   = (x_cont[:, :3] * _DATA_STD_XYZ   + _DATA_MEAN_XYZ).clamp(-1, 1)
    rot6d      = x_cont[:, 3:9].clone()  # no yaw_project -- matches 3% probe
    scale_norm = (x_cont[:, 9:12] * _DATA_STD_SCALE + _DATA_MEAN_SCALE).clamp(-1, 1)
    pres       = x_cont[:, 12] > presence_threshold

    if pres.sum() < 1:
        return None

    quats  = rot6d_to_quat_wxyz(rot6d)
    poses  = torch.cat([xyz_norm, quats], dim=-1)
    # object_types: not returned by sampler.sample -- use zeros (unknown type)
    type_ids = torch.zeros(N_MAX, dtype=torch.long)
    st_norm  = SceneTensor(object_types=type_ids, poses=poses, scales=scale_norm, presence=pres)
    st_phys  = st_norm.denormalize(bounds)
    return st_norm, st_phys


def sample_with_ug(
    model: SceneDenoiser,
    sampler: DDIMSampler,
    guidance_scale: float,
    batch_size: int,
    seed: int,
    device: torch.device,
    presence_threshold: float,
) -> torch.Tensor:
    """Sample using noise_prediction_fn (matches 3% baseline) + optional UG.
    
    For guidance_scale=0: pure sampler.sample (exact baseline).
    For guidance_scale>0: manual DDIM loop with UG gradient applied at each step.
    Returns x0 tensor (B, N_MAX, 13).
    """
    fn = model.noise_prediction_fn(text_emb=None)
    
    if guidance_scale == 0.0:
        # Exact baseline path
        with torch.no_grad():
            x0 = sampler.sample(fn, (batch_size, N_MAX, 13), seed=seed, device=device)
        return x0
    
    # UG path: manual DDIM loop.
    # NOTE: CosineSchedule with zero_terminal_snr=True gives alpha_bar=0 at t=T-1.
    # Any Tweedie x0_hat = (x - sigma * eps) / sqrt(alpha_bar) is undefined (div by 0)
    # at those steps. We guard with alpha_min and skip UG there.
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None

    x = torch.randn(batch_size, N_MAX, 13, device=device, generator=gen)
    sched = sampler.schedule
    timesteps = sampler._timesteps
    n_steps = len(timesteps)
    guide_start = int(0.1 * n_steps)
    guide_end   = int(0.9 * n_steps)
    ALPHA_MIN   = 1e-4  # below this, Tweedie x0 is undefined; skip UG and use fallback step

    type_ids_dummy = torch.zeros(batch_size, N_MAX, dtype=torch.long, device=device)

    for i, t_val in enumerate(timesteps):
        t_tensor = torch.full((batch_size,), t_val, dtype=torch.long, device=device)
        alpha_bar = sched.alpha_bar(t_tensor).to(device).view(batch_size, 1, 1)
        sigma_bar = (1.0 - alpha_bar).sqrt()

        with torch.no_grad():
            out = fn(x, t_tensor, None)  # noise_prediction_fn(x_t, t, _text_emb) -> eps
            eps_pred = out[0] if isinstance(out, tuple) else out

        # Compute UG guidance only when Tweedie estimate is numerically valid
        eps_guided = eps_pred
        a_scalar = float(alpha_bar[0, 0, 0])
        if guide_start <= i < guide_end and a_scalar > ALPHA_MIN:
            x0_hat = (x - sigma_bar * eps_pred) / alpha_bar.sqrt()
            x0_g = x0_hat.detach().clone().requires_grad_(True)
            energy = pairwise_overlap_energy(x0_g, type_ids_dummy, presence_threshold)
            grad = torch.autograd.grad(energy.sum(), x0_g)[0]
            grad[:, :, 12] = 0.0  # FIX: presence_logit (ch12) must not be guided
            grad_norm = grad.abs().max()
            if grad_norm > 1e-8:
                grad = grad / grad_norm  # L-inf normalize
            eps_guided = eps_pred - guidance_scale * sigma_bar * grad.detach()

        # DDIM step -- use Tweedie form when alpha is safe, else eps-only fallback
        if i < n_steps - 1:
            t_next = timesteps[i + 1]
            t_next_t = torch.full((batch_size,), t_next, dtype=torch.long, device=device)
            alpha_bar_next = sched.alpha_bar(t_next_t).to(device).view(batch_size, 1, 1)
            sigma_bar_next = (1.0 - alpha_bar_next).sqrt()
            if a_scalar > ALPHA_MIN:
                x0_ddim = (x - sigma_bar * eps_guided) / alpha_bar.sqrt()
                x = alpha_bar_next.sqrt() * x0_ddim + sigma_bar_next * eps_guided
            else:
                # Pure eps step: x_{t-1} = x_t - sigma_bar * eps_guided (ignoring x0 term)
                # This is only hit at t=999 (i=0) with zero_terminal_snr.
                x = x - (sigma_bar - sigma_bar_next) * eps_guided
        else:
            # Final step: return x0 estimate
            if a_scalar > ALPHA_MIN:
                x = (x - sigma_bar * eps_guided) / alpha_bar.sqrt()
            # else: x already close to x0 at near-zero sigma

    return x.detach()


def run_one_config(
    model: SceneDenoiser,
    sampler: DDIMSampler,
    guidance_scale: float,
    seed: int,
    n_scenes: int,
    batch_size: int,
    device: torch.device,
    bounds: WorkspaceBounds,
    rrt_budget: float,
    presence_threshold: float,
) -> dict:
    all_phys = []
    remaining = n_scenes
    batch_idx = 0
    while remaining > 0:
        bs = min(batch_size, remaining)
        batch_seed = (seed * 10007 + batch_idx) & 0xFFFFFFFF
        x0 = sample_with_ug(model, sampler, guidance_scale, bs,
                             batch_seed, device, presence_threshold)
        x0 = x0.cpu()
        for i in range(bs):
            result = decode_scene(x0[i], bounds, presence_threshold)
            all_phys.append(result[1] if result is not None else None)
        remaining -= bs
        batch_idx += 1

    non_empty_phys = [s for s in all_phys if s is not None]
    accepted = interp_pass = stable_pass = ik_pass = rrt_pass = 0

    if non_empty_phys:
        def _val(st: SceneTensor) -> dict:
            v = SceneValidator(rrt_budget_s=rrt_budget)
            r = v.validate(st)
            return {"accepted": r.accepted, "no_interpenetration": r.no_interpenetration,
                    "stable_rest": r.stable_rest, "ik_reachable": r.ik_reachable,
                    "rrt_solvable": r.rrt_solvable}
        
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        if len(non_empty_phys) > 8:
            # Use process pool for speed
            from validator.core import SceneValidator as SV
            reports = [_val(s) for s in non_empty_phys]
        else:
            reports = [_val(s) for s in non_empty_phys]
        
        for r in reports:
            if r["accepted"]:            accepted    += 1
            if r["no_interpenetration"]: interp_pass += 1
            if r["stable_rest"]:         stable_pass += 1
            if r["ik_reachable"]:        ik_pass     += 1
            if r["rrt_solvable"]:        rrt_pass    += 1

    return {
        "guidance_scale": guidance_scale, "seed": seed,
        "n_scenes": n_scenes, "non_empty": len(non_empty_phys),
        "accepted": accepted, "validity_rate": accepted / n_scenes,
        "interp_pass": interp_pass, "stable_pass": stable_pass,
        "ik_pass": ik_pass, "rrt_pass": rrt_pass,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-checkpoint", required=True)
    p.add_argument("--guidance-scales", nargs="+", type=float, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--n-scenes", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--presence-threshold", type=float, default=-0.589)
    p.add_argument("--rrt-budget", type=float, default=2.0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--budget-cap-hours", type=float, default=5.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bounds = WorkspaceBounds.default()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model  = load_model(args.model_checkpoint, device)
    sched  = CosineSchedule(T=1000, zero_terminal_snr=True)
    sampler = DDIMSampler(sched, n_steps=50, prediction_type="v")

    total = len(args.guidance_scales) * len(args.seeds)
    print(f"\n=== Phase E-fix UG Sweep (v9 uncond, noise_prediction_fn) ===")
    print(f"Scales:        {args.guidance_scales}")
    print(f"Seeds:         {args.seeds}")
    print(f"Scenes/config: {args.n_scenes}")
    print(f"Yaw project:   False (disabled -- caused stability failures)")
    print(f"Sampler:       noise_prediction_fn + sampler.sample (3% baseline match)")
    print(f"Budget cap:    {args.budget_cap_hours}h\n")

    summary_path = out_dir / "summary.jsonl"
    t0 = time.time()

    with open(summary_path, "w") as f:
        for ci, scale in enumerate(args.guidance_scales):
            for si, seed in enumerate(args.seeds):
                elapsed = (time.time() - t0) / 3600
                if elapsed > args.budget_cap_hours:
                    print(f"Budget cap ({elapsed:.2f}h), stopping")
                    break
                idx = ci * len(args.seeds) + si + 1
                print(f"[{idx}/{total}] scale={scale}, seed={seed}  (elapsed={elapsed:.2f}h)")
                result = run_one_config(model, sampler, scale, seed, args.n_scenes,
                                        args.batch_size, device, bounds,
                                        args.rrt_budget, args.presence_threshold)
                v = result["validity_rate"] * 100
                print(f"  -> {result['accepted']}/{result['n_scenes']} = {v:.1f}%  "
                      f"[ne={result['non_empty']} st={result['stable_pass']} "
                      f"ik={result['ik_pass']} rrt={result['rrt_pass']}]")
                f.write(json.dumps(result) + "\n")
                f.flush()
            else:
                continue
            break

    by_scale: dict[float, list[float]] = {}
    with open(summary_path) as f:
        for line in f:
            r = json.loads(line)
            by_scale.setdefault(r["guidance_scale"], []).append(r["validity_rate"])

    print(f"\n=== AGGREGATE (ref: v9 3% uncond, 11% non-empty) ===")
    print(f"{'Scale':<8} {'Mean%':>7} {'N':>3}")
    for scale in sorted(by_scale):
        vals = by_scale[scale]
        mean = statistics.mean(vals) * 100
        print(f"{scale:<8.1f} {mean:>7.2f} {len(vals):>3}")
    print(f"\nResults: {summary_path}")


if __name__ == "__main__":
    main()
