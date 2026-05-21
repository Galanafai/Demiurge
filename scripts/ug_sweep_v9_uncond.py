"""Phase E: UG sweep for v9 step 170k in unconditional mode.

Uses sample_with_universal_guidance (Bansal et al. ICML 2023) with
pairwise_overlap_energy. Applies yaw projection at decode. No text/CFG.

Output: JSONL compatible with plot_ug_pareto.py.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import statistics
import sys
import time
from pathlib import Path

import torch

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
    type_ids: torch.Tensor,
    bounds: WorkspaceBounds,
    presence_threshold: float,
    yaw_project: bool,
) -> tuple[SceneTensor, SceneTensor] | None:
    """Decode continuous tensor to SceneTensor. Returns (norm, phys) or None if empty."""
    xyz_norm  = (x_cont[:, :3] * _DATA_STD_XYZ + _DATA_MEAN_XYZ).clamp(-1, 1)
    rot6d     = x_cont[:, 3:9].clone()
    if yaw_project:
        # Zero the out-of-plane components so rotation is yaw-only.
        rot6d[:, 2] = 0.0
        rot6d[:, 5] = 0.0
    scale_norm = (x_cont[:, 9:12] * _DATA_STD_SCALE + _DATA_MEAN_SCALE).clamp(-1, 1)
    pres       = x_cont[:, 12] > presence_threshold

    if pres.sum() < 1:
        return None

    quats  = rot6d_to_quat_wxyz(rot6d)
    poses  = torch.cat([xyz_norm, quats], dim=-1)
    st_norm = SceneTensor(object_types=type_ids, poses=poses,
                          scales=scale_norm, presence=pres)
    st_phys = st_norm.denormalize(bounds)
    return st_norm, st_phys


_global_validator: SceneValidator | None = None

def _pool_init(rrt_budget: float) -> None:
    global _global_validator
    _global_validator = SceneValidator(rrt_budget_s=rrt_budget)

def _pool_validate(st_phys: SceneTensor) -> dict:
    assert _global_validator is not None
    rpt = _global_validator.validate(st_phys)
    return {
        "accepted":          rpt.accepted,
        "no_interpenetration": getattr(rpt, "no_interpenetration", False),
        "stable_rest":       getattr(rpt, "stable_rest", False),
        "ik_reachable":      getattr(rpt, "ik_reachable", False),
        "rrt_solvable":      getattr(rpt, "rrt_solvable", False),
    }


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
    yaw_project: bool,
    drake_workers: int,
) -> dict:
    fn = model.conditional_sampling_fn(text_emb=None)  # unconditional

    all_norm, all_phys = [], []
    remaining = n_scenes
    batch_idx = 0
    while remaining > 0:
        bs = min(batch_size, remaining)
        batch_seed = (seed * 10007 + batch_idx) & 0xFFFFFFFF
        with torch.no_grad():
            x_cont, type_ids = sampler.sample_with_universal_guidance(
                fn,
                (bs, N_MAX, 13),
                seed=batch_seed,
                device=device,
                type_init="uniform",
                guidance_scale=guidance_scale,
            )
        x_cont  = x_cont.cpu()
        type_ids = type_ids.cpu()
        for i in range(bs):
            result = decode_scene(x_cont[i], type_ids[i], bounds, presence_threshold, yaw_project)
            if result is not None:
                all_norm.append(result[0])
                all_phys.append(result[1])
            else:
                all_norm.append(None)
                all_phys.append(None)
        remaining  -= bs
        batch_idx  += 1

    # Validate
    non_empty_phys = [s for s in all_phys if s is not None]
    accepted = interp_pass = stable_pass = ik_pass = rrt_pass = 0

    if non_empty_phys:
        def _validate_one(st: SceneTensor) -> dict:
            v = SceneValidator(rrt_budget_s=rrt_budget)
            r = v.validate(st)
            return {
                "accepted":            r.accepted,
                "no_interpenetration": r.no_interpenetration,
                "stable_rest":         r.stable_rest,
                "ik_reachable":        r.ik_reachable,
                "rrt_solvable":        r.rrt_solvable,
            }
        ctx = multiprocessing.get_context("spawn")
        if drake_workers > 1:
            with ctx.Pool(
                processes=drake_workers,
                initializer=_pool_init,
                initargs=(rrt_budget,),
            ) as pool:
                reports = pool.map(_pool_validate, non_empty_phys)
        else:
            reports = [_validate_one(s) for s in non_empty_phys]

        for r in reports:
            if r["accepted"]:            accepted    += 1
            if r["no_interpenetration"]: interp_pass += 1
            if r["stable_rest"]:         stable_pass += 1
            if r["ik_reachable"]:        ik_pass     += 1
            if r["rrt_solvable"]:        rrt_pass    += 1

    return {
        "guidance_scale":  guidance_scale,
        "seed":            seed,
        "n_scenes":        n_scenes,
        "non_empty":       len(non_empty_phys),
        "accepted":        accepted,
        "validity_rate":   accepted / n_scenes,
        "interp_pass":     interp_pass,
        "stable_pass":     stable_pass,
        "ik_pass":         ik_pass,
        "rrt_pass":        rrt_pass,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase E: UG sweep for v9 uncond")
    p.add_argument("--model-checkpoint", required=True)
    p.add_argument("--guidance-scales",  nargs="+", type=float, required=True)
    p.add_argument("--seeds",            nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--n-scenes",         type=int, default=300)
    p.add_argument("--batch-size",       type=int, default=50)
    p.add_argument("--presence-threshold", type=float, default=-0.589)
    p.add_argument("--yaw-project",      action="store_true", default=True)
    p.add_argument("--rrt-budget",       type=float, default=2.0)
    p.add_argument("--drake-workers",    type=int, default=6)
    p.add_argument("--output-dir",       required=True)
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
    print(f"\n=== Phase E UG Sweep (unconditional v9) ===")
    print(f"Scales:        {args.guidance_scales}")
    print(f"Seeds:         {args.seeds}")
    print(f"Scenes/config: {args.n_scenes}")
    print(f"Total scenes:  {total * args.n_scenes}")
    print(f"Yaw project:   {args.yaw_project}")
    print(f"Drake workers: {args.drake_workers}")
    print(f"Budget cap:    {args.budget_cap_hours}h\n")

    summary_path = out_dir / "summary.jsonl"
    t0 = time.time()

    with open(summary_path, "w") as f:
        for ci, scale in enumerate(args.guidance_scales):
            for si, seed in enumerate(args.seeds):
                elapsed_h = (time.time() - t0) / 3600
                if elapsed_h > args.budget_cap_hours:
                    print(f"Budget cap reached ({elapsed_h:.2f}h), stopping")
                    break
                idx = ci * len(args.seeds) + si + 1
                print(f"[{idx}/{total}] scale={scale}, seed={seed}  (elapsed={elapsed_h:.2f}h)")
                result = run_one_config(
                    model, sampler, scale, seed,
                    args.n_scenes, args.batch_size,
                    device, bounds, args.rrt_budget,
                    args.presence_threshold, args.yaw_project,
                    args.drake_workers,
                )
                v = result["validity_rate"] * 100
                print(f"  -> {result['accepted']}/{result['n_scenes']} = {v:.1f}%  "
                      f"[ne={result['non_empty']} ik={result['ik_pass']} rrt={result['rrt_pass']}]")
                f.write(json.dumps(result) + "\n")
                f.flush()
            else:
                continue
            break

    # Aggregate summary
    by_scale: dict[float, list[float]] = {}
    with open(summary_path) as f:
        for line in f:
            r = json.loads(line)
            by_scale.setdefault(r["guidance_scale"], []).append(r["validity_rate"])

    print(f"\n=== AGGREGATE (ref: v9 uncond 3.0%) ===")
    print(f"{'Scale':<8} {'Mean%':>7} {'Std%':>6} {'N':>3}")
    for scale in sorted(by_scale):
        vals = by_scale[scale]
        mean = statistics.mean(vals) * 100
        std  = statistics.stdev(vals) * 100 if len(vals) > 1 else 0.0
        bar  = "#" * int(mean * 2)
        print(f"{scale:<8.1f} {mean:>7.2f} {std:>6.2f} {len(vals):>3}  {bar}")

    print(f"\nResults: {summary_path}")


if __name__ == "__main__":
    main()
