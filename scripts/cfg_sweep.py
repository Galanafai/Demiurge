"""CFG (Classifier-Free Guidance) inference sweep on a Demiurge checkpoint.

Tests whether reducing text influence at inference recovers Drake validity.
For each cfg_scale in [0.0, 0.25, 0.5, 0.75, 1.0]:
  eps_guided = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

cfg_scale=0.0 -> pure unconditional (should match --text-mode none baseline)
cfg_scale=1.0 -> pure conditional (should match --text-mode full-cond baseline)
cfg_scale>1.0 -> over-conditioned (not tested here)

Usage:
    PYTHONPATH=/workspace/Demiurge/src python3 -u scripts/cfg_sweep.py \\
        --config configs/train/conditional_v6.yaml \\
        --checkpoint checkpoints/conditional_v6/latest.pt \\
        --n-scenes 100 \\
        --prompt-file artifacts/ablation_prompts_full_cond.json \\
        --out artifacts/cfg_sweep_v6.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from model.denoiser import DenoiserConfig, SceneDenoiser  # noqa: E402
from model.rotations import rot6d_to_quat_wxyz  # noqa: E402
from model.schedule import CosineSchedule, DDIMSampler  # noqa: E402
from model.text_encoder import TextEncoder  # noqa: E402
from scene.schema import N_MAX, SceneTensor, WorkspaceBounds  # noqa: E402
from validator.core import SceneValidator  # noqa: E402

CFG_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _validate_worker(args_tuple: tuple) -> dict:
    """Drake validation worker -- must be top-level for pickling."""
    st_dict, rrt_budget = args_tuple
    st = SceneTensor(
        object_types=st_dict["object_types"],
        poses=st_dict["poses"],
        scales=st_dict["scales"],
        presence=st_dict["presence"],
    )
    bounds = WorkspaceBounds.default()
    st_phys = st.denormalize(bounds)
    validator = SceneValidator(rrt_budget_s=rrt_budget)
    rpt = validator.validate(st_phys)
    return {
        "accepted": rpt.accepted,
        "no_interpenetration": rpt.no_interpenetration,
        "ik_reachable": rpt.ik_reachable,
        "rrt_solvable": rpt.rrt_solvable,
        "stable_rest": rpt.stable_rest,
    }


def _make_cfg_fn(
    model: SceneDenoiser,
    text_emb: torch.Tensor,
    cfg_scale: float,
) -> object:
    """Return a sampling_fn that applies CFG at each denoising step.

    eps_guided = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    cfg_scale=0.0: pure unconditional
    cfg_scale=1.0: pure conditional
    """
    uncond_fn = model.conditional_sampling_fn(text_emb=None)
    cond_fn = model.conditional_sampling_fn(text_emb=text_emb)

    if cfg_scale == 0.0:
        return uncond_fn
    if cfg_scale == 1.0:
        return cond_fn

    def _cfg_fn(x_t: torch.Tensor, type_ids: torch.Tensor, t: torch.Tensor):
        eps_uncond, logits_uncond = uncond_fn(x_t, type_ids, t)
        eps_cond, logits_cond = cond_fn(x_t, type_ids, t)
        eps_guided = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        # Use conditional type logits -- type prediction is downstream of cross-attn
        return eps_guided, logits_cond

    return _cfg_fn


def run_one_scale(
    model: SceneDenoiser,
    sampler: DDIMSampler,
    text_emb: torch.Tensor,
    cfg_scale: float,
    n_scenes: int,
    batch_size: int,
    seed: int,
    rrt_budget: float,
    drake_workers: int,
    device: torch.device,
) -> dict:
    """Sample n_scenes at a given cfg_scale and validate with Drake."""
    sampling_fn = _make_cfg_fn(model, text_emb, cfg_scale)

    scenes_as_dicts: list[dict] = []
    remaining = n_scenes
    rng_seed = seed

    t_sample = time.monotonic()
    with torch.no_grad():
        while remaining > 0:
            b = min(batch_size, remaining)
            x_cont, type_ids_batch = sampler.sample_with_types(
                sampling_fn, (b, N_MAX, 13),
                seed=rng_seed, device=device,
                type_init="uniform",
            )
            rng_seed += 1
            remaining -= b

            xyz = x_cont[:, :, :3].clamp(-1.0, 1.0)
            rot6d_pred = x_cont[:, :, 3:9]
            scale_pred = x_cont[:, :, 9:12].clamp(-1.0, 1.0)
            pres_bit = x_cont[:, :, 12]

            for i in range(b):
                pres_mask = pres_bit[i] > 0.0
                quats = rot6d_to_quat_wxyz(rot6d_pred[i])
                poses_raw = torch.cat([xyz[i], quats], dim=-1)
                st_norm = SceneTensor(
                    object_types=type_ids_batch[i].cpu(),
                    poses=poses_raw.cpu(),
                    scales=scale_pred[i].cpu(),
                    presence=pres_mask.cpu(),
                )
                scenes_as_dicts.append({
                    "object_types": st_norm.object_types,
                    "poses": st_norm.poses,
                    "scales": st_norm.scales,
                    "presence": st_norm.presence,
                })
    sample_elapsed = time.monotonic() - t_sample

    # Drake validation
    t_drake = time.monotonic()
    validate_args = [(sd, rrt_budget) for sd in scenes_as_dicts]
    if drake_workers > 1:
        with mp.Pool(processes=drake_workers) as pool:
            reports = pool.map(_validate_worker, validate_args)
    else:
        reports = [_validate_worker(a) for a in validate_args]
    drake_elapsed = time.monotonic() - t_drake

    accepted = sum(1 for r in reports if r["accepted"])
    rejection_reasons: dict[str, int] = {
        "interpenetration": 0,
        "ik_unreachable": 0,
        "rrt_failed": 0,
        "unstable": 0,
    }
    for r in reports:
        if not r["accepted"]:
            if not r["no_interpenetration"]:
                rejection_reasons["interpenetration"] += 1
            elif not r["ik_reachable"]:
                rejection_reasons["ik_unreachable"] += 1
            elif not r["rrt_solvable"]:
                rejection_reasons["rrt_failed"] += 1
            elif not r["stable_rest"]:
                rejection_reasons["unstable"] += 1

    validity_rate = accepted / n_scenes
    print(
        f"  cfg={cfg_scale:.2f}  accepted={accepted}/{n_scenes} "
        f"({100 * validity_rate:.1f}%)  "
        f"sample={sample_elapsed:.1f}s  drake={drake_elapsed:.1f}s"
    )
    return {
        "cfg_scale": cfg_scale,
        "n_scenes": n_scenes,
        "accepted": accepted,
        "validity_rate": validity_rate,
        "rejection_reasons": rejection_reasons,
        "sample_elapsed_s": sample_elapsed,
        "drake_elapsed_s": drake_elapsed,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-scenes", type=int, default=100)
    p.add_argument("--head", choices=["model", "ema"], default="ema")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--rrt-budget", type=float, default=2.0)
    p.add_argument("--drake-workers", type=int, default=4)
    p.add_argument(
        "--prompt-file",
        default=str(_ROOT / "artifacts" / "ablation_prompts_full_cond.json"),
        help="JSON file with prompt list to embed (uses first prompt as fixed conditioning)",
    )
    p.add_argument(
        "--cfg-scales",
        nargs="+",
        type=float,
        default=CFG_SCALES,
        help="List of CFG scales to sweep",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ──────────────────────────────────────────────────────────────────
    cfg = _load_yaml(args.config)
    mcfg_raw = cfg.get("model", {})
    dcfg = DenoiserConfig(
        n_layers=mcfg_raw.get("n_layers", 6),
        d_model=mcfg_raw.get("d_model", 256),
        n_heads=mcfg_raw.get("n_heads", 8),
        ffn_mult=mcfg_raw.get("ffn_mult", 4),
        dropout=mcfg_raw.get("dropout", 0.1),
        use_type_grad_isolation=mcfg_raw.get("use_type_grad_isolation", False),
    )
    model = SceneDenoiser(dcfg).to(device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if args.head == "ema":
        state = ckpt.get("ema_state", ckpt.get("model_state"))
        head_label = "EMA"
    else:
        state = ckpt.get("model_state", ckpt)
        head_label = "model"
    model.load_state_dict(state, strict=False)
    model.eval()
    step = ckpt.get("step", "?")
    print(f"Loaded {head_label} from {ckpt_path} (step={step})")

    # ── Sampler ────────────────────────────────────────────────────────────────
    ddim_cfg = cfg.get("diffusion", {})
    T = int(ddim_cfg.get("T", 1000))
    ddim_steps = int(ddim_cfg.get("ddim_steps", 50))
    schedule = CosineSchedule(T=T)
    sampler = DDIMSampler(schedule, n_steps=ddim_steps)

    # ── Text embedding (use SINGLE representative prompt for all scales) ───────
    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_file}\n"
            "Run scripts/build_ablation_prompts.py first."
        )
    prompts = json.loads(prompt_file.read_text())
    # Use first 5 prompts averaged -- representative conditioning signal
    sample_prompts = [p["prompt"] for p in prompts[:5]]
    emb_cache = _ROOT / "data" / "v1" / "text_embeddings.pt"
    encoder = TextEncoder(
        device=device,
        cache_path=str(emb_cache) if emb_cache.exists() else None,
    )
    # Encode and average to get a single representative embedding
    embs = encoder.encode_batch(sample_prompts).to(device)  # (5, 384)
    text_emb = embs.mean(dim=0, keepdim=True)  # (1, 384)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)  # re-normalize
    print(f"Text emb shape: {text_emb.shape}  norm: {text_emb.norm():.4f}")

    # ── Sweep ──────────────────────────────────────────────────────────────────
    print(f"\nCFG sweep: {args.cfg_scales}")
    print(f"Scenes per scale: {args.n_scenes}  Drake workers: {args.drake_workers}")
    results = []

    for cfg_scale in args.cfg_scales:
        r = run_one_scale(
            model=model,
            sampler=sampler,
            text_emb=text_emb,
            cfg_scale=cfg_scale,
            n_scenes=args.n_scenes,
            batch_size=args.batch_size,
            seed=args.seed,
            rrt_budget=args.rrt_budget,
            drake_workers=args.drake_workers,
            device=device,
        )
        results.append(r)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== CFG Sweep Summary ===")
    print(f"  {'cfg_scale':>10}  {'validity':>10}  {'accepted':>10}")
    for r in results:
        print(
            f"  {r['cfg_scale']:>10.2f}  "
            f"  {r['validity_rate'] * 100:>8.1f}%  "
            f"  {r['accepted']:>4}/{r['n_scenes']}"
        )

    # Detect trend: is recovery monotone as cfg_scale decreases?
    vrs = [r["validity_rate"] for r in results]
    uncond_vr = vrs[0]  # cfg=0.0
    cond_vr = vrs[-1]   # cfg=1.0
    recovery_ratio = uncond_vr / max(cond_vr, 1e-6)
    if recovery_ratio > 2.0:
        conclusion = (
            f"OUTCOME A: CFG recovery detected. "
            f"Unconditional ({100 * uncond_vr:.1f}%) >> Conditional ({100 * cond_vr:.1f}%). "
            f"Text pathway degrades geometry. Use low CFG scale at inference."
        )
    else:
        conclusion = (
            f"OUTCOME B: No CFG recovery. "
            f"Uncond={100 * uncond_vr:.1f}%  Cond={100 * cond_vr:.1f}%  ratio={recovery_ratio:.2f}. "
            f"Text pathway fully poisoning geometry -- retrain required."
        )
    print(f"\n  Conclusion: {conclusion}")

    output = {
        "checkpoint": str(ckpt_path),
        "step": step,
        "head": head_label,
        "prompt_file": str(prompt_file),
        "n_scenes": args.n_scenes,
        "cfg_scales_tested": args.cfg_scales,
        "results": results,
        "uncond_validity": uncond_vr,
        "cond_validity": cond_vr,
        "recovery_ratio": recovery_ratio,
        "conclusion": conclusion,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"  Results written to {out_path}")


if __name__ == "__main__":
    main()
