"""Milestone Drake validity probe for a Demiurge checkpoint.

Replaces the inline run_validation() call removed from the training loop.
Loads a checkpoint, samples N scenes via DDIM, runs Drake validation,
prints validity rate and rejection breakdown.

Usage:
    PYTHONPATH=/workspace/Demiurge/src python3 scripts/probe_drake.py \
        --config configs/train/conditional_v4.yaml \
        --checkpoint checkpoints/conditional_v4/latest.pt \
        --n-scenes 500 \
        --head ema \
        --seed 42 \
        --out artifacts/probe_v4_step50k.json
"""
from __future__ import annotations

import argparse
import json
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
from scene.schema import N_MAX, SceneTensor, WorkspaceBounds  # noqa: E402
from validator.core import SceneValidator  # noqa: E402


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Training config YAML (for model arch)")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    p.add_argument("--n-scenes", type=int, default=500)
    p.add_argument("--head", choices=["model", "ema"], default="ema",
                   help="Which checkpoint head to use for sampling")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--rrt-budget", type=float, default=2.0,
                   help="Seconds per RRT attempt in Drake validator")
    p.add_argument("--type-init", choices=["uniform", "zeros"], default="uniform",
                   help="Type ID initialisation: 'uniform' (fixed sampler) or "
                        "'zeros' (legacy broken behavior for regression comparison)")
    p.add_argument("--out", default=None, help="Optional JSON output path")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bounds = WorkspaceBounds.default()

    cfg = _load_yaml(args.config)
    mcfg_raw = cfg.get("model", {})
    dcfg = DenoiserConfig(
        n_layers=mcfg_raw.get("n_layers", 6),
        d_model=mcfg_raw.get("d_model", 256),
        n_heads=mcfg_raw.get("n_heads", 8),
        ffn_mult=mcfg_raw.get("ffn_mult", 4),
        dropout=mcfg_raw.get("dropout", 0.1),
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
    print(f"Loaded {head_label} head from {ckpt_path} (step={step})")

    ddim_cfg = cfg.get("diffusion", {})
    T = int(ddim_cfg.get("T", 1000))
    ddim_steps = int(ddim_cfg.get("ddim_steps", 50))
    schedule = CosineSchedule(T=T)
    sampler = DDIMSampler(schedule, n_steps=ddim_steps)
    # Use the type-aware sampling fn. noise_prediction_fn is the legacy broken
    # variant (type_ids frozen at zero) -- kept only for regression comparison.
    sampling_fn = model.conditional_sampling_fn(text_emb=None)

    validator = SceneValidator(rrt_budget_s=args.rrt_budget)

    print(f"Probing {args.n_scenes} scenes (head={head_label}, seed={args.seed}) ...")
    t0 = time.monotonic()

    accepted = 0
    rejection_reasons: dict[str, int] = {
        "interpenetration": 0,
        "ik_unreachable": 0,
        "rrt_failed": 0,
        "unstable": 0,
    }
    remaining = args.n_scenes
    rng_seed = args.seed

    with torch.no_grad():
        while remaining > 0:
            b = min(args.batch_size, remaining)
            x_cont, type_ids_batch = sampler.sample_with_types(
                sampling_fn, (b, N_MAX, 13),
                seed=rng_seed, device=device,
                type_init=args.type_init,
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
                st = st_norm.denormalize(bounds)
                rpt = validator.validate(st)
                if rpt.accepted:
                    accepted += 1
                else:
                    if not rpt.no_interpenetration:
                        rejection_reasons["interpenetration"] += 1
                    elif not rpt.ik_reachable:
                        rejection_reasons["ik_unreachable"] += 1
                    elif not rpt.rrt_solvable:
                        rejection_reasons["rrt_failed"] += 1
                    elif not rpt.stable_rest:
                        rejection_reasons["unstable"] += 1

    elapsed = time.monotonic() - t0
    validity_rate = accepted / args.n_scenes
    total_rejected = args.n_scenes - accepted

    print("\n=== Drake Probe Results ===")
    print(f"  Head:            {head_label}")
    print(f"  Step:            {step}")
    print(f"  N scenes:        {args.n_scenes}")
    print(f"  Accepted:        {accepted} ({100*validity_rate:.1f}%)")
    print(f"  Rejected:        {total_rejected}")
    if total_rejected > 0:
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"    {reason:20s}: {count:4d} ({100*count/total_rejected:.1f}%)")
    print(f"  Elapsed:         {elapsed:.1f}s ({elapsed/args.n_scenes:.2f}s/scene)")
    print(f"  Validity rate:   {validity_rate:.4f}")

    result = {
        "step": step,
        "head": head_label,
        "checkpoint": str(ckpt_path),
        "n_scenes": args.n_scenes,
        "accepted": accepted,
        "validity_rate": validity_rate,
        "rejection_reasons": rejection_reasons,
        "elapsed_s": elapsed,
        "seed": args.seed,
        "type_init": args.type_init,
        "sampler": "sample_with_types",
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  Results written to {out_path}")

    return result


if __name__ == "__main__":
    main()
