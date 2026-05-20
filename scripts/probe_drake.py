"""Milestone Drake validity probe for a Demiurge checkpoint.

Extended with:
  --text-mode: controls how text conditioning is applied during sampling
    none          unconditional (text_emb=None)
    type-only     type-only prompts from ablation_prompts_type_only.json
    position-only position-only prompts from ablation_prompts_position_only.json
    full-cond     both type + position from ablation_prompts_full_cond.json
    held-out      descriptions from the held-out eval set (default ablation)
  --drake-workers: parallel CPU workers for Drake validation (default 1)

Usage:
    PYTHONPATH=/workspace/Demiurge/src python3 -u scripts/probe_drake.py \\
        --config configs/train/conditional_v6.yaml \\
        --checkpoint checkpoints/conditional_v6/latest.pt \\
        --n-scenes 200 \\
        --head ema \\
        --seed 42 \\
        --text-mode none \\
        --out artifacts/ablation_v6_none.json
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
import torch as _torch
_DATA_MEAN_XYZ   = _torch.tensor([-0.049867, +0.378790, -0.751155])
_DATA_STD_XYZ    = _torch.tensor([+0.419080, +0.457588, +0.127651])
_DATA_MEAN_SCALE = _torch.tensor([-0.038706, -0.038706, -0.038706])
_DATA_STD_SCALE  = _torch.tensor([+0.221619, +0.221619, +0.221619])


# Whitening constants -- must mirror train.py _DATA_MEAN_* / _DATA_STD_*
# Model output is in whitened space. Inverse: x_norm = x_white * std + mean.
import torch as _torch
_DATA_MEAN_XYZ   = _torch.tensor([-0.049867, +0.378790, -0.751155])  # (3,)
_DATA_STD_XYZ    = _torch.tensor([+0.419080, +0.457588, +0.127651])  # (3,) z_std is 3x smaller -- was the mismatch
_DATA_MEAN_SCALE = _torch.tensor([-0.038706, -0.038706, -0.038706])  # (3,)
_DATA_STD_SCALE  = _torch.tensor([+0.221619, +0.221619, +0.221619])  # (3,)

# ── Held-out description templates (same as probe_vlm.py) ─────────────────────
_HELD_OUT_TEMPLATES = [
    ("cluttered_pick", [
        "retrieve the mustard bottle from the cluttered central area",
        "pick the sugar box from among the scattered objects",
        "find and grasp the tomato soup can in the cluttered workspace",
        "grab the bleach cleanser from the crowded table",
        "pick out the banana from the cluttered arrangement",
    ]),
    ("obstacle_avoidance", [
        "reach the master chef can while avoiding the surrounding objects",
        "get the gelatin box from behind the obstacles",
        "retrieve the cube with obstacles on both sides",
        "pick the sphere while navigating around the tall boxes",
        "grasp the cylinder past the blocking objects",
    ]),
    ("tabletop_reach", [
        "place an object 32cm out and 10cm to the right",
        "place an object 28cm out and 8cm to the left",
        "place an object 40cm out and 15cm to the right",
        "place an object 20cm out and 5cm to the left",
        "place an object 35cm out and 12cm to the right",
    ]),
]


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_prompts(text_mode: str, artifacts_dir: Path) -> list[str] | None:
    """Return list of prompt strings or None for unconditional."""
    if text_mode == "none":
        return None

    mode_map = {
        "type-only":     "ablation_prompts_type_only.json",
        "position-only": "ablation_prompts_position_only.json",
        "full-cond":     "ablation_prompts_full_cond.json",
    }

    if text_mode in mode_map:
        path = artifacts_dir / mode_map[text_mode]
        if not path.exists():
            raise FileNotFoundError(
                f"Ablation prompt file not found: {path}\n"
                "Run scripts/build_ablation_prompts.py first."
            )
        data = json.loads(path.read_text())
        return [entry["prompt"] for entry in data]

    if text_mode == "held-out":
        prompts = []
        for _tmpl, descs in _HELD_OUT_TEMPLATES:
            prompts.extend(descs)
        return prompts

    raise ValueError(f"Unknown --text-mode: {text_mode!r}")


def _validate_worker(args_tuple: tuple) -> dict:
    """Top-level fn for Drake validation in a subprocess (must be picklable)."""
    st_dict, rrt_budget = args_tuple
    # Reconstruct SceneTensor from dict of tensors.
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Training config YAML")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    p.add_argument("--n-scenes", type=int, default=200)
    p.add_argument("--head", choices=["model", "ema"], default="ema")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--rrt-budget", type=float, default=2.0)
    p.add_argument(
        "--type-init", choices=["uniform", "zeros"], default="uniform",
        help="Type ID initialisation strategy",
    )
    p.add_argument(
        "--text-mode",
        choices=["none", "type-only", "position-only", "full-cond", "held-out"],
        default="none",
        help="Text conditioning regime for ablation study",
    )
    p.add_argument(
        "--drake-workers", type=int, default=1,
        help="Parallel CPU workers for Drake validation (1 = serial)",
    )
    p.add_argument(
        "--cfg-scale", type=float, default=1.0,
        help="CFG guidance scale: 0.0=unconditional, 1.0=pure conditional (default). "
             "For v7+ checkpoints trained with cfg_dropout>0, values in [0,5] give "
             "smooth monotone tradeoff between validity and text-following.",
    )
    p.add_argument("--presence-threshold", type=float, default=0.0,
                        help="Logit threshold for presence binarisation")
    p.add_argument("--out", default=None, help="JSON output path")
    p.add_argument(
        "--presence-threshold", type=float, default=-0.589,
        help=(
            "Logit threshold for the presence decode step. "
            "The training dataset has 24.4%% slot occupancy (mean 2.93 objects "
            "per 12-slot scene). The optimal logit threshold that reproduces this "
            "occupancy at inference is log(0.244/0.756) = -1.13, but empirically "
            "-0.589 matches the model output distribution at step 50k. "
            "The old default of 0.0 (sigmoid > 0.5) decoded only ~12%% of slots "
            "as present, causing false-sparse scenes and invalid Drake probes. "
            "Set to 0.0 to recover the legacy behaviour."
        ),
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts_dir = _ROOT / "artifacts"

    # ── Load model ────────────────────────────────────────────────────────────
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

    # ── Build DDIM sampler ───────────────────────────────────────────────────
    ddim_cfg = cfg.get("diffusion", {})
    T = int(ddim_cfg.get("T", 1000))
    ddim_steps = int(ddim_cfg.get("ddim_steps", 50))
    prediction_type = ddim_cfg.get("prediction_type", "epsilon")
    zero_terminal_snr = bool(ddim_cfg.get("zero_terminal_snr", False))
    schedule = CosineSchedule(T=T, zero_terminal_snr=zero_terminal_snr)
    sampler = DDIMSampler(schedule, n_steps=ddim_steps, prediction_type=prediction_type)

    # ── Text conditioning setup ───────────────────────────────────────────────
    prompts = _load_prompts(args.text_mode, artifacts_dir)
    text_emb: torch.Tensor | None = None

    if prompts is not None:
        emb_cache = _ROOT / "data" / "v1" / "text_embeddings.pt"
        encoder = TextEncoder(
            device=device,
            cache_path=str(emb_cache) if emb_cache.exists() else None,
        )
        print(f"Text mode: {args.text_mode} ({len(prompts)} prompts available)")
    else:
        encoder = None
        print("Text mode: none (unconditional)")

    # ── Sampling loop ─────────────────────────────────────────────────────────
    print(f"Sampling {args.n_scenes} scenes ...")
    t0 = time.monotonic()
    scenes_as_dicts: list[dict] = []
    remaining = args.n_scenes
    rng_seed = args.seed
    prompt_idx = 0

    with torch.no_grad():
        while remaining > 0:
            b = min(args.batch_size, remaining)

            # Pick text embedding for this batch.
            if prompts is not None and encoder is not None:
                prompt = prompts[prompt_idx % len(prompts)]
                prompt_idx += 1
                # encode_batch returns (1, D_TEXT); expand handled inside fn
                text_emb = encoder.encode_batch([prompt]).to(device)
            else:
                text_emb = None

            # Build sampling function with optional CFG interpolation.
            cfg_scale = args.cfg_scale
            if cfg_scale == 0.0 or text_emb is None:
                sampling_fn = model.conditional_sampling_fn(text_emb=None)
            elif cfg_scale == 1.0:
                sampling_fn = model.conditional_sampling_fn(text_emb=text_emb)
            else:
                _uncond_fn = model.conditional_sampling_fn(text_emb=None)
                _cond_fn = model.conditional_sampling_fn(text_emb=text_emb)
                def sampling_fn(x_t, type_ids_arg, t):
                    eps_u, logits_u = _uncond_fn(x_t, type_ids_arg, t)
                    eps_c, logits_c = _cond_fn(x_t, type_ids_arg, t)
                    return eps_u + cfg_scale * (eps_c - eps_u), logits_c
            x_cont, type_ids_batch = sampler.sample_with_types(
                sampling_fn, (b, N_MAX, 13),
                seed=rng_seed, device=device,
                type_init=args.type_init,
            )
            rng_seed += 1
            remaining -= b

            # Inverse whitening: x_norm = x_white * std + mean, then clamp to [-1,1].
            xyz = x_cont[:, :, :3] * _DATA_STD_XYZ.to(x_cont.device) + _DATA_MEAN_XYZ.to(x_cont.device)
            xyz = xyz.clamp(-1.0, 1.0)
            rot6d_pred = x_cont[:, :, 3:9]
            scale_pred = x_cont[:, :, 9:12] * _DATA_STD_SCALE.to(x_cont.device) + _DATA_MEAN_SCALE.to(x_cont.device)
            scale_pred = scale_pred.clamp(-1.0, 1.0)
            pres_bit = x_cont[:, :, 12]

            for i in range(b):
                pres_mask = pres_bit[i] > args.presence_threshold
                quats = rot6d_to_quat_wxyz(rot6d_pred[i])
                poses_raw = torch.cat([xyz[i], quats], dim=-1)
                st_norm = SceneTensor(
                    object_types=type_ids_batch[i].cpu(),
                    poses=poses_raw.cpu(),
                    scales=scale_pred[i].cpu(),
                    presence=pres_mask.cpu(),
                )
                # Pass normalized SceneTensor to dict. The _validate_worker
                # subprocess calls denormalize(bounds) before running Drake,
                # so physical conversion happens exactly once in the worker.
                scenes_as_dicts.append({
                    "object_types": st_norm.object_types,
                    "poses": st_norm.poses,
                    "scales": st_norm.scales,
                    "presence": st_norm.presence,
                })

    sample_elapsed = time.monotonic() - t0
    print(f"Sampling done in {sample_elapsed:.1f}s. Running Drake validation ...")

    # ── Drake validation (serial or parallel) ─────────────────────────────────
    t1 = time.monotonic()
    validate_args = [(sd, args.rrt_budget) for sd in scenes_as_dicts]
    reports: list[dict]

    if args.drake_workers > 1:
        with mp.Pool(processes=args.drake_workers) as pool:
            reports = pool.map(_validate_worker, validate_args)
    else:
        reports = [_validate_worker(a) for a in validate_args]

    drake_elapsed = time.monotonic() - t1
    total_elapsed = time.monotonic() - t0

    # ── Tally results ─────────────────────────────────────────────────────────
    accepted = sum(1 for r in reports if r["accepted"])
    rejection_reasons: dict[str, int] = {
        "interpenetration": 0,
        "ik_unreachable":   0,
        "rrt_failed":       0,
        "unstable":         0,
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

    validity_rate = accepted / args.n_scenes
    total_rejected = args.n_scenes - accepted

    print("\n=== Drake Probe Results ===")
    print(f"  Text mode:       {args.text_mode}")
    print(f"  Head:            {head_label}")
    print(f"  Step:            {step}")
    print(f"  N scenes:        {args.n_scenes}")
    print(f"  Accepted:        {accepted} ({100 * validity_rate:.1f}%)")
    print(f"  Rejected:        {total_rejected}")
    if total_rejected > 0:
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100 * count / total_rejected
                print(f"    {reason:20s}: {count:4d} ({pct:.1f}%)")
    print(f"  Sample elapsed:  {sample_elapsed:.1f}s")
    print(f"  Drake elapsed:   {drake_elapsed:.1f}s ({args.drake_workers} workers)")
    print(f"  Total elapsed:   {total_elapsed:.1f}s ({total_elapsed / args.n_scenes:.2f}s/scene)")
    print(f"  Validity rate:   {validity_rate:.4f}")

    result = {
        "step": step,
        "head": head_label,
        "checkpoint": str(ckpt_path),
        "text_mode": args.text_mode,
        "n_scenes": args.n_scenes,
        "accepted": accepted,
        "validity_rate": validity_rate,
        "rejection_reasons": rejection_reasons,
        "sample_elapsed_s": sample_elapsed,
        "drake_elapsed_s": drake_elapsed,
        "total_elapsed_s": total_elapsed,
        "seed": args.seed,
        "type_init": args.type_init,
        "drake_workers": args.drake_workers,
        "sampler": "sample_with_types",
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  Results written to {out_path}")


if __name__ == "__main__":
    main()
