"""Universal Guidance sweep script.

Generates scenes with the v7 model under Universal Guidance at multiple
guidance scales, validates via Drake, computes diversity, and logs
per-config results to a JSONL file for Pareto plotting.

Usage:
    python3 -u scripts/run_universal_guidance_sweep.py \\
        --model-checkpoint checkpoints/conditional_v7/latest.pt \\
        --cfg-scale 1.0 \\
        --guidance-scales 0.0 0.5 1.0 2.0 4.0 8.0 \\
        --seeds 42 123 456 \\
        --n-prompts 200 \\
        --n-per-prompt 5 \\
        --output-dir artifacts/ug_sweep_v1/ \\
        --budget-cap 8.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.reader import ShardReader
from guidance.energy import pairwise_overlap_energy
from model.denoiser import DenoiserConfig, SceneDenoiser
from model.rotations import rot6d_to_quat_wxyz
from model.schedule import CosineSchedule, DDIMSampler
from scene.schema import SceneTensor, WorkspaceBounds
from validator.core import SceneValidator

DDIM_STEPS = 50
N_MAX = 8


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(ckpt_path: Path, device: torch.device) -> SceneDenoiser:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    cfg = DenoiserConfig(
        n_layers=arch["n_layers"], d_model=arch["d_model"],
        n_heads=arch["n_heads"], ffn_mult=arch["ffn_mult"], dropout=0.0,
        use_type_grad_isolation=arch.get("use_type_grad_isolation", False),
    )
    model = SceneDenoiser(cfg).to(device)
    model.load_state_dict(ckpt["ema_state"])
    model.eval()
    print(f"Loaded EMA from {ckpt_path.name} (step={ckpt['step']})")
    return model


# ---------------------------------------------------------------------------
# Scene decoding
# ---------------------------------------------------------------------------


def decode_scene(
    x_cont_i: torch.Tensor,
    type_ids_i: torch.Tensor,
    bounds: WorkspaceBounds,
) -> SceneTensor:
    presence = x_cont_i[:, 12] > 0.0
    quats = rot6d_to_quat_wxyz(x_cont_i[:, 3:9])
    poses = torch.cat([x_cont_i[:, :3].clamp(-1, 1), quats], dim=-1)
    return SceneTensor(
        object_types=type_ids_i.cpu(),
        poses=poses.cpu(),
        scales=x_cont_i[:, 9:12].clamp(-1, 1).cpu(),
        presence=presence.cpu(),
    )


# ---------------------------------------------------------------------------
# Diversity metric: mean pairwise L2 distance between scene centroids
# ---------------------------------------------------------------------------


def compute_diversity(scenes: list[torch.Tensor]) -> float:
    """Mean pairwise distance between scene position tensors.

    Args:
        scenes: List of (N, 3) position tensors (present objects only).

    Returns:
        Mean pairwise L2 distance across scene pairs. Higher = more diverse.
    """
    if len(scenes) < 2:
        return 0.0
    # Use scene centroid as scene embedding
    centroids = [s.mean(dim=0) if s.shape[0] > 0 else torch.zeros(3) for s in scenes]
    centroids_t = torch.stack(centroids)  # (K, 3)
    K = centroids_t.shape[0]
    total = 0.0
    count = 0
    for i in range(K):
        for j in range(i + 1, K):
            total += (centroids_t[i] - centroids_t[j]).norm().item()
            count += 1
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Per-config run
# ---------------------------------------------------------------------------


def run_config(
    model: SceneDenoiser,
    sampler: DDIMSampler,
    held_out: list[dict],
    text_cache: dict[str, torch.Tensor],
    validator: SceneValidator,
    bounds: WorkspaceBounds,
    device: torch.device,
    cfg_scale: float,
    guidance_scale: float,
    n_per_prompt: int,
    seed: int,
) -> dict:
    """Run one (guidance_scale, seed) config and return metrics."""
    t0 = time.monotonic()
    drake_accepted = 0
    drake_total = 0
    n_empty = 0
    scene_positions: list[torch.Tensor] = []
    all_energies: list[float] = []
    records: list[dict] = []

    for d_idx, desc_info in enumerate(held_out):
        desc = desc_info["description"]
        desc_id = desc_info["desc_id"]

        key = hashlib.sha256(desc.encode()).hexdigest()
        if key not in text_cache:
            continue
        text_emb = text_cache[key].unsqueeze(0).to(device)

        per_prompt_seed = (seed * 10007 + d_idx) & 0xFFFFFFFF

        if cfg_scale == 1.0:
            fn = model.conditional_sampling_fn(text_emb.expand(n_per_prompt, -1))
        else:
            fn_uncond = model.conditional_sampling_fn(text_emb=None)
            fn_cond = model.conditional_sampling_fn(text_emb.expand(n_per_prompt, -1))
            _cfg = cfg_scale  # capture

            def fn(x_t, tids, t, _fn_u=fn_uncond, _fn_c=fn_cond, _s=_cfg):
                eu, lu = _fn_u(x_t, tids, t)
                ec, lc = _fn_c(x_t, tids, t)
                return eu + _s * (ec - eu), lc

        with torch.no_grad():
            x_cont, type_ids = sampler.sample_with_universal_guidance(
                fn,
                (n_per_prompt, N_MAX, 13),
                seed=per_prompt_seed,
                device=device,
                type_init="uniform",
                guidance_scale=guidance_scale,
            )

        for s_idx in range(n_per_prompt):
            st_norm = decode_scene(x_cont[s_idx], type_ids[s_idx], bounds)
            st_phys = st_norm.denormalize(bounds)

            pres = st_norm.presence.bool()
            if not pres.any():
                n_empty += 1
            else:
                scene_positions.append(st_norm.poses[pres, :3])

                # Per-scene energy on normalised positions
                xi = x_cont[s_idx : s_idx + 1]
                ti = type_ids[s_idx : s_idx + 1]
                e = pairwise_overlap_energy(xi, ti).item()
                all_energies.append(e)

            rpt = validator.validate(st_phys)
            drake_total += 1
            if rpt.accepted:
                drake_accepted += 1

            records.append({
                "desc_id": desc_id,
                "description": desc,
                "sample_idx": s_idx,
                "drake_accepted": rpt.accepted,
                "energy": all_energies[-1] if all_energies else 0.0,
                "n_present": int(pres.sum().item()) if not (not pres.any()) else 0,
                "type_ids": type_ids[s_idx].cpu().tolist(),
            })

        if (d_idx + 1) % 25 == 0:
            elapsed = time.monotonic() - t0
            rate = drake_accepted / max(1, drake_total)
            print(
                f"  [{d_idx+1}/{len(held_out)}] "
                f"drake={drake_accepted}/{drake_total} ({rate*100:.1f}%) "
                f"elapsed={elapsed:.0f}s"
            )

    diversity = compute_diversity(scene_positions)
    mean_energy = sum(all_energies) / max(1, len(all_energies))
    elapsed = time.monotonic() - t0

    return {
        "guidance_scale": guidance_scale,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "n_prompts": len(held_out),
        "n_scenes": drake_total,
        "n_empty": n_empty,
        "drake_accepted": drake_accepted,
        "drake_validity_rate": drake_accepted / max(1, drake_total),
        "diversity": diversity,
        "mean_energy": mean_energy,
        "elapsed_s": elapsed,
        "records": records,
    }


# ---------------------------------------------------------------------------
# Held-out set
# ---------------------------------------------------------------------------


def build_held_out(data_dir: Path, n_prompts: int) -> list[dict]:
    per_template: dict[str, list[dict]] = defaultdict(list)
    for scene, desc, report, sdf in ShardReader(data_dir):
        tmpl = report.get("task_family", "unknown")
        per_template[tmpl].append({"description": desc, "template": tmpl})

    rng = torch.Generator()
    rng.manual_seed(42)
    total_held = sum(len(v) for v in per_template.values())

    sampled: list[dict] = []
    for tmpl in sorted(per_template):
        items = per_template[tmpl][-100:]  # last 100 per template = held-out
        n_take = max(1, round(n_prompts * len(items) / total_held))
        idx = torch.randperm(len(items), generator=rng).tolist()[:n_take]
        sampled.extend(items[i] for i in idx)

    seen: set[str] = set()
    deduped: list[dict] = []
    for item in sampled:
        key = hashlib.sha256(item["description"].encode()).hexdigest()[:16]
        if key not in seen:
            seen.add(key)
            item["desc_id"] = key
            deduped.append(item)

    return deduped[:n_prompts]


def load_text_cache(data_dir: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(data_dir / "text_embeddings.pt", weights_only=True)
    return {k: v.float().cpu() for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-checkpoint", required=True)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--guidance-scales", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--n-prompts", type=int, default=200)
    p.add_argument("--n-per-prompt", type=int, default=5)
    p.add_argument("--rrt-budget", type=float, default=2.0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-dir", default="data/v1")
    p.add_argument("--budget-cap", type=float, default=8.0,
                   help="Max wall-clock hours before halting.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(args.model_checkpoint), device)
    schedule = CosineSchedule(T=1000)
    sampler = DDIMSampler(schedule, n_steps=DDIM_STEPS)
    bounds = WorkspaceBounds.default()
    validator = SceneValidator(rrt_budget_s=args.rrt_budget)

    data_dir = Path(args.data_dir)
    held_out = build_held_out(data_dir, args.n_prompts)
    text_cache = load_text_cache(data_dir)
    print(f"Held-out: {len(held_out)} descriptions")

    # Build encoder for text cache (already precomputed, just verify)
    total_configs = len(args.guidance_scales) * len(args.seeds)
    print(f"Configs: {len(args.guidance_scales)} scales x {len(args.seeds)} seeds = {total_configs}")
    print(f"Total scenes: ~{total_configs * len(held_out) * args.n_per_prompt}")
    print()

    summary_path = out_dir / "summary.jsonl"
    run_idx = 0
    t_global = time.monotonic()

    for guidance_scale in args.guidance_scales:
        for seed in args.seeds:
            run_idx += 1
            elapsed_h = (time.monotonic() - t_global) / 3600
            if elapsed_h > args.budget_cap:
                print(f"HALT: budget cap {args.budget_cap}h reached at run {run_idx}/{total_configs}")
                break

            print(f"=== Run {run_idx}/{total_configs}: guidance_scale={guidance_scale} seed={seed} ===")

            result = run_config(
                model=model,
                sampler=sampler,
                held_out=held_out,
                text_cache=text_cache,
                validator=validator,
                bounds=bounds,
                device=device,
                cfg_scale=args.cfg_scale,
                guidance_scale=guidance_scale,
                n_per_prompt=args.n_per_prompt,
                seed=seed,
            )

            # Save full record per run
            run_path = out_dir / f"run_gs{guidance_scale:.1f}_s{seed}.json"
            records = result.pop("records")
            with open(run_path, "w") as f:
                json.dump({"config": result, "records": records}, f)
            result["records"] = records  # restore for summary

            # Append summary row
            summary_row = {k: v for k, v in result.items() if k != "records"}
            with open(summary_path, "a") as f:
                f.write(json.dumps(summary_row) + "\n")

            print(
                f"  --> Drake {result['drake_validity_rate']*100:.1f}%  "
                f"diversity={result['diversity']:.4f}  "
                f"energy={result['mean_energy']:.4f}  "
                f"time={result['elapsed_s']:.0f}s"
            )
            print()

    print(f"Sweep complete. Results in {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
