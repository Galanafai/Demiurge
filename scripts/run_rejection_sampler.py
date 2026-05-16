"""Rejection sampling baseline for Pareto comparison.

Runs the v7 model with standard sampling (no guidance) and retains
only Drake-valid scenes. Reports: acceptance rate, mean attempts per
valid scene, wall-clock cost, and diversity of accepted scenes.

Usage:
    python3 -u scripts/run_rejection_sampler.py \\
        --model-checkpoint checkpoints/conditional_v7/latest.pt \\
        --cfg-scale 1.0 \\
        --n-prompts 200 \\
        --target-per-prompt 5 \\
        --max-attempts 50 \\
        --seeds 42 123 456 \\
        --out artifacts/rejection_baseline.json
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
from model.denoiser import N_MAX, DenoiserConfig, SceneDenoiser
from model.rotations import rot6d_to_quat_wxyz
from model.schedule import CosineSchedule, DDIMSampler
from scene.schema import SceneTensor, WorkspaceBounds
from validator.core import SceneValidator

DDIM_STEPS = 50



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
        items = per_template[tmpl][-100:]
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


def compute_diversity(positions_list: list[torch.Tensor]) -> float:
    if len(positions_list) < 2:
        return 0.0
    centroids = [p.mean(dim=0) if p.shape[0] > 0 else torch.zeros(3) for p in positions_list]
    ct = torch.stack(centroids)
    K = ct.shape[0]
    total, count = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            total += (ct[i] - ct[j]).norm().item()
            count += 1
    return total / max(count, 1)


def run_seed(
    model: SceneDenoiser,
    sampler: DDIMSampler,
    held_out: list[dict],
    text_cache: dict[str, torch.Tensor],
    validator: SceneValidator,
    bounds: WorkspaceBounds,
    device: torch.device,
    cfg_scale: float,
    target_per_prompt: int,
    max_attempts: int,
    seed: int,
) -> dict:
    t0 = time.monotonic()
    total_attempts = 0
    total_accepted = 0
    total_prompts = 0
    attempts_to_first: list[int] = []   # attempts until first valid scene
    scene_positions: list[torch.Tensor] = []

    for d_idx, desc_info in enumerate(held_out):
        desc = desc_info["description"]
        key = hashlib.sha256(desc.encode()).hexdigest()
        if key not in text_cache:
            continue

        text_emb = text_cache[key].unsqueeze(0).to(device)
        total_prompts += 1
        prompt_accepted = 0
        first_accept_attempt = None

        for attempt in range(max_attempts):
            per_seed = (seed * 99991 + d_idx * 1009 + attempt) & 0xFFFFFFFF

            # Sample a single scene
            if cfg_scale == 1.0:
                fn = model.conditional_sampling_fn(text_emb)
            else:
                fn_u = model.conditional_sampling_fn(text_emb=None)
                fn_c = model.conditional_sampling_fn(text_emb)
                _s = cfg_scale
                def fn(x, t_ids, t, _u=fn_u, _c=fn_c, _sc=_s):
                    eu, lu = _u(x, t_ids, t)
                    ec, lc = _c(x, t_ids, t)
                    return eu + _sc * (ec - eu), lc

            with torch.no_grad():
                x_cont, type_ids = sampler.sample_with_types(
                    fn, (1, N_MAX, 13),
                    seed=per_seed, device=device, type_init="uniform",
                )

            st_norm = decode_scene(x_cont[0], type_ids[0], bounds)
            st_phys = st_norm.denormalize(bounds)
            total_attempts += 1

            rpt = validator.validate(st_phys)
            if rpt.accepted:
                total_accepted += 1
                prompt_accepted += 1
                if first_accept_attempt is None:
                    first_accept_attempt = attempt + 1
                pres = st_norm.presence.bool()
                if pres.any():
                    scene_positions.append(st_norm.poses[pres, :3])
                if prompt_accepted >= target_per_prompt:
                    break

        if first_accept_attempt is not None:
            attempts_to_first.append(first_accept_attempt)

        if (d_idx + 1) % 25 == 0:
            elapsed = time.monotonic() - t0
            rate = total_accepted / max(1, total_attempts)
            print(
                f"  [{d_idx+1}/{len(held_out)}] "
                f"accepted={total_accepted}/{total_attempts} ({rate*100:.1f}%) "
                f"elapsed={elapsed:.0f}s"
            )

    elapsed = time.monotonic() - t0
    diversity = compute_diversity(scene_positions)
    mean_attempts = sum(attempts_to_first) / max(1, len(attempts_to_first))

    return {
        "seed": seed,
        "cfg_scale": cfg_scale,
        "n_prompts": total_prompts,
        "total_attempts": total_attempts,
        "total_accepted": total_accepted,
        "acceptance_rate": total_accepted / max(1, total_attempts),
        "mean_attempts_to_first": mean_attempts,
        "diversity": diversity,
        "elapsed_s": elapsed,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-checkpoint", required=True)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--n-prompts", type=int, default=200)
    p.add_argument("--target-per-prompt", type=int, default=5)
    p.add_argument("--max-attempts", type=int, default=50)
    p.add_argument("--rrt-budget", type=float, default=2.0)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--data-dir", default="data/v1")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(Path(args.model_checkpoint), device)
    schedule = CosineSchedule(T=1000)
    sampler = DDIMSampler(schedule, n_steps=DDIM_STEPS)
    bounds = WorkspaceBounds.default()
    validator = SceneValidator(rrt_budget_s=args.rrt_budget)

    data_dir = Path(args.data_dir)
    held_out = build_held_out(data_dir, args.n_prompts)
    text_cache = load_text_cache(data_dir)
    print(f"Held-out: {len(held_out)} descriptions")

    all_results = []
    for seed in args.seeds:
        print(f"\n=== Rejection sampling seed={seed} ===")
        result = run_seed(
            model=model, sampler=sampler, held_out=held_out,
            text_cache=text_cache, validator=validator, bounds=bounds,
            device=device, cfg_scale=args.cfg_scale,
            target_per_prompt=args.target_per_prompt,
            max_attempts=args.max_attempts,
            seed=seed,
        )
        all_results.append(result)
        print(
            f"  --> acceptance={result['acceptance_rate']*100:.1f}%  "
            f"mean_attempts={result['mean_attempts_to_first']:.1f}  "
            f"diversity={result['diversity']:.4f}"
        )

    # Aggregate across seeds
    rates = [r["acceptance_rate"] for r in all_results]
    diversities = [r["diversity"] for r in all_results]
    attempts = [r["mean_attempts_to_first"] for r in all_results]
    agg = {
        "method": "rejection_sampling",
        "cfg_scale": args.cfg_scale,
        "seeds": args.seeds,
        "acceptance_rate_mean": sum(rates) / len(rates),
        "acceptance_rate_std": (sum((r - sum(rates)/len(rates))**2 for r in rates) / max(1, len(rates)-1))**0.5,
        "diversity_mean": sum(diversities) / len(diversities),
        "mean_attempts_mean": sum(attempts) / len(attempts),
        "per_seed": all_results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(agg, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Acceptance rate: {agg['acceptance_rate_mean']*100:.1f}% +/- {agg['acceptance_rate_std']*100:.1f}%")
    print(f"Diversity: {agg['diversity_mean']:.4f}")
    print(f"Mean attempts to first valid: {agg['mean_attempts_mean']:.1f}")


if __name__ == "__main__":
    main()
