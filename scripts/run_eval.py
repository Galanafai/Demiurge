"""Full evaluation harness for Demiurge samplers.

Runs 8 scenes per held-out prompt x 3 seeds for each sampler config
and persists results to artifacts/eval_v1/<sampler_name>.json.

Usage:
    python3 scripts/run_eval.py \
        --suite data/eval_suite_v1/eval_suite.jsonl \
        --checkpoint-v7 checkpoints/conditional_v7/latest.pt \
        --checkpoint-v8c checkpoints/conditional_v8c_warminit/latest.pt \
        --ug-scale 0.5 \
        --out-dir artifacts/eval_v1 \
        --seeds 42 123 456 \
        --n-per-prompt 8 \
        --drake-workers 4 \
        --rrt-budget 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.reader import ShardReader  # noqa: E402
from eval.downstream import DownstreamEvaluator  # noqa: E402
from eval.metrics import diversity, raw_validity_rate  # noqa: E402
from model.denoiser import N_MAX, DenoiserConfig, SceneDenoiser  # noqa: E402
from model.rotations import rot6d_to_quat_wxyz  # noqa: E402
from model.schedule import CosineSchedule, DDIMSampler  # noqa: E402
from scene.schema import SceneTensor, WorkspaceBounds  # noqa: E402
from validator.core import SceneValidator  # noqa: E402

DDIM_STEPS = 50


# ---------------------------------------------------------------------------
# Model loading (same as sweep script)
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
# Scene generation helpers
# ---------------------------------------------------------------------------


def generate_scenes(
    model: SceneDenoiser,
    sampler: DDIMSampler,
    text_emb: torch.Tensor | None,
    n_scenes: int,
    seed: int,
    device: torch.device,
    cfg_scale: float,
    guidance_scale: float,
    bounds: WorkspaceBounds,
) -> list[SceneTensor]:
    """Generate n_scenes for a single prompt embedding."""
    from guidance.energy import pairwise_overlap_energy

    if text_emb is not None:
        text_batch = text_emb.expand(n_scenes, -1)
        if cfg_scale == 1.0:
            fn = model.conditional_sampling_fn(text_batch)
        else:
            fn_u = model.conditional_sampling_fn(text_emb=None)
            fn_c = model.conditional_sampling_fn(text_batch)
            _s = cfg_scale

            def fn(x_t, tids, t):
                eu, lu = fn_u(x_t, tids, t)
                ec, lc = fn_c(x_t, tids, t)
                return eu + _s * (ec - eu), lc
    else:
        fn = model.conditional_sampling_fn(text_emb=None)

    x_cont, type_ids = sampler.sample_with_universal_guidance(
        fn,
        (n_scenes, N_MAX, 13),
        seed=seed,
        device=device,
        type_init="uniform",
        guidance_scale=guidance_scale,
    )

    scenes = []
    for i in range(n_scenes):
        st_norm = decode_scene(x_cont[i], type_ids[i], bounds)
        scenes.append(st_norm.denormalize(bounds))
    return scenes


# ---------------------------------------------------------------------------
# Load held-out suite
# ---------------------------------------------------------------------------


def load_suite(suite_path: Path) -> list[dict]:
    with open(suite_path) as f:
        return [json.loads(line) for line in f]


def load_text_cache(data_dir: Path) -> dict[str, torch.Tensor]:
    """Load pre-cached text embeddings if available; return empty dict otherwise.

    The caller falls back to on-the-fly encoding via TextEncoder when a key
    is missing from the cache.
    """
    pt_path = data_dir / "text_embeddings.pt"
    if not pt_path.exists():
        print(f"  text_embeddings.pt not found at {pt_path} - will encode on the fly")
        return {}
    raw = torch.load(pt_path, weights_only=True)
    return {k: v.float().cpu() for k, v in raw.items()}


_text_encoder = None  # lazy singleton


def encode_text_onthefly(text: str, device: torch.device) -> torch.Tensor:
    """Encode a prompt on the fly via TextEncoder (singleton)."""
    global _text_encoder
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from model.text_encoder import TextEncoder
    if _text_encoder is None:
        _text_encoder = TextEncoder()
    return _text_encoder.encode(text).to(device)


# ---------------------------------------------------------------------------
# Sampler configs
# ---------------------------------------------------------------------------


SAMPLER_CONFIGS = [
    {"name": "v7_uncond",        "model": "v7", "cfg_scale": 0.0, "ug_scale": 0.0},
    {"name": "v7_cond_baseline", "model": "v7", "cfg_scale": 1.0, "ug_scale": 0.0},
    {"name": "v7_rejection",     "model": "v7", "cfg_scale": 1.0, "ug_scale": 0.0, "rejection": True},
    {"name": "v7_ug_best",       "model": "v7", "cfg_scale": 1.0, "ug_scale": None},  # filled from --ug-scale
    {"name": "v7_ug_scale1",     "model": "v7", "cfg_scale": 1.0, "ug_scale": 1.0},
]


# ---------------------------------------------------------------------------
# Parallel Drake validation
# ---------------------------------------------------------------------------


def _worker_validate(args: tuple) -> tuple[int, bool, float | None]:
    """Subprocess worker: create a fresh Drake context and validate one scene."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from scene.schema import WorkspaceBounds, SceneTensor
    from validator.core import SceneValidator
    import torch

    idx, scene_dict, rrt_budget = args
    bounds = WorkspaceBounds.default()
    val = SceneValidator(workspace_bounds=bounds, rrt_budget_s=rrt_budget)
    sc = SceneTensor(
        object_types=torch.tensor(scene_dict["object_types"]),
        poses=torch.tensor(scene_dict["poses"]),
        scales=torch.tensor(scene_dict["scales"]),
        presence=torch.tensor(scene_dict["presence"]),
    )
    try:
        report = val.validate(sc)
        return idx, bool(report.accepted), getattr(report, "rrt_plan_length", None)
    except Exception:
        return idx, False, None


def parallel_validate(
    scenes: list["SceneTensor"],
    rrt_budget: float,
    n_workers: int,
) -> list[tuple[bool, float | None]]:
    """Validate scenes in parallel. Returns (accepted, plan_len) per scene."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    args_list = [
        (
            i,
            {
                "object_types": s.object_types.tolist(),
                "poses": s.poses.tolist(),
                "scales": s.scales.tolist(),
                "presence": s.presence.tolist(),
            },
            rrt_budget,
        )
        for i, s in enumerate(scenes)
    ]
    results: list[tuple[bool, float | None]] = [(False, None)] * len(scenes)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_worker_validate, a): a[0] for a in args_list}
        for fut in as_completed(futs):
            try:
                idx, accepted, plan_len = fut.result()
                results[idx] = (accepted, plan_len)
            except Exception:
                pass
    return results


# ---------------------------------------------------------------------------
# Evaluate one sampler x seed
# ---------------------------------------------------------------------------


def evaluate_sampler(
    sampler_cfg: dict,
    suite: list[dict],
    text_cache: dict[str, torch.Tensor],
    models: dict[str, SceneDenoiser],
    ddim_sampler: DDIMSampler,
    validator: SceneValidator,
    bounds: WorkspaceBounds,
    device: torch.device,
    seed: int,
    n_per_prompt: int,
    n_drake_workers: int = 4,
    rrt_budget: float = 2.0,
) -> dict:
    import hashlib

    t0 = time.monotonic()
    model_key = sampler_cfg["model"]
    if model_key not in models:
        print(f"  SKIP {sampler_cfg['name']}: model {model_key} not loaded")
        return {}

    model = models[model_key]
    cfg_scale = sampler_cfg["cfg_scale"]
    ug_scale = sampler_cfg["ug_scale"]
    is_rejection = bool(sampler_cfg.get("rejection"))
    n_gen = n_per_prompt * 4 if is_rejection else n_per_prompt

    # ---- GENERATION (GPU-bound, serial per-prompt) --------------------------
    candidates_per_prompt: list[list[SceneTensor]] = []
    prompts_per_group: list[str] = []

    for i, entry in enumerate(suite):
        desc = entry["description"]
        key = hashlib.sha256(desc.encode()).hexdigest()
        text_emb = text_cache.get(key)
        if text_emb is not None:
            text_emb = text_emb.to(device)
        if text_emb is None:
            try:
                text_emb = encode_text_onthefly(desc, device)
            except Exception as enc_e:
                print(f"    encode failed for '{desc[:40]}': {enc_e}")

        per_seed = (seed * 10007 + i) & 0xFFFFFFFF
        scenes_i = generate_scenes(
            model, ddim_sampler, text_emb, n_gen,
            per_seed, device, cfg_scale, ug_scale or 0.0, bounds,
        )
        candidates_per_prompt.append(scenes_i)
        prompts_per_group.append(desc)

    # ---- PARALLEL VALIDATION ------------------------------------------------
    flat: list[SceneTensor] = []
    group_idx: list[int] = []
    for g, group in enumerate(candidates_per_prompt):
        flat.extend(group)
        group_idx.extend([g] * len(group))

    n_flat = len(flat)
    print(f"  Generation done ({n_flat} scenes). Validating with {n_drake_workers} workers ...")
    flat_results = parallel_validate(flat, rrt_budget, n_drake_workers)
    # flat_results[i] = (accepted, rrt_plan_length | None)

    # ---- REASSEMBLE --------------------------------------------------------
    scenes_all: list[SceneTensor] = []
    prompts_all: list[str] = []
    accepted_all: list[bool] = []

    for g, desc in enumerate(prompts_per_group):
        g_idxs = [j for j, gi in enumerate(group_idx) if gi == g]
        g_scenes = [flat[j] for j in g_idxs]
        g_accepted = [flat_results[j][0] for j in g_idxs]

        if is_rejection:
            valid = [s for s, a in zip(g_scenes, g_accepted) if a]
            chosen = (valid or g_scenes)[:n_per_prompt]
            chosen_acc = [True] * len(valid) if valid else [False] * len(chosen)
            chosen_acc = chosen_acc[:n_per_prompt]
        else:
            chosen = g_scenes[:n_per_prompt]
            chosen_acc = g_accepted[:n_per_prompt]

        scenes_all.extend(chosen)
        prompts_all.extend([desc] * len(chosen))
        accepted_all.extend(chosen_acc)

    # ---- METRICS -----------------------------------------------------------
    val_rate = sum(accepted_all) / max(len(accepted_all), 1)
    div = diversity(scenes_all)

    # RRT plan lengths from parallel_validate results (if available)
    plan_lens = [
        flat_results[j][1]
        for j in range(n_flat)
        if flat_results[j][0] and flat_results[j][1] is not None
    ]
    downstream_success = len(plan_lens) / max(n_flat, 1)
    mean_plan_len = float(sum(plan_lens) / len(plan_lens)) if plan_lens else 0.0

    elapsed = time.monotonic() - t0

    # Serialize for gallery
    scene_records = [
        {
            "description": pr,
            "drake_accepted": acc,
            "object_types": sc.object_types.tolist(),
            "poses": sc.poses.tolist(),
            "scales": sc.scales.tolist(),
            "presence": sc.presence.tolist(),
        }
        for sc, pr, acc in zip(scenes_all, prompts_all, accepted_all)
    ]

    return {
        "sampler": sampler_cfg["name"],
        "seed": seed,
        "n_scenes": len(scenes_all),
        "validity_rate": val_rate,
        "diversity": div,
        "downstream_success_rate": downstream_success,
        "mean_plan_length": mean_plan_len,
        "mean_planning_time_s": rrt_budget,
        "elapsed_s": elapsed,
        "scene_records": scene_records,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="data/eval_suite_v1/eval_suite.jsonl")
    p.add_argument("--data-dir", default="data/v1")
    p.add_argument("--checkpoint-v7", default="checkpoints/conditional_v7/latest.pt")
    p.add_argument("--checkpoint-v8c", default="checkpoints/conditional_v8c_warminit/latest.pt")
    p.add_argument("--ug-scale", type=float, default=0.5,
                   help="UG guidance scale for v7_ug_best config.")
    p.add_argument("--out-dir", default="artifacts/eval_v1")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--n-per-prompt", type=int, default=8)
    p.add_argument("--drake-workers", type=int, default=4)
    p.add_argument("--rrt-budget", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    models: dict[str, SceneDenoiser] = {}
    v7_path = Path(args.checkpoint_v7)
    if v7_path.exists():
        models["v7"] = load_model(v7_path, device)
    else:
        print(f"WARNING: v7 checkpoint not found at {v7_path}")

    v8c_path = Path(args.checkpoint_v8c)
    if v8c_path.exists():
        models["v8c"] = load_model(v8c_path, device)
    else:
        print(f"WARNING: v8c checkpoint not found at {v8c_path}")

    schedule = CosineSchedule(T=1000)
    ddim_sampler = DDIMSampler(schedule, n_steps=DDIM_STEPS)
    bounds = WorkspaceBounds.default()
    validator = SceneValidator(
        workspace_bounds=bounds,
        rrt_budget_s=args.rrt_budget,
    )

    suite = load_suite(Path(args.suite))
    text_cache = load_text_cache(Path(args.data_dir))
    print(f"Suite: {len(suite)} entries | Seeds: {args.seeds} | Drake workers: {args.drake_workers}")

    # Fill in ug_scale for the UG best config
    for cfg in SAMPLER_CONFIGS:
        if cfg["name"] == "v7_ug_best":
            cfg["ug_scale"] = args.ug_scale

    for sampler_cfg in SAMPLER_CONFIGS:
        name = sampler_cfg["name"]
        all_seed_results = []

        print(f"\n=== Evaluating: {name} ===")
        for seed in args.seeds:
            print(f"  seed={seed}")
            result = evaluate_sampler(
                sampler_cfg, suite, text_cache, models, ddim_sampler,
                validator, bounds, device, seed, args.n_per_prompt,
                n_drake_workers=args.drake_workers,
                rrt_budget=args.rrt_budget,
            )
            if result:
                all_seed_results.append(result)

        if not all_seed_results:
            continue

        out_path = out_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump({"sampler": name, "runs": all_seed_results}, f, indent=2)
        print(f"  --> saved to {out_path}")

        # Quick aggregate
        val_rates = [r["validity_rate"] for r in all_seed_results]
        print(f"  validity: {sum(val_rates)/len(val_rates)*100:.1f}% mean")

    print("\nEval complete.")


if __name__ == "__main__":
    main()
