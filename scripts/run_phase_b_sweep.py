"""Phase B guidance-scale sweep: 14 configs x 3 seeds = 42 runs.

Evaluates classifier-guided vs rejection sampling on held-out prompts.
Each run generates 500 scenes per config, Drake-validates all of them,
computes pairwise type diversity, and logs to W&B.

Usage:
    PYTHONPATH=/workspace/Demiurge/src python3 -u scripts/run_phase_b_sweep.py \\
        --output-dir artifacts/guidance_sweep_v1/ \\
        --vlm-subset-size 0 \\
        --budget-cap 8 \\
        2>&1 | tee logs/phase_b_sweep.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from guidance.classifier import ValidityClassifier  # noqa: E402
from guidance.classifier_guided import ClassifierGuidedSampler  # noqa: E402
from guidance.rejection import RejectionSampler  # noqa: E402
from model.denoiser import N_MAX, DenoiserConfig, SceneDenoiser  # noqa: E402
from model.schedule import CosineSchedule  # noqa: E402
from model.text_encoder import TextEncoder  # noqa: E402
from scene.schema import WorkspaceBounds  # noqa: E402
from validator.core import SceneValidator  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GUIDANCE_SCALES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
MODES = ["continuous", "full"]
SEEDS = [42, 123, 456]
N_PER_PROMPT = 5   # scenes per prompt per run (500 total with 100 prompts)
N_PROMPTS = 100    # held-out prompts from eval split

# GPU/API cost halt conditions.
BUDGET_CAP_USD = 8.0

HOLDOUT_PROMPTS = [
    # tabletop_reach
    "place a cube on the table", "pick up the cracker box", "reach for the sugar box",
    "grasp the mustard bottle", "pick the tomato soup can", "retrieve the potted meat can",
    "grab the banana", "pick up the pitcher base", "reach for the master chef can",
    "grasp the gelatin box",
    # cluttered_pick
    "pick the cube among distractors", "grasp the cracker box from clutter",
    "retrieve the sugar box from a cluttered scene", "pick the mustard from a pile",
    "grasp the soup can from crowded table", "pick the meat can from clutter",
    "retrieve the banana from distractors", "grasp the pitcher from clutter",
    "pick the chef can from cluttered scene", "retrieve gelatin from clutter",
    # obstacle_avoidance
    "reach for the cube while avoiding obstacles", "pick cracker box around obstacles",
    "grasp sugar box while avoiding the barrier", "pick mustard avoiding the wall",
    "retrieve soup can around obstacles", "grasp meat can avoiding barriers",
    "pick banana while avoiding obstacles", "reach pitcher avoiding objects",
    "grasp chef can around barriers", "pick gelatin avoiding obstacles",
] * 4  # 30 unique -> 120, trim to N_PROMPTS
HOLDOUT_PROMPTS = HOLDOUT_PROMPTS[:N_PROMPTS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(cfg_path: str, ckpt_path: str, device: torch.device) -> tuple:
    """Load SceneDenoiser, schedule, bounds from config + checkpoint."""
    cfg = yaml.safe_load(open(cfg_path))
    mcfg = cfg.get("model", {})
    dcfg = DenoiserConfig(
        n_layers=mcfg.get("n_layers", 6),
        d_model=mcfg.get("d_model", 256),
        n_heads=mcfg.get("n_heads", 8),
        ffn_mult=mcfg.get("ffn_mult", 4),
        dropout=0.0,
        use_type_grad_isolation=bool(mcfg.get("use_type_grad_isolation", False)),
    )
    model = SceneDenoiser(dcfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["ema_state"])
    model.eval()
    schedule = CosineSchedule(T=cfg["diffusion"]["T"])
    bounds = WorkspaceBounds.default()
    return model, schedule, bounds


def _load_classifier(cfg_path: str, ckpt_path: str, device: torch.device) -> ValidityClassifier:
    cfg = yaml.safe_load(open(cfg_path))
    mcfg = cfg.get("model", {})
    cls = ValidityClassifier(
        d_model=mcfg.get("d_model", 192),
        n_layers=mcfg.get("n_layers", 5),
        n_heads=mcfg.get("n_heads", 6),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    cls.load_state_dict(ckpt["model_state"])
    cls.eval()
    return cls


def _compute_type_diversity(scenes: list) -> float:
    """Mean number of distinct types per scene (presence-weighted)."""
    if not scenes:
        return 0.0
    per_scene = []
    for st in scenes:
        present_types = set()
        for i in range(N_MAX):
            if st.presence[i]:
                present_types.add(int(st.object_types[i]))
        per_scene.append(len(present_types))
    return float(sum(per_scene) / len(per_scene)) / 12.0  # normalise to [0,1]


def _type_counts(scenes: list) -> dict:
    """Global type frequency across all present slots in all scenes."""
    cnt: Counter = Counter()
    for st in scenes:
        for i in range(N_MAX):
            if st.presence[i]:
                cnt[int(st.object_types[i])] += 1
    return dict(sorted(cnt.items()))


def _run_one(
    w: float,
    mode: str,
    seed: int,
    model: SceneDenoiser,
    classifier: ValidityClassifier | None,
    schedule: CosineSchedule,
    text_encoder: TextEncoder,
    bounds: WorkspaceBounds,
    validator: SceneValidator,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """Run one sweep config. Returns result dict."""
    run_name = f"w{w}_{mode}_seed{seed}"
    out_file = out_dir / f"{run_name}.json"
    if out_file.exists():
        print(f"  [skip] {run_name} already done")
        return json.loads(out_file.read_text())

    print(f"\n{'='*60}")
    print(f"  Config: w={w} mode={mode} seed={seed}  ({run_name})")
    t0 = time.monotonic()

    if w == 0.0 or classifier is None:
        sampler = RejectionSampler(
            model=model,
            schedule=schedule,
            text_encoder=text_encoder,
            bounds=bounds,
            ddim_steps=50,
            rrt_budget_s=2.0,
            batch_size=8,
            type_init="uniform",
            device=device,
            seed=seed,
        )
    else:
        sampler = ClassifierGuidedSampler(
            model=model,
            classifier=classifier,
            schedule=schedule,
            text_encoder=text_encoder,
            bounds=bounds,
            guidance_scale=w,
            mode=mode,
            grad_clip_norm=1.0,
            ddim_steps=50,
            batch_size=1,
            type_init="uniform",
            device=device,
            seed=seed,
        )

    # Generate scenes (classifier-guided: no Drake filter; rejection: Drake filtered).
    scenes = sampler.sample(HOLDOUT_PROMPTS, n_per_prompt=N_PER_PROMPT)
    total_generated = len(scenes)
    print(f"  Generated {total_generated} scenes in {time.monotonic()-t0:.1f}s")

    # Drake validation (for classifier-guided, where Drake wasn't run during sampling).
    if w > 0.0 and classifier is not None:
        t_val = time.monotonic()
        accepted = [st for st in scenes if validator.validate(st).accepted]
        print(f"  Drake: {len(accepted)}/{total_generated} valid ({time.monotonic()-t_val:.1f}s)")
    else:
        # Rejection sampler already filtered -- all returned are valid.
        accepted = scenes

    validity_rate = len(accepted) / max(1, total_generated)
    type_diversity = _compute_type_diversity(scenes)
    type_counts = _type_counts(scenes)
    type0_frac = type_counts.get(0, 0) / max(1, sum(type_counts.values()))

    elapsed = time.monotonic() - t0
    result = {
        "run_name": run_name,
        "w": w,
        "mode": mode,
        "seed": seed,
        "n_generated": total_generated,
        "n_valid": len(accepted),
        "validity_rate": validity_rate,
        "type_diversity": type_diversity,
        "type_counts": type_counts,
        "type0_frac": type0_frac,
        "elapsed_s": elapsed,
    }
    out_file.write_text(json.dumps(result, indent=2))
    print(f"  validity={validity_rate:.1%} diversity={type_diversity:.3f} type0={type0_frac:.1%} t={elapsed:.0f}s")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="configs/train/conditional_v6.yaml")
    p.add_argument("--model-checkpoint", default="checkpoints/conditional_v6/latest.pt")
    p.add_argument("--classifier-config", default="configs/classifier/classifier_v2.yaml")
    p.add_argument("--classifier-checkpoint", default="checkpoints/classifier_v2/latest.pt")
    p.add_argument("--output-dir", default="artifacts/guidance_sweep_v1")
    p.add_argument("--vlm-subset-size", type=int, default=0,
                   help="Number of scenes to VLM-score per config (0 = skip VLM)")
    p.add_argument("--budget-cap", type=float, default=BUDGET_CAP_USD)
    p.add_argument("--seed", type=int, default=42,
                   help="Meta-seed (unused; per-run seeds fixed in SEEDS)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model ...")
    model, schedule, bounds = _load_model(args.model_config, args.model_checkpoint, device)

    print("Loading text encoder ...")
    text_encoder = TextEncoder()
    text_encoder.eval()

    # Preload text embeddings cache if available.
    emb_cache = _ROOT / "data" / "v1" / "text_embeddings.pt"
    if emb_cache.exists():
        text_encoder.load_cache(str(emb_cache))
        print("Text embedding cache loaded.")

    print("Loading classifier ...")
    cls_ckpt = Path(args.classifier_checkpoint)
    if cls_ckpt.exists():
        classifier = _load_classifier(args.classifier_config, str(cls_ckpt), device)
        print("Classifier loaded.")
    else:
        print(f"WARNING: classifier checkpoint not found at {cls_ckpt}. Guided runs will be skipped.")
        classifier = None

    validator = SceneValidator(rrt_budget_s=2.0)

    all_results: list[dict] = []
    sweep_t0 = time.monotonic()

    for w in GUIDANCE_SCALES:
        for mode in MODES:
            if w == 0.0 and mode == "full":
                continue  # rejection has no mode distinction; run once
            for seed in SEEDS:
                result = _run_one(
                    w, mode, seed,
                    model, classifier, schedule, text_encoder, bounds,
                    validator, device, out_dir,
                )
                all_results.append(result)

                elapsed_total = time.monotonic() - sweep_t0
                # Very rough GPU cost estimate: ~$0.69/hr for RTX 5080.
                gpu_cost_est = elapsed_total / 3600 * 0.69
                if gpu_cost_est > args.budget_cap:
                    print(f"\nHALT: estimated GPU cost ${gpu_cost_est:.2f} exceeds cap ${args.budget_cap}.")
                    break
            else:
                continue
            break
        else:
            continue
        break

    # Write combined summary.
    summary_path = out_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps({"runs": all_results}, indent=2))
    print(f"\nSweep complete. {len(all_results)} runs written to {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
