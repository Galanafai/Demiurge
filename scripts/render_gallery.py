"""Gallery renderer for Phase B best config.

Generates N scenes with the best classifier-guided config and renders
them as text-serialized scene descriptions (+ Drake validity summary).
Drake offscreen rendering is not used; text format is accepted for portfolio.

Usage:
    PYTHONPATH=/workspace/Demiurge/src python3 -u scripts/render_gallery.py \\
        --config configs/train/conditional_v6.yaml \\
        --checkpoint checkpoints/conditional_v6/latest.pt \\
        --classifier-config configs/classifier/classifier_v2.yaml \\
        --classifier-checkpoint checkpoints/classifier_v2/latest.pt \\
        --guidance-scale 2.0 \\
        --mode continuous \\
        --n-scenes 30 \\
        --output artifacts/gallery/ \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from guidance.classifier import ValidityClassifier  # noqa: E402
from guidance.classifier_guided import ClassifierGuidedSampler  # noqa: E402
from model.denoiser import N_MAX, DenoiserConfig, SceneDenoiser  # noqa: E402
from model.schedule import CosineSchedule  # noqa: E402
from model.text_encoder import TextEncoder  # noqa: E402
from scene.schema import WorkspaceBounds  # noqa: E402
from validator.core import SceneValidator  # noqa: E402

# Diverse gallery prompts across all 3 templates.
GALLERY_PROMPTS = [
    # tabletop_reach (10)
    "place a cube on the table",
    "pick up the cracker box",
    "reach for the sugar box",
    "grasp the mustard bottle",
    "pick the tomato soup can",
    "retrieve the potted meat can",
    "grab the banana",
    "pick up the pitcher base",
    "reach for the master chef can",
    "grasp the gelatin box",
    # cluttered_pick (10)
    "pick the cube among distractors",
    "grasp the cracker box from clutter",
    "retrieve the sugar box from a cluttered scene",
    "pick the mustard from a pile",
    "grasp the soup can from crowded table",
    "pick the meat can from clutter",
    "retrieve the banana from distractors",
    "grasp the pitcher from clutter",
    "pick the chef can from cluttered scene",
    "retrieve gelatin from clutter",
    # obstacle_avoidance (10)
    "reach for the cube while avoiding obstacles",
    "pick cracker box around obstacles",
    "grasp sugar box while avoiding the barrier",
    "pick mustard avoiding the wall",
    "retrieve soup can around obstacles",
    "grasp meat can avoiding barriers",
    "pick banana while avoiding obstacles",
    "reach pitcher avoiding objects",
    "grasp chef can around barriers",
    "pick gelatin avoiding obstacles",
]


def _scene_to_dict(st, prompt: str, valid: bool, rejection_reason: str | None) -> dict:
    """Serialize a SceneTensor to a JSON-friendly dict."""
    objects = []
    for i in range(N_MAX):
        if not st.presence[i]:
            continue
        pose = st.poses[i].tolist()
        objects.append({
            "slot": i,
            "type_id": int(st.object_types[i]),
            "xyz": pose[:3],
            "quat_wxyz": pose[3:7],
            "scale": st.scales[i].tolist(),
        })
    return {
        "prompt": prompt,
        "n_objects": len(objects),
        "objects": objects,
        "drake_valid": valid,
        "rejection_reason": rejection_reason,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--classifier-config", default="configs/classifier/classifier_v2.yaml")
    p.add_argument("--classifier-checkpoint", default="checkpoints/classifier_v2/latest.pt")
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--mode", default="continuous", choices=["continuous", "full"])
    p.add_argument("--n-scenes", type=int, default=30)
    p.add_argument("--output", default="artifacts/gallery/")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model.
    cfg = yaml.safe_load(open(args.config))
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
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["ema_state"])
    model.eval()
    schedule = CosineSchedule(T=cfg["diffusion"]["T"])
    bounds = WorkspaceBounds.default()

    # Load classifier.
    cls_ckpt = Path(args.classifier_checkpoint)
    classifier = None
    if cls_ckpt.exists():
        ccfg = yaml.safe_load(open(args.classifier_config))
        cm = ccfg.get("model", {})
        classifier = ValidityClassifier(
            d_model=cm.get("d_model", 192),
            n_layers=cm.get("n_layers", 5),
            n_heads=cm.get("n_heads", 6),
        ).to(device)
        cls_state = torch.load(str(cls_ckpt), map_location=device)
        classifier.load_state_dict(cls_state["model_state"])
        classifier.eval()
        print(f"Classifier loaded (w={args.guidance_scale} mode={args.mode})")
    else:
        print(f"WARNING: classifier not found at {cls_ckpt}. Using unguided sampling.")

    emb_cache_path = _ROOT / "data" / "v1" / "text_embeddings.pt"
    text_encoder = TextEncoder(
        device=device,
        cache_path=str(emb_cache_path) if emb_cache_path.exists() else None,
    )

    sampler = ClassifierGuidedSampler(
        model=model,
        classifier=classifier,  # type: ignore[arg-type]
        schedule=schedule,
        text_encoder=text_encoder,
        bounds=bounds,
        guidance_scale=args.guidance_scale if classifier is not None else 0.0,
        mode=args.mode,
        grad_clip_norm=1.0,
        ddim_steps=50,
        batch_size=1,
        type_init="uniform",
        device=device,
        seed=args.seed,
    )

    validator = SceneValidator(rrt_budget_s=2.0)

    n = min(args.n_scenes, len(GALLERY_PROMPTS))
    prompts = GALLERY_PROMPTS[:n]
    print(f"Generating {n} gallery scenes ...")
    scenes = sampler.sample(prompts, n_per_prompt=1)

    gallery = []
    n_valid = 0
    for i, (st, prompt) in enumerate(zip(scenes, prompts)):
        rpt = validator.validate(st)
        is_valid = rpt.accepted
        reason = None if is_valid else rpt.failure_reason
        if is_valid:
            n_valid += 1
        entry = _scene_to_dict(st, prompt, is_valid, reason)
        gallery.append(entry)
        print(f"  Scene {i+1:3d}: {'VALID' if is_valid else f'INVALID ({reason})'} | "
              f"{entry['n_objects']} objects | {prompt[:40]}")

    print(f"\nGallery: {n_valid}/{n} valid ({100*n_valid/n:.0f}%)")

    # Write outputs.
    gallery_path = out_dir / "gallery.json"
    gallery_path.write_text(json.dumps({"scenes": gallery, "n_valid": n_valid, "n_total": n}, indent=2))
    print(f"Gallery JSON saved to {gallery_path}")

    # Write one text file per scene for portfolio readability.
    for i, entry in enumerate(gallery):
        scene_txt = out_dir / f"scene_{i+1:03d}_{entry['drake_valid']}.txt"
        lines = [
            f"Scene {i+1}",
            f"Prompt: {entry['prompt']}",
            f"Drake valid: {entry['drake_valid']}",
            f"Objects: {entry['n_objects']}",
            "",
        ]
        for obj in entry["objects"]:
            lines.append(
                f"  [{obj['slot']}] type={obj['type_id']} "
                f"xyz=[{', '.join(f'{v:.3f}' for v in obj['xyz'])}] "
                f"scale=[{', '.join(f'{v:.3f}' for v in obj['scale'])}]"
            )
        scene_txt.write_text("\n".join(lines))

    print(f"Wrote {n} text scene files to {out_dir}")


if __name__ == "__main__":
    main()
