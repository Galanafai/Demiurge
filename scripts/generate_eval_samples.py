"""Generate VLM evaluation samples for Week 3 Task 9.

Loads held-out task descriptions, generates 8 scenes per description per model
(conditional_v2 EMA and conditional_v3 EMA), runs Drake validation, serializes
scenes to text, and writes a JSONL file for downstream VLM judging.

Usage:
    PYTHONPATH=src python3 scripts/generate_eval_samples.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from model.denoiser import DenoiserConfig, SceneDenoiser, N_MAX
from model.rotations import rot6d_to_quat_wxyz
from model.schedule import CosineSchedule, DDIMSampler
from scene.schema import WorkspaceBounds, SceneTensor
from scene.vocab import OBJECT_VOCAB
from validator.core import SceneValidator
from data.reader import ShardReader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = pathlib.Path("data/v1")
OUT_DIR = pathlib.Path("artifacts/week3_vlm_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "samples.jsonl"

CHECKPOINTS = {
    "conditional_v2": pathlib.Path("checkpoints/conditional_v2/latest.pt"),
    "conditional_v3": pathlib.Path("checkpoints/conditional_v3/latest.pt"),
}

N_DESCRIPTIONS = 200   # total held-out descriptions to evaluate
SCENES_PER_DESC = 8    # DDIM samples per description per model
HELD_OUT_PER_TEMPLATE = 100  # last N per template reserved for held-out
DDIM_STEPS = 50
RRT_BUDGET = 2.0

OBJECT_TYPE_NAMES: dict[int, str] = {i: v.name for i, v in enumerate(OBJECT_VOCAB)}

# ---------------------------------------------------------------------------
# Scene serialization
# ---------------------------------------------------------------------------


def serialize_scene(st: SceneTensor, bounds: WorkspaceBounds) -> str:
    """Convert SceneTensor to compact human-readable text for VLM judging."""
    physical = st.denormalize(bounds) if st.poses[:, :3].abs().max() <= 1.0 + 1e-3 else st
    parts = ["Scene contents:"]
    n_present = 0
    for i in range(N_MAX):
        if not physical.presence[i].item():
            continue
        n_present += 1
        type_id = int(physical.object_types[i].item())
        name = OBJECT_TYPE_NAMES.get(type_id, f"obj_{type_id}")
        p = physical.poses[i]
        s = physical.scales[i]
        parts.append(
            f"  - {name} at position ({p[0].item():.2f}m, {p[1].item():.2f}m, {p[2].item():.2f}m)"
            f" scale ({s[0].item():.2f}, {s[1].item():.2f}, {s[2].item():.2f})"
        )
    if n_present == 0:
        parts.append("  - (empty scene)")
    parts.append("Workspace: 0.6m x 0.6m tabletop, UR5e robot at origin")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Held-out split
# ---------------------------------------------------------------------------


def build_held_out(data_dir: pathlib.Path, n_per_template: int, total: int) -> list[dict]:
    """Return stratified sample of held-out descriptions."""
    print(f"Loading dataset from {data_dir}...")
    reader = ShardReader(data_dir)
    per_template: dict[str, list[dict]] = defaultdict(list)

    for scene, desc, report, sdf in reader:
        tmpl = report.get("task_family", "unknown")
        per_template[tmpl].append({"description": desc, "template": tmpl})

    print(f"Templates: {', '.join(f'{k}={len(v)}' for k,v in per_template.items())}")

    # Take last N per template as held-out
    held_out_by_tmpl: dict[str, list[dict]] = {}
    for tmpl, items in per_template.items():
        held_out_by_tmpl[tmpl] = items[-n_per_template:]

    # Stratified sample: proportional to template counts
    all_templates = sorted(held_out_by_tmpl.keys())
    total_held = sum(len(v) for v in held_out_by_tmpl.values())
    sampled: list[dict] = []
    rng = torch.Generator()
    rng.manual_seed(42)
    for tmpl in all_templates:
        items = held_out_by_tmpl[tmpl]
        n_take = max(1, round(total * len(items) / total_held))
        # Shuffle deterministically
        idx = torch.randperm(len(items), generator=rng).tolist()[:n_take]
        sampled.extend(items[i] for i in idx)

    # Deduplicate by description text
    seen: set[str] = set()
    deduped = []
    for item in sampled:
        key = hashlib.sha256(item["description"].encode()).hexdigest()[:16]
        if key not in seen:
            seen.add(key)
            item["desc_id"] = key
            deduped.append(item)

    # Trim to exactly total
    deduped = deduped[:total]
    print(f"Held-out sample: {len(deduped)} descriptions")
    per_tmpl_count = defaultdict(int)
    for d in deduped:
        per_tmpl_count[d["template"]] += 1
    for tmpl, n in sorted(per_tmpl_count.items()):
        print(f"  {tmpl}: {n}")
    return deduped


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------


def load_model(ckpt_path: pathlib.Path, device: torch.device) -> tuple[SceneDenoiser, DenoiserConfig]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    dcfg = DenoiserConfig(
        n_layers=arch["n_layers"], d_model=arch["d_model"],
        n_heads=arch["n_heads"], ffn_mult=arch["ffn_mult"], dropout=0.0,
    )
    model = SceneDenoiser(dcfg).to(device)
    model.load_state_dict(ckpt["ema_state"])
    model.eval()
    print(f"  Loaded {ckpt_path.name} at step {ckpt['step']}")
    return model, dcfg


def load_text_cache(data_dir: pathlib.Path) -> dict[str, torch.Tensor]:
    cache_path = data_dir / "text_embeddings.pt"
    raw = torch.load(cache_path, weights_only=True)
    return {k: v.float().cpu() for k, v in raw.items()}


def get_text_emb(desc: str, text_cache: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    key = hashlib.sha256(desc.encode()).hexdigest()
    if key not in text_cache:
        raise KeyError(f"Description hash not in cache: {key[:12]}...")
    return text_cache[key].unsqueeze(0).to(device)  # (1, 384)


def generate_scenes(
    model: SceneDenoiser,
    schedule: CosineSchedule,
    text_emb: torch.Tensor,
    n_scenes: int,
    device: torch.device,
    seed: int,
) -> list[SceneTensor]:
    """Generate n_scenes via DDIM with text conditioning."""
    sampler = DDIMSampler(schedule, n_steps=DDIM_STEPS)
    fn = model.noise_prediction_fn(text_emb.expand(n_scenes, -1))
    x0 = sampler.sample(fn, (n_scenes, N_MAX, 13), seed=seed, device=device)
    bounds = WorkspaceBounds.default()

    scenes = []
    for i in range(n_scenes):
        pm = x0[i, :, 12] > 0.0
        quats = rot6d_to_quat_wxyz(x0[i, :, 3:9])
        poses = torch.cat([x0[i, :, :3].clamp(-1, 1), quats], dim=-1)
        st_norm = SceneTensor(
            object_types=torch.zeros(N_MAX, dtype=torch.long),
            poses=poses.cpu(),
            scales=x0[i, :, 9:12].clamp(-1, 1).cpu(),
            presence=pm.cpu(),
        )
        scenes.append(st_norm)
    return scenes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    bounds = WorkspaceBounds.default()
    validator = SceneValidator(rrt_budget_s=RRT_BUDGET)
    schedule = CosineSchedule(T=1000)

    # Load held-out descriptions
    held_out = build_held_out(DATA_DIR, HELD_OUT_PER_TEMPLATE, N_DESCRIPTIONS)
    text_cache = load_text_cache(DATA_DIR)

    # Open output file
    total_written = 0
    model_stats: dict[str, dict] = {}

    with open(OUT_PATH, "w") as out_f:
        for model_name, ckpt_path in CHECKPOINTS.items():
            print(f"\n{'='*60}")
            print(f"Model: {model_name}")
            model, _ = load_model(ckpt_path, device)
            accepted_count = 0
            total_scenes = 0
            tmpl_counts: dict[str, dict] = defaultdict(lambda: {"accepted": 0, "total": 0})

            for desc_idx, desc_info in enumerate(held_out):
                desc = desc_info["description"]
                tmpl = desc_info["template"]
                desc_id = desc_info["desc_id"]

                try:
                    text_emb = get_text_emb(desc, text_cache, device)
                except KeyError as e:
                    print(f"  WARN: {e} -- skipping desc {desc_id}")
                    continue

                seed = hash((model_name, desc_id)) & 0xFFFFFFFF
                scenes_norm = generate_scenes(model, schedule, text_emb, SCENES_PER_DESC, device, seed)

                for s_idx, st_norm in enumerate(scenes_norm):
                    sample_id = f"{model_name}_{desc_id}_{s_idx}"
                    st_phys = st_norm.denormalize(bounds)
                    rpt = validator.validate(st_phys)

                    scene_text = serialize_scene(st_norm, bounds)

                    record = {
                        "sample_id": sample_id,
                        "model": model_name,
                        "description": desc,
                        "scene_text": scene_text,
                        "drake": {
                            "accepted": rpt.accepted,
                            "rejection_reason": (
                                None if rpt.accepted else
                                ("interpenetration" if not rpt.no_interpenetration else
                                 "stability" if not rpt.stable_rest else
                                 "ik" if not rpt.ik_reachable else "rrt")
                            ),
                        },
                        "template": tmpl,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    total_written += 1
                    if rpt.accepted:
                        accepted_count += 1
                        tmpl_counts[tmpl]["accepted"] += 1
                    total_scenes += 1
                    tmpl_counts[tmpl]["total"] += 1

                if (desc_idx + 1) % 20 == 0:
                    rate = accepted_count / max(1, total_scenes)
                    print(f"  [{desc_idx+1}/{len(held_out)}] {total_scenes} scenes, "
                          f"validity={rate*100:.1f}%")
                    out_f.flush()

            rate = accepted_count / max(1, total_scenes)
            model_stats[model_name] = {
                "total": total_scenes, "accepted": accepted_count, "validity": rate,
                "by_template": {k: v for k, v in tmpl_counts.items()},
            }
            print(f"\n{model_name} DONE: {accepted_count}/{total_scenes} ({rate*100:.1f}%)")
            for tmpl, cnts in sorted(tmpl_counts.items()):
                tr = cnts["accepted"] / max(1, cnts["total"])
                print(f"  {tmpl}: {cnts['accepted']}/{cnts['total']} ({tr*100:.1f}%)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Total samples written: {total_written}")
    print(f"Output: {OUT_PATH}")
    for model_name, stats in model_stats.items():
        print(f"  {model_name}: {stats['validity']*100:.1f}% validity ({stats['accepted']}/{stats['total']})")

    # Save stats
    with open(OUT_DIR / "generation_stats.json", "w") as f:
        json.dump(model_stats, f, indent=2, default=str)


if __name__ == "__main__":
    main()
