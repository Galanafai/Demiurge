"""Build ablation prompt sets for the Phase A-prime conditioning study.

Generates two prompt collections:
  - type_only (100): mentions object types but no positions
  - position_only (100): mentions positions but no object types

Writes to artifacts/ablation_prompts_{name}.json.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OBJECT_TYPES = [
    "cube",
    "sphere",
    "cylinder",
    "tall_box",
    "flat_box",
    "mustard_bottle",
    "sugar_box",
    "tomato_soup_can",
    "bleach_cleanser",
    "banana",
    "master_chef_can",
    "gelatin_box",
]

DIRECTIONS = ["left", "right"]


def main() -> None:
    rng = random.Random(42)

    # --- Type-only (100): types mentioned, no coordinates ---
    type_only: list[dict] = []
    for _ in range(100):
        n = rng.choice([1, 2, 3])
        chosen = rng.sample(OBJECT_TYPES, n)
        if n == 1:
            prompt = f"place a {chosen[0]} on the table"
        elif n == 2:
            prompt = f"arrange a {chosen[0]} and a {chosen[1]} on the table"
        else:
            prompt = (
                f"set down a {chosen[0]}, a {chosen[1]}, and a {chosen[2]} on the table"
            )
        type_only.append({"prompt": prompt, "n_objects": n, "types": chosen})

    # --- Position-only (100): coordinates mentioned, no named object types ---
    position_only: list[dict] = []
    for _ in range(100):
        x = rng.uniform(-0.20, 0.20)
        y = rng.uniform(0.15, 0.45)
        x_cm = int(abs(x) * 100)
        y_cm = int(y * 100)
        side = "right" if x > 0 else "left"
        prompt = (
            f"place an object {y_cm}cm out and {x_cm}cm to the {side}"
        )
        position_only.append({
            "prompt": prompt,
            "position_x_m": round(x, 4),
            "position_y_m": round(y, 4),
        })

    # --- Full-conditioning (100): both type and position ---
    full_cond: list[dict] = []
    for _ in range(100):
        obj = rng.choice(OBJECT_TYPES)
        x = rng.uniform(-0.20, 0.20)
        y = rng.uniform(0.15, 0.45)
        x_cm = int(abs(x) * 100)
        y_cm = int(y * 100)
        side = "right" if x > 0 else "left"
        prompt = (
            f"place a {obj} {y_cm}cm out and {x_cm}cm to the {side}"
        )
        full_cond.append({
            "prompt": prompt,
            "obj_type": obj,
            "position_x_m": round(x, 4),
            "position_y_m": round(y, 4),
        })

    # --- Null prompt (100 identical empty strings for unconditional baseline) ---
    null_cond = [{"prompt": ""} for _ in range(100)]

    out_dir = Path(__file__).resolve().parent.parent / "artifacts"
    out_dir.mkdir(exist_ok=True)

    datasets = {
        "type_only": type_only,
        "position_only": position_only,
        "full_cond": full_cond,
        "null": null_cond,
    }
    for name, prompts in datasets.items():
        path = out_dir / f"ablation_prompts_{name}.json"
        path.write_text(json.dumps(prompts, indent=2))
        print(f"  {name}: {len(prompts)} prompts -> {path}")


if __name__ == "__main__":
    main()
