"""Week 1 smoke run: validate 500 randomly sampled scenes.

Generates scenes via the procedural sampler stub, validates each through the
full four-check SceneValidator pipeline, and writes summary statistics plus
a per-scene CSV to artifacts/week1_smoke.csv.

Usage:
    uv run python scripts/smoke_run.py --seed 42 --n-scenes 500
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import time

from data.sampler_stub import sample_batch
from scene.schema import WorkspaceBounds
from validator.core import SceneValidator


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 1 smoke run: 500-scene validation.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for sampling.")
    parser.add_argument("--n-scenes", type=int, default=500, help="Number of scenes to sample.")
    parser.add_argument("--rrt-budget", type=float, default=5.0, help="RRT budget per scene (seconds).")
    parser.add_argument("--output", type=str, default="artifacts/week1_smoke.csv", help="Output CSV path.")
    args = parser.parse_args()

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bounds = WorkspaceBounds.default()
    print(f"Sampling {args.n_scenes} scenes with seed={args.seed}...")
    scenes = sample_batch(args.n_scenes, seed=args.seed, bounds=bounds)

    validator = SceneValidator(workspace_bounds=bounds, rrt_budget_s=args.rrt_budget)

    print(f"Validating {len(scenes)} scenes (rrt_budget={args.rrt_budget}s)...")
    t0 = time.monotonic()

    reports = []
    for i, scene in enumerate(scenes):
        report = validator.validate(scene, rrt_seed=i)
        reports.append(report)
        if (i + 1) % 50 == 0:
            elapsed = time.monotonic() - t0
            accepted = sum(1 for r in reports if r.accepted)
            rate = accepted / (i + 1)
            print(f"  [{i+1}/{len(scenes)}] accepted={accepted} ({rate:.1%}) elapsed={elapsed:.1f}s")

    total_time = time.monotonic() - t0

    # Summary statistics.
    n_total = len(reports)
    n_interp = sum(1 for r in reports if r.no_interpenetration)
    n_stable = sum(1 for r in reports if r.stable_rest)
    n_ik = sum(1 for r in reports if r.ik_reachable)
    n_rrt = sum(1 for r in reports if r.rrt_solvable)
    n_accepted = sum(1 for r in reports if r.accepted)

    print()
    print("=" * 60)
    print(f"Week 1 Smoke Run: {n_total} scenes, seed={args.seed}")
    print(f"  no_interpenetration:  {n_interp:>4d} / {n_total}  ({n_interp/n_total:.1%})")
    print(f"  stable_rest:          {n_stable:>4d} / {n_total}  ({n_stable/n_total:.1%})")
    print(f"  ik_reachable:         {n_ik:>4d} / {n_total}  ({n_ik/n_total:.1%})")
    print(f"  rrt_solvable:         {n_rrt:>4d} / {n_total}  ({n_rrt/n_total:.1%})")
    print(f"  ACCEPTED (all four):  {n_accepted:>4d} / {n_total}  ({n_accepted/n_total:.1%})")
    print(f"  Total time:           {total_time:.1f}s ({total_time/n_total:.2f}s/scene)")
    print("=" * 60)

    # Write CSV.
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_idx",
            "n_present",
            "no_interpenetration",
            "stable_rest",
            "ik_reachable",
            "rrt_solvable",
            "accepted",
            "min_signed_distance_m",
            "max_pose_drift_trans_m",
            "max_pose_drift_rot_rad",
            "rrt_seed",
            "elapsed_s",
            "error",
        ])
        for i, (scene, report) in enumerate(zip(scenes, reports)):
            writer.writerow([
                i,
                scene.n_present,
                int(report.no_interpenetration),
                int(report.stable_rest),
                int(report.ik_reachable),
                int(report.rrt_solvable),
                int(report.accepted),
                f"{report.min_signed_distance_m:.6f}" if not math.isnan(report.min_signed_distance_m) else "",
                f"{report.max_pose_drift_trans_m:.6f}" if not math.isnan(report.max_pose_drift_trans_m) else "",
                f"{report.max_pose_drift_rot_rad:.6f}" if not math.isnan(report.max_pose_drift_rot_rad) else "",
                report.rrt_seed,
                f"{report.elapsed_s:.4f}",
                report.error,
            ])

    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
