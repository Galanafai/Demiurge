#!/usr/bin/env python3
"""Profile RRT budget impact on per-template acceptance rates.

Runs the generation pipeline twice (5s and 2s RRT budgets) on 500-scene
smoke runs and prints the decision-gate table required by Task 5.

Decision gates:
  < 20% acceptance per template at 2s -> use 5s budget
  20-30% acceptance per template at 2s -> proceed with flag
  > 30% acceptance per template at 2s -> use 2s budget (2.5x throughput gain)
  Total wall-clock projection > 12h for 50k scenes -> reduce target to 25k/30k

Usage:
    uv run python scripts/profile_rrt_budget.py \\
        --config configs/dataset/profile_500.yaml \\
        --seed 0

Output:
    Prints a CSV-formatted result table to stdout and a human summary to stderr.
    The CSV is written to <output_dir>/profile_results.csv for later reference.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from generate_dataset import _validate_task, _worker_init  # noqa: E402

from data.sampler import (  # noqa: E402
    ClutteredPickTemplate,
    ObstacleAvoidanceTemplate,
    ProceduralSampler,
    TabletopReachTemplate,
)

logger = logging.getLogger(__name__)

_BUDGETS = [5.0, 2.0]
_FAMILIES = ["tabletop_reach", "cluttered_pick", "obstacle_avoidance"]


def _run_arm(
    budget_s: float,
    cfg: dict,
    seed: int,
    n: int,
) -> tuple[Counter, Counter, float]:
    """Run one budget arm and return (attempts, accepted, elapsed_s)."""
    mix_cfg = cfg.get("mix", {})
    templates = [
        TabletopReachTemplate(),
        ClutteredPickTemplate(),
        ObstacleAvoidanceTemplate(),
    ]
    mix = [
        float(mix_cfg.get("tabletop_reach", 1.0 / 3)),
        float(mix_cfg.get("cluttered_pick", 1.0 / 3)),
        float(mix_cfg.get("obstacle_avoidance", 1.0 / 3)),
    ]
    sampler = ProceduralSampler(templates=templates, mix=mix, seed=seed)
    rrt_rng = np.random.default_rng(seed + 2)
    batch_size: int = int(cfg.get("batch_size", 32))
    num_workers: int = int(cfg.get("num_workers", 4))

    template_attempts: Counter = Counter()
    template_accepted: Counter = Counter()
    total_attempts = 0
    start = time.monotonic()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(budget_s,),
    ) as executor:
        while total_attempts < n:
            remaining = n - total_attempts
            batch_n = min(batch_size, remaining)
            batch = sampler.sample_batch(batch_n)
            rrt_seeds = [int(rrt_rng.integers(0, 2**31)) for _ in batch]

            futures = {
                executor.submit(_validate_task, (c, s)): c
                for c, s in zip(batch, rrt_seeds)
            }
            for future, candidate in futures.items():
                try:
                    report = future.result()
                except Exception as exc:
                    logger.warning("Validation raised: %s", exc)
                else:
                    template_attempts[candidate.task_family] += 1
                    if report.accepted:
                        template_accepted[candidate.task_family] += 1
                total_attempts += 1

    elapsed = time.monotonic() - start
    return template_attempts, template_accepted, elapsed


def _print_decision_table(
    results: dict,
    target_n: int,
) -> None:
    """Print the decision-gate table and projection."""
    print("\n=== RRT Budget Profiling Results ===\n")
    header = (
        f"{'Template':<22} "
        f"{'5s Att':>8} {'5s Acc':>8} {'5s Rate':>8}  "
        f"{'2s Att':>8} {'2s Acc':>8} {'2s Rate':>8}  "
        f"{'Decision'}"
    )
    print(header)
    print("-" * len(header))

    for family in _FAMILIES:
        r5 = results[5.0]
        r2 = results[2.0]
        att5 = r5["attempts"][family]
        acc5 = r5["accepted"][family]
        rate5 = acc5 / att5 if att5 > 0 else 0.0
        att2 = r2["attempts"][family]
        acc2 = r2["accepted"][family]
        rate2 = acc2 / att2 if att2 > 0 else 0.0

        if rate2 < 0.20:
            decision = "USE 5s (2s < 20%)"
        elif rate2 < 0.30:
            decision = "FLAG (2s 20-30%)"
        else:
            decision = "USE 2s (>30%)"

        print(
            f"  {family:<20} "
            f"{att5:>8,d} {acc5:>8,d} {rate5:>7.1%}  "
            f"{att2:>8,d} {acc2:>8,d} {rate2:>7.1%}  "
            f"{decision}"
        )

    # Wall-clock projection.
    for budget in _BUDGETS:
        r = results[budget]
        elapsed = r["elapsed_s"]
        n_sampled = sum(r["attempts"].values())
        n_accepted = sum(r["accepted"].values())
        rate = n_accepted / n_sampled if n_sampled > 0 else 0.0
        # Time to generate target_n at this acceptance rate and throughput.
        scenes_per_s = n_accepted / elapsed if elapsed > 0 else 0.001
        projected_s = target_n / (scenes_per_s * max(rate, 1e-6) / rate)
        projected_h = projected_s / 3600
        flag = "EXCEED 12h" if projected_h > 12 else "OK"
        print(
            f"\n  {budget}s budget: "
            f"acceptance_rate={rate:.1%}, "
            f"throughput={scenes_per_s:.2f} accepted/s, "
            f"projected={projected_h:.1f}h for {target_n:,d} scenes  [{flag}]"
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Profile RRT budget impact.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n: int = int(cfg.get("target_n", 500))
    target_50k: int = 50000

    results: dict = {}
    for budget in _BUDGETS:
        logger.info("Running arm: rrt_budget_s=%.1f, n=%d", budget, n)
        attempts, accepted, elapsed = _run_arm(budget, cfg, args.seed, n)
        results[budget] = {
            "attempts": attempts,
            "accepted": accepted,
            "elapsed_s": elapsed,
        }
        logger.info(
            "Arm %.1fs done: accepted=%d/%d in %.0fs",
            budget, sum(accepted.values()), sum(attempts.values()), elapsed,
        )

    _print_decision_table(results, target_50k)

    # Write CSV for reproducibility.
    output_dir = Path(cfg.get("output_dir", "data/profile_500"))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "profile_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "budget_s", "template", "attempts", "accepted", "rate", "elapsed_s"
        ])
        for budget in _BUDGETS:
            r = results[budget]
            for family in _FAMILIES:
                att = r["attempts"][family]
                acc = r["accepted"][family]
                rate = acc / att if att > 0 else 0.0
                writer.writerow([budget, family, att, acc, f"{rate:.4f}", f"{r['elapsed_s']:.1f}"])
    logger.info("Results written to %s", csv_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
