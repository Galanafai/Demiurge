#!/usr/bin/env python3
"""Profile RRT budget impact on per-template acceptance rates.

Runs the generation pipeline twice (5s and 2s RRT budgets) on 500-scene
smoke runs and prints the full decision-gate table required by Task 2.

Outputs per run:
  - Per-template acceptance rates (both budgets)
  - Per-check rejection breakdown (no_interpenetration, stable_rest,
    ik_reachable, rrt_solvable) across all attempts
  - Mean elapsed time per candidate (whole scene, not just accepted)
  - Wall-clock projection for 50k scenes on target hardware

Decision gates:
  < 20% acceptance per template at 2s -> HALT: surface before Task 3
  20-30% acceptance per template at 2s -> FLAG: proceed with note
  > 30% acceptance per template at 2s -> proceed clean
  50k projection > 12h on CCX33 (7 workers) -> surface as anomalous

Usage:
    uv run python scripts/profile_rrt_budget.py \\
        --config configs/dataset/profile_500.yaml \\
        --seed 0

Output:
    Full table printed to stdout; CSV written to <output_dir>/profile_results.csv.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import logging
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
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
from validator.core import ValidityReport  # noqa: E402

logger = logging.getLogger(__name__)

_BUDGETS = [5.0, 2.0]
_FAMILIES = ["tabletop_reach", "cluttered_pick", "obstacle_avoidance"]
_CHECKS = ["no_interpenetration", "stable_rest", "ik_reachable", "rrt_solvable"]

# CCX33 hardware parameters for wall-clock projection.
_CCX33_WORKERS = 7      # 8 dedicated AMD cores minus 1 for OS / writer
_LAPTOP_WORKERS = 4     # measured workers in the 8-vocab profile run


@dataclass
class ArmResult:
    """Aggregated statistics for one (budget, n_attempts) profiling arm."""

    budget_s: float
    attempts: Counter = field(default_factory=Counter)       # by template family
    accepted: Counter = field(default_factory=Counter)       # by template family
    check_pass: Counter = field(default_factory=Counter)     # by check name (all attempts)
    check_fail_first: Counter = field(default_factory=Counter)  # first failing check per scene
    elapsed_total_s: float = 0.0   # total wall-clock (arm level)
    candidate_elapsed_s: list[float] = field(default_factory=list)  # per-candidate elapsed_s


# ---------------------------------------------------------------------------
# Arm runner
# ---------------------------------------------------------------------------


def _run_arm(budget_s: float, cfg: dict, seed: int, n: int) -> ArmResult:
    """Run one budget arm and return an ArmResult with full stats."""
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

    result = ArmResult(budget_s=budget_s)
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
                    report: ValidityReport = future.result()
                except Exception as exc:
                    logger.warning("Validation raised: %s", exc)
                    total_attempts += 1
                    continue

                fam = candidate.task_family
                result.attempts[fam] += 1
                if report.accepted:
                    result.accepted[fam] += 1

                # Per-check pass tracking.
                for check in _CHECKS:
                    if getattr(report, check):
                        result.check_pass[check] += 1

                # First failing check (sequential: interp -> stable -> ik -> rrt).
                first_fail = None
                for check in _CHECKS:
                    if not getattr(report, check):
                        first_fail = check
                        break
                if first_fail is not None:
                    result.check_fail_first[first_fail] += 1

                # Per-candidate elapsed time (includes RRT timeout cost).
                if not math.isnan(report.elapsed_s):
                    result.candidate_elapsed_s.append(report.elapsed_s)

                total_attempts += 1

    result.elapsed_total_s = time.monotonic() - start
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_acceptance_table(results: dict[float, ArmResult]) -> None:
    print("\n=== Acceptance Rates ===\n")
    header = (
        f"{'Template':<22} "
        f"{'5s Att':>8} {'5s Acc':>8} {'5s Rate':>8}  "
        f"{'2s Att':>8} {'2s Acc':>8} {'2s Rate':>8}  "
        f"{'Gate'}"
    )
    print(header)
    print("-" * len(header))

    halt_triggered = False
    for family in _FAMILIES:
        r5 = results[5.0]
        r2 = results[2.0]
        att5 = r5.attempts[family]
        acc5 = r5.accepted[family]
        rate5 = acc5 / att5 if att5 > 0 else 0.0
        att2 = r2.attempts[family]
        acc2 = r2.accepted[family]
        rate2 = acc2 / att2 if att2 > 0 else 0.0

        if rate2 < 0.20:
            gate = "HALT (<20%)"
            halt_triggered = True
        elif rate2 < 0.30:
            gate = "FLAG (20-30%)"
        else:
            gate = "OK (>30%)"

        print(
            f"  {family:<20} "
            f"{att5:>8,d} {acc5:>8,d} {rate5:>7.1%}  "
            f"{att2:>8,d} {acc2:>8,d} {rate2:>7.1%}  "
            f"{gate}"
        )

    for budget in _BUDGETS:
        r = results[budget]
        total_att = sum(r.attempts.values())
        total_acc = sum(r.accepted.values())
        rate = total_acc / total_att if total_att > 0 else 0.0
        print(f"  {'TOTAL':<20} {'':>8} {'':>8} {'':>8}  {total_att:>8,d} {total_acc:>8,d} {rate:>7.1%}  (overall, {budget}s)")

    return halt_triggered


def _print_rejection_breakdown(results: dict[float, ArmResult]) -> None:
    print("\n=== Per-Check Rejection Breakdown ===\n")
    for budget in _BUDGETS:
        r = results[budget]
        total_att = sum(r.attempts.values())
        print(f"  Budget {budget}s  (n_attempts={total_att:,d})")
        print(f"  {'Check':<24} {'Pass':>8} {'Pass%':>8} {'First-fail':>12}")
        print(f"  {'-'*55}")
        for check in _CHECKS:
            passes = r.check_pass[check]
            pct = passes / total_att if total_att > 0 else 0.0
            first_fail = r.check_fail_first[check]
            print(f"  {check:<24} {passes:>8,d} {pct:>7.1%}  {first_fail:>12,d}")
        # Error-exception count.
        exception_count = total_att - sum(r.attempts.values())
        if exception_count > 0:
            print(f"  {'exception':24} {exception_count:>8,d}")
        print()


def _print_timing(results: dict[float, ArmResult]) -> None:
    print("=== Candidate Timing ===\n")
    for budget in _BUDGETS:
        r = results[budget]
        et = r.candidate_elapsed_s
        if not et:
            print(f"  {budget}s: no timing data")
            continue
        mean_t = sum(et) / len(et)
        et_sorted = sorted(et)
        p50 = et_sorted[len(et_sorted) // 2]
        p95 = et_sorted[int(len(et_sorted) * 0.95)]
        p99 = et_sorted[min(int(len(et_sorted) * 0.99), len(et_sorted) - 1)]
        print(
            f"  {budget}s budget: mean={mean_t:.3f}s  p50={p50:.3f}s  "
            f"p95={p95:.3f}s  p99={p99:.3f}s  "
            f"(n={len(et):,d} candidates)"
        )


def _print_cloud_projection(results: dict[float, ArmResult], target_n: int) -> None:
    """Project wall-clock time for target_n scenes on CCX33 (7 workers)."""
    print("\n=== Wall-Clock Projection ===\n")
    print(f"  Target:  {target_n:,d} accepted scenes")
    print(f"  Hardware: Hetzner CCX33 (8 dedicated AMD cores, {_CCX33_WORKERS} workers)")
    print(f"  Measured: laptop (6 cores, {_LAPTOP_WORKERS} workers)\n")

    # Laptop throughput -> CCX33 throughput.
    # Assumption: throughput scales linearly with workers (conservative; cache effects
    # on dedicated cores may improve this). We do NOT apply a clock-speed multiplier;
    # the worker-count ratio is the only adjustment, which is verifiable.
    worker_scale = _CCX33_WORKERS / _LAPTOP_WORKERS

    halt_triggered = False
    for budget in _BUDGETS:
        r = results[budget]
        total_att = sum(r.attempts.values())
        total_acc = sum(r.accepted.values())
        rate = total_acc / total_att if total_att > 0 else 0.0
        # Laptop: accepted/s observed
        laptop_acc_per_s = total_acc / r.elapsed_total_s if r.elapsed_total_s > 0 else 0.001
        # CCX33 projection (linear worker scale)
        cloud_acc_per_s = laptop_acc_per_s * worker_scale
        projected_s = target_n / cloud_acc_per_s
        projected_h = projected_s / 3600

        if projected_h > 12:
            gate = "ANOMALOUS (>12h): halt, surface before Task 3"
            halt_triggered = True
        elif projected_h > 8:
            gate = "FLAG (8-12h)"
        else:
            gate = "OK (<8h)"

        print(
            f"  {budget}s budget:\n"
            f"    acceptance_rate    = {rate:.1%}\n"
            f"    laptop_throughput  = {laptop_acc_per_s:.3f} acc/s "
            f"({_LAPTOP_WORKERS} workers)\n"
            f"    cloud_throughput   = {cloud_acc_per_s:.3f} acc/s "
            f"({_CCX33_WORKERS} workers, {worker_scale:.2f}x scale)\n"
            f"    projected_50k      = {projected_h:.1f}h  [{gate}]\n"
        )

    return halt_triggered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Profile RRT budget impact (Task 2).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-n", type=int, default=50000,
        help="Target scene count for wall-clock projection (default: 50000)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n: int = int(cfg.get("target_n", 500))

    results: dict[float, ArmResult] = {}
    for budget in _BUDGETS:
        logger.info("Running arm: rrt_budget_s=%.1f, n=%d", budget, n)
        result = _run_arm(budget, cfg, args.seed, n)
        results[budget] = result
        total_acc = sum(result.accepted.values())
        total_att = sum(result.attempts.values())
        logger.info(
            "Arm %.1fs done: accepted=%d/%d in %.0fs",
            budget, total_acc, total_att, result.elapsed_total_s,
        )

    # Print full report.
    halt_accept = _print_acceptance_table(results)
    _print_rejection_breakdown(results)
    _print_timing(results)
    halt_proj = _print_cloud_projection(results, args.target_n)

    # Gate summary.
    print("=== Gate Summary ===\n")
    if halt_accept:
        print("  HALT: one or more templates below 20% at 2s budget.")
        print("  Do not proceed to Task 3 without user review.")
    elif halt_proj:
        print("  HALT: 50k projection exceeds 12h. Likely acceptance-rate regression.")
        print("  Do not proceed to Task 3 without user review.")
    else:
        print("  All gates PASSED. Await user review before Task 3.")

    # Write CSV.
    output_dir = Path(cfg.get("output_dir", "data/profile_500"))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "profile_results_16vocab.csv"
    with csv_path.open("w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "budget_s", "template", "attempts", "accepted", "rate",
            "check_interp_pass", "check_stable_pass", "check_ik_pass", "check_rrt_pass",
            "mean_elapsed_s", "elapsed_total_s",
        ])
        for budget in _BUDGETS:
            r = results[budget]
            total_att = sum(r.attempts.values())
            mean_t = (
                sum(r.candidate_elapsed_s) / len(r.candidate_elapsed_s)
                if r.candidate_elapsed_s else float("nan")
            )
            for family in _FAMILIES:
                att = r.attempts[family]
                acc = r.accepted[family]
                rate = acc / att if att > 0 else 0.0
                writer.writerow([
                    budget, family, att, acc, f"{rate:.4f}",
                    r.check_pass["no_interpenetration"],
                    r.check_pass["stable_rest"],
                    r.check_pass["ik_reachable"],
                    r.check_pass["rrt_solvable"],
                    f"{mean_t:.3f}",
                    f"{r.elapsed_total_s:.1f}",
                ])
    logger.info("Results written to %s", csv_path)

    return 1 if (halt_accept or halt_proj) else 0


if __name__ == "__main__":
    sys.exit(main())
