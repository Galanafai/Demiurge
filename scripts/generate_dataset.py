#!/usr/bin/env python3
"""Generate the Demiurge training dataset.

Procedurally samples scenes from three task templates, validates each scene
through the full Drake pipeline (interpenetration, stability, IK, RRT), and
writes accepted scenes to sharded WebDataset tars.

Usage:
    uv run python scripts/generate_dataset.py \\
        --config configs/dataset/v1.yaml \\
        --seed 42

The script is deterministic for a given --config and --seed. Every run logs
the git SHA, config hash, and seed to stderr; optionally to Weights & Biases.

Resume:
    The ShardWriter maintains a manifest.json in output_dir. Re-running the
    script with the same --config and --output_dir resumes from the last
    completed shard. Partial shards are deleted and restarted automatically.

Decision gate (per-template acceptance rate):
    Rates are printed in a table after every checkpoint. If any template falls
    below 20% acceptance after the first 200 attempts for that template, the
    script prints a WARNING and continues (does not halt; the profiling step
    in profile_rrt_budget.py is the halt gate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

# Ensure src/ is on the path when run as a script.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.descriptions import generate_description  # noqa: E402
from data.sampler import (  # noqa: E402
    CandidateScene,
    ClutteredPickTemplate,
    ObstacleAvoidanceTemplate,
    ProceduralSampler,
    TabletopReachTemplate,
)
from data.writer import ShardWriter  # noqa: E402
from scene.schema import WorkspaceBounds  # noqa: E402
from validator.core import SceneValidator, ValidityReport  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-worker state (initialised once per process)
# ---------------------------------------------------------------------------

_worker_validator: SceneValidator | None = None


def _worker_init(rrt_budget_s: float) -> None:
    """Initialise a SceneValidator in each worker process.

    Called once per worker process before any tasks are dispatched.
    Drake plant construction is expensive; amortise it here.

    With multiprocessing.Pool(maxtasksperchild=N), workers are recycled
    after N tasks. _worker_init fires again in each replacement worker,
    so the validator is always initialised before use.
    """
    global _worker_validator
    _worker_validator = SceneValidator(
        workspace_bounds=WorkspaceBounds.default(),
        rrt_budget_s=rrt_budget_s,
    )


def _validate_task(
    args: tuple[CandidateScene, int],
) -> tuple[CandidateScene, ValidityReport]:
    """Validate a single candidate scene. Called in a worker process.

    Returns (candidate, report) so the caller can match results to inputs
    without relying on result order (imap_unordered does not preserve order).
    """
    candidate, rrt_seed = args
    assert _worker_validator is not None, "Worker not initialised"
    report = _worker_validator.validate(candidate.scene, rrt_seed=rrt_seed)
    return candidate, report


# ---------------------------------------------------------------------------
# ValidityReport -> JSON-serialisable dict
# ---------------------------------------------------------------------------


def _report_to_dict(report: ValidityReport) -> dict[str, Any]:
    """Convert ValidityReport to a JSON-serialisable dict.

    ik_solution (ndarray | None) is stored as a list of floats or null.
    NaN floats are preserved as the JSON string "nan" for exact round-trip.
    """

    def _float(v: float) -> Any:
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v

    return {
        "no_interpenetration": report.no_interpenetration,
        "stable_rest": report.stable_rest,
        "ik_reachable": report.ik_reachable,
        "rrt_solvable": report.rrt_solvable,
        "accepted": report.accepted,
        "min_signed_distance_m": _float(report.min_signed_distance_m),
        "max_pose_drift_trans_m": _float(report.max_pose_drift_trans_m),
        "max_pose_drift_rot_rad": _float(report.max_pose_drift_rot_rad),
        "ik_solution": (
            report.ik_solution.tolist()
            if report.ik_solution is not None
            else None
        ),
        "rrt_seed": report.rrt_seed,
        "elapsed_s": _float(report.elapsed_s),
        "error": report.error,
    }


# ---------------------------------------------------------------------------
# Config loading and validation
# ---------------------------------------------------------------------------


def _load_config(config_path: str, overrides: list[str]) -> dict[str, Any]:
    """Load YAML config and apply CLI key=value overrides."""
    with open(config_path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override {override!r} must be in key=value format")
        key, val = override.split("=", 1)
        # Attempt to cast to int/float/bool before storing as string.
        for cast in (int, float):
            try:
                val = cast(val)  # type: ignore[assignment]
                break
            except ValueError:
                pass
        if val == "true":
            val = True  # type: ignore[assignment]
        elif val == "false":
            val = False  # type: ignore[assignment]
        cfg[key] = val

    return cfg


def _config_hash(cfg: dict[str, Any]) -> str:
    """Return a short SHA-256 of the serialised config (for reproducibility logs)."""
    canon = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(canon).hexdigest()[:12]


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# W&B integration (optional)
# ---------------------------------------------------------------------------


def _init_wandb(cfg: dict[str, Any], seed: int, config_hash: str, git_sha: str) -> Any:
    """Initialise a W&B run if enabled. Returns the run object or None."""
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        logger.info("W&B logging disabled")
        return None

    try:
        import wandb  # type: ignore[import]
    except ImportError:
        logger.warning("wandb not installed; W&B logging skipped")
        return None

    run = wandb.init(
        project=wandb_cfg.get("project", "demiurge"),
        entity=wandb_cfg.get("entity") or None,
        tags=wandb_cfg.get("tags", []),
        config={
            **cfg,
            "seed": seed,
            "config_hash": config_hash,
            "git_sha": git_sha,
        },
        resume="allow",
    )
    return run


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def _print_stats_table(
    attempts: Counter,
    accepted: Counter,
    elapsed_s: float,
) -> None:
    """Print a per-template acceptance rate table to stderr."""
    header = f"{'Template':<22} {'Attempts':>10} {'Accepted':>10} {'Rate':>8}  {'Flag'}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)
    for family in ["tabletop_reach", "cluttered_pick", "obstacle_avoidance"]:
        att = attempts[family]
        acc = accepted[family]
        rate = acc / att if att > 0 else float("nan")
        flag = ""
        if att >= 200:
            if rate < 0.20:
                flag = "HALT-GATE (<20%)"
            elif rate < 0.30:
                flag = "FLAG (20-30%)"
        print(
            f"  {family:<20} {att:>10,d} {acc:>10,d} {rate:>7.1%}  {flag}",
            file=sys.stderr,
        )
    total_att = sum(attempts.values())
    total_acc = sum(accepted.values())
    total_rate = total_acc / total_att if total_att > 0 else float("nan")
    print(
        f"  {'TOTAL':<20} {total_att:>10,d} {total_acc:>10,d} {total_rate:>7.1%}",
        file=sys.stderr,
    )
    scenes_per_hour = (total_acc / elapsed_s * 3600) if elapsed_s > 0 else 0
    print(
        f"\n  Elapsed: {elapsed_s:.0f}s  |  Rate: {scenes_per_hour:,.0f} accepted/hr",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate the Demiurge training dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a YAML config file (e.g. configs/dataset/v1.yaml)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for scene sampling and RRT (default: 42)",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Override config fields, e.g. --override target_n=1000 num_workers=2",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config, args.override or [])
    seed = args.seed
    config_hash = _config_hash(cfg)
    git_sha = _git_sha()

    # Log run identity to stderr (always; W&B may also capture this).
    logger.info(
        "generate_dataset | git_sha=%s | config_hash=%s | seed=%d | "
        "target_n=%s | rrt_budget_s=%s | num_workers=%s | max_tasks_per_worker=%s",
        git_sha, config_hash, seed,
        cfg["target_n"], cfg["rrt_budget_s"], cfg["num_workers"],
        cfg.get("max_tasks_per_worker", 200),
    )

    # W&B (optional).
    wandb_run = _init_wandb(cfg, seed, config_hash, git_sha)

    # Build sampler from mix config.
    mix_cfg: dict[str, float] = cfg.get("mix", {})
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
    desc_rng = np.random.default_rng(seed + 1)

    output_dir = Path(cfg["output_dir"])
    target_n: int = int(cfg["target_n"])
    num_workers: int = int(cfg["num_workers"])
    batch_size: int = int(cfg["batch_size"])
    shard_size_mb: float = float(cfg["shard_size_mb"])
    rrt_budget_s: float = float(cfg["rrt_budget_s"])
    max_attempts: int = target_n * int(cfg["max_attempts_multiplier"])
    max_tasks_per_worker: int = int(cfg.get("max_tasks_per_worker", 200))

    # Stats.
    template_attempts: Counter = Counter()
    template_accepted: Counter = Counter()
    checkpoint_interval = 500  # print stats table every N accepted scenes
    start_time = time.monotonic()

    # Seed for RRT calls: deterministic sequence derived from base seed.
    rrt_rng = np.random.default_rng(seed + 2)

    with ShardWriter(
        output_dir,
        shard_size_mb=shard_size_mb,
        resume=True,
    ) as writer:

        already_accepted = writer.accepted_count
        if already_accepted >= target_n:
            logger.info(
                "Already at target (%d >= %d). Nothing to do.",
                already_accepted, target_n,
            )
            return 0

        logger.info(
            "Resuming from %d accepted scenes. Need %d more.",
            already_accepted, target_n - already_accepted,
        )

        pbar = tqdm(
            total=target_n,
            initial=already_accepted,
            unit="scene",
            desc="Accepted",
            file=sys.stderr,
        )

        total_attempts = 0

        with multiprocessing.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(rrt_budget_s,),
            maxtasksperchild=max_tasks_per_worker,
        ) as pool:

            while writer.accepted_count < target_n:
                if total_attempts >= max_attempts:
                    logger.error(
                        "Hard stop: total_attempts=%d >= max_attempts=%d. "
                        "Check per-template acceptance rates.",
                        total_attempts, max_attempts,
                    )
                    pbar.close()
                    _print_stats_table(
                        template_attempts, template_accepted,
                        time.monotonic() - start_time,
                    )
                    return 1

                # Sample a batch of candidates.
                batch: list[CandidateScene] = sampler.sample_batch(batch_size)
                rrt_seeds = [
                    int(rrt_rng.integers(0, 2**31)) for _ in batch
                ]

                # Dispatch via imap_unordered: results arrive as they complete,
                # preserving the same unordered processing as as_completed.
                # chunksize=1 ensures maxtasksperchild counts individual tasks,
                # not chunks (chunked imap would decrement the counter once per
                # chunk regardless of chunk size).
                args_iter = [(c, s) for c, s in zip(batch, rrt_seeds)]

                for candidate, report_or_exc in pool.imap_unordered(
                    _validate_task, args_iter, chunksize=1
                ):
                    # imap_unordered re-raises worker exceptions in the main
                    # process; wrap in try/except to match the existing
                    # future.result() exception handling.
                    try:
                        report: ValidityReport = report_or_exc
                    except Exception as exc:
                        logger.warning(
                            "Validation raised unexpected exception: %s", exc
                        )
                        template_attempts[candidate.task_family] += 1
                        total_attempts += 1
                        continue

                    template_attempts[candidate.task_family] += 1
                    total_attempts += 1

                    if report.accepted:
                        desc = generate_description(candidate, desc_rng)
                        report_dict = _report_to_dict(report)
                        report_dict["task_family"] = candidate.task_family
                        writer.write(candidate.scene, desc, report_dict)
                        template_accepted[candidate.task_family] += 1
                        pbar.update(1)

                        # Periodic stats table.
                        n_acc = writer.accepted_count
                        if n_acc % checkpoint_interval == 0:
                            elapsed = time.monotonic() - start_time
                            _print_stats_table(
                                template_attempts, template_accepted, elapsed
                            )
                            if wandb_run is not None:
                                _log_wandb(
                                    wandb_run, n_acc, template_attempts,
                                    template_accepted, elapsed,
                                )

                        if writer.accepted_count >= target_n:
                            break  # inner for-loop; outer while checks condition

        pbar.close()

    # Final stats.
    elapsed = time.monotonic() - start_time
    logger.info("Generation complete. Accepted=%d in %.0fs.", writer.accepted_count, elapsed)
    print("\n=== Final acceptance rates ===", file=sys.stderr)
    _print_stats_table(template_attempts, template_accepted, elapsed)

    if wandb_run is not None:
        _log_wandb(
            wandb_run, writer.accepted_count, template_attempts,
            template_accepted, elapsed,
        )
        wandb_run.finish()

    return 0


def _log_wandb(
    run: Any,
    n_accepted: int,
    attempts: Counter,
    accepted: Counter,
    elapsed_s: float,
) -> None:
    metrics = {"accepted_total": n_accepted, "elapsed_s": elapsed_s}
    for family in ["tabletop_reach", "cluttered_pick", "obstacle_avoidance"]:
        att = attempts[family]
        acc = accepted[family]
        metrics[f"acceptance_rate/{family}"] = acc / att if att > 0 else 0.0
        metrics[f"accepted/{family}"] = acc
    run.log(metrics, step=n_accepted)


if __name__ == "__main__":
    sys.exit(main())
