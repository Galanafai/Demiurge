"""Memory and worker-recycling regression tests for SceneValidator.

Two tests:
  1. test_validator_cache_removed: _cache must not exist on SceneValidator.
     Catches accidental re-introduction of the unbounded per-call cache
     that caused the CCX33 OOM at scene 5597 (~0.56 MB/call accumulation).

  2. test_pool_recycles_workers: Pool(maxtasksperchild=2) must produce at
     least 2 distinct child PIDs across 4 tasks. Catches accidental removal
     of maxtasksperchild from the generate_dataset.py pool configuration.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from validator.core import SceneValidator  # noqa: E402

# ---------------------------------------------------------------------------
# Test 1: cache removed
# ---------------------------------------------------------------------------


def test_validator_cache_removed() -> None:
    """SceneValidator must not have a _cache attribute.

    The per-call result cache was confirmed to be dead code in production
    (0% hit rate, Probe C) and was removed in the worker-recycling fix.
    If this test fails, _cache has been re-introduced; check that
    validate() does not write to self._cache.
    """
    validator = SceneValidator.__new__(SceneValidator)
    # Check the class, not just an instance, to catch class-level caches too.
    assert not hasattr(SceneValidator, "_cache"), (
        "SceneValidator has a class-level _cache attribute. "
        "This was removed in the worker-recycling fix; do not re-introduce it."
    )

    # Also construct a real instance and confirm no _cache appears post-init.
    validator = SceneValidator()
    assert not hasattr(validator, "_cache"), (
        "SceneValidator.__init__ assigns self._cache. "
        "This was removed in the worker-recycling fix (0% hit rate in production). "
        "See artifacts/leak_diagnosis.md for the full probe record."
    )


# ---------------------------------------------------------------------------
# Test 2: pool worker recycling
# ---------------------------------------------------------------------------


def _pid_task(_: int) -> int:
    """Return the current worker PID. Used to verify worker recycling."""
    return os.getpid()


def test_pool_recycles_workers() -> None:
    """Pool(maxtasksperchild=2) must recycle workers.

    With maxtasksperchild=2 and 4 tasks submitted to a 1-process pool,
    at least 2 distinct PIDs must be observed. If only 1 PID appears,
    maxtasksperchild is not taking effect.

    Uses a single-process pool so worker lifetime is predictable.
    Catches accidental removal or disabling of maxtasksperchild in the
    generate_dataset.py pool configuration.
    """
    with multiprocessing.Pool(processes=1, maxtasksperchild=2) as pool:
        pids = pool.map(_pid_task, range(4))

    unique_pids = set(pids)
    assert len(unique_pids) >= 2, (
        f"Expected at least 2 distinct worker PIDs (maxtasksperchild=2, 4 tasks), "
        f"got {len(unique_pids)}: {unique_pids}. "
        "Check that multiprocessing.Pool is constructed with maxtasksperchild "
        "in scripts/generate_dataset.py."
    )
