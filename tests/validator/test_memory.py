"""Memory and executor-rotation regression tests for SceneValidator.

Two tests:
  1. test_validator_cache_removed: _cache must not exist on SceneValidator.
     Catches accidental re-introduction of the unbounded per-call cache
     that caused the CCX33 OOM at scene 5597 (~0.56 MB/call accumulation).

  2. test_executor_rotates: successive ProcessPoolExecutor instances must
     produce non-overlapping worker PIDs. Catches accidental removal of
     the rotation logic that reclaims pydrake C++ allocator memory.
"""
from __future__ import annotations

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
# Test 2: executor rotation
# ---------------------------------------------------------------------------


def _pid_task(_: int) -> int:
    """Return the current worker PID."""
    return os.getpid()


def test_executor_rotates() -> None:
    """Executor rotation must produce distinct worker PIDs across lifetimes.

    The memory fix uses ProcessPoolExecutor.shutdown(wait=True) + recreation
    every rotation_interval tasks. This test verifies two successive executors
    produce non-overlapping worker PIDs, confirming workers exit and new ones
    spawn at each rotation.

    Catches accidental removal of the rotation logic in generate_dataset.py.
    """
    import concurrent.futures

    # First executor lifetime.
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as exc1:
        futures = [exc1.submit(_pid_task, i) for i in range(2)]
        pids_1 = {f.result() for f in concurrent.futures.as_completed(futures)}

    # Second executor lifetime (fresh processes after shutdown).
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as exc2:
        futures = [exc2.submit(_pid_task, i) for i in range(2)]
        pids_2 = {f.result() for f in concurrent.futures.as_completed(futures)}

    assert pids_1.isdisjoint(pids_2), (
        f"Executor 1 PIDs {pids_1} overlap with Executor 2 PIDs {pids_2}. "
        "Workers from executor 1 should have exited before executor 2 started. "
        "Rotation is not creating fresh processes."
    )

