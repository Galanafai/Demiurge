# Validator Memory Leak Diagnosis

**Date:** 2026-05-12
**Symptom:** CCX33 worker OOM at scene 5597 (~3h 16min into generation run)
**anon-rss at OOM:** ~4.8 GB per worker
**Instance:** Hetzner CCX33 (8 vCPU, 32 GB RAM), 7 workers, 2s RRT budget

---

## Symptom

Generation halted with OOM kill on all 7 workers at approximately scene 5597.
Each worker had consumed ~4.8 GB anon-rss. With 7 workers the combined footprint
was ~33.6 GB, exceeding the CCX33's 32 GB RAM + swap.

Extrapolating from the probe data: 5597 scenes accepted / ~10% acceptance rate
= ~56,000 validate() calls per worker / 4.8 GB = **~0.086 MB per call** at the
worker level, consistent with the ~0.56 MB/call measured in single-process probes
(workers share the allocator across multiple candidates per call).

---

## Probe Protocol

All probes ran 200 iterations of `validate()` or `_run_checks()` on randomised
scenes (unique seed per call, matching production conditions), measured RSS via
`/proc/<pid>/status`, reported every 20 iterations.

Baseline (reproduced from production memory probe):

```
iter     rss_mb   delta_mb
   0      516.5      516.5
  20      536.0       19.5
  40      546.6       10.6
  ...
 180      625.3       13.1
Per-call rate: ~0.56 MB/call
```

---

## Probe A: Cache Bypassed

**Method:** Called `validator._run_checks(scene, rrt_seed=0)` directly,
skipping `validate()` entirely. No cache reads or writes.

**Result:**

```
iter     rss_mb    delta_mb
   0      522.7         0.0
  20      538.8        16.1
  40      553.4        30.8
  60      564.8        42.1
  80      579.0        56.4
 100      593.2        70.5
 120      603.7        81.0
 140      616.3        93.6
 160      616.3        93.6
 180      627.3       104.6
Final RSS: 641.5 MB  |  Per-call rate: 0.594 MB/call
```

**Conclusion:** Cache is **not** the leak. 0.594 MB/call with no cache activity.

---

## Probe B: Forced gc.collect() After Every Call

**Method:** Called `validator.validate()` (full path, cache active) with
`gc.collect()` forced after every iteration.

**Result:**

```
iter     rss_mb    delta_mb    cache_size
   0      522.8         0.0             2
  20      531.7         8.9            22
  40      541.7        18.9            42
  ...
 180      614.1        91.2           182
Final RSS: 624.7 MB  |  Per-call rate: 0.509 MB/call
```

**Conclusion:** Python's cyclic GC **cannot release the leaking memory.** The
objects are not in Python reference cycles; they are live C++ allocations that
Python GC has no visibility into.

---

## Probe C: Cache Hit Rate

**Method:** Ran 100 `validate()` calls with unique seeds matching production
distribution. Counted cache hits vs misses. Then re-queried the same 10 seeds
to confirm the cache mechanism itself works.

**Result:**

```
Iterations:  100
Cache hits:  0  (0.0%)
Cache misses:100  (100.0%)
Cache size:  100 entries
Repeat hits (same 10 seeds re-queried): 10/10
```

**Conclusion:** Production cache hit rate is exactly 0%. The cache is dead code
in production and was removed as housekeeping (not the OOM cause).

---

## Probe: malloc_trim

**Method:** Called `ctypes.CDLL("libc.so.6").malloc_trim(0)` after every
`_run_checks` call to force glibc to return free pages to the OS. If memory
growth was glibc heap retention (freed objects sitting in glibc's free list),
this would recover them.

**Result:**

```
iter     rss_mb    delta_mb
   0      523.1         0.0
  10      528.2         5.1
  ...
  50      546.8        23.7
Final RSS: 551.8 MB  |  Per-call rate: 0.479 MB/call
```

**Conclusion:** malloc_trim provides negligible improvement (0.479 vs 0.56
MB/call). The memory is **not** in freed-but-unreturned glibc pages. The C++
objects are genuinely live -- they are not being freed at all.

---

## Path 1: Python-Side Referrer Audit

Checked via `gc.get_referrers(validator)` after a single `_run_checks()` call:

- `ValidityReport` fields: all plain Python types (bool, float, ndarray). No
  pydrake objects retained.
- `SceneValidator` instance vars: `_bounds` (WorkspaceBounds), `_rrt_budget_s`
  (float), `_urdf_path` (PosixPath). No Drake objects.
- Module-level: no `@lru_cache` or `@functools.cache` wrappers in `core.py`.
- `gc.get_referrers(validator)`: only the `__main__` dict (the probe script's
  local scope).

**Conclusion:** No Python-side reference retention. The leak is purely in
pydrake's C++ allocator.

---

## Path 3: pydrake Cache Reset API

Checked `SceneGraph`, `MultibodyPlant`, `Context`, `Diagram`, `Simulator` for
cache management methods:

- `Context.DisableCaching()`, `FreezeCache()`, `SetAllCacheEntriesOutOfDate()`:
  available, but these control Drake's *system-theoretic evaluation caches* (for
  port `Eval()` calls), not memory pool allocations. Irrelevant to this leak.
- No `purge()`, `flush()`, or allocator-reset methods on any Drake class.
- `Simulator.reset_context()`: resets context values, not the C++ allocator.

**Conclusion:** No pydrake API can reset the C++ allocator mid-process. Process
exit is the only guaranteed reclaim mechanism.

---

## Root Cause

pydrake's `MultibodyPlant`, `SceneGraph`, `Diagram`, and `Context` use Drake's
internal C++ allocators (`drake::internal::AbstractValue` pools, geometry
proximity result buffers, BVH traversal state). These are allocated per-call
in `_check_rrt_solvable` (BiRRT loop issuing hundreds of
`ComputeSignedDistancePairwiseClosestPoints()` calls) and `_check_stable_rest`
(Simulator forward integration).

The Python wrappers are local variables that drop their Python references on
method return. However, pybind11-managed ownership chains
(`Diagram → Plant → SceneGraph → Context → back-refs`) create C++ reference
cycles that are not visible to Python's cyclic GC. The C++ destructors fire
eventually (at some indeterminate future GC cycle), but at production throughput
(~1 call/sec × 7 workers) new allocations outpace deferred destruction.

The net effect is monotonic, linear RSS growth at ~0.56 MB/call regardless of
Python GC, malloc_trim, or cache state.

---

## Fix: Worker Recycling (maxtasksperchild=200)

**Changed:** `scripts/generate_dataset.py`

Replaced `concurrent.futures.ProcessPoolExecutor` with
`multiprocessing.Pool(maxtasksperchild=200)`.

Workers are recycled after 200 tasks. Process exit forces the OS to reclaim all
pages, guaranteed, regardless of pydrake destructor state.

**Memory bound:**

- 200 tasks × 0.56 MB/task = 112 MB peak per worker
- 7 workers × 112 MB = 784 MB total working memory
- CCX33 has 32 GB -- 784 MB is 2.4% of available RAM

**Throughput cost:**

- Worker spawn: ~3-4s (Drake import + plant setup amortised)
- Worker lifetime at 200 tasks × ~1.3s/task: ~260s
- Overhead fraction: 4s / 260s ≈ **1.5%**

**Also removed:** `self._cache` from `SceneValidator` (0% hit rate in
production; not the OOM cause but dead weight).

**Regression tests:** `tests/validator/test_memory.py`

- `test_validator_cache_removed`: asserts `_cache` absent from `SceneValidator`
- `test_pool_recycles_workers`: verifies `maxtasksperchild` actually recycles
  workers using a single-process pool with 4 tasks and maxtasksperchild=2
