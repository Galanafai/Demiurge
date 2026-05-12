# Validator Profiling Report: RRT Budget Comparison

**Run date:** 2026-05-11
**Script:** `scripts/profile_rrt_budget.py`
**Config:** `configs/dataset/profile_500.yaml`
**Seed:** 0
**Attempts per arm:** 500

## Purpose

This document records the Task 5 profiling results that determined the dataset
generation parameters for `data/v1`. It exists so future contributors understand
why the dataset is 15k scenes and not 50k.

## Setup

- **Validator:** `SceneValidator` with hardened RRT plant (Task 0). All present
  scene objects are welded into the RRT planning plant. This is the correct plant
  and differs from the Week 1 smoke run, which used a robot-only plant.
- **Workers:** 4 `ProcessPoolExecutor` workers on a 6-core machine.
- **Templates:** Unchanged from the procedural sampler (Task 1). No tuning was
  applied before or after this measurement.

## Raw Results

| Template | 5s att | 5s acc | 5s rate | 2s att | 2s acc | 2s rate |
|---|---|---|---|---|---|---|
| tabletop_reach | 178 | 33 | 18.5% | 178 | 32 | 18.0% |
| cluttered_pick | 172 | 14 | 8.1% | 172 | 11 | 6.4% |
| obstacle_avoidance | 150 | 19 | 12.7% | 150 | 19 | 12.7% |
| **Total** | **500** | **66** | **13.2%** | **500** | **62** | **12.4%** |

**Elapsed:** 5s arm = 210s (0.31 accepted/s), 2s arm = 116s (0.54 accepted/s).

Full results in: `data/profile_500/profile_results.csv`

## Comparison with Week 1 Baseline

| | Week 1 (robot-only RRT) | Week 2 (hardened RRT) |
|---|---|---|
| Overall acceptance | 38.2% | 13.2% |
| RRT pass rate (given IK pass) | 45.4% | ~15% estimated |

The drop from 38.2% to 13.2% is entirely attributable to adding real scene-object
collision geometry to the RRT planning plant (Task 0). The hardened plant is
correct; the Week 1 number was inflated by the APPROX annotation.

## Decision Gate Analysis

Per the gate thresholds in the implementation plan:

| Template | 5s gate | 2s gate |
|---|---|---|
| tabletop_reach | HALT (< 20%) | HALT (< 20%) |
| cluttered_pick | HALT (< 20%) | HALT (< 20%) |
| obstacle_avoidance | HALT (< 20%) | HALT (< 20%) |

Wall-clock projections at 4 workers:

| Budget | Rate | Throughput | Proj. (50k) | Proj. (15k) |
|---|---|---|---|---|
| 5s | 13.2% | 0.31 acc/s | ~44h | ~13h |
| 2s | 12.4% | 0.54 acc/s | ~26h | **~7.7h** |

## Decision: Option B

**Target:** 15,000 accepted scenes
**Budget:** 2s RRT per candidate
**Workers:** 4 (on 6-core machine; 4 workers avoids oversubscription)
**Projected wall-clock:** ~7.7h

### Why not tune templates?

The cluttered_pick floor at 8% is a load-bearing property of the dataset.
Dense object placement is the entire design intent of that template. If we
increased min_separation to raise acceptance toward 20%, the resulting scenes
would be less cluttered and less challenging for the robot arm. The Week 5
evaluation question is "does the model produce valid scenes across a difficulty
spectrum?" A tuned dataset would compress that spectrum.

### Why not add more workers?

A 6-core machine with 4 Python + Drake workers leaves 2 cores for the OS and the
main process. Adding a 5th or 6th worker causes Drake's internal parallelism
(used in collision query broadphase) to contend with worker processes. The
profiling run at 4 workers showed 0.54 accepted/s; oversubscription would reduce
this. 8+ workers requires a larger machine, which is out of scope for Week 2.

### Why 15k and not 30k?

15k is sufficient for a Week 3 proof-of-concept baseline. The DDPM transformer
target is ~10-50M parameters. At 15k training examples with standard augmentation
(random rotation of the tabletop plane, scale jitter), the model will either learn
the scene distribution or not. If Week 5 shows data starvation, we extend with
seed offsets on cloud hardware without changing the pipeline.

## Reproducibility

To reproduce this exact run:

```bash
uv run python scripts/profile_rrt_budget.py \
    --config configs/dataset/profile_500.yaml \
    --seed 0
```

The profiling script is deterministic for fixed seed and config. The results above
were generated with git SHA `31e67cc` (feat: parallel generation harness and
profiling script).

## Production Run Command

```bash
nohup uv run python scripts/generate_dataset.py \
    --config configs/dataset/v1.yaml \
    --seed 42 \
    > logs/generate_v1.log 2>&1 &
echo "PID: $!"
```

Monitor progress:
```bash
tail -f logs/generate_v1.log
```

Resume (if interrupted): re-run the same command. ShardWriter reads
`data/v1/manifest.json`, deletes any partial shards, and continues from the
last completed shard.
