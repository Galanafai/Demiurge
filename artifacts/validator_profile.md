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

---

## Week 2.5 Profiling: 16-Vocab Expansion (Halted)

**Run date:** 2026-05-12
**Vocab size:** 16 (8 original + 8 new YCB entries)

### Raw Results

| Template | 5s rate | 2s rate | Gate |
|---|---|---|---|
| tabletop_reach | 8.6% | 8.6% | HALT |
| cluttered_pick | 0.7% | 1.4% | HALT |
| obstacle_avoidance | 6.6% | 5.9% | HALT |
| **Overall** | **5.8%** | **5.8%** | HALT |

### Per-Check Breakdown (16-vocab, 5s arm)

| Check | Pass% | First-fail count |
|---|---|---|
| no_interpenetration | 90.0% | 50 |
| stable_rest | **67.8%** | 111 |
| ik_reachable | 72.8% | 68 |
| rrt_solvable | 7.8% | 242 |

### Root Cause

**stable_rest collapsed from ~95% to 67.8%.** A single-object stability probe
(36 trials across BOX_TALL, MUSTARD_BOTTLE, SUGAR_BOX, BLEACH_CLEANSER at 3 scales
and 3 yaw values) showed **0/36 failures** in isolation. The regression is a
multi-object contact interaction effect: dense scenes with multiple large-volume
objects cause Drake's contact solver to assign residual forces that drift objects
past the 5mm / 5-degree drift threshold during the 1.5s forward simulation.

**RRT solvability collapsed in cluttered_pick to 0.7-1.4%.** The four new entries
with the largest bounding volumes (cracker_box: 0.158m lateral, power_drill:
0.175m height, cracker_box: 0.170m bounding radius, power_drill: 0.204m bounding
radius) weld disproportionate volume into the RRT planning plant, blocking the
BiRRT from finding collision-free paths in the tight cluster_x=(-0.20, 0.20)
region of ClutteredPickTemplate.

### Decision: Vocabulary Reduction (Option B)

Dropped from OBJECT_VOCAB: PUDDING_BOX (12), CRACKER_BOX (13), POTTED_MEAT_CAN (14),
POWER_DRILL (15). Final vocab size: 12 (IDs 0-11). N_MAX reduced from 16 to 12.

Also added: `upright_constrained: bool` field to ObjectEntry. BLEACH_CLEANSER
(hz=0.125m > 0.10m threshold) flagged True. This is documentation and enforcement
against future SO(3) augmentation; the sampler already produces yaw-only orientations
for all entries.

---

## Week 2.5 Re-Profile: 12-Vocab (Production)

**Run date:** 2026-05-12
**Vocab size:** 12 (IDs 0-11; IDs 12-15 dropped)
**Git SHA:** 0d92866 (feat: vocab 16->12, upright_constrained flag, _sample_orientation wrapper)

### Raw Results

| Template | 5s att | 5s acc | 5s rate | 2s att | 2s acc | 2s rate |
|---|---|---|---|---|---|---|
| tabletop_reach | 209 | 19 | 9.1% | 209 | 18 | 8.6% |
| cluttered_pick | 150 | 8 | 5.3% | 150 | 8 | 5.3% |
| obstacle_avoidance | 141 | 24 | 17.0% | 141 | 23 | 16.3% |
| **Total** | **500** | **51** | **10.2%** | **500** | **49** | **9.8%** |

**Elapsed:** 5s arm = 167s (0.305 accepted/s at 4 workers), 2s arm = 93s (0.529 accepted/s).

### Per-Check Breakdown (12-vocab, 5s arm)

| Check | Pass% | First-fail count | vs 16-vocab |
|---|---|---|---|
| no_interpenetration | 97.2% | 14 | +7.2pp |
| stable_rest | **97.2%** | 0 | **+29.4pp recovered** |
| ik_reachable | 82.4% | 74 | +9.6pp |
| rrt_solvable | 10.2% | 361 | +2.4pp |

**stable_rest fully recovered to 97.2%.** Dropping the 4 large-volume entries
eliminated the multi-object contact instability entirely. No first-fail count
for stable_rest (0) confirms objects are not failing stability before
interpenetration in this run.

### Candidate Timing

| Budget | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| 5s | 1.057s | 0.332s | 5.319s | 5.432s |
| 2s | 0.630s | 0.343s | 2.337s | 2.418s |

### Wall-Clock Projections (50k scenes)

| Budget | Laptop acc/s | CCX33 -> h | CCX43 -> h | Gate |
|---|---|---|---|---|
| 5s | 0.305 | 26.1h | 14.0h | WARNING (12-20h) |
| **2s** | **0.529** | **15.0h** | **8.1h** | **OK (<12h)** |

CCX43: 16 dedicated AMD cores, 15 workers (1 reserved for OS + writer).
Scale assumption: 13x vs single-worker baseline (not 15x; writer bottleneck
caps throughput). This is 3.25x vs the 4-worker laptop measurement.

**Production parameters: 2s budget, 50k target, 15 workers on CCX43. ~8.1h projected.**

### Decision Gate Interpretation

The per-template acceptance gate fires HALT for all three templates at both
budget levels (all rates are below 20%). This gate is **deliberately overridden**
for the following documented reason:

The gate was calibrated in the implementation plan to catch catastrophic regression,
specifically the 16-vocab cluttered_pick collapse to 0.7-1.4%. It was not intended
to gatekeep templates that are structurally hard by design.

The 12-vocab numbers (5.3% cluttered_pick / 9.1% tabletop_reach / 17.0%
obstacle_avoidance) are structurally similar to the 8-vocab Week 2 baseline
(8.1% / 18.5% / 12.7%). In both runs, all templates fell below 20%. In Week 2,
the decision record approved proceeding with the explanation that the
cluttered_pick floor is a load-bearing property of the dataset: dense placement
is the design intent of that template, and tuning it toward 20% would collapse
the difficulty stratification that the Week 5 evaluation depends on.

The same reasoning applies here. The difficulty stratification is preserved.
The RRT bottleneck (10.2% conditional pass) reflects real scene complexity,
not a pipeline defect.

**Gate override approved by Galanafai on 2026-05-12. Proceed to cloud setup (Task 3).**

### Production Run Command (CCX43)

See `docs/cloud_run.md` for full provisioning and execution instructions.

```bash
# On the CCX43 instance after bootstrap:
nohup uv run python scripts/generate_dataset.py \
    --config configs/dataset/v1.yaml \
    --seed 42 \
    > logs/generate_v1.log 2>&1 &
echo "PID: $!"
```
