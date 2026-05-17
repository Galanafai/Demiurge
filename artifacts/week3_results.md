# Week 3 Results: Conditional Diffusion Baseline

## Summary

Week 3 ships a conditional DDPM that generates 6-DoF tabletop scenes conditioned on task
descriptions, validated through Drake (pydrake) for physical validity.

**Shipped baseline:** `conditional_v3` at step 115,000  
**EMA validity:** 17.4% [14.1, 20.7]% (500-scene Drake probe, unconditional sampling)  
**Improvement over unconditional:** 3.2x over `unconditional_v3` (5.4%)

---

## Full Ablation Table

| Model | Validity (EMA, 500-scene) | 95% CI | Mean obj count | Notes |
|---|---|---|---|---|
| unconditional_v1 | 0.0% | [0.0, 0.7]% | ~10.5 | presence_bce too low (0.05), all slots filled |
| unconditional_v2 | 1.4% | [0.4, 2.4]% | 6.40 | presence_bce=0.3, still overcrowded |
| unconditional_v3 | 5.4% | [3.4, 7.4]% | 4.66 | presence_bce=1.0, Pareto optimum |
| unconditional_v4 | 8.6% | [6.1, 11.1]% | 3.73 | presence_bce=2.0, Probe B degraded vs v3 |
| conditional_v1 | 1.0% | -- | 10.11 | CFG dropout missing, model ignored conditioning |
| conditional_v2 | 17.0% | [13.7, 20.3]% | 2.66 | CFG dropout=0.15 fix, 3.1x over uncond baseline |
| **conditional_v3 (model)** | **18.0%** | **[14.6, 21.4]%** | **2.71** | step 115k, bf16, bs256, LR restart |
| **conditional_v3 (EMA)** | **17.4%** | **[14.1, 20.7]%** | **2.66** | step 115k, EMA head |

---

## Experiment Narrative

### Unconditional ablation (Week 3 original)

Four variants over `presence_bce` weight. `unconditional_v3` at `presence_bce=1.0` was the
Pareto optimum: 5.4% validity with mean 4.66 objects per scene. `v4` improved raw validity to
8.6% but degraded mean object count to 3.73 and Probe B task relevance -- rejected as ship
candidate due to task-diversity regression.

### conditional_v1 failure

First conditional attempt trained without CFG dropout. The model learned to ignore the text
conditioning entirely (text embedding passed every step, model found it easier to ignore than
use). Result: 1.0% validity, mean 10.11 objects -- effectively an unconditional model that
memorized "fill all slots." Identified in post-run debrief.

### conditional_v2: CFG dropout fix

Added `cfg_dropout=0.15` -- 15% of training batches pass `text_emb=None`, forcing the model
to maintain an unconditional fallback while learning the conditional path. Result: 17.0% EMA
validity [13.7, 20.3]%. 3.1x improvement over `unconditional_v3`. Mean object count collapsed
to 2.66 (model learned to place fewer, more valid objects). Shipped as the conditional baseline.

### conditional_v3: longer training + engineering improvements

Three changes from v2:
1. `max_steps=200000` (double v2's 100k)
2. Cosine LR warm restart at step 100k (`cosine_restart_lr=1e-4`)
3. `bfloat16` mixed precision + `batch_size=256` (up from 128)

Stopped at step 115,000 (budget constraint). Results:
- **model_state:** 18.0% [14.6, 21.4]%
- **EMA:** 17.4% [14.1, 20.7]%

**Decision: Case B (statistical tie with v2).** v3 EMA (17.4%) and v2 EMA (17.0%) overlap
within their 95% CIs. v3 does not demonstrate a clear improvement at step 115k. The cosine
LR restart was not reached (set at step 100k; training stopped at 115k with insufficient
post-restart signal). v3 is documented as a validated extension experiment. **v2 remains the
official shipped baseline for Week 4 planning purposes.**

---

## Throughput Investigation

During v3 training, real wall-clock throughput was measured at 5.6-6.9 steps/sec against a
logged 18-22 steps/sec (active training window only). GPU utilization: consistently 0-8%
across 10 consecutive nvidia-smi readings at 2s intervals.

**Root cause:** `run_validation()` called synchronously every 10 epochs (1940 steps). Each
pass runs 100 scenes through Drake RRT at `rrt_budget_s=2.0`. At 17% validity, 83% of scenes
exhaust the full 2s budget. Estimated 290s per pass, 33 passes in 65k steps = ~9,570s of
GPU-idle Drake work = 57% of total wall-clock.

**Mitigation applied mid-run:** `val_every_epochs: 10 -> 50`, `val_n_scenes: 100 -> 50`.
Expected ~10x reduction in validation overhead. Full fix deferred to Week 4.

**Week 4 recommendation:** Decouple Drake validation from the training loop entirely. Run a
single large probe (500 scenes) on checkpoints at defined milestones rather than inline. Or
run validation in a background subprocess so training continues unblocked.

---

## Rejection Analysis (conditional_v3, EMA head, 500 scenes)

| Failure mode | Count | % of rejections |
|---|---|---|
| RRT planning failed | 225 | 54% |
| IK unreachable | 92 | 22% |
| Interpenetration | 84 | 20% |
| Unstable rest | 12 | 3% |

The dominant failure mode is RRT (robot can't reach the scene from home config), followed by
IK failure. This is consistent with v2. The model is generating scenes in physically plausible
poses but not yet constrained to robot-reachable configurations. Classifier guidance targeting
reachability is the highest-value Week 4 improvement.

---

## CRITICAL FINDING: Object Type Collapse

Task 9 VLM evaluation revealed that the model collapses to `type_id=0` (generic cube) in
~99% of generated scenes, despite the dataset containing 6 distinct object types.

### Evidence

| Signal | Value |
|---|---|
| VLM mean score (Haiku 4.5, 3,054 samples) | 1.09/5 |
| Score distribution | 93% score 1, 6% score 2, 1% score 4 |
| Score-4 samples origin | 100% from `cluttered_pick` (generic cube prompts) |
| Joint Drake-valid + VLM>=4 | ~0% |
| `type_ce` loss at convergence | near-zero (majority-class collapse) |

### Root Cause

Training data imbalance toward `type_id=0` combined with `type_ce` loss weight of 0.1.
The model minimizes total loss by always predicting type_id=0 (majority class) and concentrating
capacity on pose and presence prediction. Standard imbalanced classification failure mode.

### Interpretation

Drake geometric validity numbers remain fully meaningful -- they measure spatial coherence,
not object identity. Text conditioning works on spatial relationships (positions, orientations,
object counts) as evidenced by the 9.8% conditioned validity rate (vs. 5.4% unconditional).
Object identity conditioning is simply not implemented.

This is not a flaw in the evaluation methodology. The VLM harness correctly identified a
real model limitation that was invisible to the Drake validity metric alone. The two evaluation
axes (geometric validity, semantic alignment) are complementary, not redundant.

### Week 4 Remediation -- OUTCOME: FAILED

Attempted interventions in `conditional_v4` (50k steps, warm init from v2):
1. Class-balanced sampling: inverse-frequency type weighting
2. Weighted cross-entropy: `type_ce` raised from 0.1 to 1.0
3. Warm init from `conditional_v2` checkpoint

**Phase A gate evaluation (1,536 conditioned scenes, 200 descriptions):**

| Gate | Threshold | v4 result | Pass? |
|---|---|---|---|
| VLM mean score | > 2.0 | 1.12 | FAIL |
| Joint Drake+VLM>=4 | > 5% | 0.00% | FAIL |
| Per-type diversity | >=5/12 types | 6/12 (only 2 with real presence) | FAIL |
| Type 0 fraction | < 50% | 88.7% | FAIL |

Drake validity regressed from 9.8% (v2 conditioned) to 7.49% (v4 conditioned).
Interpenetration rate tripled (20% to 49.3% of rejections).

**`conditional_v2` remains the shipped conditional baseline.** `conditional_v4` is abandoned.

Loss reweighting + class-balanced sampling are insufficient interventions. Root cause
investigation is required to identify whether the failure is in the text encoder embedding
space, the type embedding matrix, the output head bias, or the cross-attention conditioning
pathway. See `artifacts/conditional_v4_phase_a_results.md` for full gate results.

---

## Artifacts

- `checkpoints/conditional_v2/latest.pt` -- **shipped conditional baseline** (100k steps)
- `checkpoints/conditional_v3/latest.pt` -- v3 final checkpoint (115k steps)
- `checkpoints/conditional_v4/latest.pt` -- v4 abandoned (50k steps, failed Phase A)
- `artifacts/conditional_v3_final_probe.json` -- raw probe results (both heads)
- `artifacts/conditional_v4_phase_a_results.md` -- full Phase A gate results for v4
- `artifacts/conditional_v4_vlm_probe.json` -- raw v4 Phase A probe (1,536 scenes)
- `artifacts/conditional_vlm_eval.md` -- Task 9 VLM evaluation report (type collapse finding)
- `artifacts/week3_vlm_eval/haiku_scores.summary.json` -- Haiku bulk judge summary
- `artifacts/week3_vlm_eval/sonnet_cross_val.summary.json` -- Sonnet CV summary
- `artifacts/week3_vlm_eval/sonnet_disagreement.summary.json` -- 290-case Sonnet re-judge
- `logs/v3_final_probe.log` -- full probe stdout
- `logs/v4_vlm_probe.log` -- full v4 Phase A probe stdout
- `configs/train/conditional_v3.yaml` -- v3 training config
- `configs/train/conditional_v4.yaml` -- v4 training config (failed intervention)
- W&B project: `galanafai-self/demiurge`, runs `iqfebjel` (v2), `wvdoaybk` (v3), `9idj9jem` (v4)
- Total VLM API cost: $3.47 (Week 3) + $1.43 (v4 Phase A) = $4.90
