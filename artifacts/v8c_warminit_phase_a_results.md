# v8c Warm-Init Phase A Results

## Architecture
- d_model=384, n_layers=16, n_heads=6, ffn_mult=4
- 48.65M parameters (5.4x v7)
- Warm-init: v7 weights zero-padded from d_model=256 to d_model=384
- Training: 200,000 steps (resumed from step 2,000 smoke test)
- Config: configs/train/conditional_v8c_warminit.yaml
- WandB run: qxmwbe4x

## Final Training Losses (step 200k)

| Component | Value |
|---|---|
| loss/total | 2.107 |
| loss/pose_rot | 0.773 |
| loss/pose_xyz | 0.366 |
| loss/scale | 0.240 |
| loss/type_ce | 0.586 |
| loss/presence_bce | 0.215 |
| train/grad_norm | 3.580 |

## Drake Probe Trajectory

| Step | Scenes | Validity | Interp% | RRT% |
|---|---|---|---|---|
| 2,000 (smoke) | 50 | 0.0% | 98% | 0% |
| 10,000 | 50 | 0.0% | 100% | 0% |
| 20,000 | 100 | 0.0% | 16% | 52% |
| 50,000 | 100 | 5.0% | 22% | 64% |
| 100,000 | 100 | 8.0% | 18% | 68% |
| 200,000 (final) | 500 | **0.2%** | 97% | 2.4% |

Note: Model showed promising trajectory at 50k-100k (interp declining, validity appearing)
but regressed at 200k. The spatial prior collapsed in the unconditional path during
extended conditional training.

## Phase A Results

### Unconditional Drake (500 scenes, text-mode=none, step 200k)

- Accepted: 1/500 (**0.2%**)
- Interpenetration: 484/500 (97.0%)
- RRT failed: 12/500 (2.4%)
- IK unreachable: 2/500 (0.4%)

### CFG Sweep (100 scenes per scale)

| cfg_scale | validity |
|---|---|
| 0.0 | 0.0% |
| 0.5 | 1.0% |
| 1.0 | 0.0% |
| **1.5** | **2.0%** (peak) |
| 2.0 | 1.0% |
| 3.0 | 0.0% |
| 5.0 | 0.0% |

### Eps Correlation (5 prompts)

Mean: **0.9942** -- CFG training succeeded. Collapse is a sampling issue, not weights.

### 5-Regime Ablation (cfg=1.5, 200 scenes each)

| Regime | v7 | v8c_WI | Delta | Interp | RRT | IK |
|---|---|---|---|---|---|---|
| none | 6.5% | 0.0% | -6.5% | 193 | 4 | 2 |
| type-only | 0.5% | 0.0% | -0.5% | 163 | 28 | 8 |
| position-only | 1.0% | 1.0% | +0.0% | 149 | 32 | 11 |
| full-cond | 1.0% | 1.0% | +0.0% | 119 | 75 | 4 |
| held-out | 1.5% | 2.0% | +0.5% | 164 | 22 | 10 |

### Phase 8: VLM Probe

**SKIPPED** -- uncond validity 0.2% < 2% threshold. Scores on a collapsed
geometry distribution are not meaningful.

## Outcome Classification

CASE B (parity) for conditioned regimes + CASE C (regression) for unconditional.
Scaling from 9M to 49M via warm-init did NOT improve Drake validity.

## Root Cause

The 49M model learned CFG conditioning perfectly (eps_corr=0.994) but the
unconditional spatial prior collapsed during extended training. Position-only
and full-cond conditioning reduces interpenetration from 97% to 60-75%,
enabling 1-2% validity -- confirming the model CAN produce valid scenes
when the text signal is strong enough to override the collapsed prior.

## Week 5 Direction

Abandon scaling. Return to v7 (9M) as base for Universal Guidance.
