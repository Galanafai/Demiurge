# conditional_v4 Phase A Gate Results

## Overview

`conditional_v4` was trained to fix the type collapse observed in v2/v3. It added:
- `type_ce` loss weight raised from 0.1 to 1.0
- Class-balanced sampler (inverse-frequency weighting)
- Warm init from `conditional_v2` (step 100k)
- 50,000 training steps on the full 50k-scene `data/v1` dataset

**Evaluation:** 200 held-out task descriptions, 8 scenes per description = 1,536 scenes total. VLM judged via Claude Haiku.

---

## Gate Results

| Gate | Threshold | v2 baseline | v4 measured | Pass? |
|---|---|---|---|---|
| 1. VLM mean score | > 2.0 | 1.09 | 1.12 | FAIL |
| 2. Joint Drake+VLM>=4 | > 5% | 0% | 0.00% | FAIL |
| 3. Per-type diversity | >=5/12 types seen | 2/12 | 6/12 | PASS |
| Type collapse proxy | < 50% type 0 | 99% | 88.7% | FAIL |

**1/3 gates passed.**

The single passing gate (type diversity: 6/12 types technically seen) is not meaningful in practice. Only types 0 and 1 have material presence (88.7% and 11.2% respectively). Types 3, 5, 6, 9, 10, 11 were never generated. Types 2, 4, 7, 8 appeared as traces (1-2 objects total across 1,536 scenes).

---

## VLM Score Distribution (1,524 scored)

| Score | Count | Fraction | Interpretation |
|---|---|---|---|
| 1 -- No match | 1,374 | 90.2% | Scene completely unrelated to description |
| 2 -- Poor | 132 | 8.7% | Most objects missing or misplaced |
| 3 -- Partial | 1 | 0.1% | Some objects present but composition wrong |
| 4 -- Good | 15 | 1.0% | Minor issues only |
| 5 -- Perfect | 2 | 0.1% | Matches description |

VLM mean: **1.12** (v2: 1.09). The improvement is within noise. The model is still generating scenes that are almost entirely unrelated to the input task description. This is consistent with type collapse: if the model always generates the same object types regardless of conditioning, it cannot follow type-specific instructions.

---

## Drake Validity Regression

| Metric | v2 conditioned | v4 conditioned |
|---|---|---|
| Drake validity rate | 9.8% | **7.49%** |
| Interpenetration | ~20% of rejections | **49.3%** |
| RRT failed | ~54% | **28.6%** |
| Stability | ~3% | **13.1%** |
| IK unreachable | ~22% | **8.9%** |

> [!WARNING]
> Drake validity regressed by 2.3 percentage points under conditioned sampling. The interpenetration rate nearly tripled (20% to 49.3% of rejections). This suggests that the class-balanced sampler or the higher `type_ce` weight damaged the spatial layout learned by the model -- the geometric loss weight balance may have shifted during fine-tuning from v2.

The standalone unconditioned probe (500 scenes) showed 17.2% validity. The conditioned probe shows 7.49%. The gap suggests the text-conditioning path is actively harmful to geometric validity when type collapse has not been fixed.

---

## Type Distribution (9,207 present objects)

| Type | Count | % |
|---|---|---|
| 0 | 8,168 | 88.7% |
| 1 | 1,034 | 11.2% |
| 2 | 1 | ~0% |
| 4, 7, 8 | 1-2 each | ~0% |
| 3, 5, 6, 9, 10, 11 | 0 | 0% |

The collapse from 99% to 88.7% type 0 in v4 is real but insufficient. Raising `type_ce` by 10x (from 0.1 to 1.0) only moved 10 percentage points. The model is still degenerate.

---

## Conclusion

**Loss reweighting and class-balanced sampling are insufficient interventions for type collapse.** The model has learned to minimize total loss by concentrating on spatial layout while treating type prediction as a constant. The `type_ce` gradient is either not reaching the responsible parameters, or there is a structural bottleneck preventing type discrimination.

Root cause candidates (to investigate):
1. **Text encoder semantic bottleneck** -- type names may be nearly indistinguishable in embedding space, giving the model no gradient signal to differentiate them via cross-attention.
2. **Type embedding degeneracy** -- the learned type embedding matrix may have collapsed, with all type vectors pointing in the same direction.
3. **Output head bias collapse** -- the type head bias for type 0 may be large enough to dominate regardless of logit differences.
4. **Architectural conditioning pathway failure** -- cross-attention on text embeddings is not routing type-specific information to the type prediction head.

---

## Status

- `conditional_v2` remains the shipped conditional baseline (17.0% Drake validity, 1.09 VLM mean).
- `conditional_v4` is **superseded** -- it did not fix type collapse and regressed Drake validity under conditioned sampling.
- Phase B (classifier guidance) is deferred to Week 5 pending root cause fix.
- Investigation phase begins (see `week4_preview.md`).
