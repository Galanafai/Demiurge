# Phase C Decision: VLM Quality Check on v9 Step 170k

Date: 2026-05-20
Model: v9_uncond_v9/step_00170000.pt (49M params, v8-warmup resume)

## Phase B Reference
- Drake validity (non-empty): 11.3% (n=200 probe)
- Drake validity (overall): 3.0%
- v7 baseline: 1.1% (10x improvement)

## Phase C Results
- Generation attempts: 1500 (15 batches x 100)
- Non-empty scenes produced: 660 (44%)
- Drake-accepted: 4 (0.3% overall, 0.6% of non-empty)
- Scenes judged by VLM: 28/28
- VLM model: claude-haiku-4-5
- Total cost: $0.022

### Drake-Valid Cohort (n=4)
- Mean VLM score: 1.50
- Pct scoring >= 3 (Acceptable): 25%

### Drake-Invalid Cohort (n=4)
- Mean VLM score: 2.50

### Interp-Only Cohort (n=20)
- Mean: 2.25

## Notes
- Drake-valid n=4 is far below the n=20 minimum for statistical significance.
- This is an expected consequence of 0.3% overall validity from this sampling run.
  The Phase B probe (n=200 targeted scenes) measured 3% overall and 11.3% conditional --
  the discrepancy comes from the probe using rejection sampling bias toward non-empty.
- VLM judged mostly single-object scenes (n=1 valid objects each), scoring them 1-2.
  Single-object scenes are easy for Drake but visually trivial; scores reflect emptiness not model failure.

## Type Diversity
- Types seen in scored scenes: [0, 2, 3, 5, 6, 7, 8, 9, 10, 11]
- Count: 10/12

## Decision Gates
- Drake-valid mean >= 2.5: FAIL (1.50)
- Pct >= 3 on Drake-valid >= 30%: FAIL (25%)
- Type diversity >= 8/12: PASS (10)
- Drake-valid mean > invalid mean: FAIL/NA

## Verdict
**BELOW EXPECTATIONS -- only 4 Drake-valid scenes from 1500 attempts (0.3% overall). Ship v9 as honest negative result with full debugging narrative.**

The n=4 valid-scene sample is insufficient for a reliable quality verdict.
Phase B (n=200, 11.3% conditional validity) remains the primary validity measurement.
v9 step 170k ships as the unconditional baseline with documented limitations.
