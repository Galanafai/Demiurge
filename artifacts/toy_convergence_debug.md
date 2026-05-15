# Toy Convergence Debug Report

**Generated:** automatically on FAIL

## Summary

| Metric | Value |
|---|---|
| Steps completed | 24001 |
| Wall-clock | 30.8 min |
| Validity rate | 0.000 (0/100) |
| Threshold | 0.90 |
| Status | FAIL |

## Loss Curve

| Window | Avg Loss |
|---|---|
| First 100 steps | 4.8367 |
| Last 100 steps | 1.3909 |
| Reduction | 71.2% |

## Failure Mode Hypotheses

Check the following in order before patching:

1. **Rotation projection gradient issue**: If rot6d loss is not decreasing,
   verify that gradients flow through `rot6d_to_matrix` (Gram-Schmidt).
   Run `tests/model/test_rotations.py::test_gradient_flow`.

2. **Presence head collapse**: If all sampled scenes have presence_bit < 0 for
   all slots, the presence BCE loss may be dominating and the model predicts
   all-absent. Increase `presence_bce` weight or verify BCE target is correct.

3. **Scale normalisation**: Toy scenes have scale=0 in normalised space.
   If the model is predicting large scale noise, the denormalized scene may
   place objects outside workspace bounds. Check `denormalize()` output.

4. **Validation failure mode**: Run a small manual validation to see which
   Drake checks are failing (non-interpenetration, stability, IK, RRT).
   The rrt_budget_s=1.0 in toy mode may be too tight; verify against
   the profile run budget.

## Next Steps

- Do NOT start full GPU training until toy passes.
- Do NOT extend the 30-minute budget. Diagnose first.
- Address the highest-probability failure mode above, re-run.
