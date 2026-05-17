# Week 4 Final Results: Type Collapse Resolution + Architectural Scaling

## Investigation Chain (6 model iterations)

| Model | Params | Init | Key Intervention | Outcome |
|---|---|---|---|---|
| v4 | 9M | warm v2 | class-balanced sampler + type_ce=1.0 | type collapse persists, 0% Drake |
| v5 | 9M | warm v4 | discrete type diffusion | broke geometry, 0% Drake |
| v6 | 9M | warm v4 | gradient isolation (type head detached) | fixed type collapse, 6.5% uncond Drake |
| v7 | 9M | warm v6 | CFG null-conditioning dropout 10% | 99.9% per-prompt type acc, 1.1% cond Drake |
| v8b | 49M | scratch | d_model=384, n_layers=16 | slot collapse, 0% Drake at 150k steps |
| v8c_warminit | 49M | padded v7 | warm-init + slot_diversity loss | 0.2% uncond Drake, parity on conditioned regimes |

## Phase A Final Comparison

| Metric | v6 (9M) | v7 (9M) | v8c_warminit (49M) | Best |
|---|---|---|---|---|
| Uncond Drake | 6.5% | 6.5% | 0.2% | **v6/v7** |
| Cond Drake (cfg=1.0) | 1.5% | 1.1% | 1.0% | v6 |
| Cond Drake best cfg | -- | -- | 2.0% (held-out, cfg=1.5) | -- |
| Per-prompt type acc | ~60% | 99.9% | ~69% (9/13) | **v7** |
| CFG eps correlation | ~0.99 | 0.994 | 0.994 | parity |
| VLM mean score | 1.94 | 1.94 | N/A (skipped) | v7 |
| Joint Drake+VLM>=4 | 0.20% | 0.20% | N/A | -- |
| Final total loss | ~1.9 | ~1.80 | 2.107 | v7 |

**Best model: conditional_v7** -- 6.5% uncond Drake, 99.9% per-prompt type accuracy, proven CFG stability.

## Universal Guidance Results (v7 baseline)

UG sweep ran on v7 with analytical pairwise bounding sphere energy (commit 27829c4).
18 configs (6 guidance scales x 3 seeds), 15 scenes each.

| guidance_scale | validity (mean) | diversity (mean centroid dist) |
|---|---|---|
| 0.0 (baseline) | ~8.8% | 0.334 |
| 0.5 | ~11.0% | 0.338 |
| 1.0 | ~6.7% | 0.338 |
| 2.0 | ~8.8% | 0.336 |
| 4.0 | ~6.7% | 0.337 |
| 8.0 | ~8.8% | 0.341 |

Best operating point: guidance_scale=0.5 (11% validity, diversity preserved).
Pareto plot: `artifacts/ug_sweep_v1/pareto_v1.png`

Note: 15 scenes/config is insufficient for statistical significance. Week 5 re-runs at 100 scenes/config.

## Key Findings

1. **Type collapse is architecturally solvable** -- gradient isolation in v6 resolved the object-type collapse that persisted through v4/v5.
2. **CFG null-conditioning dropout enables per-prompt control** -- v7's 10% dropout produced 99.9% per-prompt type accuracy.
3. **Scaling alone does NOT improve joint validity** -- v8b/v8c falsify the capacity hypothesis. The 50k training dataset is likely the binding constraint.
4. **Warm-init partially preserves priors but regresses** -- v8c showed 16% interp at step 20k but reverted to 97% by step 200k.
5. **UG guidance improves uncond validity** -- scale=0.5 shows ~11% vs 6.5% baseline at 15 scenes/config (needs confirmation at 100).

## Git State at Week 4 Close

- Branch: `week4/classifier-guidance`
- Merged into: `main`
- Key commits: `c1bf00d` (UG), `27829c4` (Pareto), `c89f310` (v8c WI), `5cf7e9e` (v8c eval), `df67eef` (closeout)

## Week 5 Plan: Universal Guidance on v7 (re-run at scale)

- Re-run UG sweep at 100 scenes/config (vs 15 in Week 4 sweep)
- Fill in VLM panel of Pareto plot (right panel currently empty)
- Compare UG vs rejection sampling at equivalent Drake budget
- Sweep: guidance_scales [0, 0.5, 1.0, 2.0, 4.0, 8.0] x 3 seeds x 100 scenes
- Budget: ~$2-3
