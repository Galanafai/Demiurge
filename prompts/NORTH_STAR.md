# North Star — Six-Week Build Plan

This is the full arc. Each week ships a concrete artifact. Each week's prompt assumes the prior week shipped. Treat the prompts as a North Star, not a contract: re-read and revise the next prompt after each week's actual numbers land.

## The Arc

| Week | Theme | Headline Artifact | Decision Gate to Next Week |
|------|-------|-------------------|----------------------------|
| 1 | Foundations | Validator + 500-scene smoke run | Validator passes all 6 fixture tests; smoke run completes |
| 2 | Data scale-up | 50k-scene sharded dataset + dataset card | Per-template acceptance rate above 20%; diversity baseline established |
| 3 | Model and conditioning | Unconditional + conditional checkpoints with measured validity | Toy convergence passed; conditional validity rate beats unconditional |
| 4 | Classifier guidance | Pareto plot validity rate vs diversity | Honest result reported (guidance dominates or it does not) |
| 5 | Evaluation | Final results table + ablations + gallery | Downstream RRT success metric computed; ablations done |
| 6 | Ship | Live demo + technical post + walkthrough video | Demo URL is public and the post is published |

## Decision Gates Worth Highlighting

**End of Week 2.** If a task template has below 20% acceptance, you have a choice: tune the template (preferred), relax the validator (rarely justified), or drop the template (acceptable if the remaining two cover the story). Do not silently accept low acceptance. Tune the Week 3 prompt's training mix to match what Week 2 actually produced.

**End of Week 3.** If the toy convergence test fails, the project is stuck until it passes. Common failure modes: rotation representation gradient flow, scale normalization, presence head dominance. Do not skip the toy.

**End of Week 4.** If classifier guidance does not Pareto-dominate rejection sampling, the project pivots. Two honest pivots: (a) reframe the contribution around the rejection-sampling-vs-guidance comparison as a negative result with a clean methodology, or (b) implement an energy-based guidance or a distilled validity rejector in a Week 4.5 sprint before committing to Week 5. Decide explicitly in the Week 4 results doc.

**End of Week 5.** If the downstream RRT success rate is not meaningfully higher for generated scenes vs. random scenes, the portfolio story weakens. Surface this honestly. The writeup can still work as a methodology contribution, but the framing changes from "this improves robot training" to "this is a rigorous framework for studying generative scene synthesis under physical validity constraints."

**End of Week 6.** Ship even if the demo is rough. A live URL plus an honest writeup beats a polished plan that never publishes.

## How to Use the Prompts

1. After each week ships, open the next week's prompt in the `prompts/` directory.
2. Re-read it with the prior week's actual results in hand.
3. Edit any task that the data invalidates. The prompts are starting points, not contracts.
4. Paste the (possibly edited) prompt into Antigravity Agent Manager in Plan Mode.
5. Review the Plan Artifact carefully. Push back hard on anything that violates `AGENTS.md`.
6. Approve when satisfied. Execute task by task with brief progress Artifacts in between.
7. After the week ships, update `AGENTS.md` only if an invariant changed. Update the skill only if procedural knowledge changed.

## Total Estimated Time and Compute

- Total wall-clock: 6 weeks of focused work assuming 4 productive hours per day. Realistic real-world span: 8 to 10 weeks.
- Training compute: roughly 50 to 100 GPU-hours total across all experiments (A100 or 4090).
- Validator compute: roughly 40 to 80 CPU-hours total across all dataset generations.
- API costs (VLM judge): under $100 if responses are cached properly. Above $500 if you forget to cache.

## Risk Map

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Validator too slow | Medium | Week 2 task 8 profiles and optimizes |
| Procedural sampler too narrow | Medium | Week 2 task 7 measures diversity explicitly |
| Toy convergence fails | Low | Week 3 task 6 catches it before full training |
| Classifier guidance underperforms rejection | Medium | Week 4 task 8 mandates honest reporting and pivot path |
| Frontend scope creep | High | Week 6 task 2 hard time-box and fallback to notebook demo |
| Writeup gets skipped | High | Treat the writeup as the deliverable, not the model |

The two highest-likelihood risks are frontend creep and writeup skipping. Both are entirely under the agent's control. Both are addressed by the explicit time-boxes and definitions of done in the Week 6 prompt.
