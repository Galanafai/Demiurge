# Week 4 Preview: Classifier Guidance and Evaluation

**Status:** Planning. Week 3 shipped `conditional_v3` (EMA 17.4% [14.1, 20.7]%) and
`conditional_v2` (EMA 17.0% [13.7, 20.3]%) as the established conditional baseline.

**Week 3 final state:**
- Unconditional baseline: `unconditional_v3`, 5.4% validity
- Conditional baseline: `conditional_v2` / `conditional_v3`, 17.0-17.4% EMA validity
- 3.1-3.2x improvement over unconditional baseline confirmed via 500-scene Drake probe

---

## Week 3 Lessons Applied to Week 4

### 1. Drake validation is not an inline training metric

The biggest practical finding of Week 3: synchronous Drake validation inside the training
loop consumed 57% of wall-clock time (GPU idle at 0%, CPU-bound BiRRT at 2s/scene).

**Week 4 validation architecture:**
- Remove inline `run_validation()` calls from the training loop entirely
- Track loss as the training-time proxy for model quality
- Run 500-scene Drake probes as explicit evaluation milestones (not during training)
- If intermediate checkpoints need validity estimates: use a fast headless check
  (interpenetration + stable_rest only, skip IK/RRT) as a cheap proxy

### 2. CFG dropout is load-bearing

CFG dropout at 15% was the single fix that took conditional validity from 1.0% to 17%.
Week 4 must preserve CFG dropout in all classifier-guided variants. The guidance multiplier
is applied at inference time; training always uses 15% null conditioning.

### 3. Architecture capacity is not the bottleneck at step 100k

v3 at 115k steps is statistically tied with v2 at 100k steps. Loss plateau at 1.20-1.24 from
epoch 50 onward. The model is not underfit -- 8.89M parameters is sufficient for 50k scenes
at this scene complexity. Scaling to d_model=512 is not the next lever; better guidance is.

---

## Week 4 Objectives

### Primary: Classifier guidance for reachability

**Motivation:** Dominant rejection mode in v3 is RRT failure (54%) followed by IK
unreachability (22%). Together, 76% of rejections are robot-reachability failures. The model
generates physically plausible scenes but does not know what the UR5e can reach.

**Approach:** Train a lightweight Drake-validity classifier on the 50k dataset (binary label:
did the scene pass all four Drake checks?). Use its gradient (or logit) as a guidance signal
during DDIM sampling.

**Implementation options:**
- Gradient-based classifier guidance (Dhariwal and Nichol 2021): `x_t <- x_t + s * grad_x log p(valid | x_t)`. Requires the classifier to be differentiable with respect to the noisy scene input.
- Logit-based rejection sampling: run classifier at inference time, filter scenes below a
  validity threshold before Drake validation. Simpler, no gradient needed.

**Baseline comparison:** Compare guided sampling validity rate vs. unguided conditional
sampling at the same number of DDIM steps. Success criterion: >25% validity with guidance
vs. 17.4% without.

### Secondary: Evaluation harness (Week 3.5 deferred)

Per the six-layer architecture, Layer 6 (evaluation) was deferred. Week 4 should produce:
- `src/eval/validity_rate.py` -- 500-scene Drake probe, wraps the probe script into a library
- `src/eval/diversity.py` -- mean pairwise distance in scene space
- `src/eval/task_relevance.py` -- embedding similarity between generated scene description
  and target text prompt
- `src/eval/rrt_success.py` -- downstream RRT success rate on accepted scenes

### Stretch: Larger model variant

If classifier guidance does not break the validity plateau:
- Scale to d_model=512 (32M params, 4x current)
- Requires new training run; ~6h with fixed validation architecture
- Keep d_model=256 as the comparison baseline

---

## Week 4 Config Starting Point

Baseline: `conditional_v3.yaml` with these changes:
```yaml
training:
  max_steps: 100000         # guidance training is fast
  val_every_epochs: 999999  # disable inline validation
  val_n_scenes: 0           # disabled
  lr: 1.0e-4                # lower LR for fine-tuning on top of v3
  warm_init_from: checkpoints/conditional_v3/latest.pt
  warm_init_keys: model_state
```

Classifier head: separate script, separate checkpoint. Does not modify the denoiser.

---

## Week 4 Build Order

1. Dataset labeling: add Drake validity label to each scene in the 50k dataset
2. Classifier training: small MLP on top of frozen scene features
3. Guidance integration: plug classifier gradient into `DDIMSampler.sample()`
4. Evaluation harness: `src/eval/` modules
5. Final ablation: guided vs. unguided validity rates, diversity, task relevance
6. Ship Week 4 with classifier guidance + evaluation numbers

---

## Open Questions

- Does the classifier need to be trained on noisy inputs (x_t) for gradient guidance, or can
  it be trained on clean scenes (x_0) and used in an x_0-prediction guidance scheme?
- What is the right guidance strength `s`? Too high drives samples out of the data manifold.
- Does the guidance interact well with CFG? (Two guidance signals: text conditioning + validity)
