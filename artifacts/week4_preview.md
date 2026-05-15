# Week 4 Preview: Classifier Guidance and Evaluation

**Status:** Planning. Week 3 shipped `conditional_v3` (EMA 17.4% [14.1, 20.7]%) and
`conditional_v2` (EMA 17.0% [13.7, 20.3]%) as the established conditional baseline.

**Week 3 final state:**
- Unconditional baseline: `unconditional_v3`, 5.4% validity
- Conditional baseline: `conditional_v2` / `conditional_v3`, 17.0-17.4% EMA validity
- 3.1-3.2x improvement over unconditional baseline confirmed via 500-scene Drake probe
- VLM prompt-following: 1.09/5 mean -- driven by object type collapse (all type_id=0)
- Joint Drake-valid + VLM>=4: ~0% outside cluttered_pick template

---

## CRITICAL FINDING: Object Type Collapse

Week 3 Task 9 VLM evaluation exposed a fundamental model limitation: the model collapses
to generating only `type_id=0` (generic cube) objects in ~99% of samples. Task descriptions
reference specific named objects (mug, cracker_box, mustard_bottle, soup_can). The VLM
judge correctly assigns score 1 to these mismatches.

**Evidence:**
- 2,839 of 3,054 scored samples (93%) received VLM score 1
- The 26 score-4 samples are exclusively `cluttered_pick` prompts using generic "cube" references
- `type_ce` loss converged near-zero (trivially by always predicting majority class)
- Confirmed via Sonnet cross-validation (290 disagreement cases re-judged)

**Root cause:** Training data imbalance toward type_id=0 combined with low `type_ce` weight
(0.1). Standard majority-class collapse in imbalanced multi-class classification.

**Impact:** Drake geometric validity (17% unconditional, 9.8% text-conditioned) remains
meaningful. Text conditioning works on spatial relationships. Object identity is not conditioned.
Joint metric is blocked until this is fixed.

**This is Week 4 priority 0** -- it must be resolved before classifier guidance numbers
are meaningful, since a model that always generates cubes cannot be fairly evaluated on
type-diverse prompts.

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

## Week 4 Objectives (Priority Order)

### Priority 0: Fix object type collapse (NEW -- blocks meaningful VLM metric)

**Problem:** 93% of generated scenes contain only `type_id=0` (cube). `joint Drake+VLM>=4`
rate is ~0% as a result. The VLM evaluation harness works correctly; the model does not.

**Fix options (in order of preference):**
1. **Class-balanced sampling:** During each batch, ensure all 6 object types are represented
   proportionally. Weighted sampler on the dataset keyed by object type distribution per scene.
2. **Weighted cross-entropy:** Explicit inverse-frequency class weights in `type_ce` loss.
   Estimated weights: cube ~0.05, others ~1.0 (roughly 20x upweight on rare types).
3. **Increase `type_ce` loss weight:** From 0.1 to 1.0. May require LR adjustment.
4. **Fine-tune only:** Load `conditional_v2` checkpoint, freeze denoiser layers, fine-tune
   only the type head with balanced batches for 10k steps.

**Success criterion:** VLM mean score > 2.5 on `tabletop_reach` and `cluttered_pick`
templates. `joint Drake+VLM>=4` rate > 5%.

### Priority 1: Classifier guidance for reachability

**Motivation:** Dominant rejection mode in v3 is RRT failure (54%) followed by IK
unreachability (22%). Together, 76% of rejections are robot-reachability failures. The model
generates physically plausible scenes but does not know what the UR5e can reach.

**Approach:** Train a lightweight Drake-validity classifier on the 50k dataset (binary label:
did the scene pass all four Drake checks?). Use its gradient (or logit) as a guidance signal
during DDIM sampling.

**Implementation options:**
- Gradient-based classifier guidance (Dhariwal and Nichol 2021): `x_t <- x_t + s * grad_x log p(valid | x_t)`. Requires differentiability w.r.t. noisy scene input.
- Logit-based rejection sampling: run classifier at inference, filter below threshold. Simpler.

**Baseline comparison:** guided vs. unguided validity rate. Success criterion: >25% with
guidance vs. 17.4% without.

### Priority 2: Decoupled validation architecture

Remove inline `run_validation()` from training loop entirely. 57% of Week 3 wall-clock was
GPU-idle Drake work. Week 4 trains faster with milestone-based 500-scene probes.

### Priority 3: Evaluation harness (Layer 6)

- `src/eval/validity_rate.py` -- wraps probe into a library callable
- `src/eval/diversity.py` -- mean pairwise distance in scene space
- `src/eval/task_relevance.py` -- embedding similarity between scene and prompt
- `src/eval/rrt_success.py` -- downstream RRT success rate on accepted scenes

### Priority 4: Larger model variant (stretch)

Only if priorities 0-2 do not break the plateau:
- Scale to d_model=512 (32M params, 4x current)
- Keep d_model=256 as comparison baseline

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

1. **Type collapse fix:** weighted cross-entropy + class-balanced sampler, 10k fine-tune steps
2. **VLM re-evaluation:** run Task 9 harness on fixed model; confirm mean VLM > 2.5
3. **Decoupled validation:** remove inline Drake from training loop, milestone probe script
4. **Dataset labeling:** add Drake validity label to each scene in the 50k dataset
5. **Classifier training:** small MLP/transformer on noisy scenes, AUC > 0.85 target
6. **Guidance integration:** plug classifier gradient into `DDIMSampler.sample()`
7. **Evaluation harness:** `src/eval/` modules (validity_rate, diversity, task_relevance, rrt_success)
8. **Ablation + Pareto sweep:** guided vs. unguided, guidance scale 0-16
9. Ship Week 4 with all results including honest comparison to rejection sampling

---

## Open Questions

- Does the classifier need to be trained on noisy inputs (x_t) for gradient guidance, or can
  it be trained on clean scenes (x_0) and used in an x_0-prediction guidance scheme?
- What is the right guidance strength `s`? Too high drives samples out of the data manifold.
- Does the guidance interact well with CFG? (Two guidance signals: text conditioning + validity)
