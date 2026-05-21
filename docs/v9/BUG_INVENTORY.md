# Demiurge v9: Bug Inventory

**15 bugs documented** across phases v1 through Phase E.  
Each entry: phase, file, symptom, root cause, fix, lesson.

**Final result after all fixes:** 3.0% Drake validity overall, 11.3% on non-empty (n=200).

---

## Severity Summary

| Severity | Count | Examples |
|---|---|---|
| Critical (project-blocking) | 6 | Bugs 6, 7, 13, 14, 15 |
| High (significant impact) | 5 | Bugs 3, 5, 8, 10, 12 |
| Medium (measurement impact) | 4 | Bugs 2, 4, 9, 11 |

## Bugs By Phase

| Phase | Bugs |
|---|---|
| v1-v2 (initial) | #1 |
| v1-v5 (probe pipeline) | #2, #3, #4, #5 |
| v5 (sampler) | #6 |
| v6 (Drake API, presence) | #7, #8 |
| v7-v9 audit | #9, #10, #11, #12 |
| Phase E (UG sweep) | #13, #14, #15 |

---

## Bug #1: type_grad_isolation memorization shortcut 🔴

| Field | Detail |
|---|---|
| **Severity** | critical |
| **Phase** | v1-v2 |
| **File** | `src/model/denoiser.py` |
| **Symptom** | type_ce stuck at 0; model outputs constant scenes |

**Diagnosis:** Gradient isolation on type embedding created a shortcut: the model learned to copy type labels without denoising, bypassing the diffusion process entirely.

**Fix:** Removed use_type_grad_isolation flag from DenoiserConfig

> **Lesson:** Architectural ablations can introduce silent memorization paths that look like convergence

---

## Bug #2: Probe presence threshold 0.0 vs trained -0.589 🟡

| Field | Detail |
|---|---|
| **Severity** | medium |
| **Phase** | v1-v4 |
| **File** | `scripts/probe_drake.py` |
| **Symptom** | Probe undercounted non-empty scenes by ~30% |

**Diagnosis:** Training used logit threshold -0.589 (calibrated to 24.4% positive class rate via -log((1-p)/p)). Probe used naive >0 check. This caused the probe to miss scenes the model considered non-empty.

**Fix:** Parameterized threshold at -0.589 = -log((1-0.244)/0.244) to match class balance

> **Lesson:** Every decode-time threshold must trace to a training-time hyperparameter

---

## Bug #3: Double denormalization in probe pipeline 🟠

| Field | Detail |
|---|---|
| **Severity** | high |
| **Phase** | v4 |
| **File** | `scripts/probe_drake.py` |
| **Symptom** | Probe output at std=5+; all scenes rejected as interpenetrating |

**Diagnosis:** denormalize() was called twice: once inline before SceneTensor construction, then again via SceneTensor.denormalize(bounds). xyz coordinates scaled to ~5m instead of 0.5m.

**Fix:** Removed the redundant inline denormalize() call

> **Lesson:** Pipeline composition with side-effecting methods needs typed stages to prevent double-application

---

## Bug #4: Inline validator denormalization mismatch 🟡

| Field | Detail |
|---|---|
| **Severity** | medium |
| **Phase** | v4 |
| **File** | `src/training/inline_validator.py` |
| **Symptom** | Training-time validity % diverged from probe % |

**Diagnosis:** The inline validator (run during training to log validity) skipped the workspace-centering step that probe_drake.py applied. Scenes appeared valid inline but failed the probe.

**Fix:** Matched all denormalization steps between inline validator and probe paths

> **Lesson:** Multiple validation paths must share a single decode function

---

## Bug #5: Whitening constants hardcoded differently in train vs probe 🟠

| Field | Detail |
|---|---|
| **Severity** | high |
| **Phase** | v4-v5 |
| **File** | `scripts/train.py, scripts/probe_drake.py` |
| **Symptom** | Probe validity dropped even after threshold fix |

**Diagnosis:** Training used DATA_STD_XYZ from the corpus statistics file. Probe had the same constants hardcoded but with 3 vs 4 significant figures rounding differences.

**Fix:** Single source of truth: import from scene/schema.py in both paths

> **Lesson:** Magic numbers must live in exactly one place and be imported everywhere else

---

## Bug #6: DDIM sampler hardwired epsilon decode for v-prediction model 🔴

| Field | Detail |
|---|---|
| **Severity** | critical |
| **Phase** | v5 |
| **File** | `src/model/schedule.py:DDIMSampler.sample()` |
| **Symptom** | Model output exploded to std=5.32; all scenes NaN after 5 steps |

**Diagnosis:** The DDIMSampler.sample() method unconditionally used epsilon decode: x0 = (x_t - sigma*eps) / alpha. When the model was trained with v-prediction (output = sqrt(alpha)*eps - sqrt(1-alpha)*x0), this formula produces the wrong x0 estimate, diverging rapidly.

**Fix:** Added explicit branch: if prediction_type=='v': convert v_pred to eps first, then use standard DDIM. Verified with std checks after step 1.

> **Lesson:** Samplers must branch on prediction_type. Never hardcode decode convention.

---

## Bug #7: Drake API SetDefaultFloatingBaseBodyPose renamed 🔴

| Field | Detail |
|---|---|
| **Severity** | critical |
| **Phase** | v6 (discovered) |
| **File** | `src/validator/core.py:SceneValidator.validate()` |
| **Symptom** | 100% validator failures (AttributeError) across all model versions |

**Diagnosis:** Drake renamed SetDefaultFreeBodyPose to SetDefaultFloatingBaseBodyPose between versions. Every validator call silently failed with AttributeError. Affected all probes run before the fix.

**Fix:** Updated to SetDefaultFreeBodyPose (the new API) with deprecation comment

> **Lesson:** Pin physics simulator versions or add an API surface smoke test on import

---

## Bug #8: Presence collapse: insufficient pos_weight in BCE loss 🟠

| Field | Detail |
|---|---|
| **Severity** | high |
| **Phase** | v6-v7 |
| **File** | `src/model/loss.py:PresenceLoss` |
| **Symptom** | Non-empty scenes degraded from 40 to 2 per 100 over 50k steps |

**Diagnosis:** BCE loss with pos_weight=3.1 was insufficient for 24.4% positive rate. The negative gradient on absent slots (76% of all slots) overwhelmed the positive gradient on present slots. The model learned the trivial solution of predicting all-absent.

**Fix:** Increased pos_weight from 3.1 to 5.0 (= 0.756/0.244 balanced). Added presence_bce weight 1.5 in total loss.

> **Lesson:** For class-imbalanced BCE, set pos_weight = neg_rate/pos_rate as a starting point, then tune

---

## Bug #9: X-coordinate spread deficit 🟡

| Field | Detail |
|---|---|
| **Severity** | medium |
| **Phase** | v6-v9 |
| **File** | `Training distribution vs loss function` |
| **Symptom** | x_w_std stuck at 0.637 (target 1.0); objects concentrated at x=0 |

**Diagnosis:** Model consistently predicted x coordinates too close to zero. The training data has good x spread (std=1.0 whitened) but the diffusion loss does not explicitly penalize low variance. The model minimized MSE by shrinking toward the conditional mean.

**Fix:** Partially addressed with pose_rot=2.0; fundamental fix requires IK-feasibility filter in sampler.py

> **Lesson:** Implicit diversity requirements are not learnable from MSE alone; need explicit variance loss or data filtering

---

## Bug #10: Yaw-only constraint not enforced at inference 🟠

| Field | Detail |
|---|---|
| **Severity** | high |
| **Phase** | v9 audit |
| **File** | `scripts/probe_drake.py and inference scripts` |
| **Symptom** | Objects tilted at arbitrary angles; stable_rest=5% on v7 |

**Diagnosis:** Training data has q_x=q_y=0 exactly (yaw-only rotations). The diffusion model predicts full 6D rotation but the learned distribution is yaw-only. Without inference projection, the model occasionally outputs small but nonzero pitch/roll which Drake detects as unstable.

**Fix:** Project rot6d[2]=0, rot6d[5]=0 at inference time (Bug #13 later showed this can also backfire)

> **Lesson:** Soft constraints learned from data may need hard projection at inference, but verify it doesn't introduce degenerate cases

---

## Bug #11: Loss balance measurement at untrained model misinterpreted 🟡

| Field | Detail |
|---|---|
| **Severity** | medium |
| **Phase** | v9 audit Phase 5 |
| **File** | `artifacts/pipeline_audit.md` |
| **Symptom** | Audit claimed presence dominated 60% of gradient at init |

**Diagnosis:** The audit measured loss component magnitudes at a randomly initialized model. At init, presence BCE dominates because untrained type logits and geometry are near-uniform. The trained model shows 11.6% presence contribution, which is fine.

**Fix:** No change needed; added note in audit that gradient balance must be measured at trained model

> **Lesson:** Always specify the training step when reporting gradient balance measurements

---

## Bug #12: LR scheduler fast-forward destabilizes warm-init optimizer state 🟠

| Field | Detail |
|---|---|
| **Severity** | high |
| **Phase** | v8 |
| **File** | `scripts/train.py:resume logic` |
| **Symptom** | v8 presence collapsed from 40 to 7 non-empty over 20k steps after warm-init |

**Diagnosis:** When resuming from a v7 checkpoint with a fresh CosineAnnealingLR scheduler, the training script replayed scheduler.step() N times to catch up to the correct LR. But Adam's first and second moment estimates (m1, m2) were initialized fresh, not loaded. The mismatch between large LR and fresh momentum caused oscillations that collapsed the presence BCE loss weight.

**Fix:** Either (a) load optimizer state alongside model state for identical-config resume, or (b) start at a lower LR and warm up. v8 used approach (b) implicitly by checkpointing at step 170k before collapse deepened.

> **Lesson:** Warm-init must preserve optimizer-scheduler alignment. Never replay scheduler without matching optimizer state.

---

## Bug #13: yaw_project at inference creates degenerate rotations 🔴

| Field | Detail |
|---|---|
| **Severity** | critical |
| **Phase** | Phase E |
| **File** | `scripts/ug_sweep_v9_uncond.py` |
| **Symptom** | stable_rest dropped to 6/97=6% with yaw_project=True (vs 30% baseline) |

**Diagnosis:** Hard-zeroing rot6d[:,2] and rot6d[:,5] projects to yaw-only rotations, but for rot6d vectors that are not already near yaw-only, this creates degenerate (near-singular) rotation matrices. Drake's stability check uses the rotation matrix to compute the moment of inertia tensor; degenerate rotations produce unstable inertia estimates.

**Fix:** Removed yaw_project from inference. The v9 model has learned the soft constraint and doesn't need hard projection.

> **Lesson:** Hard projection of soft constraints can create degenerate cases worse than the unconstrained solution

---

## Bug #14: UG presence channel gradient leak 🔴

| Field | Detail |
|---|---|
| **Severity** | critical |
| **Phase** | Phase E |
| **File** | `scripts/ug_sweep_v9_uncond.py:sample_with_ug()` |
| **Symptom** | 0/200 non-empty scenes at guidance_scale=2 (all empty after sampling) |

**Diagnosis:** pairwise_overlap_energy(x0_hat, ...) depends physically only on xyz (channels 0-2) and scale (channels 9-11). But autograd.grad(energy, x0_hat) flows nonzero gradient through all 13 channels including presence_logit (channel 12). Each DDIM step applied eps_guided = eps_pred - scale * sigma * grad, pushing presence_logit negative. After 50 steps, all logits were << -0.589 (threshold), making all scenes empty.

**Fix:** Added grad[:, :, 12] = 0.0 after autograd.grad() call. Energy functions must mask gradient on channels they don't physically depend on.

> **Lesson:** When applying UG to structured tensors, explicitly mask gradient channels the energy function is agnostic to

---

## Bug #15: NaN propagation at zero terminal SNR boundary in Tweedie estimate 🔴

| Field | Detail |
|---|---|
| **Severity** | critical |
| **Phase** | Phase E |
| **File** | `scripts/ug_sweep_v9_uncond.py:sample_with_ug()` |
| **Symptom** | All scenes NaN; ne=0 even after Bug #14 fix |

**Diagnosis:** CosineSchedule with zero_terminal_snr=True gives alpha_bar[t=999]=0.0. The Tweedie x0 estimate x0_hat = (x_t - sqrt(1-alpha_bar)*eps) / sqrt(alpha_bar) evaluates to inf/NaN when sqrt(alpha_bar)=0. This NaN propagated through pairwise_overlap_energy and autograd.grad, corrupting eps_guided. All subsequent x tensors became NaN; x[:,12] > -0.589 returns False for NaN inputs, so ne=0.

**Fix:** Added ALPHA_MIN=1e-4 guard: skip Tweedie/UG when alpha_bar < ALPHA_MIN and use eps-only fallback DDIM step.

> **Lesson:** Zero terminal SNR creates a mathematical singularity at t=T. Any Tweedie-based computation needs an explicit numerical guard at the schedule boundary.

---


## Recurring Themes

### 1. Decode/encode asymmetry (Bugs 2, 3, 4, 5, 13)
Every time two code paths applied different decode transformations to the same
model output, they disagreed on which scenes were valid. The fix was typed
scene wrappers (`PhysicalScene`, `NormalizedScene`, `WhitenedScene`) that make
conversions explicit and prevent double-application.

### 2. API drift (Bug 7)
Physics simulator APIs change. A single renamed method caused 100% validator
failure silently across all runs. Mitigation: pin versions and add an API smoke
test on import.

### 3. Numerical boundary conditions (Bug 15)
Zero terminal SNR is mathematically principled but creates a singularity at
t=T (alpha_bar=0). Any computation involving `1/sqrt(alpha_bar)` needs a guard.

### 4. Implicit data constraints not encoded in loss (Bugs 9, 10, 13)
Training data had yaw-only rotations (constraint learned softly). The loss had no
explicit yaw penalty. Inference projection (Bug 10 fix) worked but later caused
degenerate rotations (Bug 13). The fundamental fix is encoding the constraint in
the data filter, not the inference code.

### 5. Warm-init optimizer disequilibrium (Bugs 8, 12)
Changing loss weights or LR schedule at warm-init restart without matching
optimizer momentum state caused training instability. This pattern appeared twice.
