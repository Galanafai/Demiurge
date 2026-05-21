# Demiurge: Design Decisions

The **why** behind key implementation choices. Cross-referenced with the
bug inventory where relevant.

## `src/model/denoiser.py` -- SceneDenoiser

**49M parameters, d_model=384:** Scaling sweet spot. 10M showed underfitting
on rotation distribution (type_ce converged, geometry did not). 100M showed
no measurable improvement and tripled compute.

**ffn_mult=4:** Standard transformer ratio. Increasing to 6 showed no improvement
in v8c ablations.

**Slot ID embeddings:** Slots are technically permutation-invariant but ID
embeddings help the model develop slot-specific representations. Audit Phase 10
verified symmetry is preserved at init (slot correlation ratio = 0.014).

**No type_grad_isolation:** v1 used this and the model memorized type labels
without denoising (Bug #1). Removed permanently.

## `src/model/schedule.py` -- CosineSchedule + DDIMSampler

**zero_terminal_snr:** Per Lin 2024, standard cosine schedules leak signal
at t=T (alpha_bar(T) > 0). With narrow distributions like tabletop scenes,
the model can always partially infer the conditional mean even at max noise.
Zero terminal SNR forces the model to learn from pure noise at T.

**Caution:** alpha_bar=0 at t=T creates a Tweedie singularity. Every
computation involving `(x_t - sigma*eps) / sqrt(alpha_bar)` needs the
`ALPHA_MIN=1e-4` guard (Bug #15 lesson).

**prediction_type='v' must be threaded everywhere:** Bug #6 showed that
hardcoding epsilon decode in the sampler silently corrupts v-prediction
output. The DDIMSampler branches on `prediction_type` and converts v->eps
before the DDIM step.

**50 DDIM steps:** Tested 20, 50, 100. Quality plateaus at 50 for our
distribution. 20 showed visible artifacts in rotation.

## `src/model/rotations.py` -- rot6d representation

**rot6d over quaternions:** Avoids the sign ambiguity (q and -q represent
the same rotation) and the antipodal singularity in quaternion loss. Any
6D vector maps to a valid rotation via Gram-Schmidt, making the representation
continuous.

**Why not just predict yaw?** Training data has yaw-only rotations but
predicting full rotation lets the model learn the constraint softly. Hard
projection at inference (Bug #10 fix) worked for v7 but caused degenerate
rotations in v9 (Bug #13). The v9 model learned the constraint sufficiently
that hard projection is unnecessary and harmful.

## `src/scene/typed_scene.py` -- Typed wrappers

Bugs #2-5 all involved different code paths applying different normalization
sequences to the same model output. The fix: explicit types that prevent
implicit re-application of transformations.

```python
# Pipeline (enforced by types):
WhitenedScene   -- model output space (zero mean, unit variance per channel)
    -> .unwhiten()
NormalizedScene  -- [-1, 1] per workspace dimension
    -> .denormalize(bounds)
PhysicalScene    -- meters, quaternions, ready for Drake
```

Each conversion is a one-way function on the type. Calling `.denormalize()`
on a PhysicalScene is a type error.

## `src/validator/core.py` -- SceneValidator

**Validation order (interpenetration -> stable -> IK -> RRT):** Ordered
by compute cost. Interpenetration is geometric (cheap). Stability requires
forward sim (~100ms). IK is iterative (~10ms). RRT is stochastic (~1s).
Sequential rejection means most scenes never reach RRT.

**Fresh Context per call:** Drake Contexts hold mutable state. Reusing a
Context across calls with mutated plant positions causes stale collision
queries. SceneValidator creates a fresh plant per scene (expensive but correct).

**SetDefaultFreeBodyPose (not SetDefaultFloatingBaseBodyPose):** Bug #7.
Drake renamed this API. Wrapped in a try/except with fallback to old name.

## `src/training/train.py` -- Loss weights

**pose_rot=2.0 (not 1.0):** rot6d has 6 channels vs xyz's 3. With equal
weights, each rot6d channel receives half the per-channel gradient. Doubling
pose_rot equalizes per-channel gradient magnitude.

**presence_bce=1.5, pos_weight=5.0:** Starting point: pos_weight = neg_rate/pos_rate
= 0.756/0.244 = 3.1. v7 showed this was insufficient (Bug #8). Increased
to 5.0. The 1.5 global weight on presence BCE balances against the geometry MSE.

**EMA (exponential moving average):** EMA weights (ema_state in checkpoint)
are more stable than the active training weights. All probes use EMA.
All warm-init resumes load from ema_state.

**Warm-init from ema_state:** Each version inherits learned representations
from the previous. v8 inherits v7's IK distribution. Without warm-init,
fresh training (v10) failed to learn IK reachability (architectural ceiling).

**Caution (Bug #12):** Changing loss weights during warm-init disrupts the
Adam optimizer state (m1, m2 mismatch). Keep loss weights identical at
warm-init or restart Adam.

## `src/data/sampler.py` -- Procedural scene generation

**No IK feasibility filter (known limitation):** All 234k scenes satisfy
interpenetration, stability, and RRT constraints. But we did NOT filter
for IK reachability beyond procedural placement heuristics.

This is the primary architectural ceiling. The model learns "scenes look
like training data" but the loss doesn't enforce "scenes must be IK-reachable".

**The unimplemented fix:** In `ProceduralSampler.sample()`, after placing
objects, call `SceneValidator.check_ik(scene)` and reject if any object
is outside the UR5e workspace. This makes IK feasibility a data constraint
rather than a loss constraint.

## `src/guidance/energy.py` -- pairwise_overlap_energy

**Squared penalty:** Smooth gradient (zero at non-overlap, quadratic growth).
Avoids discontinuities.

**Channel 12 must be masked:** Bug #14. The energy depends physically on
xyz (channels 0-2) and scale (9-11), but autograd.grad flows through all 13
channels. Without masking, the gradient pushes presence_logit (channel 12)
negative, collapsing all scenes to empty.

```python
grad = torch.autograd.grad(energy.sum(), x0_g)[0]
grad[:, :, 12] = 0.0  # REQUIRED: presence_logit must not be guided
```

**Why overlap energy doesn't help:** At v9 step 170k, interpenetration pass
rate is 100%. The overlap energy targets a constraint that is already satisfied.
UG with overlap energy is optimization in the wrong direction.

## `configs/train/` -- YAML configs per version

One config file per training version (v9_uncond_v7.yaml, v9_uncond_v8.yaml,
v9_uncond_v9.yaml). This makes every run reproducible without git archaeology.
"What was the config at step 170k?" is answered by reading the file, not by
checking out a commit.

**warm_init_keys: [ema_state]:** We warm-init from EMA weights (not raw
training weights). EMA represents the model's "settled" state; raw weights
can be temporarily perturbed by large gradient steps.
