# Demiurge: Technical Writeup

**A 49M parameter conditional diffusion model for 6-DoF tabletop scene
generation, validated by Drake physics. Final result: 3.0% overall Drake
validity (11.3% on non-empty scenes). 15 bugs documented across 11+
training runs. ~$60 total compute.**

This document is the deep technical narrative. For project overview and
decisions, see [README.md](README.md). For specific bugs, see
[BUGS.md](BUGS.md). For design rationale, see [DESIGN.md](DESIGN.md).

---

## Problem Statement

Train a generative model that produces physically valid 6-DoF tabletop
scenes for robot manipulation. Validity is defined by Drake (MIT/TRI
physics simulator) and requires passing four sequential gates:

```
Gate 1: Interpenetration  -- minimum pairwise signed distance > 1mm
Gate 2: Stable rest       -- max pose drift < 5mm/5deg after 1.5s forward sim
Gate 3: IK reachable      -- UR5e InverseKinematics solves to goal above scene
Gate 4: RRT solvable      -- BiRRT finds collision-free path from home to grasp
```

The difficulty: these constraints are global properties of a scene
depending on the UR5e workspace geometry and physics dynamics, while
the diffusion loss is local (MSE on denoised noise vectors per channel
per slot). Diffusion MSE can be minimized while violating any of the
four global constraints.

### Baselines

```
Random procedural sampler with hand-tuned filters: 9% validity
v7 model (smaller, earlier iteration):             1.1% validity
Phase B gate (target for downstream eligibility):  5% validity
```

---

## The Scaling Journey

The project moved through three distinct configurations. Each revealed
a different failure mode and motivated the next change.

### Stage 1: Smaller architecture, limited data (v1 - v7)

**Configuration:** ~9M parameter transformer (12 layers, d_model=256),
v1 procedural corpus (~50k Drake-validated scenes).

**Final result (v7 step 140k):** 1.1% overall Drake validity.

**Diagnostic from v7:**
- Stable rest pass: 5% on non-empty (rotations were unstable)
- IK reachable: 73% on non-empty
- x-coordinate spread (whitened): 0.43 vs target 1.0
- Objects clustered near x=0, falling outside the UR5e workspace

**What this revealed:** The 9M architecture lacked capacity to jointly
model position and 6D rotation. With equal loss weights, rotation
gradients absorbed less per-channel signal than position. The model
prioritized position learning and never adequately learned the rotation
distribution shape.

### Stage 2: 49M parameters, ~50k data (overfitting transition)

**Configuration:** Scaled architecture to 49M parameters (16 layers,
d_model=384, 8 heads, ffn_mult=4). Initially trained on same v1 corpus.

**Diagnostic:**
- Training loss decreased rapidly (overfitting signature)
- Validation loss diverged after ~30k steps
- Drake validity plateaued at very low rates despite low training loss
- Model produced scene configurations matching training scenes exactly
  (verified by k-nearest neighbor distance analysis)

**What this revealed:** A 49M parameter model on 50k scenes has a ~1000:1
ratio of parameters to training examples. Beyond a critical threshold,
the model memorizes configurations rather than learning the underlying
distribution. This is textbook overfitting, but notably visible in this
domain because Drake validity is sensitive to exact pose values - memorized
poses pass; novel-but-similar poses fail edge constraints.

### Stage 3: 49M parameters, 234k data (v8/v9, the working configuration)

**Configuration:** Same 49M architecture. Combined v1 + v2 procedural
corpora into 234k Drake-validated scenes. Approximately 4.7x more
training data than Stage 2.

**Key intervention identified by audit:** `pose_rot=2.0`. The audit
discovered that rot6d (6 channels) was receiving half the per-channel
gradient that xyz (3 channels) received with equal loss weights:

```
Per-channel gradient with equal weight:
  xyz:   weight=1.0 / 3 channels = 0.333 per channel
  rot6d: weight=1.0 / 6 channels = 0.167 per channel  <- starved

Per-channel gradient with pose_rot=2.0:
  xyz:   weight=1.0 / 3 channels = 0.333 per channel
  rot6d: weight=2.0 / 6 channels = 0.333 per channel  <- balanced
```

**Result at v9 step 170k:** 3.0% overall Drake validity, 11.3% on
non-empty scenes. This was the peak. Subsequent training showed presence
channel collapse from warm-initialization disequilibrium (Bug #12).

### Stage 4: Confirming the ceiling (v10, Phase D, Phase E)

Three additional experiments confirmed the ceiling is loss function
underspecification, not architecture or training procedure.

**v10 (fresh training, no warm-init):** Same 49M architecture, same
234k data, same hyperparameters. Trained from random initialization
instead of warm-initialized from v7.

```
Result: 0% Drake validity across 110k-150k steps
IK pass rate: 0% (vs v9's 81%)
```

Implication: warm-init from v7 was transmitting an implicit
IK-reachable pose distribution that v7 had learned from the procedural
corpus. Without warm-init, diffusion MSE alone could not teach IK
reachability. The constraint is not learnable from MSE because the loss
has no representation for "this xyz is outside the UR5e workspace."

**Phase D (text conditioning):** Added cross-attention layers to v9
step 170k for text conditioning. 17 probes across 200k steps. Best
result: 1.0% at 30k steps. Final: 0.33% at 200k.

```
Phase D step 80k:
  IK pass:      4% (vs v9's 81%)
  Stable rest:  1% (vs v9's 30%)

Phase D step 200k:
  IK pass:      1%
  Stable rest:  0%
```

The cross-attention layers required gradient signal that perturbed the
pretrained geometric features. The pretrained model had carefully
balanced position, rotation, scale, and presence learning. Adding new
capabilities via gradient descent disrupted this balance.

**Phase E (Universal Guidance):** Applied Bansal et al. 2023 framework
with pairwise overlap energy. Three implementation bugs found and fixed
before the clean result:

- Bug #13: yaw_project at inference destroyed stability
- Bug #14: presence channel gradient leak collapsed scenes to empty
- Bug #15: NaN propagation at zero terminal SNR boundary (Tweedie singularity)

After all bugs fixed:
```
scale=0.0: 4/200 = 2.0%  (baseline, no guidance)
scale=2.0: 0/200 = 0.0%  (Universal Guidance with overlap energy)
  - Non-empty:  200/200   (presence restored from Bug #14 fix)
  - Stable:       0/200   (UG destroyed stability)
  - IK:          10/200   (UG did not help reachability)
```

The overlap energy targeted interpenetration, which was already at 100%
pass rate - not the binding constraint. The 50 DDIM guidance steps spent
compute optimizing a non-bottleneck while pushing objects into physically
impossible configurations.

### The architectural ceiling conclusion

The loss function is underspecified. Diffusion MSE can be minimized
locally while violating global physical constraints. The model satisfies
the training objective without satisfying the validation criterion.

**The fix is not at the architecture or loss level.** The fix is at the
data level: filter training scenes for IK feasibility before they enter
the corpus. With IK-filtered training data, the model learns the
constraint by exclusion, not by trying to learn it from gradient signal
that does not exist.

This mirrors other generative modeling problems where the constraint
cannot be encoded in differentiable loss (discrete structure, hard
physical bounds, exact equality constraints). The solution is data
filtering or rejection sampling, not loss engineering.

---

## Architecture Details

```
SceneDenoiser (49M parameters)

  Input:  x_t in R^{12 x 13}         -- 12 slots, 13 channels per slot
          type_ids in Z^{12}          -- per-slot object type (12 YCB classes)
          t in Z                      -- diffusion timestep
          text_emb (optional)         -- text conditioning (Phase D only)
  Output: v_t in R^{12 x 13}         -- v-prediction target
          type_logits in R^{12 x 13} -- per-slot type classification

Channel layout (per slot):
  [0:3]   xyz (whitened: zero mean, unit std)
  [3:9]   rot6d (6D rotation, Gram-Schmidt orthonormalization -> SO(3))
  [9:12]  scale (whitened per-axis)
  [12]    presence_logit (sigmoid > -0.589 -> object present)

Internal architecture:
  Sinusoidal embed(t) -> MLP -> t_emb (B, 384)
  type_embed(type_ids) + Linear(13, 384) -> slot tokens
  + slot_id_embed(0..11)  [symmetry breaking]
  Append [NOISE_T] token at position 12

  16 x TransformerBlock:
    AdaLN(t_emb) -> Self-Attention (12+1 tokens, 8 heads, head_dim=48)
    Cross-Attention to text_emb (Phase D only)
    AdaLN(t_emb) -> FFN (mult=4)

  Output heads (applied to scene tokens 0..11):
    head_xyz      -> (B, 12, 3)   v-prediction for position
    head_rot6d    -> (B, 12, 6)   v-prediction for rotation
    head_scale    -> (B, 12, 3)   v-prediction for scale
    head_presence -> (B, 12, 1)   presence logit
    head_type     -> (B, 12, 13)  type classification logits

Diffusion schedule: CosineSchedule(T=1000, zero_terminal_snr=True)
Sampler: DDIMSampler(n_steps=50, prediction_type='v')
```

### Why v-prediction (Salimans and Ho 2022)

Epsilon prediction parameterizes the model as predicting the noise:
```
model(x_t, t) = eps_hat
loss = MSE(eps_hat, eps_true)
```

At t near T (pure noise), the SNR is near zero. eps_hat = eps_true =
random noise. The loss has near-zero gradient signal exactly where the
model most needs to learn the global distribution shape.

v-prediction reparameterizes using a target that mixes epsilon and x_0:
```
v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x_0
model(x_t, t) = v_hat
loss = MSE(v_hat, v_true)
```

The v-target has more uniform loss magnitude across all noise levels.
The model gets gradient signal from both endpoints (clean data and pure
noise) rather than just the noise end. This was the fix in Bug #6 - the
original implementation used epsilon prediction, which was particularly
harmful for a narrow distribution like tabletop scenes.

### Why zero terminal SNR (Lin et al. 2024)

Standard cosine schedules satisfy `alpha_bar(T) > 0`. Even at maximum
training noise, the model sees a faint preserved signal from the clean
data. With narrow distributions like tabletop scenes, this lets the model
partially infer the conditional mean at all timesteps, amplifying mode
collapse.

Zero terminal SNR fixes this: `alpha_bar(T) = 0`. At t=T the input is
pure Gaussian noise with no preserved signal. The model must learn to
generate samples from scratch, not from a partially-inferred mean.

**Numerical hazard (discovered as Bug #15):** `alpha_bar(T) = 0` creates
a Tweedie singularity. Any computation of the form:
```
x_0_hat = (x_t - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)
```
evaluates to infinity at t=T. The Phase E Universal Guidance
implementation hit this immediately - all generated scenes were NaN
until we added an `ALPHA_MIN=1e-4` guard that skips Tweedie estimation
at the schedule boundary.

The lesson: zero terminal SNR is mathematically principled but requires
explicit handling of the schedule boundary in any downstream computation
that uses Tweedie denoising. Treat this as a known pitfall.

### Why per-dimension whitening (Everaert et al. 2024)

The 13 channels have different natural scales after normalization:
```
xyz (normalized to [-1, 1]):               std ~= 0.42
rot6d (Gram-Schmidt output):               std ~= 0.62
scale (normalized):                        std ~= 0.22
presence_logit (logit of 24.4% positive): std ~= 1.50
```

Without per-channel whitening, the diffusion MSE loss is dominated by
the highest-variance channels (presence_logit, rot6d). The model learns
to fit these while ignoring xyz and scale - which explains the z-axis
collapse observed in v3 runs before whitening was introduced.

Per-channel whitening: `z = (x - mean) / std` brings all channels to
zero mean and unit variance. The diffusion loss is then balanced across
channels. Whitening statistics are computed once on the training corpus
and stored as constants (verified in pipeline audit Phase 2).

---

## The 15 Bugs

See [BUGS.md](BUGS.md) for full diagnoses and lessons. Summary by category:

```
Category                     Count   Key examples
-----------------------------------------------------------
Pipeline / decode bugs         6     Bugs #2, #3, #4, #5, #7
Sampler / math bugs            4     Bugs #6, #13, #14, #15
Architecture bugs              1     Bug #1
Loss function bugs             3     Bugs #8, #9, #11
Optimizer / training bugs      1     Bug #12
```

**Most expensive (by cost):** Bug #7 - Drake API rename (`QueryObject`
method renamed between Drake versions). 100% validator failures across
v1-v6, silently. Detected only when the validator suddenly produced 5%
pass rates after the API fix. All v1-v6 validity measurements were
reporting 0% because the validator was throwing caught exceptions rather
than running the actual physics checks.

**Most subtle:** Bug #15 - NaN at zero terminal SNR. Mathematically
correct schedule. Mathematically correct Tweedie formula. But division
by zero at the schedule boundary. Caught by tracing NaN propagation
through the UG implementation rather than by any test.

**Most surprising:** Bug #13 - yaw_project at decode creates degenerate
rotations. The Bug #10 fix (yaw projection to keep objects upright) was
correct for v7 - it raised stable_rest from 5% to 30%. But at v9, where
the model had learned the stability constraint precisely, the same
projection created near-singular rotation matrices (two nearly-identical
columns in the rotation matrix) that destroyed the stability Drake had
learned to validate. Fixes are context-dependent: a correct fix for one
model state can be incorrect for another.

---

## Defensive Infrastructure

Built incrementally as bugs were found. Each component targets a specific
failure class observed in earlier bugs.

### Typed scene wrappers

After Bugs #2-5 all involved different code paths applying different
normalization sequences to the same model output:

```python
# src/scene/typed_scene.py
# Enforced conversion pipeline:
WhitenedScene    -- model output space (zero mean, unit variance per channel)
    -> .unwhiten()
NormalizedScene  -- [-1, 1] per workspace dimension
    -> .denormalize(bounds)
PhysicalScene    -- meters, quaternions, ready for Drake

# Each conversion is a one-way function on the type.
# Calling .denormalize() on a PhysicalScene is a type error caught at import time.
```

This prevents the class of bug where two probes apply different
normalization sequences to the same model output and report different
validity numbers with no error.

### 12-phase pipeline audit

Located in `artifacts/pipeline_audit.md`. Tests one invariant per phase:

```
Phase 1:  Encode/decode round-trip integrity
Phase 2:  Whitening constants verification (training corpus vs probe constants)
Phase 3:  Cross-file consistency (train.py vs probe.py constants)
Phase 4:  Rotation encoding deep dive (rot6d -> quat -> rot6d identity)
Phase 5:  Loss component balance at trained model (corrects Bug #11)
Phase 6:  Drake validator constraint inventory (4-gate funnel breakdown)
Phase 7:  Config vs code mismatch (yaml keys vs script reads)
Phase 8:  Gradient flow per dimension (catches Bug #9 class)
Phase 9:  Model input verification (shapes, dtypes, ranges)
Phase 10: Slot ID embedding symmetry (ratio 0.014, correct behavior)
Phase 11: Type embedding behavior
Phase 12: Failure mode inventory (synthesizes findings)
```

The audit discovered Bugs #9, #10, and #11. Without the systematic
inventory these would have been found one at a time through training
failure investigation.

### 38 unit tests

Each test was added after a bug was found that it would have caught.
Organized by component:

```
tests/
├── test_diffusion_integration.py   - End-to-end diffusion sanity check
├── test_full_sampling_pipeline.py  - Catches Bug #2-#5 class (encode/decode)
├── test_output_sanity.py           - Catches Bug #6 symptoms (std explosion)
├── test_pipeline_invariants.py     - Single-source-of-truth constant verification
├── test_presence_loss.py           - Catches Bug #8 (pos_weight calibration)
├── test_rotation_encoding.py       - rot6d <-> quat identity (Bug #13 vicinity)
├── test_v7_prerequisites.py        - v7 -> v8 transition prerequisites
├── test_validator_api.py           - Catches Drake API renames (Bug #7)
├── test_vpred_conventions.py       - v-prediction math identities (Bug #6 core)
└── ...
```

### Output sanity checks

Runtime detection in the training loop:

```python
# src/training/output_sanity.py
def check_output_sanity(v_pred: torch.Tensor, step: int) -> None:
    """Catch the v5-style output explosion within 100 steps of onset."""
    if v_pred.std() > 3.0:  # observed value: 5.32 at v5 failure point
        raise SanityCheckFailure(
            f"Step {step}: model output std={v_pred.std():.2f}, "
            f"expected ~1.0. Possible v-prediction decode mismatch."
        )
```

This would have caught Bug #6 at step 100 instead of step ~5000 where
we first noticed it via a Drake validity probe.

---

## Cost Discipline

Total project cost: **~$60 across 11+ training runs, audits, sweeps.**

```
v1-v7 runs (smaller arch, multiple iterations):   ~$15
v8 / v9 training (49M, 234k data, 170k steps):     ~$8
v10 fresh training experiment:                     ~$8
Phase D conditional training (200k steps):         ~$10
Phase E UG sweep + debugging:                      ~$8
Audits, probes, smoke tests:                       ~$5
VLM API (Claude Haiku 4.5 for Phase C scoring):   ~$0.022
                                                   ------
Total:                                             ~$54-60
```

Mitigations that kept this manageable:

**Smoke tests before full sweeps.** Phase E smoke test (n=200, 1 scale
value, 1 seed) ran first and immediately showed UG hurt stability. If we
had run the full 18,000-scene sweep without the smoke test, that would
have been an additional $5 wasted on a known-bad configuration.

**Early kill criteria.** Training loops auto-kill if presence collapses
at step 10k (a clear signal of optimizer instability). This would have
killed v8 early if the warm-init hypothesis had been wrong.

**Strategic probe scheduling.** Probes at decision points (20k, 50k,
100k, 130k, 150k, 200k steps) instead of every 5k. Each probe of 200
scenes takes ~3 minutes on Drake. 6 probes vs 40 probes is a meaningful
time saving across a long training run.

**Single pod, multiple background processes.** Training and autonomous
probes ran in parallel on the same pod. No need to spin up separate
infrastructure for evaluation.

**VLM caching.** Cached Claude Haiku responses by scene content hash.
Avoided re-judging identical scenes across multiple runs of the Phase C
evaluation pipeline.

Without these mitigations, this project would have cost $300-500 instead
of $60.

---

## What I Would Do Differently

**1. Build typed wrappers first.** Bugs #2-5 cost roughly 2 weeks of
debugging. The fix (typed scene wrappers) is 200 lines of code. Building
it on day 1 would have prevented the entire class of encode/decode bugs.

**2. Filter training data for IK feasibility.** The architectural ceiling
came from the loss not enforcing IK reachability. Filtering the 234k
training scenes through `SceneValidator.check_ik()` before including them
in the corpus would have removed the ceiling entirely. The hook already
exists in `src/data/sampler.py` - it was marked as "unimplemented next
step" and never executed.

**3. Never change loss weights at warm-init.** v8 (changed `pose_rot`
from 1.0 to 2.0 during warm-init from v7) and Phase D (added
cross-attention layers during warm-init from v9) both demonstrated that
this destabilizes the optimizer. Future warm-init should either keep
config identical or accept restarting Adam from scratch.

**4. Analyze the rejection funnel before designing guidance.** Phase E
spent compute on overlap energy when interpenetration was at 100% pass
rate. The actual binding constraints at v9 were stability and IK.
Designing the energy function for the actual bottleneck would have been
more productive. One analysis query before running the sweep would have
caught this.

**5. Probe with 50 scenes during development, 300+ for final results.**
Statistical significance matters less than iteration velocity during
debugging. A 50-scene probe takes 45 seconds; a 200-scene probe takes
3 minutes. Over 20 debugging iterations this is a meaningful difference.
Save the high-n probes for the headline numbers.

---

## What Is Next (If This Project Continued)

The unimplemented work that would push past the current ceiling:

**1. IK-filtered training corpus.** Add `SceneValidator.check_ik()` to
the procedural sampler's accept/reject loop. Regenerate the 234k corpus
with all scenes passing IK as well as the other three gates. Retrain v9
on the filtered corpus. Expected: IK pass rate climbs from 81% toward
100%, overall validity climbs to ~5-8%.

**2. Larger architecture (100M parameters).** Once data is no longer the
bottleneck (after IK filtering), test whether a 100M model can learn the
implicit IK distribution from IK-filtered data without warm-init
dependency.

**3. Different guidance energy functions.** Universal Guidance with energy
targeting the actual binding constraint (stability or IK reachability)
instead of interpenetration. Potentially: differentiable relaxations of
the stability check or an approximate IK feasibility function as guidance
signal.

**4. Phase D revisit with frozen backbone.** Train cross-attention layers
in isolation with backbone weights frozen. Separates the "learn text
conditioning" problem from the "do not disrupt geometry features" problem.
Once text conditioning converges, unfreeze and fine-tune jointly at a
lower learning rate.

---

## References

- Lin et al. 2024. "Common Diffusion Noise Schedules and Sample Steps are
  Flawed." (zero terminal SNR motivation; Bug #15 lesson)
- Everaert et al. 2024. "Covariance Mismatch in Diffusion Models."
  (per-dimension whitening; motivation for v3+ runs)
- Salimans and Ho 2022. "Progressive Distillation for Fast Sampling of
  Diffusion Models." (v-parameterization; addressed by Bug #6 fix)
- Bansal et al. 2023. "Universal Guidance for Diffusion Models."
  (UG framework used in Phase E)
- Zhou et al. 2019. "On the Continuity of Rotation Representations in
  Neural Networks." (rot6d representation choice)
- Nichol and Dhariwal 2021. "Improved Denoising Diffusion Probabilistic
  Models." (cosine schedule baseline)
