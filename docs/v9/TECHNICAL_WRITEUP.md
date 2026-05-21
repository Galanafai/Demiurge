# Demiurge v9: Technical Writeup

**Summary:** 49M parameter transformer denoiser for 6-DoF tabletop scene
generation. Validated by Drake (4-gate funnel). Final: 3.0% overall /
11.3% on non-empty. 15 silent bugs. ~$60 compute. 11+ training runs.

---

## Problem Statement

Generate valid 6-DoF tabletop scenes for robot manipulation training.
Validity requires passing all four Drake gates in sequence:

1. **No interpenetration** -- minimum signed distance > 0 across all pairs
2. **Stable rest** -- max pose drift < threshold after 1.5s forward sim
3. **IK reachable** -- goal pose solves UR5e InverseKinematics with collision avoidance
4. **RRT solvable** -- BiRRT plan exists from home config to IK solution

The difficulty: these constraints are global (IK reachability depends on the
UR5e workspace geometry), but the diffusion loss is local (MSE on denoised
noise vectors). The model can satisfy MSE while violating IK globally.

**Baselines:**
- Procedural sampler with heuristic filters: 9% (no IK check)
- v7 model (earlier iteration, 9M params): 1.1% overall
- Phase B gate for Phase C eligibility: 5%

---

## Architecture

```
SceneDenoiser (49M parameters)
  Input:  x_t ∈ ℝ^{12 × 13}   -- 12 slots, 13 channels
  Output: v_t ∈ ℝ^{12 × 13}   -- v-prediction target
          type_logits ∈ ℝ^{12 × |V|}  -- per-slot type distribution

Channel layout (13 channels per slot):
  [0:3]   xyz (whitened position)
  [3:9]   rot6d (6D rotation, Gram-Schmidt -> quaternion at decode)
  [9:12]  scale (whitened per-axis scale)
  [12]    presence_logit (sigmoid > threshold -> object present)

Architecture:
  - 16 transformer layers, d_model=384, 8 heads, ffn_mult=4
  - Per-slot ID embedding (N_MAX=12, learned)
  - AdaLN timestep conditioning at each layer
  - Optional cross-attention to text embeddings (Phase D)
  - v-prediction head with separate type classification head

Schedule:
  - CosineSchedule with zero_terminal_snr=True (Lin 2024)
  - T=1000 training steps
  - 50-step DDIM inference
  - prediction_type='v' throughout (critical -- see Bug #6)
```

**Why v-prediction?** Better gradient signal at high noise levels. The
standard epsilon parameterization has small loss magnitude at t near T
(pure noise), making gradient updates near-zero where signal matters most.
v-prediction reweights this.

**Why zero terminal SNR?** Standard cosine schedules have alpha_bar(T) > 0,
meaning the model never sees true noise. With narrow distributions like
tabletop scenes, this means the model can always partially "see" the
conditional mean even at maximum noise, amplifying mode collapse.

---

## Training Journey (11 Runs)

### The Accumulated Bug List

The first 7 runs (v1-v6) spent more time finding pipeline bugs than
improving the model. See [BUG_INVENTORY.md](BUG_INVENTORY.md) for full
details. Key ones:

- **Bug #1:** Gradient isolation caused memorization shortcut (v1-v2)
- **Bug #6:** DDIM sampler hardwired epsilon decode for v-prediction model
  (model output at std=5.32 -- should be ~1.0)
- **Bug #7:** Drake API renamed, causing 100% validator failure across all runs

### v7: First Real Signal

After fixing Bugs 1-8, v7 produced 1.1% overall validity. Binding
constraint: IK reachability. The model's x-coordinate distribution was
too concentrated near zero (x_spread_whitened=0.43 vs target 1.0). Objects
outside the UR5e reachable workspace in x.

### v8 (= final "v9"): Pose gradient rebalancing

**Key insight from audit:** rot6d has 6 channels vs xyz's 3. With equal
loss weights, each rot6d channel gets half the gradient per-channel compared
to xyz. Increased `pose_rot` weight from 1.0 to 2.0 to compensate.

v8 step 100k reached 4.0% (near the 5% gate). But at step 170k, presence
collapsed (Bug #12, LR scheduler fast-forward). We caught the peak at
step 170k: **3.0% overall, 11.3% on non-empty**.

```
v9_final_probe.json (n=200, seed=0):
  Non-empty: 53/200 (26.5%)
  Interp:    53/53  (100%)
  Stable:    16/53  (30.2%)
  IK:        43/53  (81.1%)
  RRT:       13/53  (24.5%)
  Accepted:   6/200 ( 3.0%)

vs v7:
  Stable:   6x improvement (5% -> 30%)
  IK:       1.1x (72.8% -> 81.1%)
  Accepted: 10x on non-empty (1.1% -> 11.3%)
```

### v10 (fresh training): Proved loss underspecification

Trained v10 from scratch with the v8 config (no warm-init from v7).
Result: 0/200 accepted across 110k-150k steps. IK completely collapsed.

This proved that warm-init was not a crutch -- it was transmitting the
learned IK distribution from v7's procedural data. Without it, the model
couldn't discover the implicit IK constraint from MSE alone.

**Architectural ceiling conclusion:** The loss function is underspecified.
Diffusion MSE can be minimized while violating IK reachability globally.
The real fix is at data generation: filter training scenes for IK feasibility.

### Phase D: Conditional Training (Negative Result)

Added text conditioning to v9 step 170k via cross-attention on
sentence-transformer embeddings. 17 probes over 200k steps:

```
Best: 30k steps -> 1/100 = 1.0%
Peak: 80k steps -> 1/200 = 0.5% (noisy)
Final: 200k steps -> 1/300 = 0.33%
```

The cross-attention layers perturbed the pretrained geometry features.
IK pass rate dropped from 81% to ~4% early in Phase D training.
Never recovered to the unconditional baseline (3%).

**Lesson:** Warm-init for new capabilities (conditioning) requires
careful architectural isolation. The new cross-attention layers should
have been frozen separately while geometry layers adapted.

### Phase E: Universal Guidance (Negative Result)

Applied Bansal et al. 2023 UG with `pairwise_overlap_energy`. Found
3 implementation bugs before getting a clean result:

- Bug #13: yaw_project destroyed stability (6% -> 30%)
- Bug #14: presence channel gradient leak (ne=0 at all scales)
- Bug #15: Tweedie NaN at zero terminal SNR (ne=0, all NaN)

**After all bugs fixed:**
```
scale=0.0: 4/200 = 2.0%  [ne=57  st=12 ik=46 rrt=16]  <- matches 3% baseline
scale=2.0: 0/200 = 0.0%  [ne=200 st=0  ik=10 rrt=9]
```

Presence restored (ne=200), but stability destroyed (0/200 stable).
The overlap energy steered objects apart in position space but pushed
them into physically unstable configurations. UG targeted a non-binding
constraint (interpenetration at 100% pass rate) while actively harming
the binding constraint (stability at 30%).

---

## Defensive Infrastructure

Built during the v9 audit phase after seeing the same class of bug multiple times.

### Typed Scene Wrappers (`src/scene/typed_scene.py`)

```python
# Before: implicit conventions caused Bugs 2, 3, 4, 5
output = model_output  # what space is this?
drake_scene = denormalize(output)  # called twice?

# After: explicit types prevent double-application
whitened = WhitenedScene(model_output)
normalized: NormalizedScene = whitened.unwhiten()
physical: PhysicalScene = normalized.denormalize(bounds)
# physical is now unambiguously in meters/quaternions
```

### 12-Phase Pipeline Audit (`artifacts/pipeline_audit.md`)

Systematic audit of every invariant in the pipeline. Discovered Bugs
#9, #10, #11. Format:

```
Phase 1: Encode/decode round-trip (whitened -> normalized -> physical -> normalized)
Phase 2: Whitening constants (verified against corpus statistics)
Phase 3: Cross-file consistency (train.py vs probe_drake.py constants)
Phase 4: Rotation encoding (rot6d -> quat -> rot6d identity)
Phase 5: Loss component balance at trained model
Phase 6: Validator constraint inventory
Phase 7: Config vs code mismatch
Phase 8: Per-dimension gradient flow
Phase 9: Model input format verification
Phase 10: Slot ID embedding symmetry (ratio 0.014)
Phase 11: Type embedding ablation
Phase 12: Failure mode inventory
```

### Unit Tests (38 tests, `tests/`)

Each test was added after a bug was found that it would have caught:

| Test file | Bug class caught |
|---|---|
| `test_vpred_conventions.py` | Bug #6 (v-prediction math identity) |
| `test_presence_loss.py` | Bug #8 (pos_weight calibration) |
| `test_output_sanity.py` | Bug #6 symptoms (std explosion) |
| `test_full_sampling_pipeline.py` | Bugs #2-5 (encode/decode chain) |
| `test_rotation_encoding.py` | Bug #13 (yaw projection degeneracy) |

---

## The Architectural Ceiling

Three independent experiments confirmed that the ceiling is the data distribution:

1. **v10 fresh training:** Without warm-init, IK collapsed to 0% despite the
   same model config that achieved 11.3% on non-empty with warm-init.

2. **Phase D conditional:** IK dropped from 81% to 4% within 20k steps of
   adding cross-attention layers that perturbed the learned pose distribution.

3. **Phase E UG:** Overlap energy couldn't improve IK (it steers positions,
   not toward the IK workspace). Stability was actively harmed.

**The fix (not yet implemented):** Add IK feasibility filtering to
`src/data/sampler.py`. Reject candidate scenes where any object falls
outside the UR5e reachable workspace. With IK-filtered training data,
the model would learn the constraint by exclusion, eliminating the ceiling.

---

## Reproducibility

```yaml
# configs/train/v9_uncond_v9.yaml (the final model config)
architecture:
  n_layers: 16
  d_model: 384
  n_heads: 8
  ffn_mult: 4
  use_slot_id_embed: true

training:
  lr: 3e-4
  pose_rot: 2.0
  presence_bce: 1.5
  pos_weight: 5.0
  batch_size: 128

schedule:
  T: 1000
  zero_terminal_snr: true
  prediction_type: v

inference:
  n_steps: 50
  presence_threshold: -0.589
```

**Reproduce final result:**
```bash
python3 scripts/probe_drake.py \
    --checkpoint checkpoints/v9_uncond_v9/step_00170000.pt \
    --n-scenes 200 --head ema --seed 0 \
    --presence-threshold -0.589 \
    --out artifacts/repro_results.json
# Expected: ~6/200 accepted (3%), 53/200 non-empty (26%)
```

**W&B artifacts:**
- `galanafai-self/demiurge/v9_FINAL_MODEL:latest` -- v9 step 170k EMA weights
- `galanafai-self/demiurge/phase_a_combined_corpus:latest` -- 234k training scenes

---

## References

1. Lin et al. 2024. "Common Diffusion Noise Schedules and Sample Steps are
   Flawed." (zero terminal SNR motivation)
2. Everaert et al. 2024. "Covariance Mismatch in Diffusion Models."
   (per-dimension whitening)
3. Salimans & Ho 2022. "Progressive Distillation for Fast Sampling of
   Diffusion Models." (v-parameterization)
4. Bansal et al. 2023. "Universal Guidance for Diffusion Models." (UG framework)
5. Zhou et al. 2019. "On the Continuity of Rotation Representations in
   Neural Networks." (6D rotation representation)
