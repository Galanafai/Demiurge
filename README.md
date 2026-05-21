# Demiurge

> A conditional diffusion model for 6-DoF tabletop scene generation, with
> every output validated by Drake physics simulation (interpenetration,
> stability, IK reachability, RRT path solvability).

**49M parameters | 234k training scenes | 3.0% Drake validity (11.3% on non-empty) | 15 bugs documented | ~$60 total compute | 11+ training runs**

---

## Why This Project Exists

I wanted end-to-end ML engineering experience on a problem that was
genuinely difficult, not a portfolio project chosen because it works.
Physics-grounded generative modeling is a class of problem where the
loss function and the evaluation criterion measure fundamentally
different things, and where most failures are silent.

The result is not a working production model. The result is a
documented engineering process across 11+ training runs, a 15-bug
inventory with technical diagnoses, and an architectural finding about
why this class of model hits a ceiling.

This is portfolio work demonstrating senior ML engineering capability
on hard problems - not "look at this cool demo."

## What This System Does

Demiurge takes random noise and produces a 3D scene of household objects
on a tabletop. Each generated scene is then validated by Drake (a physics
simulator commonly used in robotics research) against four constraints
that all must pass:

1. **No interpenetration** - object collision volumes do not overlap
2. **Stable rest** - objects remain in place after 1.5s of physics simulation
3. **IK reachable** - a UR5e robot arm can solve inverse kinematics to grasp each object
4. **RRT solvable** - the arm can find a collision-free motion plan to reach each object

A scene that passes all four gates is "valid" - usable as training data
for a robot manipulation policy. The hypothesis was that a diffusion
model trained on procedurally-validated scenes could learn to produce
valid scenes directly, removing the cost of rejection sampling.

## The Hard Problem

Generative modeling problems usually look like:
1. Define a loss function
2. Minimize it
3. Get a working model

Physics-grounded scene generation does not work this way. The loss is
diffusion MSE on scene tensors. The validator is Drake running forward
dynamics, IK solving, and motion planning. These two functions measure
different things, and minimizing the loss does not minimize validator
rejection rate.

Three specific challenges drove the difficulty:

**1. Global constraints from local loss.** The diffusion MSE loss is
computed per-channel, per-slot. IK reachability is a global property
of a scene relative to the UR5e kinematic chain. The loss has no
representation for "this object is outside the workspace." The model
must learn this implicitly from the training distribution.

**2. Narrow distribution geometry.** The valid scene manifold is a small
fraction of the 12-slot x 13-channel latent space. Getting the model to
learn this without mode collapse required v-prediction with zero terminal
SNR, per-dimension whitening, and 11+ rounds of debugging.

**3. Silent failure modes.** The pipeline has six independent stages where
an off-by-one in a normalization constant, a renamed Drake API, or an
incorrect decode convention produces zero error message but 0% downstream
validity. Most of the 15 bugs documented in this repo were silent failures
caught only by careful comparison of expected vs measured outputs.

---

## The Scaling Story

The project went through three distinct compute and data scales. Each
revealed a different failure mode.

### Stage 1: 9M parameters, limited data (v1 - v7)

Initial architecture was a smaller transformer denoiser (~9M parameters,
12 layers, d_model=256). Training data was the initial procedural corpus
(v1 dataset, ~50k Drake-validated scenes).

**Result at v7 step 140k:** 1.1% overall Drake validity. The model
learned object types and presence reasonably but failed to spread the
x-coordinate distribution adequately - generated objects clustered near
x=0, falling outside the UR5e reachable workspace.

**What this stage revealed:** The architecture lacked capacity for the
6D rotation distribution. Rotation gradients were being absorbed by
position learning, and the smaller model could not represent both jointly.

### Stage 2: 49M parameters, ~50k data - overfitting

Scaled architecture to 49M parameters (16 layers, d_model=384, 8 heads).
Initially attempted to train on the same v1 corpus (~50k scenes).

**Result:** Clear overfitting symptoms. Validation loss diverged from
training loss after ~30k steps. The larger model memorized scene
configurations rather than learning the underlying distribution. Drake
validity plateaued at very low rates despite low training loss.

**What this stage revealed:** A 49M parameter model with no inductive
bias for physical validity needs significantly more data than 50k scenes
to generalize. The dataset itself became the bottleneck.

### Stage 3: 49M parameters, 234k data (v8/v9, final)

Combined v1 and v2 procedural datasets into a 234k Drake-validated scene
corpus. With ~4.7x more training data, the 49M model could be trained
without overfitting.

**Key intervention:** `pose_rot=2.0` in the loss function. The audit
phase discovered that rot6d (6 channels) was receiving half the
per-channel gradient that xyz (3 channels) received with equal loss
weights. Doubling the rotation loss weight equalized this:

```
Per-channel gradient with equal weights:
  xyz:   1.0 / 3 channels = 0.333 per channel
  rot6d: 1.0 / 6 channels = 0.167 per channel   <- starved

Per-channel gradient with pose_rot=2.0:
  xyz:   1.0 / 3 channels = 0.333 per channel
  rot6d: 2.0 / 6 channels = 0.333 per channel   <- balanced
```

**Result at v9 step 170k:** 3.0% overall Drake validity (10x improvement
over v7 baseline on the conditional rate, 11.3% on non-empty scenes).
This was the peak. Subsequent training showed presence channel collapse
from warm-initialization disequilibrium (Bug #12).

### Why the architecture hit its ceiling here

Three subsequent experiments confirmed the ceiling is a property of the
loss function, not the architecture or training procedure:

- **v10 fresh training:** The same 49M architecture trained from scratch
  on the same 234k data achieved 0% validity. Warm-init from v7 had been
  transmitting the implicit IK-reachable pose distribution. Without it,
  diffusion MSE alone could not teach IK reachability in 150k steps.

- **Phase D conditional training:** Adding text conditioning via
  cross-attention to v9 step 170k dropped IK pass rate from 81% to ~4%.
  The cross-attention layers, having trained as effective no-ops during
  unconditional pretraining, required relearning that perturbed the
  geometric features.

- **Phase E Universal Guidance:** Pairwise overlap energy guidance restored
  presence (after fixing bugs #14 and #15) but dropped stable_rest from
  30% to 0%. The energy function targeted a non-binding constraint
  (interpenetration was already at 100%) while breaking the binding
  constraint (stability).

The architectural conclusion: **the loss function is underspecified.**
Diffusion MSE can be minimized locally while violating global physical
constraints. The fix is at the data level - filter training scenes for
IK feasibility - not at the architecture or loss level.

---

## Technical Stack and Why

### Architecture: 49M parameter transformer denoiser

```
SceneDenoiser (49M parameters)

  Input:  x_t in R^{12 x 13}    -- 12 slots, 13 channels
          type_ids in Z^{12}     -- per-slot object type
          t in Z                 -- diffusion timestep
          text_emb (optional)    -- text conditioning embedding

  Per slot, 13 channels:
    [0:3]  xyz (whitened position)
    [3:9]  rot6d (6D rotation, Zhou et al. 2019)
    [9:12] scale (whitened per-axis)
    [12]   presence_logit

  Layers: 16 transformer blocks
    AdaLN(t) -> Self-Attention (12+1 tokens)
    Cross-Attention to text_emb (Phase D only)
    AdaLN(t) -> FFN (mult=4)

  Output heads:
    head_xyz       -> v-prediction for position
    head_rot6d     -> v-prediction for rotation
    head_scale     -> v-prediction for scale
    head_presence  -> logit
    head_type      -> per-slot type distribution
```

**Why transformer not UNet:** Scenes are sets of slots with no spatial
grid structure. UNet inductive biases (locality, translation equivariance)
do not apply. Transformers naturally handle variable-cardinality sets via
attention, and the slot structure maps cleanly to tokens.

**Why separate type head:** Object types are categorical (12 YCB-style
classes); positions and rotations are continuous. Mixing them in a single
regression loss destabilizes training. Separate heads preserve the right
inductive bias for each.

**Why AdaLN timestep conditioning:** Replaces bias-only timestep injection
with learned scale-and-shift, providing stronger conditioning signal at
every layer. Standard in modern diffusion transformers (DiT, MDT, etc.).

### Diffusion: v-prediction with zero terminal SNR

Epsilon prediction gives near-zero gradient signal at t near T (pure
noise) - exactly where the model most needs to learn the global
distribution shape. v-prediction (Salimans and Ho 2022) uses a target
that mixes epsilon and x_0, giving uniform gradient signal across all
noise levels.

Zero terminal SNR (Lin et al. 2024) forces `alpha_bar(T) = 0` so the
model sees true Gaussian noise at maximum noise level rather than a
faint preserved signal. On narrow distributions like tabletop scenes,
that residual signal amplifies mode collapse.

**Critical caveat (Bug #15):** Zero terminal SNR creates a division-by-zero
singularity in Tweedie-based computations at t=T. The Phase E Universal
Guidance implementation produced all-NaN outputs until we added an
`ALPHA_MIN=1e-4` guard. Any future use of zero terminal SNR should treat
this as a known pitfall.

### Normalization: per-dimension whitening (Everaert et al. 2024)

The 13 channels span very different natural scales (xyz in meters,
rot6d in [-1,1], presence_logit on a logit scale). Without per-channel
whitening the diffusion loss is dominated by highest-variance channels.
Whitening brings all channels to zero mean, unit variance - making the
loss balanced across channels and preventing the z-axis collapse observed
in v3.

### Validation: Drake 4-gate funnel

Drake (MIT/TRI) provides exact physics for rigid body dynamics, IK, and
motion planning. Gate order is cheap to expensive to minimize evaluation
cost:

```
Gate 1: Interpenetration  --  5ms  (geometric, signed distance)
Gate 2: Stable rest       -- 100ms (forward dynamics for 1.5s)
Gate 3: IK reachable      --  10ms (per-object IK with collision)
Gate 4: RRT solvable      --   1s  (bidirectional RRT for full path)
```

Using Drake as ground truth means validity numbers transfer directly to
downstream robotic policy training. Approximate physics would bias the
training signal in ways that might not generalize.

### Experiment tracking: Weights and Biases

All training runs logged with full config, seed, git SHA, and dataset
hash to W&B. Public artifacts:

- `galanafai-self/demiurge/v9_FINAL_MODEL:latest` - v9 step 170k checkpoint
- `galanafai-self/demiurge/phase_a_combined_corpus:latest` - 234k training corpus

### Evaluation: VLM-based quality scoring (Claude Haiku 4.5)

Phase C used Claude Haiku 4.5 to score generated scene renders for visual
plausibility (1-5 scale). Drake validates physical correctness; VLM scoring
catches "technically valid but visually trivial" scenes (e.g. a single
object at one valid pose) that pure Drake validation misses. Total Phase C
API cost: $0.022 for 28 scenes, with caching to avoid re-judging identical
scenes.

### Infrastructure

- **Compute:** RTX 4000 Ada (20GB VRAM) on RunPod at $0.21/hr
- **Total cost:** ~$60 across 11+ training runs, audits, sweeps
- **No distributed training:** Single GPU, batch size 64-128
- **No team:** Solo project

---

## What I Built

### Core ML
- `src/model/denoiser.py` - 49M parameter transformer scene denoiser
- `src/model/schedule.py` - Cosine schedule with zero terminal SNR, DDIM sampler
- `src/model/rotations.py` - rot6d <-> quaternion (Gram-Schmidt + Shepperd)
- `src/model/loss.py` - Weighted multi-component diffusion loss

### Scene representation
- `src/scene/schema.py` - SceneTensor and WorkspaceBounds (single source of truth)
- `src/scene/typed_scene.py` - Typed wrappers (PhysicalScene/NormalizedScene/WhitenedScene)
- `src/scene/vocab.py` - Object type definitions (12 YCB-style classes)

### Validation
- `src/validator/core.py` - 4-gate Drake validator (interpenetration, stability, IK, RRT)
- `src/validator/api_compat.py` - Drake API compatibility layer (Bug #7 mitigation)

### Data
- `src/data/sampler.py` - Procedural scene generation with Drake validation
- `src/data/reader.py` - WebDataset shard reader for 234k corpus
- `src/data/descriptions.py` - Scene-to-text description generator (Phase D)

### Training and evaluation
- `scripts/train.py` - Training loop with warm-init, EMA, autonomous probes
- `scripts/probe_drake.py` - Drake validity probe at any checkpoint
- `scripts/probe_vlm.py` - VLM scoring via Claude Haiku 4.5
- `scripts/ug_sweep_v9_uncond.py` - Universal Guidance sweep (Phase E)

### Defensive infrastructure (built incrementally as bugs were found)
- 38 unit tests in `tests/` covering math identities, loss components, integration
- 12-phase pipeline audit framework in `artifacts/pipeline_audit.md`
- Output sanity checks in `src/training/output_sanity.py` (catches mid-training explosions)

---

## Results

### Final model: v9 step 170k

```
Probe: n=200, seed=0, EMA weights, no yaw projection at decode

Validation funnel (sequential, on non-empty):
  Generated:         200 scenes
  Non-empty:          53 / 200   (26.5%)   <- presence ceiling
  Interpenetration:   53 / 53    (100.0%)
  Stable rest:        16 / 53    ( 30.2%)
  IK reachable:       43 / 53    ( 81.1%)
  RRT solvable:       13 / 53    ( 24.5%)

  Accepted (all gates):  6 / 200 = 3.0% overall
                        6 / 53  = 11.3% on non-empty
```

### Comparison to baselines

```
v7 baseline (9M params, 50k data):          1.1% overall validity
Random procedural sampler (heuristic):       9.0% (biased toward easy scenes)
v9 step 170k (49M params, 234k data):        3.0% overall, 11.3% on non-empty
Phase B target:                              5.0% overall
```

### What did not work

- **v10 (fresh training without warm-init):** 0% validity. Proved warm-init
  was transmitting the implicit IK-reachable pose distribution.
- **Phase D (text conditioning):** Best result 1.0% at 30k steps. Cross-attention
  layers disrupted pretrained geometric features.
- **Phase E (Universal Guidance):** After fixing 3 implementation bugs, confirmed
  that overlap energy targets a non-binding constraint while destroying stability.

---

## Key Engineering Lessons

**1. Pipeline bugs dominate model bugs.** 15 bugs total; 7 of them caused 0%
validity across v1-v6 but were entirely in the pipeline, not the model. The
implication: build bug-detection infrastructure first, not model optimizations.

**2. Defensive types beat heroic debugging.** After Bug #5 (the 5th
encode/decode normalization bug), I built typed scene wrappers that make
conversions explicit and type-checked. Should have built this after Bug #1.
Explicit types are how you scale beyond what one person can hold in their head.

**3. Warm-initialization transmits implicit constraints.** v7 learned the
approximate IK-reachable workspace shape from 234k procedural scenes. That
transferred to v9 via warm-init. Without it (v10), diffusion MSE alone could
not teach IK reachability. Useful technique but fragile: changing loss weights
at warm-init disrupts optimizer momentum (Bug #12). Either fresh training or
matched-config warm-init - no mixing.

**4. Energy functions must target the binding constraint.** At v9 step 170k,
interpenetration was at 100% pass rate - not the binding constraint. Phase E
spent 50 DDIM steps optimizing something already perfect while destroying
stability. Analyze the validator rejection funnel first to identify the actual
bottleneck, then design guidance to target it.

**5. Negative results have signal.** v10 proved warm-init was load-bearing.
Phase D revealed that warm-init for new capabilities requires architectural
isolation. Phase E found 3 implementation bugs before confirming the
wrong-energy hypothesis. A portfolio piece that only documents successes is
less honest and less educational than one that documents the full process.

**6. Cost discipline is real engineering.** ~$60 total across 11+ runs.
Mitigations: smoke tests before full sweeps, early kill criteria, strategic
probe scheduling (at decision points, not every 5k steps), single pod with
parallel background processes, VLM caching. Without these: $300-500.

---

## What I Would Do Differently

1. **Build typed wrappers first.** Bugs #2-5 cost ~2 weeks. The fix is
   200 lines of code that would have prevented the entire class.
2. **Filter training data for IK feasibility.** Encode the constraint in the
   data rather than trying to learn it from a loss that has no representation
   for it. This was the unimplemented next step in `src/data/sampler.py`.
3. **Never change loss weights at warm-init.** v8 and Phase D both demonstrated
   this destabilizes the optimizer. Keep configs identical across warm-init.
4. **Analyze the rejection funnel before designing guidance.** Phase E spent
   compute on the wrong objective because we did not check the actual bottleneck.
5. **Probe with 50 scenes during development, 300+ for final results.**
   Iteration velocity beats statistical significance during debugging.

---

## Documentation Map

For technical reviewers (recommended reading order):

1. **[README.md](README.md)** - This file. Project overview and decisions.
2. **[TECHNICAL_WRITEUP.md](TECHNICAL_WRITEUP.md)** - Deep technical writeup:
   full training narrative, math, architectural ceiling analysis.
3. **[BUGS.md](BUGS.md)** - Complete inventory of 15 bugs with diagnoses
   and lessons.
4. **[DESIGN.md](DESIGN.md)** - Rationale for 12 key architectural choices,
   cross-referenced with the bug inventory.

For code review:
- `src/model/denoiser.py` - The core architecture
- `src/validator/core.py` - The validation pipeline
- `src/scene/typed_scene.py` - Example of defensive infrastructure
- `tests/` - 38 unit tests

For reproducibility:
- `configs/train/v9_uncond_v9.yaml` - Exact config of the final model
- `artifacts/v9_final_results.json` - Final probe results

---

## Reproduce

```bash
git clone https://github.com/Galanafai/Demiurge && cd Demiurge
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
pytest tests/ -v

python3 scripts/probe_drake.py \
    --checkpoint checkpoints/v9_uncond_v9/step_00170000.pt \
    --n-scenes 200 --head ema --seed 0 \
    --presence-threshold -0.589 \
    --out artifacts/repro.json
# Expected: ~6/200 accepted (3%), 53/200 non-empty (26%)
```

---

## References

- Lin et al. 2024. "Common Diffusion Noise Schedules and Sample Steps are Flawed."
- Everaert et al. 2024. "Covariance Mismatch in Diffusion Models."
- Salimans and Ho 2022. "Progressive Distillation for Fast Sampling of Diffusion Models."
- Bansal et al. 2023. "Universal Guidance for Diffusion Models."
- Zhou et al. 2019. "On the Continuity of Rotation Representations in Neural Networks."
- Nichol and Dhariwal 2021. "Improved Denoising Diffusion Probabilistic Models."

---

## Contact

Galanafai - actively looking for opportunities in robotics, simulation,
ML, and systems engineering. Find me on LinkedIn or open an issue here.

*Solo project. ~$60 on cloud compute. 11+ training runs. 15 bugs. 38 tests.*
*Drake is the single source of truth for validity.*
