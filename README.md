# Demiurge

> Conditional diffusion model for 6-DoF robot manipulation scene generation,
> validated by Drake physics simulator.

**49M params | 3.0% Drake validity | 11.3% on non-empty | 15 bugs documented | ~$60 compute**

---

## What This Is

Demiurge generates 3D tabletop scenes -- positions, orientations, and scales
of household objects -- validated for physical correctness by Drake:

1. No interpenetration between objects
2. Stable resting configuration (1.5s forward sim)
3. IK reachable by UR5e arm
4. RRT path solvable from home config to grasp

Portfolio work demonstrating senior ML engineering on a hard, constrained
generative modeling problem.

---

## Architecture

```
Input:  x_t ∈ ℝ^{12 × 13}   (12 object slots, 13 channels)
        channels: xyz | rot6d | scale | presence_logit

Backbone: 16-layer transformer
          d_model=384, 8 heads, ffn_mult=4
          AdaLN timestep conditioning
          Per-slot ID embeddings

Schedule: CosineSchedule (zero_terminal_snr=True, T=1000)
          v-prediction (Salimans & Ho 2022)
          50-step DDIM at inference

Output:   v_t ∈ ℝ^{12 × 13}  (v-prediction target)
          type_logits ∈ ℝ^{12 × |V|}
```

See [architecture diagram](docs/v9/diagrams/architecture.png).

---

## Results

### Final Model: v9 step 170k (unconditional)

```
n=200, seed=0, from artifacts/v9_final_probe.json:

  Generated:       200 scenes
  Non-empty:        53 / 200   (26.5%)
  Interpenetration: 53 / 53    (100.0%)
  Stable rest:      16 / 53    (30.2%)
  IK reachable:     43 / 53    (81.1%)
  RRT solvable:     13 / 53    (24.5%)
  Accepted:          6 / 200   ( 3.0%)

vs v7 baseline (1.1%): 10x on non-empty, 3x overall
```

See [validation funnel chart](docs/v9/charts/validation_funnel.png).

### Phase D: Text Conditioning (Negative Result)

Best: 1/100 = 1.0% at step 30k. Final: 1/300 = 0.33% at step 200k.
Cross-attention layers disrupted pretrained geometry features.

### Phase E: Universal Guidance (Negative Result)

3 implementation bugs found (Bugs #13-15). After all fixed:
- scale=0: 4/200 (2.0%) -- confirms baseline
- scale=2: 0/200 (0.0%) -- overlap energy destroys stability

Overlap energy was the wrong choice: interpenetration was already at 100%.

---

## Key Findings

**1. Loss underspecification is the architectural ceiling.**
The model satisfies diffusion MSE while violating IK reachability globally.
Proven by v10 (0% fresh training without warm-init).
Fix: IK-feasibility filter in `src/data/sampler.py`.

**2. Warm-initialization transmits implicit constraints.**
v7's learned IK distribution was transmitted to v8 via warm-init.
Without it (v10), the model couldn't discover IK feasibility from MSE alone.

**3. Hard projection of soft constraints can backfire.**
yaw_project at inference (Bug #13) made stable_rest drop from 30% to 6%.
The model had already learned yaw-only rotations softly.

**4. UG energy function must target the binding constraint.**
Overlap energy targeted interpenetration (100% pass) while the binding
constraint was stability (30%) and IK (81%). UG actively hurt results.

---

## Structure

```
Demiurge/
 src/
   ├── model/        denoiser.py, schedule.py, rotations.py, loss.py
   ├── scene/        schema.py, typed_scene.py, vocab.py
   ├── validator/    core.py (IK, RRT, stability, interpenetration)
   ├── training/     train.py, invariants.py, output_sanity.py
   ├── guidance/     energy.py, classifier.py, rejection.py
   ├── data/         sampler.py, reader.py, writer.py
   └── eval/         metrics.py, judge.py, downstream.py
 scripts/          probe_drake.py, train.py, ug_sweep_v9_uncond.py, ...
 configs/train/    v9_uncond_v{4..10}.yaml, v9_cond_d.yaml
 tests/            38 tests across math, loss, integration
 artifacts/        probe JSONs, audit, VLM scores, UG sweep results
 docs/v9/
    ├── TECHNICAL_WRITEUP.md    full narrative (~300 lines)
    ├── BUG_INVENTORY.md        15 bugs with diagnoses (~300 lines)
    ├── DESIGN_DECISIONS.md     the why behind choices (~150 lines)
    ├── diagrams/architecture.png
    └── charts/training_trajectory.png, validation_funnel.png
```

---

## Defensive Infrastructure

**Typed scene wrappers** (`src/scene/typed_scene.py`): `WhitenedScene`,
`NormalizedScene`, `PhysicalScene`. Prevents the class of bug (5 instances)
where two code paths applied different normalization sequences.

**12-phase pipeline audit** (`artifacts/pipeline_audit.md`): Systematic
audit of every invariant. Found Bugs #9-12.

**38 unit tests** (`tests/`): Added incrementally as bugs were found.
Each test catches a specific bug class.

**Output sanity checks** (`src/training/output_sanity.py`): Catches
std explosion (Bug #6 symptom) within 100 training steps.

---

## Reproduce

```bash
# Clone and install
git clone https://github.com/Galanafai/Demiurge
cd Demiurge
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run the validated 3% probe
python3 scripts/probe_drake.py \
    --checkpoint checkpoints/v9_uncond_v9/step_00170000.pt \
    --n-scenes 200 --head ema --seed 0 \
    --presence-threshold -0.589 \
    --out artifacts/repro_results.json
# Expected: ~6/200 accepted (3%), 53/200 non-empty
```

**W&B:** `galanafai-self/demiurge/v9_FINAL_MODEL:latest`

---

## References

- Lin et al. 2024. *Common Diffusion Noise Schedules and Sample Steps are Flawed*
- Everaert et al. 2024. *Covariance Mismatch in Diffusion Models*
- Salimans & Ho 2022. *Progressive Distillation for Fast Sampling of Diffusion Models*
- Bansal et al. 2023. *Universal Guidance for Diffusion Models*
- Zhou et al. 2019. *On the Continuity of Rotation Representations in Neural Networks*

---

*Built solo on consumer compute (~$60). Claude used as debugging copilot.*
