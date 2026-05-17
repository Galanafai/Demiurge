# Demiurge Week 5 Results

**Branch:** `week5/universal-guidance`  
**Model:** `conditional_v7` (9M parameters, 200k steps on 50k scenes)  
**Eval:** 200 held-out prompts, 5 samplers, 3 seeds, 8 scenes/prompt

---

## Headline Results

| Sampler | Validity (%) | Diversity |
|---|---|---|
| `v7_uncond` | 1.2 +/- 0.3 | 0.059 |
| `v7_cond_baseline` | 1.4 +/- 0.3 | 0.140 |
| `v7_rejection` | **5.5 +/- 0.2** | 0.138 |
| `v7_ug_best` (scale=0.5) | 1.3 +/- 0.4 | **0.140** |
| `v7_ug_scale1` (scale=1.0) | 1.3 +/- 0.1 | **0.140** |

Validity = fraction of generated scenes passing all four Drake checks
(no interpenetration, stable rest, IK reachable, RRT solvable). 3 seeds x 200
prompts x 8 scenes. Mean +/- std across seeds.

---

## Key Findings

### 1. Universal Guidance does not generalize to held-out prompts

The UG sweep (Phase A) ran on familiar in-distribution prompts using
cached text embeddings. Best result: 6.5% (uncond) -> 11.1% (UG scale=0.5),
a 70% relative gain.

On the held-out evaluation suite (200 novel procedural prompts encoded
on-the-fly), UG validity collapsed to 1.3% - indistinguishable from the
1.2% unconditional baseline.

The pairwise overlap energy used as the UG surrogate is physics-motivated
but too weak a gradient signal to overcome out-of-distribution text
embeddings. The model's prior dominates; UG energy cannot compensate.

### 2. Rejection sampling is the practical winner on novel prompts

At 5.5% validity on held-out prompts, rejection sampling outperforms all
other strategies by 4x. This is expected: rejection sampling is
model-agnostic and benefits from the same Drake validator used during
training data generation. It simply samples more and filters harder.

Cost: 4x scene generation overhead vs. single-pass methods.

### 3. Text conditioning drives diversity; UG energy does not

Unconditional generation: diversity = 0.059.
Any conditioned sampler: diversity = 0.138-0.140.
UG energy scale (0.5 vs 1.0): no effect on diversity.

The text encoder is the sole driver of output diversity. The energy
surrogate influences spatial arrangement but not object-type selection,
which dominates the diversity metric.

### 4. Downstream RRT planning is equivalent to raw validity in this formulation

For the current task setup (single robot, tabletop scene, fixed IK goal),
a scene that passes Drake validity is also RRT-solvable by construction.
The validator runs RRT as check 4; there is no separate downstream planning
task to evaluate. This metric is therefore dropped. Future work that decouples
scene validity from task-specific planning (e.g., pick-and-place trajectories
for specific object configurations) could reintroduce a meaningful downstream
metric.

---

## Limitations

- **UG surrogate too weak for OOD text.** The pairwise overlap energy provides
  insufficient gradient signal when text embeddings are novel. A learned
  validity classifier trained on diverse prompts would be more robust.

- **9M model on 50k scenes has hit capacity.** Validity has plateaued at 1-6%
  across all sampling strategies. The architectural ceiling was established in
  Week 4 (v7 vs v8c ablation). More parameters alone do not help without more
  and more diverse training data.

- **Rejection sampling cost is real.** 4x oversampling at inference time trades
  compute for validity. At 200 prompts x 8 scenes x 4x = 6400 generation calls
  per eval run. At scale this is prohibitive.

---

## Week 6 Options

| Option | Cost | Expected gain |
|---|---|---|
| LinkedIn writeup + portfolio piece | 0 compute | Portfolio value |
| 500k scene dataset generation | ~$12-15, 25-33 hrs | Training data for v9 |
| Retrain v9 (25-50M params) on 500k | ~$20-30 | Validity ceiling lifted |
| Learned validity classifier for UG | ~$5-8 | UG generalization |

**Recommendation:** 500k dataset first, then v9 training. The validity ceiling
is a data and capacity problem, not a guidance problem. More data at the
current 9M scale is unlikely to yield meaningful gains; the jump to 25-50M
params on 500k scenes is the next meaningful experiment.

---

## Reproducibility

```bash
# Re-run full eval
python3 scripts/run_eval.py \
  --suite data/eval_suite_v1/eval_suite.jsonl \
  --data-dir data/v1 \
  --checkpoint-v7 checkpoints/conditional_v7/latest.pt \
  --ug-scale 0.5 \
  --out-dir artifacts/eval_v1 \
  --seeds 42 123 456 \
  --n-per-prompt 8 \
  --drake-workers 32 \
  --rrt-budget 2.0

# Rebuild results table
python3 scripts/build_results_table.py \
  --eval-dir artifacts/eval_v1 \
  --out artifacts/results_table.md
```

Config, seed, and git SHA are logged to W&B for all training runs.
Eval suite is deterministic given fixed seeds.
