# Week 3 Preview: Diffusion Model Architecture

**Status:** Planning artifact. No implementation work begins until the Week 3
prompt is issued and a Plan Artifact is approved.

**Dataset provenance:** This preview is based on the 50,000-scene v1 dataset
described in `artifacts/dataset_v1_card.md`. The dataset has full provenance
documentation including two production bugs, their diagnoses, and their fixes.
See `artifacts/leak_diagnosis.md` for the memory leak and CPython 3.11
`multiprocessing.Pool` deadlock diagnostic records.

---

## Dataset Characteristics Driving Architecture Decisions

| Property | Value | Implication |
|---|---|---|
| Total scenes | 50,000 | Small-to-medium; model capacity must not exceed data |
| Object slots (N_MAX) | 12 | Fixed sequence length -- fits in a transformer with low overhead |
| Vocab size | 12 types + 1 PAD | Object type embedding table: 13 rows |
| Per-slot feature dim | ~17 continuous/discrete dims | Compact; no dimensionality bottleneck |
| Task templates | 3 (tabletop_reach, cluttered_pick, obstacle_avoidance) | Multimodal scene distribution; conditioning on text is essential |
| Template imbalance | 20,318 / 15,921 / 8,164 | Requires weighted sampling; see below |
| Text MPD (baseline) | 1.0585 (all-MiniLM-L6-v2) | Text embeddings are well-separated; conditioning should help |
| Scene MPD (baseline) | 2.9105 (204-dim hand-crafted) | Pose space is well-covered; model must reproduce this coverage |

---

## Input/Output Shape

Per-slot representation (12 slots, N_MAX=12):

| Field | Dims | Encoding |
|---|---|---|
| Object type | 1 (int) | Embedded via learned table (vocab size 13, d_model dim) |
| XYZ position | 3 | Raw float, normalised to workspace bounds |
| Quaternion (wxyz) | 4 raw -> 6 in training | 6D rotation representation (Zhou et al. 2019) during diffusion |
| Scale (xyz) | 3 | Raw float |
| Presence bit | 1 | Binary; masked out in loss for absent slots |

Total per-slot continuous dims during diffusion: 3 + 6 + 3 + 1 = 13 (type
handled separately via embedding lookup).

Full scene tensor in model space: 12 slots x 13 continuous dims = 156 dims
(plus per-slot type embedding injected as a learned offset, not concatenated).

The output shape is identical to the input: the denoiser predicts the noise
vector epsilon of the same 156-dim scene tensor. Type logits are predicted
from the final slot representation via a separate linear head (cross-entropy
loss, not MSE).

Quaternions are projected to the unit quaternion sphere after each DDIM
step by normalising the 6D prediction and converting to wxyz.

---

## Denoiser Architecture

Transformer with object slots as tokens, conditioned on diffusion timestep
and task description.

| Hyperparameter | Value | Justification |
|---|---|---|
| Sequence length | 12 + 1 | 12 object tokens + 1 [NOISE_T] timestep token |
| d_model | 256 | Sufficient expressivity for 12-slot scenes; fits on 4090 at batch=128 |
| n_heads | 8 | d_head = 32; standard for d_model=256 |
| n_layers | 6 | Empirical: 6 layers covers 3-4 object interaction hops |
| FFN expansion | 4x | d_ffn = 1024 |
| Timestep conditioning | AdaLN | Applied to each layer; timestep is a scalar, not a sequence |
| Text conditioning | Cross-attention | One cross-attention sublayer per transformer block |

**Parameter count estimate:**

- Self-attention per layer: 4 x (256 x 256) = 262,144 params per layer, x6 = 1.6M
- Cross-attention per layer: Q (256x256) + KV (384x256 x2) = 459,776 per layer, x6 = 2.8M
- FFN per layer: 2 x (256x1024) = 524,288 per layer, x6 = 3.1M
- AdaLN per layer: 2 x 256 x 2 = ~3K per layer, negligible
- Input embeddings (type table 13x256, continuous projection 13x256): ~7K
- Output heads (noise 156-dim, type logits 13x12): ~2K

**Total: approximately 7.5M trainable parameters.**

This is in the lower portion of the 10-50M target range from the skill. If
Week 3 validation shows the model is underfit (validity rate plateaus below
0.5 at epoch 100), increase d_model to 512 (roughly 4x parameter count,
~30M) before changing depth.

---

## Text Conditioning

**Encoder:** `sentence-transformers/all-MiniLM-L6-v2` (frozen, 384-dim output,
22M parameters). Not trained; embeddings are precomputed and cached per run.

**Rationale:** The baseline text MPD of 1.0585 confirms descriptions are
well-separated in this embedding space. Training a text encoder from scratch
on 50k examples would underfit. A larger encoder (e.g., CLIP ViT-L/14) is
deferred to Week 5 if task relevance scores are low.

**Cross-attention strategy:** Each of the 6 transformer blocks contains one
cross-attention sublayer positioned after the self-attention sublayer. Q
comes from the 12 scene tokens; K and V come from the frozen 384-dim text
embedding projected to d_model=256 via a learned linear layer. The [NOISE_T]
token does not attend to the text (it is masked out in cross-attention).

---

## Diffusion Schedule

| Parameter | Value |
|---|---|
| Noise schedule | Cosine beta (Nichol and Dhariwal 2021) |
| Training steps T | 1000 |
| DDIM sampling steps (inference) | 50 |
| Loss | MSE on predicted noise epsilon (continuous dims) + cross-entropy on type logits |
| Type loss weight | 0.1 (tuned to prevent type prediction from dominating early training) |
| Presence loss weight | 0.05 (sparse; most slots are absent) |

---

## Training Budget

**Hardware target:** Single NVIDIA RTX 4090 (24 GB VRAM). RunPod community
cloud at approximately $0.74/hr.

**Batch and epoch math:**

| Parameter | Value |
|---|---|
| Dataset size | 50,000 scenes |
| Batch size | 128 |
| Steps per epoch | 50,000 / 128 = 391 |
| Target epochs | 100 |
| Total steps | 39,100 |

**Throughput projection (4090):** The denoiser is a small transformer (7.5M
params, sequence length 13). Estimated throughput is 150-250 steps/sec
depending on attention kernel selection (FlashAttention-2 vs PyTorch SDPA).
At 200 steps/sec (midpoint), 39,100 steps = approximately 3.3 minutes of
pure compute.

However: DataLoader I/O on WebDataset shards, validation passes every 10
epochs (5,000-scene held-out validity rate check via Drake), and checkpoint
saves will dominate wall-clock. Realistic estimate: 4 to 6 hours for 100
epochs including validation overhead.

> **Verify with a 5-epoch benchmark before committing to the full schedule.**
> Realistic 4090 throughput on this architecture is 150-250 steps/sec
> depending on attention kernel; final wall-clock may be 4-6 hours. Benchmark
> command: `uv run python scripts/train.py --config configs/train/v1.yaml
> --seed 42 --max-epochs 5` and extrapolate.

**Cost estimate:** 5 hours x $0.74/hr = approximately $3.70 for a full 100-epoch
run. Budget for 3 runs (initial + 2 ablations): approximately $11.10.

---

## Per-Template Imbalance and Weighted Sampling

The v1 dataset has a significant template imbalance:

| Template | Count | Fraction |
|---|---|---|
| tabletop_reach | 20,318 | 40.6% |
| obstacle_avoidance | 15,921 | 31.8% |
| cluttered_pick | 8,164 | 16.3% |
| (residual from mix ratio rounding) | 5,597 | 11.2% |

The imbalance arises from two sources: the configured mix (40% / 30% / 30%)
and the differing acceptance rates per template (12.2% / 12.7% / 6.5%). The
cluttered_pick template is structurally hard; at half the acceptance rate of
the others, it is underrepresented relative to its configured 30% mix
contribution.

**Decision: weighted sampling as the default training configuration.**

The DataLoader should sample from the three templates with weights inversely
proportional to their representation:

```
w_tabletop_reach    = 1 / 20318 * normalization_constant
w_obstacle_avoid    = 1 / 15921 * normalization_constant
w_cluttered_pick    = 1 / 8164  * normalization_constant
```

This exposes the model to cluttered_pick scenes at approximately 2.5x the
naive rate, ensuring the model learns the dense-obstacle regime rather than
treating it as a rare edge case.

**Rationale:** The Week 5 evaluation question is whether the model produces
valid scenes across a difficulty spectrum. A model that rarely generates
cluttered_pick scenes will score well on tabletop_reach validity but fail on
the hardest category. Weighted sampling is the standard remedy for structural
class imbalance in diffusion models trained on multi-distribution datasets.

**Alternative to document:** Train one ablation with uniform sampling and
compare cluttered_pick validity rate at inference. If the gap is small (<5pp),
simplify to uniform. If large (>10pp), weighted sampling is essential for the
paper claim.

The weighted-sampling flag should be a config parameter in
`configs/train/v1.yaml` so both regimes can be tested without code changes.

---

## Week 3 Deliverables (Preview Only)

The following are scoped for Week 3 implementation. No code is written here.

1. `src/model/denoiser.py` -- `SceneDenoiser` transformer with AdaLN and
   cross-attention to frozen text encoder.
2. `src/model/ddpm.py` -- DDPM forward process, cosine schedule, DDIM sampler.
3. `src/model/encoder.py` -- Frozen sentence-transformer wrapper with caching.
4. `scripts/train.py` -- Training loop, W&B logging, checkpoint saves, 5-epoch
   validation pass via Drake.
5. `configs/train/v1.yaml` -- Training config.
6. Unit tests: toy 3-object model converges on a trivial distribution, DDIM
   samples have correct shape, AdaLN conditions on timestep embedding.

**Layer 4 build order per `SKILL.md`:** scene schema (done), validator (done),
data pipeline (done) -> model (Week 3) -> guidance (Week 4) -> evaluation (Week 5).
