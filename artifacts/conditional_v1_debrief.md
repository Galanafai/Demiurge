# Conditional v1 Debrief -- Step 100,000

**Run:** [conditional_v1](https://wandb.ai/galanafai-self/demiurge/runs/q90jxg8q)
**Config:** `configs/train/conditional.yaml` | **Seed:** 42 | **Random init**
**Architecture:** d_model=256, 8.89M params | **DataLoader:** num_workers=2
**Wall-clock:** ~38 min | **Throughput:** 62-70 steps/sec | **Cost:** ~$0.44

---

## 1. Final Training Loss

| Component | Weight | v3 uncond | cond_v1 |
|---|---|---|---|
| total | -- | 1.173 | **1.199** |
| pose_xyz | 1.0 | 0.331 | **0.267** |
| pose_rot | 1.0 | 0.666 | **0.690** |
| scale | 0.7 | 0.161 | **0.248** |
| type_ce | 0.1 | ~0.000 | **~0.000** |
| presence_bce | 1.0 | 0.063 | **0.069** |

pose_xyz improved vs v3 (0.267 vs 0.331) -- text conditioning is providing useful spatial signal. scale regressed (0.248 vs 0.161) -- more gradient competition from cross-attention. presence_bce 0.069 is above v3's floor of 0.063, suggesting presence learning is slightly harder with conditioning.

---

## 2. In-Training Validity Trajectory

All 25 validation passes returned 0.000.

This is expected behavior. The in-training validation calls `run_validation(model, text_emb=None)` -- unconditional sampling on a model trained with `text_emb` always present. Without the text signal, the model reverts to high-count distributions (mean 10.11 objects, see Probe 1), causing immediate interpenetration rejection in Drake.

This is a known consequence of training without classifier-free guidance (CFG). The model has not learned a robust unconditional fallback path.

---

## 3. Probe B

| Timestep | v3 uncond | cond_v1 conditional | cond_v1 unconditional |
|---|---|---|---|
| t=10 | 0.199 | **0.193** | 0.226 |
| t=100 | 0.226 | **0.264** | 0.280 |
| t=999 | 0.459 | **0.411** | 0.441 |

- t=100 shows +0.038 vs v3 with conditioning -- text is helping at mid-noise.
- t=999 regresses slightly from v3.
- Average Probe B (conditional): (0.193+0.264+0.411)/3 = **0.289** vs v3's **0.295** -- essentially equal.

---

## 4. Probe 1 -- Unconditional (text_emb=None), 500 Scenes

| Metric | unconditional_v3 | cond_v1 uncond |
|---|---|---|
| Accepted | 27/500 (5.4%) | **0/500 (0.0%)** |
| Mean obj count | 4.66 | **10.11** |
| Wall-clock | 160s | 12s (all fast-fail) |
| Interpenetration | 63% | **>99%** |

The conditional model has completely lost unconditional validity. Mean count 10.11 matches unconditional_v1 (before any presence_bce tuning). Without text conditioning at inference, the presence head collapses to predicting all slots present.

---

## 5. Probe 2 -- Text-Conditioned (Held-Out), 200 Scenes

200 scenes from 3 templates (tabletop_reach, cluttered_pick, obstacle_avoidance), last 100 per template, capped at 200 total.

| Metric | unconditional_v3 | cond_v1 text-prompted |
|---|---|---|
| Accepted | N/A | **2/200 (1.0%)** |
| 95% CI | -- | [0.000, 0.024] |
| Mean obj count | 4.66 | **8.21** |
| Wall-clock | -- | 88s |

**Text-conditioned validity: 1.0% (Case 3)**

Probe 2 (1.0%) > Probe 1 (0.0%) -- text conditioning helps marginally. But 1.0% is well below the v3 unconditional baseline of 5.4%.

**Rejection breakdown (198 rejected):**

| Stage | Count | Share |
|---|---|---|
| Interpenetration | 150 | 76% |
| Stability failure | 33 | **17%** |
| RRT unsolvable | 11 | 6% |
| IK unreachable | 4 | 2% |

Stability failures at 17% is the largest spike relative to any unconditional run (always <2%). The text conditioning is influencing placement (interpenetration fell from >99% to 76%) but generating unstable configurations.

---

## 6. Final Comparison Table

| Run | Validity | Mean count | Probe B avg |
|---|---:|---:|---:|
| unconditional_v3 | **5.4%** | 4.66 | 0.295 |
| conditional_v1 unconditioned | 0.0% | 10.11 | 0.356 |
| conditional_v1 text-prompted | 1.0% | 8.21 | 0.289 |

---

## 7. Case Assessment: CASE 3

Text-conditioned validity (1.0%) < 4% threshold.

Text conditioning as implemented has regressed performance vs the v3 unconditional baseline.

---

## 8. Root Cause Analysis

### Root Cause 1: No classifier-free guidance (CFG) -- primary

During training, `text_emb` was always provided (dropout=0%). The model never learned to denoise without text conditioning. At inference with `text_emb=None`, the presence head has no learned prior and collapses to predicting all slots present.

**Fix:** Train with CFG dropout: randomly replace `text_emb` with zeros or None with probability p=0.1-0.2 per batch. This forces simultaneous learning of conditional and unconditional distributions. At inference, use CFG scale w=2-4 to amplify the conditional signal.

### Root Cause 2: 100k steps insufficient for joint learning

The model must simultaneously learn: DDPM noise prediction, cross-attention conditioning, presence from text count, pose from text objects, and rotation/scale. 100k steps at 8.89M params is the same budget as unconditional runs which showed high variance. The conditional task is strictly harder.

**Fix:** 200k steps, or initialization from v3 unconditional weights (cross-attention layers randomly initialized, rest pre-trained).

### Root Cause 3: Stability failure spike

The 17% stability rejection rate suggests the text embedding is providing horizontal placement hints (spreading objects across the workspace) but not encoding stable height/orientation priors. Objects are placed in valid XY positions but floating or tilted.

**Fix:** Same as Root Cause 1 (CFG) -- the unconditional path provides the stability prior, the conditional path specializes placement.

---

## 9. Recommended Fix: conditional_v2 with CFG

Config changes for `configs/train/conditional_cfg.yaml`:

```yaml
text_conditioning: true
cfg_dropout: 0.15        # Drop text_emb with this probability per batch
training:
  max_steps: 200000      # 2x budget
  # Or: init from v3 unconditional weights, keep cross-attn layers random
```

Training changes in `scripts/train.py`:

```python
# In training loop, after loading text_emb_b:
if text_emb_b is not None and torch.rand(1).item() < cfg_dropout:
    text_emb_b = None  # unconditional pass for CFG training
```

Inference changes: use CFG scale w=3.0 at sampling time.

Expected improvement: CFG training should lift text-conditioned validity to >5% and restore unconditional validity to near v3 levels.
