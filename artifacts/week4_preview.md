# Week 4 Preview

## Starting Point

- **Best unconditional baseline:** unconditional_v3 (presence_bce=1.0, validity 5.4%, mean count 4.66)
- **Conditional result:** conditional_v1 CASE 3 (text-conditioned validity 1.0%, root cause: no CFG)
- **Architecture:** d_model=256, 8.89M params -- confirmed adequate for unconditional, insufficient
  training procedure for conditional
- **DataLoader:** num_workers=2 validated and ready

---

## Primary Goal: conditional_v2 with CFG

The single highest-leverage change is classifier-free guidance (CFG) training.

### Implementation Plan

**1. Add CFG dropout to `scripts/train.py`:**

```python
# Read from config
cfg_dropout = float(cfg.get("cfg_dropout", 0.0))

# In training loop, after loading text_emb_b:
if text_emb_b is not None and cfg_dropout > 0.0:
    drop_mask = torch.rand(text_emb_b.shape[0]) < cfg_dropout
    if drop_mask.any():
        text_emb_b = text_emb_b.clone()
        text_emb_b[drop_mask] = 0.0  # zero-embedding for dropped samples
```

**2. Add CFG scale to `model/schedule.py` DDIMSampler:**

```python
def sample_cfg(fn_cond, fn_uncond, shape, w=3.0, ...):
    # pred = uncond + w * (cond - uncond)
```

**3. `configs/train/conditional_cfg.yaml`:**

```yaml
text_conditioning: true
cfg_dropout: 0.15       # 15% unconditional training passes
training:
  max_steps: 200000     # 2x v1 budget
dataset:
  num_workers: 2
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
loss:
  presence_bce: 1.0    # keep v3 Pareto optimum
```

**4. Initialization:** Start from unconditional_v3 weights. The cross-attention layers
initialize randomly; all other layers warm-start from v3's trained state. This gives
the presence and pose heads a head start and focuses learning on the conditioning task.

Expected improvement: With CFG training and warm initialization, text-conditioned
validity should reach 5-15% within 100k steps.

---

## Secondary Goal: Evaluation Quality

The VLM judge infrastructure (`src/eval/judge.py`) is implemented but not yet used.
Once conditional_v2 achieves acceptable validity, run the VLM scorer on accepted scenes
to measure task relevance (does "place the mug" produce a scene with a mug?).

**Required:** `CLAUDE_API_KEY` env var on the pod.

---

## Recommended Experiment Sequence

### Week 4, Phase 1: conditional_v2 with CFG

Config: `configs/train/conditional_cfg.yaml`
Init: from `checkpoints/unconditional_v3/latest.pt` (cross-attention layers random)
Steps: 200k
Cost estimate: ~$0.90

Mid-run check at step 50k:
- Text-conditioned validity >= 3%: continue
- Text-conditioned validity == 0%: halt, diagnose cross-attention initialization

### Week 4, Phase 2: CFG scale sweep (if Phase 1 succeeds)

Sample 200 scenes at CFG scales w = [1.0, 2.0, 3.0, 5.0].
Report validity vs task relevance tradeoff.
Expected: higher w = higher task relevance, lower validity (over-conditioning).

### Week 4, Phase 3: VLM scoring (if Phase 2 succeeds)

Take the best-validity conditional model. Sample 100 accepted scenes.
Score each with `judge.py` using `claude-haiku-4-5` (bulk) and `claude-sonnet-4-6` (headline).
Report: mean task relevance score, distribution by template, qualitative examples.

---

## Deferred Work

| Item | Deferred To | Reason |
|---|---|---|
| d_model=512 escalation | Week 5 if needed | v3 at d_model=256 has sufficient unconditional validity |
| Rejection sampling guidance | Week 5 | Requires working conditional baseline first |
| Full 500-scene text-conditioned probe | conditional_v2 | v1 used 200 scenes; v2 should use 500 |
| Text description quality audit | Week 4 | Check if descriptions are sufficiently discriminative for presence learning |

---

## Open Questions

1. **CFG dropout probability:** 0.15 is the standard. For this dataset with 17k unique
   descriptions and ~3 objects per scene on average, 0.1 may be too low (model rarely
   sees unconditional training). Try 0.15-0.20.

2. **CFG scale at inference:** The right scale depends on the conditioning strength.
   Start at w=3.0, measure validity vs diversity tradeoff.

3. **Warm init strategy:** Load v3 weights, keep cross-attention random. The alternative
   (full random init, longer training) is cleaner experimentally but more expensive.

4. **Presence behavior with CFG:** With p=0.15 unconditional passes, the model should
   learn that unconditioned presence is low (training data mean ~2.5). Will this fix
   the 10.11 mean count seen in conditional_v1 unconditional probe?
   Expected: yes, by ~50% reduction.

---

## Infrastructure Changes Needed for Week 4

1. Add `cfg_dropout` config key and training loop support in `scripts/train.py`
2. Add `sample_cfg()` method to `DDIMSampler` for CFG inference
3. Add `--init-from` argument to `train.py` for warm initialization
4. Add `tests/model/test_cfg_sampling.py` unit test for CFG sampler
5. Implement VLM scoring pipeline in `scripts/score_scenes.py`
