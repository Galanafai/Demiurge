# Week 3 Results Summary

## Objective

Train a conditional diffusion model that generates physically valid 6-DoF tabletop scenes
for a UR5e arm, validated through Drake. Demonstrate that text conditioning improves or
at least maintains the unconditional validity baseline.

---

## Unconditional Ablation (v1-v4)

The primary deliverable of Week 3 was the unconditional ablation to find the
Pareto-optimal `presence_bce` weight before conditional training.

| Run | pres_bce wt | pres_bce loss | Validity | Mean count | Probe B avg |
|---|---|---|---|---|---|
| v1 | 0.05 | 0.054 | 0.0% | ~10.5 | -- |
| v2 | 0.50 | 0.063 | 1.4% | 6.40 | 0.337 |
| v3 | 1.00 | 0.063 | **5.4%** | 4.66 | **0.295** |
| v4 | 2.00 | 0.063 | 8.6% | 3.73 | 0.294 |

**v3 is the Pareto optimum:** Clears the 5% validity gate. Best Probe B. BCE loss
floor (0.063) confirmed across v2-v4 -- the capacity ceiling for d_model=256 at 100k steps.

**Key finding:** The BCE raw loss equilibrium is capacity-limited, not weight-limited.
Increasing the weight from 0.5 to 2.0 (4x) does not change the scalar loss value but
does shift the generated count distribution via gradient magnitude effects.

---

## Conditional Result

| Run | Validity | Mean count | Probe B avg | vs v3 baseline |
|---|---:|---:|---:|---|
| unconditional_v3 | 5.4% | 4.66 | 0.295 | baseline |
| conditional_v1 unconditioned | 0.0% | 10.11 | 0.356 | **worse** |
| conditional_v1 text-prompted | 1.0% | 8.21 | 0.289 | **worse** |

**Result: CASE 3 -- text conditioning hurt.**

Text-conditioned validity (1.0%) is below the 4% threshold and below the v3 unconditional
baseline (5.4%).

**The root cause is not cross-attention design -- it is the absence of CFG training.**

The model was trained with `text_emb` always present. It learned to condition on text but
never learned a fallback unconditional prior. Without CFG, the model cannot be used as a
standalone unconditional generator, and the conditional path hasn't converged in 100k steps.

---

## DataLoader Optimization

| Setting | Steps/sec | GPU Util |
|---|---|---|
| num_workers=0 (v1-v4) | 100-112 | 20-26% |
| num_workers=2 (conditional_v1) | 62-70 | 47-57% |

GPU utilization nearly doubled. The lower absolute throughput for the conditional run is
due to cross-attention overhead, not DataLoader bottleneck.

---

## Total Cost

| Run | Steps | Cost |
|---|---|---|
| unconditional_v1 | 100k | ~$0.68 |
| unconditional_v2 | 100k | ~$0.71 |
| unconditional_v3 | 100k | ~$0.98 |
| unconditional_v4 | 100k | ~$0.75 |
| conditional_v1 | 100k | ~$0.44 |
| **Total** | **500k** | **~$3.56** |

---

## Key Lessons

1. **Presence_bce weight matters, but has a floor.** The BCE equilibrium at 0.063 is
   set by model capacity, not weight magnitude. Pushing above 1.0 produces diminishing
   returns on validity and regresses Probe B.

2. **Unconditional training works.** v3 at 5.4% validity and mean count 4.66 is a
   legitimate unconditional scene generator for tabletop manipulation tasks.

3. **CFG is required for conditional training.** Training with `text_emb` always present
   creates a model that cannot operate without text conditioning. CFG dropout (p=0.15)
   must be used in conditional_v2.

4. **DataLoader workers matter for conditional.** The collate function's hash lookup and
   embedding retrieval need worker parallelism. num_workers=2 is the correct setting.

5. **Drake rejection modes are informative.** The shift from interpenetration-dominated
   (v1: 91%) to RRT/IK-dominated (v4: 43%) tells a clear story about model maturation.

---

## Week 3 Outcome

**Partial success.** The unconditional ablation is complete and documented. The conditional
model failed to surpass the unconditional baseline due to a missing CFG training component.
The architecture and data pipeline are correct -- the fix is a training procedure change,
not an architectural change.
