# v9 Final Result

**Drake validity: 3.0% overall, 11.3% on non-empty scenes (n=200, seed=0)**
**10x improvement over v7 baseline on conditional rate. Training complete.**

## Checkpoint

- Artifact: `galanafai-self/demiurge/v9_final_model:latest` (W&B)
- Architecture: 49M params, d_model=384, 16 layers, 8 heads
- Training: v7 step 140k resume, pose_rot=2.0, 260k total steps
- Peak at step 170k before presence collapse

## Drake Probe Trajectory (N=100 per step, seed=step)

| Step | valid/100 | accepted (cond) | stable (cond) | ik (cond) |
|------|-----------|-----------------|---------------|-----------|
| 160k | 40 | 2.5%  | 17.5% | 75.0% |
| **170k** | **31** | **12.9%** | **32.3%** | **80.6%** |
| 180k | 11 | 9.1%  | 45.5% | 81.8% |
| 200k | 7  | 0.0%  | 71.4% | 100%  |
| 220k | 9  | 11.1% | 100%  | 100%  |
| 260k | 2  | 0.0%  | 100%  | 0%    |

Step 170k full probe (N=200): **6/53 accepted (11.3%)**, 6/200 overall (3.0%).

## What Fixed Geometry

| Fix | Impact |
|-----|--------|
| `pose_rot=2.0` | stable_rest 5% -> 30% (rot6d 6ch vs xyz 3ch gradient imbalance) |
| Yaw-only inference projection | Removes non-tabletop tilts from decoded quaternions |
| Drake API `SetDefaultFreeBodyPose` | Fixed silent crash in object placement |
| v-prediction + zero terminal SNR | Eliminated signal leak and z-collapse |

## Presence Collapse Root Cause

LR scheduler fast-forward (`for _ in range(140000): lr_sched.step()`) replays
140k scheduler updates without corresponding optimizer updates, misaligning
Adam momentum state with the new LR position. Presence BCE overshoots on
first gradient update, collapses to near-zero, never recovers.

**Fix for next run**: call `lr_scheduler.load_state_dict(ckpt["lr_state"])` directly.
Do not replay steps.

## Inference Recipe

```python
ckpt = torch.load("step_00170000.pt", map_location="cuda", weights_only=False)
model = SceneDenoiser(DenoiserConfig(**ckpt["arch"])).cuda()
model.load_state_dict(ckpt["ema_state"]); model.eval()

schedule = CosineSchedule(T=1000, zero_terminal_snr=True)
sampler  = DDIMSampler(schedule, n_steps=50, prediction_type="v")
x0 = sampler.sample(model.noise_prediction_fn(None), (N, N_MAX, 13), seed=seed)

x0[:, :, 5] = 0; x0[:, :, 8] = 0   # yaw-only projection (rot6d col 2 and 5)
pres = x0[:, :, 12] > -0.589
```

## Training History (8 runs, ~$40 total)

| Run | Steps | Outcome | Blocker |
|-----|-------|---------|---------|
| v1 | 150k | 0% | type_grad_isolation memorization |
| v2 | 100k | 0% | warm-init from broken v1 |
| v3 | 150k | 0% | covariance mismatch z-collapse |
| v4 | 100k | 4% peak | output drift |
| v5 | killed | exploded | DDIM epsilon vs v-pred bug |
| v6 | 134k | 0% | Drake API + presence collapse |
| v7 | 140k | 1.1% | yaw not enforced, x-spread deficit |
| **v9** | **170k** | **3.0% / 11.3%** | presence collapse post-resume |

## Next Step: Universal Guidance

With this unconditional base, implement classifier-guided sampling:
1. Train ValidityClassifier on Drake-validated scenes (positive) + rejected (negative)
2. Guidance gradient: `x_t += alpha * grad_x(log p(valid | x_t))`
3. Target: push from 3% -> 15%+ overall Drake validity
4. Compare vs rejection sampling baseline at equal inference compute

38 unit tests pass. Typed scene wrappers in place. Pipeline audited.
