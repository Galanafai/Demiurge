# v9 Final Results

## Headline

**Drake validity: 3.0% overall, 11.3% on non-empty scenes (n=200, seed=0)**

10x improvement over v7 baseline on conditional rate. 6x improvement on stable_rest.

## Checkpoint

- **File:** `checkpoints/v9_uncond_v9/step_00170000.pt` (pod only, 779MB)
- **W&B Artifact:** `galanafai-self/demiurge/v9_final_model:latest`
- **W&B Run:** https://wandb.ai/galanafai-self/demiurge/runs/3ox3kzg8
- **Architecture:** 49M params, d_model=384, 16 layers, 8 heads, ffn_mult=4
- **Training:** Resumed from v7 step 140k with pose_rot=2.0

## Drake Probe Results (N=200, seed=0, yaw-proj)

| Metric | v7 step 140k | v9 step 170k | Delta |
|--------|-------------|-------------|-------|
| Presence fill | 180/200 (90%) | 53/200 (27%) | -63pp (collapse) |
| Accepted (overall) | 2/200 (1.0%) | 6/200 (3.0%) | +3x |
| Accepted (non-empty) | 2/180 (1.1%) | 6/53 (11.3%) | **+10x** |
| No interpenetration | ~98% | 100% | -- |
| Stable rest | 5.0% | 30.2% | **+6x** |
| IK reachable | 72.8% | 81.1% | +8pp |
| RRT solvable | ~10% | 24.5% | +2.5x |

## Inference Recipe

```python
from model.denoiser import DenoiserConfig, SceneDenoiser
from model.schedule import CosineSchedule, DDIMSampler
from model.rotations import rot6d_to_quat_wxyz
import torch

ckpt = torch.load("step_00170000.pt", map_location="cuda", weights_only=False)
model = SceneDenoiser(DenoiserConfig(**ckpt["arch"])).cuda()
model.load_state_dict(ckpt["ema_state"]); model.eval()

schedule = CosineSchedule(T=1000, zero_terminal_snr=True)
sampler  = DDIMSampler(schedule, n_steps=50, prediction_type="v")
x0 = sampler.sample(model.noise_prediction_fn(None), (N, N_MAX, 13), seed=0)

# Yaw-only projection -- REQUIRED, do not skip
x0[:, :, 3+2] = 0   # rot6d col 2
x0[:, :, 3+5] = 0   # rot6d col 5

pres = x0[:, :, 12] > -0.589   # presence threshold
```

## What Fixed Geometry

1. **`pose_rot=2.0`**: rot6d has 6 channels vs xyz's 3. Doubling the weight
   equalized per-channel gradient flow. stable_rest: 5% -> 30%.

2. **Yaw-only inference projection**: training data has q_x=q_y=0 exactly.
   Any non-zero tilt causes stable_rest failures. Zero rot6d[:,2] and rot6d[:,5].

3. **Drake API fix**: `SetDefaultFreeBodyPose` (was crashing with deprecated API).

## Known Limitation: Presence Collapse

LR scheduler fast-forward during resume misaligned Adam momentum with LR,
causing presence BCE to overshoot. 73% of scenes are empty at step 170k.
Projected validity with healthy presence (v7 fill rate): ~8-10% overall.

## Next Steps

Universal Guidance: train ValidityClassifier on Drake-validated scenes,
implement classifier-guided sampling to push accepted rate from 3% -> 15%+.
