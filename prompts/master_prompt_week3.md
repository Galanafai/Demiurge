# Antigravity Master Prompt — Week 3: Model and Conditioning

Paste into Agent Manager in Plan Mode after Week 2 ships.

---

## Prompt

You are continuing the project. Week 2 produced a 50k-scene sharded 
dataset with task descriptions, plus full documentation of two production 
bugs discovered and fixed during generation. This session implements the 
DDPM denoiser, gets unconditional generation working on a toy 
distribution, then scales to the full text-conditioned model and runs 
the baseline training.

Before doing anything else:

1. Read `AGENTS.md` and `.agents/skills/demiurge/SKILL.md` in full.
2. Read `artifacts/dataset_v1_card.md` and `artifacts/week3_preview.md` 
   from Week 2.
3. Read `artifacts/leak_diagnosis.md` for context on Week 2's production 
   challenges (not directly relevant to Week 3 but useful for 
   understanding the dataset's generation history).
4. State in your Plan Artifact: the dataset size, the dimensionality of 
   the SceneTensor (N_MAX=12), the chosen model size in parameters, and 
   the rationale for that size given dataset scale.

### Mission Scope for This Session

1. Implement the denoiser transformer and DDPM training loop.
2. Verify on a toy 3-object no-rotation distribution that the model 
   converges to validity rate above 0.9.
3. Scale to full SceneTensor (12 objects, 6-DoF) and train the 
   unconditional baseline.
4. Add text conditioning via cross-attention and train the conditional 
   model.
5. Produce raw validity rate, fidelity, and prompt-following 
   measurements for both.

### Dataset Context (Week 2 Reality)

N_MAX is fixed at 12 from Week 2. Do not change it.

The dataset has per-template imbalance:
- tabletop_reach: 20,318 scenes (40.6%)
- obstacle_avoidance: 15,921 scenes (31.8%)
- cluttered_pick: 8,164 scenes (16.3%)
- Pre-tracking initial run: ~5,597 scenes (template distribution 
  approximate)

Consider using weighted sampling during training to compensate for the 
cluttered_pick under-representation. Default to uniform-by-template 
weighted sampling unless you find a reason to deviate. Document the 
decision in the Plan Artifact.

### Compute Environment

Cloud GPU required for full training. Default target: RunPod RTX 4090 
(24 GB VRAM, ~$0.74/hr) or A100 40GB (~$1.89-3.99/hr depending on 
availability). Per the Week 3 preview, the 4090 should handle full 
training in 4-6 hours; verify with a 5-epoch benchmark before 
committing to the full schedule.

Reference Week 2's `docs/cloud_run.md` for the general cloud workflow 
pattern (SSH + tmux + cloud_setup.sh + rsync + destroy). RunPod GPU 
provisioning differs from Hetzner CPU; a new `docs/cloud_gpu_run.md` 
may be needed.

Toy convergence test (task 6) must pass on CPU before any GPU spending. 
Cost control: total RunPod spend for Week 3 should stay under $20.

### Step-by-Step Execution Plan You Must Produce

1. **Diffusion schedule.** Implement `src/model/schedule.py`: cosine 
   beta schedule (Nichol and Dhariwal), 1000 training steps, DDIM 
   sampler with configurable inference step count. Unit tests: noise 
   variance at t=0 and t=T match expected, DDIM at 50 steps reproduces 
   DDPM at 1000 steps within tolerance on a Gaussian distribution.

2. **Rotation representation.** Implement `src/model/rotations.py`: 
   convert between quaternion (storage), 6D continuous representation 
   (Zhou et al.) for model output, and back. Projection from arbitrary 
   6D vectors to valid rotations via Gram-Schmidt. Round-trip tests, 
   gradient-flow test (gradients exist through the projection).

3. **Denoiser architecture.** Implement `src/model/denoiser.py`: 
   `SceneDenoiser` transformer.
   - Input tokens: per-object embeddings combining type embedding, 
     pose components, scale, and presence bit.
   - Positional encoding: none (set-structured; object identity is 
     permutation-equivariant).
   - AdaLN conditioning on diffusion timestep.
   - Cross-attention to text embedding (added in task 8, stubbed out 
     as identity here).
   - Output heads: type logits, pose residual (6D rotation plus xyz), 
     scale residual, presence logit.
   - Size knobs in config: layer count, hidden dim, head count, FFN 
     multiplier. Default to a model targeting 20M to 30M parameters.
   - Operates on 12-object sequences (N_MAX=12 fixed from Week 2).

4. **Loss module.** Implement `src/model/loss.py` with weighted sum of 
   MSE on continuous components (pose, scale), cross-entropy on type, 
   BCE on presence. Default weights configurable. Compute and log each 
   component separately. Test: gradients flow to all parameters, loss 
   decreases on a 10-step overfit run.

5. **Training loop.** Implement `scripts/train.py` with Hydra config 
   under `configs/train/baseline.yaml`. Mixed precision, gradient 
   accumulation, EMA weights, checkpoint every N steps. Log to Weights 
   & Biases: per-component loss, gradient norm, learning rate, sample 
   validity rate every M steps.

6. **Toy convergence test.** Configure a toy distribution: 3 objects, 
   axis-aligned only, fixed scales. Generate 5k validated scenes from 
   this restricted distribution. Train the smallest model variant for 
   30 minutes on CPU. Acceptance criterion: validity rate of 100 
   sampled scenes exceeds 0.9. If the toy fails, stop and produce a 
   debugging Artifact before touching the full model.

7. **Unconditional baseline.** Train the full 20M-parameter model on 
   the 50k dataset, conditioning disabled. Target one full epoch in 
   under 30 minutes on the 4090. Train for 100k steps or until 
   validation loss plateaus. Sample 1000 scenes via DDIM at 50 steps. 
   Measure raw validity rate. Log experiment as `unconditional_v1`.

8. **Text conditioning.** Implement `src/model/text_encoder.py` 
   wrapping a frozen `sentence-transformers/all-MiniLM-L6-v2`. 
   Pre-compute and cache embeddings for the dataset's task descriptions 
   to disk. Wire cross-attention in the denoiser from task 3. Train 
   identical model size on identical schedule. Log as `conditional_v1`.

9. **Conditional evaluation.** For 200 held-out task descriptions: 
   sample 8 scenes each, measure validity rate, measure prompt-
   following via a VLM-as-judge call to Claude with a fixed scoring 
   prompt. Use `claude-sonnet-4-6` via API (Sonnet 4.6 is sufficient 
   for scene-description matching; reserves Opus capacity for higher-
   judgment-difficulty tasks if needed later). Optionally cross-
   validate with `claude-haiku-4-5` on a 50-scene subset to verify 
   scoring stability. The judge prompt template lives in 
   `src/eval/prompts/scene_judge.txt`. Produce `artifacts/week3_results.md` 
   with both baselines side by side, including a small gallery of 20 
   generated scenes per model rendered via Drake offscreen.

10. **Verification pass.** Ruff, pyright, pytest. All checkpoints and 
    configs uploaded to W&B as artifacts.

### Ground Rules

- Plan first, wait for approval.
- If the toy convergence test fails, do not start full training. Debug 
  first. Common failure modes: rotation representation gradient issues, 
  scale normalization wrong, presence head dominating loss.
- Cache text embeddings on disk; do not re-encode every batch.
- If single-GPU training projection exceeds 24 hours per run, propose 
  a smaller model or shorter schedule in a Plan Artifact addendum 
  before launching.
- N_MAX is fixed at 12 from Week 2. Do not change it.
- The dataset has known per-template imbalance (see Dataset Context). 
  Account for this in training but do not modify the dataset.
- Toy convergence test must pass on CPU before any GPU spending.
- Honor `AGENTS.md` style rules in all prose Artifacts.
- No em dashes or double hyphens in prose.

### Definition of Done

- Both `unconditional_v1` and `conditional_v1` checkpoints saved and 
  tagged in W&B.
- `artifacts/week3_results.md` contains validity rate and VLM-judged 
  prompt-following for both, plus rendered galleries.
- Toy convergence run linked in the results doc as a sanity-check 
  appendix.
- A `week4_preview.md` Artifact identifies the gap between conditional 
  validity rate and 1.0; that gap is what guidance must close in Week 4.
- Total RunPod cloud spend for Week 3 documented and under $20.
- Total VLM judging API spend documented and under $20.

Begin with the Plan Artifact. Do not write code yet.