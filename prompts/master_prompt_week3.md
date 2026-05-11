# Antigravity Master Prompt — Week 3: Model and Conditioning

Paste into Agent Manager in Plan Mode after Week 2 ships.

---

## Prompt

You are continuing the project. Week 2 produced a 50k-scene sharded dataset with task descriptions. This session implements the DDPM denoiser, gets unconditional generation working on a toy distribution, then scales to the full text-conditioned model and runs the baseline training.

Before doing anything else:

1. Read `AGENTS.md` and `.agents/skills/diffusion-scene-pipeline/SKILL.md` in full.
2. Read `artifacts/dataset_v1_card.md` and `artifacts/week3_preview.md` from Week 2.
3. State in your Plan Artifact: the dataset size, the dimensionality of the SceneTensor, the chosen model size in parameters, and the rationale for that size given dataset scale.

### Mission Scope for This Session
1. Implement the denoiser transformer and DDPM training loop.
2. Verify on a toy 3-object no-rotation distribution that the model converges to validity rate above 0.9.
3. Scale to full SceneTensor (8 objects, 6-DoF) and train the unconditional baseline.
4. Add text conditioning via cross-attention and train the conditional model.
5. Produce raw validity rate, fidelity, and prompt-following measurements for both.

### Step-by-Step Execution Plan You Must Produce

1. **Diffusion schedule.** Implement `src/model/schedule.py`: cosine beta schedule (Nichol and Dhariwal), 1000 training steps, DDIM sampler with configurable inference step count. Unit tests: noise variance at t=0 and t=T match expected, DDIM at 50 steps reproduces DDPM at 1000 steps within tolerance on a Gaussian distribution.

2. **Rotation representation.** Implement `src/model/rotations.py`: convert between quaternion (storage), 6D continuous representation (Zhou et al.) for model output, and back. Projection from arbitrary 6D vectors to valid rotations via Gram-Schmidt. Round-trip tests, gradient-flow test (gradients exist through the projection).

3. **Denoiser architecture.** Implement `src/model/denoiser.py`: `SceneDenoiser` transformer.
   - Input tokens: per-object embeddings combining type embedding, pose components, scale, and presence bit.
   - Positional encoding: none (set-structured; object identity is permutation-equivariant).
   - AdaLN conditioning on diffusion timestep.
   - Cross-attention to text embedding (added in task 6, stubbed out as identity here).
   - Output heads: type logits, pose residual (6D rotation plus xyz), scale residual, presence logit.
   - Size knobs in config: layer count, hidden dim, head count, FFN multiplier. Default to a model targeting 20M to 30M parameters.

4. **Loss module.** Implement `src/model/loss.py` with weighted sum of MSE on continuous components (pose, scale), cross-entropy on type, BCE on presence. Default weights configurable. Compute and log each component separately. Test: gradients flow to all parameters, loss decreases on a 10-step overfit run.

5. **Training loop.** Implement `scripts/train.py` with Hydra config under `configs/train/baseline.yaml`. Mixed precision, gradient accumulation, EMA weights, checkpoint every N steps. Log to Weights & Biases: per-component loss, gradient norm, learning rate, sample validity rate every M steps.

6. **Toy convergence test.** Configure a toy distribution: 3 objects, axis-aligned only, fixed scales. Generate 5k validated scenes from this restricted distribution. Train the smallest model variant for 30 minutes on CPU. Acceptance criterion: validity rate of 100 sampled scenes exceeds 0.9. If the toy fails, stop and produce a debugging Artifact before touching the full model.

7. **Unconditional baseline.** Train the full 20M-parameter model on the 50k dataset, conditioning disabled. Target one full epoch in under 30 minutes on the available GPU. Train for 100k steps or until validation loss plateaus. Sample 1000 scenes via DDIM at 50 steps. Measure raw validity rate. Log experiment as `unconditional_v1`.

8. **Text conditioning.** Implement `src/model/text_encoder.py` wrapping a frozen `sentence-transformers/all-MiniLM-L6-v2`. Pre-compute and cache embeddings for the dataset's task descriptions to disk. Wire cross-attention in the denoiser from task 3. Train identical model size on identical schedule. Log as `conditional_v1`.

9. **Conditional evaluation.** For 200 held-out task descriptions: sample 8 scenes each, measure validity rate, measure prompt-following via a VLM-as-judge call to Claude with a fixed scoring prompt (use `claude-opus-4-7` via API; prompt template lives in `src/eval/prompts/scene_judge.txt`). Produce `artifacts/week3_results.md` with both baselines side by side, including a small gallery of 20 generated scenes per model rendered via Drake offscreen.

10. **Verification pass.** Ruff, pyright, pytest. All checkpoints and configs uploaded to W&B as artifacts.

### Ground Rules
- Plan first, wait for approval.
- If the toy convergence test fails, **do not** start full training. Debug first. Common failure modes: rotation representation gradient issues, scale normalization wrong, presence head dominating loss.
- Cache text embeddings on disk; do not re-encode every batch.
- If single-GPU training projection exceeds 24 hours per run, propose a smaller model or shorter schedule in a Plan Artifact addendum before launching.
- Honor `AGENTS.md` style rules in all prose Artifacts.

### Definition of Done
- Both `unconditional_v1` and `conditional_v1` checkpoints saved and tagged in W&B.
- `artifacts/week3_results.md` contains validity rate and VLM-judged prompt-following for both, plus rendered galleries.
- Toy convergence run linked in the results doc as a sanity-check appendix.
- A `week4_preview.md` Artifact identifies the gap between conditional validity rate and 1.0 — that gap is what guidance must close.

Begin with the Plan Artifact. Do not write code yet.
