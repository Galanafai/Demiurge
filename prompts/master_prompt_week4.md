# Antigravity Master Prompt — Week 4: Classifier Guidance

Paste into Agent Manager in Plan Mode after Week 3 ships.

---

## Prompt

You are continuing the project. Week 3 produced trained unconditional and conditional diffusion models with measured validity rates. This session implements the project's central research contribution: classifier-guided diffusion that pushes generation toward physically valid scenes.

Before doing anything else:

1. Read `AGENTS.md` and `.agents/skills/demiurge/SKILL.md` in full.
2. Read `artifacts/week3_results.md` and `artifacts/week4_preview.md`.
3. State in your Plan Artifact: the conditional validity rate from Week 3, the validity-rate gap to close, and the hypothesis for why classifier guidance should close it.

### Mission Scope for This Session
1. Build the rejection sampling baseline so guidance has a quantitative reference point.
2. Train a noise-conditioned validity classifier.
3. Implement classifier-guided sampling.
4. Sweep guidance scale and produce the validity-rate vs diversity Pareto curve.

### Step-by-Step Execution Plan You Must Produce

1. **Sampler interface.** Implement `src/guidance/base.py` defining `Sampler` protocol: `sample(prompts, n_per_prompt, **kwargs) -> List[SceneTensor]`, `name() -> str`, `config() -> dict`. All downstream samplers conform. Test: protocol enforced via runtime check.

2. **Rejection sampler.** Implement `src/guidance/rejection.py`. Generate batches from the conditional model from Week 3, validate, keep accepted up to target N, stop after a max-sample budget per prompt. Log accept rate, mean attempts per accepted scene, and total wall-clock. Test: on 100 prompts, rejection sampler hits target validity rate of 1.0 by construction.

3. **Classifier data generation.** Implement `scripts/generate_classifier_data.py`. For each scene in the training set, sample a random diffusion timestep t, apply forward noise to produce a noisy scene, label as `valid=1`. For each, also generate a corrupted version (randomly perturb one object pose by enough to cause interpenetration or unreachability), validate to confirm invalid, apply forward noise at the same t, label `valid=0`. Target 200k labeled noisy scenes. Persist alongside the v1 dataset.

4. **Validity classifier.** Implement `src/guidance/classifier.py`: a smaller transformer (5M to 10M params) operating on the noisy scene at timestep t, predicting valid/invalid. Architecture mirrors the denoiser's tokenization. Train on the data from task 3 for one epoch. Acceptance criterion: held-out AUC above 0.85. Log as `classifier_v1`.

5. **Classifier-guided sampler.** Implement `src/guidance/classifier_guided.py`. During reverse diffusion, at each step compute classifier gradient with respect to the noisy scene, mix into the denoising direction with a scale parameter. Reference: Dhariwal and Nichol classifier guidance. Implement gradient clipping to prevent runaway updates. Two modes: continuous-only gradient (perturb pose and scale, leave type and presence alone) and full gradient with straight-through estimator for discrete components.

6. **Guidance scale sweep.** Run the classifier-guided sampler across guidance scales `[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]`. For each, sample 1000 scenes across 200 held-out prompts. Measure: raw validity rate, mean pairwise diversity (hand-crafted encoding from Week 2), VLM-judged prompt-following.

7. **Pareto plot.** Generate the headline figure: x-axis diversity, y-axis validity rate, one point per sampler config (unconditional, conditional, rejection at three budgets, classifier-guided at the seven scales, continuous and full-gradient variants separately). Save as `artifacts/pareto_v1.png` and `artifacts/pareto_v1.pdf`. The plot must include error bars (3 seeds per config) and annotate the Pareto frontier.

8. **Honesty check.** If classifier guidance does not Pareto-dominate rejection sampling, do not hide it. Write up the result in `artifacts/week4_results.md` with the actual numbers and propose either a different guidance method for Week 5 (energy-based, differentiable Drake surrogate, or distilled validity rejector) or a different framing of the project's contribution.

9. **Verification pass.** Ruff, pyright, pytest. All sweeps logged to W&B grouped under `guidance_sweep_v1`.

### Ground Rules
- Plan first, wait for approval.
- The classifier must be trained on noised scenes at varying t, not clean scenes. A classifier trained only on clean data does not give useful gradients in the middle of the reverse diffusion process. If you find yourself writing the clean-only version, stop and re-read this paragraph.
- Use 3 seeds per sampler configuration. One-seed numbers are not a result.
- Render at least 30 generated scenes from the best classifier-guided configuration for the gallery.
- No em dashes or double hyphens.

### Definition of Done
- `pareto_v1.png` exists and is publication-quality.
- `week4_results.md` reports the actual numbers honestly, with a clear statement of whether guidance Pareto-dominates rejection sampling and by how much.
- All classifier checkpoints, sampler configs, and sweep runs are tagged in W&B.
- A `week5_preview.md` Artifact identifies which sampler configurations should advance to the downstream evaluation, plus any open questions about why certain guidance scales underperformed.

Begin with the Plan Artifact. Do not write code yet.
