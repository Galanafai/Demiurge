# Antigravity Master Prompt — Week 5: Full Evaluation Harness

Paste into Agent Manager in Plan Mode after Week 4 ships.

---

## Prompt

You are continuing the project. Week 4 produced the headline Pareto plot for validity rate vs diversity across guidance configurations. This session builds the full evaluation harness, runs the downstream utility study, and produces the numbers that anchor the writeup.

Before doing anything else:

1. Read `AGENTS.md` and `.agents/skills/demiurge/SKILL.md` in full.
2. Read `artifacts/week4_results.md` and `artifacts/week5_preview.md`.
3. State in your Plan Artifact: which sampler configurations are advancing to downstream evaluation (typically: unconditional baseline, conditional baseline, rejection sampler, best classifier-guided config), and the rationale.

### Mission Scope for This Session
1. Lock in the four evaluation metrics from the skill into a single reproducible harness.
2. Build the held-out task suite for downstream utility.
3. Run the downstream RRT success-rate study.
4. Produce final result tables and a generated-scene gallery suitable for the portfolio post.

### Step-by-Step Execution Plan You Must Produce

1. **Held-out task suite.** Implement `scripts/build_eval_suite.py`. Sample 200 task descriptions from a holdout split (set aside but not yet used). For each: generate a reference scene with the procedural sampler to confirm the task is solvable, store the prompt and the reference scene's IK + RRT solution. Persist under `data/eval_suite_v1/`. This becomes the fixed benchmark.

2. **Metrics module.** Implement `src/eval/metrics.py` with one function per metric, each consuming `List[SceneTensor]` plus prompts:
   - `raw_validity_rate(scenes)` using the Drake validator.
   - `diversity(scenes)` mean pairwise distance in two embedding spaces.
   - `prompt_following(scenes, prompts)` via VLM-as-judge.
   - `downstream_success(scenes, prompts)` defined in the next task.

3. **Downstream success metric.** Implement `src/eval/downstream.py`. For each (prompt, generated_scene) pair: attempt to plan a UR5e RRT trajectory from home configuration to a prompt-derived goal pose in the scene. Success means: goal pose is reachable, plan is collision-free, plan length under a budget. Time budget per task: 10 seconds. Log per-task success, mean plan length, mean planning time.

4. **Eval harness.** Implement `scripts/run_eval.py`. Hydra config selects sampler configurations to evaluate. Pipeline: sample 8 scenes per held-out prompt per sampler, run all four metrics, produce a per-sampler results dict. Persist results to `artifacts/eval_v1/<sampler_name>.json`. 3 seeds per sampler.

5. **Run the full evaluation.** Execute the harness across the advancing sampler configurations. Confirm total runtime is bounded; if a single sampler exceeds 4 hours of eval wall-clock, surface in a Plan addendum before proceeding.

6. **Results table.** Generate `artifacts/results_table.md` with one row per sampler, columns for all four metrics (mean ± std across seeds), and a "wall-clock to sample 1000 valid scenes" column. Bold the best in each column.

7. **Ablation studies.** Pick the two most informative ablations from this menu and run them:
   - Classifier size: 5M vs 10M vs 20M params (does scaling the classifier help guidance?).
   - Data scale: train conditional model on 10k, 25k, 50k scenes (where does the data curve plateau?).
   - Task template restriction: train on only `tabletop_reach`, evaluate generalization to `cluttered_pick` and `obstacle_avoidance`.
   - Guidance schedule: constant scale vs cosine-annealed scale vs linear warm-up.
   Justify the picks in the Plan Artifact. Each ablation produces a row addition to the results table and a brief paragraph in `artifacts/ablations.md`.

8. **Final gallery.** Render 100 generated scenes from the best sampler, evenly sampled across the three task templates, in a 10x10 grid PNG. Each tile shows the rendered scene and overlays the prompt. Save as `artifacts/final_gallery.png` plus individual tiles under `artifacts/gallery/`. Also produce a 30-second MP4 of the diffusion denoising process for a single representative prompt: 50 DDIM steps rendered with Meshcat, encoded with ffmpeg.

9. **Verification pass.** Ruff, pyright, pytest. All eval runs logged to W&B under `eval_v1`. Confirm `artifacts/results_table.md` matches the W&B numbers exactly.

### Ground Rules
- Plan first, wait for approval.
- Do not regenerate the held-out task suite after the first creation. The benchmark is fixed for the entire project from this point forward.
- VLM-as-judge calls cost money and add latency. Cache responses by (scene_hash, prompt_hash, judge_prompt_hash). Re-running eval should not re-bill the judge for unchanged inputs.
- If a sampler underperforms an earlier baseline on any metric, report it honestly. Negative results are still results.
- Render quality matters for the gallery. Use Drake's offscreen renderer with reasonable lighting; do not ship grainy 100x100 thumbnails.

### Definition of Done
- `artifacts/results_table.md` is final, with all advancing samplers, both ablations, all four metrics, error bars, and bolded winners.
- `artifacts/final_gallery.png` and the per-tile gallery exist and look good.
- `artifacts/denoising_demo.mp4` exists and plays cleanly.
- All eval-suite samples, classifier checkpoints, and sampler configs reproducible from W&B artifact references.
- A `week6_preview.md` Artifact outlines the writeup structure and the demo's required components.

Begin with the Plan Artifact. Do not write code yet.
