---
name: demiurge
description: Builds and maintains the Demiurge pipeline, a diffusion-based Drake scene generation system. Use when implementing the scene schema, DDPM denoiser, Drake validator, procedural data sampler, classifier guidance, or evaluation harness for the Demiurge project. Covers six-layer architecture, training discipline, and reproducibility requirements.
---

# Demiurge Pipeline Skill

This skill defines how to build, extend, and validate the six-layer Demiurge stack: a diffusion model that generates physically valid robot training scenes under Drake validation. Follow it whenever the active task touches scene representation, the DDPM model, the Drake validator, training data, guidance algorithms, or evaluation.

## When to use this skill
- Implementing or modifying any module under `src/scene/`, `src/model/`, `src/validator/`, `src/data/`, `src/guidance/`, or `src/eval/`.
- Writing or updating training scripts, eval scripts, or the demo backend.
- Debugging validity rate regressions, training instability, or Drake API surprises.
- Onboarding into the codebase from scratch.

## Six-Layer Build Order (Strict)
Build the layers in this order. Do not start a later layer before the prior one has unit tests passing.

### Layer 1: Scene Schema (`src/scene/`)
1. Define `SceneTensor` dataclass: `object_types: LongTensor[N]`, `poses: FloatTensor[N, 7]` as xyz plus quaternion wxyz, `scales: FloatTensor[N, 3]`, `presence: BoolTensor[N]`. N is fixed at 8.
2. Define `OBJECT_VOCAB`: 8 entries. Start with 5 primitives (cube, sphere, cylinder, box_tall, box_flat) and 3 YCB objects (mustard_bottle, sugar_box, tomato_soup_can). Each entry stores SDF mesh path and bounding radius.
3. Implement `SceneTensor.to_sdf(workspace_bounds)` returning an SDF string suitable for `Parser.AddModelsFromString` in Drake.
4. Implement `SceneTensor.normalize()` and `SceneTensor.denormalize()` for the diffusion training space. Quaternions stay on the 4-sphere via projection after each denoising step.
5. Unit tests: round-trip normalize/denormalize, SDF parses cleanly in Drake, quaternion projection preserves rotation.

### Layer 2: Drake Validator (`src/validator/`)
1. Build `SceneValidator` class wrapping a fresh `MultibodyPlant` plus `SceneGraph` per scene to avoid context staleness.
2. Validity checks return a `ValidityReport` with named booleans and continuous metrics:
   - `no_interpenetration`: minimum signed distance across all object pairs greater than zero. Use `SceneGraph` collision queries.
   - `stable_rest`: forward-simulate 1.5 seconds, check max pose drift below threshold per object.
   - `ik_reachable`: goal pose (sampled from scene) solves with `InverseKinematics` plus collision-free constraint.
   - `rrt_solvable`: BiRRT plan exists between a home configuration and the IK solution within budget.
3. Parallelize across CPU cores with `multiprocessing.Pool`. Each worker holds its own plant. Never share a plant across processes.
4. Cache results by scene hash to avoid re-validating during training.
5. Reuse the IK and RRT modules from the existing drake-robotics skill where signatures match; do not duplicate.
6. Unit tests: known-valid scenes pass all four checks, known-invalid scenes fail the expected check and no others.

### Layer 3: Procedural Data Sampler (`src/data/`)
1. Implement `ProceduralSampler` with task templates: `tabletop_reach`, `cluttered_pick`, `obstacle_avoidance`. Each template parameterizes object count, spatial distribution, and goal selection.
2. Pipeline: sample candidate scene, run validator, accept if all checks pass, write to dataset. Target 50k accepted scenes. Log rejection rate per check.
3. Persist as sharded WebDataset tar files. Each example stores normalized SceneTensor, raw SDF, task description string, validity metrics.
4. Generate task descriptions templated per scene: e.g., "Reach the {color} {object_type} avoiding the {n_obstacles} obstacles." Variation matters more than length.
5. Aggressive randomization on object scale, pose, and template parameters to prevent the model from memorizing the sampler.
6. Unit tests: 100-scene smoke run finishes under 90 seconds on a workstation, all output scenes re-validate.

### Layer 4: Diffusion Model (`src/model/`)
1. Implement `SceneDenoiser`: transformer with object slots as tokens, AdaLN conditioning on diffusion timestep, cross-attention to text embedding.
2. Use a frozen `sentence-transformers/all-MiniLM-L6-v2` encoder. Cache embeddings per training run.
3. DDPM with cosine beta schedule, 1000 training steps, DDIM sampling at inference with 50 steps.
4. Loss: simple MSE on noise prediction, with separate loss weights for type logits (cross-entropy), pose continuous components, scale, and presence.
5. Quaternion handling: predict in 6D rotation representation (Zhou et al.) and project to quaternion on output. Avoids antipodal ambiguity.
6. Unit tests: tiny model (3 objects, no rotation) on a toy distribution converges to validity rate above 0.9 within 5 minutes on CPU.

### Layer 5: Guidance (`src/guidance/`)
1. Baseline `RejectionSampler`: sample N scenes, run validator, keep valid. Measure throughput.
2. `ClassifierGuidedSampler`: train a lightweight validity classifier on (noisy_scene, t, valid) triples sampled from the diffusion forward process applied to procedural data. Use classifier gradient during reverse sampling with a tunable guidance scale.
3. Both samplers share an interface so eval treats them uniformly.
4. Unit tests: classifier reaches above 0.85 AUC on held-out scenes within 30 minutes of training.

### Layer 6: Evaluation (`src/eval/`)
1. Metrics module computes per-sampler:
   - Raw validity rate
   - Diversity: mean pairwise distance in scene-embedding space
   - Task relevance: VLM-as-judge using a fixed Gemini or Claude prompt scoring prompt-scene alignment 0 to 5
   - Downstream utility: RRT planning success rate on a held-out task suite of 200 tasks
2. Pareto plotter: validity rate vs diversity, one point per sampler configuration. This is the headline figure for the writeup.
3. Generate a gallery: 50 generated scenes per sampler, rendered via Drake's offscreen renderer, with task description and validity badge.

## Reproducibility Discipline
- Every script accepts `--config path/to/yaml --seed <int>`.
- Configs live under `configs/` and are checked in.
- Every run writes `run_metadata.json`: config hash, seed, git SHA, dataset hash, library versions.
- Weights & Biases logs all runs. Group by experiment name. Tag with `baseline`, `guided`, `ablation`, etc.

## Drake Pitfalls to Avoid
- Do not reuse a `Context` across collision queries with mutated state. Get a fresh `Context` per check or call `SetPositions` then re-evaluate.
- `InverseKinematics` constraints attach to the plant, not the context. Build a fresh `InverseKinematics` per goal.
- Mesh loading from SDF is slow. Pre-cache parsed `ModelInstanceIndex` lookups when validating a batch from the same scene family.
- BiRRT is non-deterministic. Pass and log the RNG seed.

## Performance Targets
- Validator throughput: greater than 20 scenes per second across 8 CPU cores. If below, profile collision queries first.
- Training: one full epoch on 50k scenes in under 30 minutes on a single A100 or 4090.
- Inference: 50-step DDIM sample under 2 seconds for a batch of 16 on the same GPU.

## When in Doubt
- Verify Drake behavior in a 10-line script in the terminal before writing the production version.
- Re-read this skill before starting a new layer.
- If a proposed change might violate an invariant in `AGENTS.md`, raise it in the Plan Artifact before executing.