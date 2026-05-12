# Antigravity Master Prompt — Week 2: Data Scale-Up

Paste into Agent Manager in Plan Mode after Week 1 is shipped and merged.

---

## Prompt

You are continuing the diffusion Drake scene generation project. Week 1 delivered the scene schema, the Drake validator, and a 500-scene smoke run. This session scales the data pipeline to production size and locks in the dataset that will train every subsequent model.

Before doing anything else:

1. Read `AGENTS.md` at the workspace root in full.
2. Read `.agents/skills/demiurge/SKILL.md` in full.
3. Read `artifacts/week1_smoke.csv` and the Week 1 summary Artifact. State in your Plan Artifact the observed rejection rate per validator check, and which check is the current bottleneck.

### Mission Scope for This Session
Build the full procedural sampler covering all three task templates, generate 50k accepted scenes, persist as sharded WebDataset, and produce a dataset card with diversity statistics.

### Step-by-Step Execution Plan You Must Produce

1. **Full procedural sampler.** Implement `src/data/sampler.py` with all three task templates from the skill: `tabletop_reach`, `cluttered_pick`, `obstacle_avoidance`. Each template is a class implementing a common `TaskTemplate` protocol with `sample_scene(rng) -> CandidateScene` and `sample_task_description(scene, rng) -> str`. Tests in `tests/data/test_templates.py` cover one round-trip per template plus a determinism test (same seed yields same scene).

2. **Task description generator.** Implement `src/data/descriptions.py` with templated prompt generation per task family. Each template carries 4 to 8 surface-form variants to prevent the model from latching onto a single phrasing. Include color, count, and spatial qualifiers drawn from scene state. Test: 1000 sampled descriptions from each template have less than 5 percent exact duplicates.

3. **Sharded dataset writer.** Implement `src/data/writer.py` using `webdataset`. Each shard is 500 MB max. Per-example payload: normalized SceneTensor as `.pt`, raw SDF as `.sdf`, task description as `.txt`, validity metrics as `.json`. Implement a corresponding `src/data/reader.py` that streams shards with prefetch and decodes back into `SceneTensor`. Round-trip test: 100 written then read examples match bit-exact on the tensor and string-exact on the description.

4. **Parallel generation harness.** Implement `scripts/generate_dataset.py`. Hydra config under `configs/dataset/v1.yaml` specifies target accepted count (50000), per-template mix (default uniform), worker count, validator timeout, and output path. Workers pull candidates from each template in round-robin, validate, and write accepted scenes to shards. Progress bar shows accepted, rejected per check, and throughput. Resume from last shard on restart.

5. **Run the 50k generation.** Execute the script targeting 50000 accepted scenes. Log total runtime, per-check rejection rates, per-template acceptance rates, and worker utilization to Weights & Biases. If total runtime exceeds 8 hours on the available machine, stop and surface a profiling Artifact before continuing.

6. **Dataset card.** Generate `artifacts/dataset_v1_card.md` containing: total accepted count, per-template count, per-check rejection rate, dataset hash, generation date, git SHA, library versions, and three histograms (object count per scene, goal distance from workspace center, task description length). Use matplotlib for the histograms; save as PNG alongside the card.

7. **Diversity sanity check.** Implement `src/data/diversity.py` computing mean pairwise distance in two embedding spaces: a simple hand-crafted scene encoding (object types one-hot concatenated with positions), and the frozen sentence-transformer embedding of task descriptions. Report both numbers in the dataset card. These become the baselines that learned generation must beat.

8. **Validator profiling pass.** If the Week 1 smoke run flagged a check as the bottleneck, run `scripts/profile_validator.py` (write it) to identify the hot path. Common suspects: SDF re-parsing, IK warm-start absence, RRT seed cost. Apply one targeted optimization, re-measure, document gain in the dataset card. If validator throughput exceeds 20 scenes per second per 8 cores, skip this task and note it in the Plan.

9. **Verification pass.** Run `ruff check`, `pyright`, `pytest -xvs`. Confirm dataset shards are readable end to end via a 1000-example streaming test.

### Ground Rules
- Stop after the Plan Artifact and wait for approval before executing.
- After each numbered task, produce a brief progress Artifact.
- Do not change `AGENTS.md` or `SKILL.md` without surfacing the change first.
- If the generation run reveals that one task template has a sub-20 percent acceptance rate, do not silently widen tolerances. Surface the finding, propose options (tune template, relax check, accept the cost), and wait for direction.
- All shards must be committed to the dataset registry path specified in the Hydra config. Do not commit the raw shards to git; use git-lfs or a `dvc` pointer if configured, otherwise document the storage location in the dataset card.
- No em dashes or double hyphens in prose.

### Definition of Done
- `data/v1/` contains the 50k-scene sharded dataset.
- `artifacts/dataset_v1_card.md` exists with all required fields, hashes, and histograms.
- All tests pass, ruff and pyright clean.
- Weights & Biases run logged under experiment `dataset_v1`.
- Plan Artifact for Week 3 sketched (model architecture sizing decision based on dataset characteristics) in a follow-up Artifact titled `week3_preview.md`.

Begin by reading the required files and producing the Plan Artifact. Do not write code yet.
