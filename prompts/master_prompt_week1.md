# Antigravity Master Prompt — Demiurge Week 1

Paste this into the Antigravity Agent Manager in **Plan Mode**. The agent will read `AGENTS.md` automatically and pull in the `demiurge` skill on demand.

---

## Prompt

You are the implementation agent for a portfolio-grade research project: a conditional diffusion model that generates physically valid 6-DoF tabletop scenes for a UR5e robot arm, validated through Drake (pydrake).

Before doing anything else:

1. Read `AGENTS.md` at the workspace root in full.
2. Read `.agents/skills/demiurge/SKILL.md` in full.
3. Confirm in your Plan Artifact that you have ingested both files and list the six layers in the architecture along with their `src/` paths.

### Mission Scope for This Session
We are executing **Week 1: Foundations** only. Do not begin Layer 2 implementation work beyond what Week 1 specifies. The deliverables for this session are:

- A clean repository skeleton with `pyproject.toml`, `ruff` config, `pyright` strict config for `src/`, and a `pytest` config.
- Layer 1 (Scene Schema) implemented and unit-tested per the skill.
- Layer 2 (Drake Validator) implemented as a working library with all four checks (no_interpenetration, stable_rest, ik_reachable, rrt_solvable) and unit tests against known-valid and known-invalid fixtures.
- A 500-scene smoke run of a stub procedural sampler that exercises the validator end to end, with a CSV of per-check rejection rates committed under `artifacts/week1_smoke.csv`.

### Step-by-Step Execution Plan You Must Produce

In your Plan Artifact, generate a numbered task list covering the items below. Each task must include: the file paths it will create or modify, the exact unit tests that will accompany it, and the verification command (pytest invocation, ruff command, or terminal probe) you will run before marking the task complete.

1. **Repo scaffold.** Initialize the project: `pyproject.toml` with pinned versions of pydrake, torch, numpy, einops, sentence-transformers, ruff, pyright, pytest, hydra-core. `.gitignore` covering `wandb/`, `data/`, `outputs/`, `.venv/`. Empty `src/scene/`, `src/validator/`, `src/data/`, `src/model/`, `src/guidance/`, `src/eval/` with `__init__.py`. Empty mirrored `tests/` tree. README with one-paragraph project summary and a run-instructions section that will fill in over time. CI workflow in `.github/workflows/ci.yml` running ruff + pyright + pytest on push.

2. **Object vocabulary.** Implement `src/scene/vocab.py` with the 8-entry `OBJECT_VOCAB` from the skill. Source SDF mesh paths from Drake's bundled YCB models for the YCB entries and inline `<box>`, `<sphere>`, `<cylinder>` primitives for the primitives. Include a Drake-loading round-trip test in `tests/scene/test_vocab.py`.

3. **Scene tensor.** Implement `src/scene/schema.py` with the `SceneTensor` dataclass, normalization, denormalization, quaternion projection, and `to_sdf(workspace_bounds)`. Cover the four unit tests called out in the skill.

4. **Validator core.** Implement `src/validator/core.py` with `SceneValidator` and `ValidityReport`. Each of the four checks lives in its own private method. Build fresh plant + scene_graph per scene as required by the skill. Add a `validate_batch(scenes, num_workers)` entry point using `multiprocessing.Pool`.

5. **Validator checks.**
   - `_check_interpenetration` uses `SceneGraph.ComputeSignedDistancePairwiseClosestPoints` at the resting pose.
   - `_check_stable_rest` calls `Simulator.AdvanceTo(1.5)` from the placed configuration and compares pose deltas against a 5 mm / 5 degree threshold per object.
   - `_check_ik_reachable` samples a goal pose above a target object, builds a fresh `InverseKinematics` with collision-free constraint, solves with SNOPT or IPOPT (whichever is available), and returns success plus solution.
   - `_check_rrt_solvable` runs BiRRT between the home configuration and the IK solution, budget 5 seconds, seed logged.

6. **Validator fixtures and tests.** Build `tests/validator/fixtures/` with three known-valid scenes and three known-invalid scenes (one violating each check: a scene with overlapping cubes, a scene with a precariously stacked tall box, a scene with the goal placed inside a wall, a scene with the goal blocked by an unreachable obstacle wall). Tests assert the correct check fires for each invalid case and all checks pass for each valid case.

7. **Stub procedural sampler.** Implement `src/data/sampler_stub.py` with one task template (`tabletop_reach`) producing 500 candidate scenes with crude randomization. Run them through `validate_batch`, write `artifacts/week1_smoke.csv` with columns `scene_id, no_interp, stable, ik, rrt, accepted`. Print rejection-rate summary.

8. **Verification pass.** Run `ruff check`, `pyright`, and `pytest -xvs` and paste the full clean output into a final Artifact. If anything fails, fix and re-run before declaring done.

### Ground Rules

- After producing the Plan Artifact, **stop and wait for explicit approval** before writing any code. I will review the plan and may request changes.
- When you do begin execution, work one numbered task at a time. After each task, produce a brief progress Artifact stating what was done, what tests pass, and what is next.
- If you hit a Drake API question you are not sure about, write a small standalone probe script in the terminal, run it, paste the output into the conversation, and only then proceed with the production code. Do not guess.
- Never modify `AGENTS.md` or the skill file without surfacing the proposed change to me first.
- All prose you write into READMEs, comments, and Artifacts must follow the style rules in `AGENTS.md` (no em dashes, no double hyphens, direct technical voice).
- Use the terminal sandbox. Do not request non-workspace file access.

### Definition of Done for This Session

- All eight numbered tasks above are complete.
- `pytest -xvs` passes with zero failures.
- `ruff check` and `pyright` return clean.
- `artifacts/week1_smoke.csv` exists with 500 rows.
- A final Artifact summarizes the state of the repo and the per-check rejection rates observed in the smoke run, plus three concrete observations about what to tune in Week 2 (data sampler scale-up to 50k).

Begin by reading the two required files and producing the Plan Artifact. Do not write code yet.
