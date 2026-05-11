# AGENTS.md — Diffusion-Based Drake Scene Generation

## Project Identity
This is a portfolio-grade research project: a conditional diffusion model that generates physically valid 6-DoF tabletop scenes for a UR5e robot arm, validated through Drake (pydrake) for non-interpenetration, static stability, IK reachability, and collision-free RRT planning. The thesis: classifier-guided diffusion produces scenes with measurably higher validity rates and equal or better task diversity compared to unconditional generation plus rejection sampling.

## Owner Context
- Lead engineer: Galanafai. Robotics and distributed systems background. Prior Drake/pydrake experience: full motion-planning take-home for Pivot Robotics with a 42-scenario Deterministic Simulation Testing (DST) harness at 42/42 passing.
- The repo should read like production research code, not a notebook dump. Aerospace-grade standards at speed.

## Non-Negotiable Engineering Standards
- Python 3.11. PyTorch 2.x. pydrake latest stable.
- Lint with ruff. Type-check with pyright in strict mode for the `src/` tree.
- Every module under `src/` must have a corresponding unit test under `tests/`.
- No silent fallbacks. Failures raise typed exceptions with context.
- Drake validator is the single source of truth for scene validity. Never approximate it elsewhere without an explicit `# APPROX:` comment that names the invariant being relaxed.
- All experiments must be reproducible: every training run logs config, seed, git SHA, and dataset hash to Weights & Biases.

## Architecture (Authoritative)
The system has six layers. Do not collapse them.
1. Scene representation: fixed-length tensor, max 8 objects, padded with presence mask. Object schema: type id, 6-DoF pose, scale. Defined once in `src/scene/schema.py`.
2. Diffusion model: custom DDPM in PyTorch, transformer denoiser, ~10-50M params, text-conditioned via cross-attention on a frozen sentence-transformer encoder. Lives in `src/model/`.
3. Drake validator: pure pydrake. Collision queries, forward sim for stability, IK, RRT. Lives in `src/validator/`. Reusable as a standalone library.
4. Data pipeline: procedural sampler generates 50k validated scenes for training. Lives in `src/data/`.
5. Guidance: rejection sampling baseline plus classifier-guided sampling. Lives in `src/guidance/`.
6. Evaluation: validity rate, diversity, task relevance, downstream RRT success rate. Lives in `src/eval/`.

## Forbidden Moves
- Do not use Isaac Sim, MuJoCo, or PyBullet. Drake only.
- Do not introduce a bimanual setup. Single UR5e.
- Do not expand the object vocabulary beyond the fixed set without explicit approval.
- Do not skip the validator to "speed up" training data generation. Invalid training data poisons the model.
- Do not use em dashes or double hyphens in any prose written into READMEs, comments, or commit messages.

## Communication Style for Plans and Artifacts
- Write plans and Artifact prose in a direct, technical voice. No marketing language.
- Surface invariant violations explicitly. If a planning step might break an invariant defined above, flag it in the Plan Artifact before executing.
- When uncertain about a Drake API, run a small script in the terminal to verify behavior before writing the production version.

## Git Conventions
- Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- One logical change per commit.
- Every PR-equivalent push must pass ruff, pyright, and pytest locally.

## Tooling Discipline
- Use `uv` for dependency management. No raw pip into the global env.
- Pin Drake and PyTorch exact versions in `pyproject.toml`.
- All long-running scripts must accept `--config` pointing at a YAML and `--seed` for determinism.
