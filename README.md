# Demiurge

A conditional diffusion model that generates physically valid 6-DoF tabletop scenes for a UR5e
robot arm. Scenes are validated through Drake (pydrake) for non-interpenetration, static
stability, inverse kinematics reachability, and collision-free BiRRT planning. The thesis:
classifier-guided diffusion produces scenes with measurably higher validity rates and equal or
better task diversity compared to unconditional generation with rejection sampling.

## Architecture

Six layers, built in strict order:

| # | Name | Path |
|---|------|------|
| 1 | Scene Schema | `src/scene/` |
| 2 | Drake Validator | `src/validator/` |
| 3 | Procedural Sampler | `src/data/` |
| 4 | Diffusion Model | `src/model/` |
| 5 | Guidance | `src/guidance/` |
| 6 | Evaluation | `src/eval/` |

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Drake (`drake==1.52.1`) requires Ubuntu 22.04 or 24.04, macOS 14+, or Windows 10/11.

## Running Tests

```bash
uv run pytest -xvs
```

## Linting and Type Checking

```bash
uv run ruff check src/ tests/
uv run pyright
```

## Smoke Run (Week 1)

Generates 500 candidate scenes through the stub sampler and validator:

```bash
uv run python src/data/sampler_stub.py --config configs/smoke.yaml --seed 42
```

Results are written to `artifacts/week1_smoke.csv`.

## Reproducibility

Every long-running script accepts `--config <path/to/yaml>` and `--seed <int>`.
Training runs log config hash, seed, git SHA, dataset hash, and library versions to
Weights and Biases.
