"""Evaluation metrics for the Demiurge diffusion model.

Four functions, one per metric. All accept lists of SceneTensor and return
scalar floats (or DownstreamResult for downstream_success). They are
intentionally stateless and composable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from scene.schema import SceneTensor

if TYPE_CHECKING:
    from eval.downstream import DownstreamResult
    from eval.judge import SceneJudge
    from validator.core import SceneValidator


# ---------------------------------------------------------------------------
# Raw validity rate
# ---------------------------------------------------------------------------


def raw_validity_rate(
    scenes: list[SceneTensor],
    validator: "SceneValidator",
) -> float:
    """Fraction of scenes that pass all four Drake validity checks.

    Args:
        scenes: Physical (denormalized) SceneTensors to evaluate.
        validator: A SceneValidator instance. Called once per scene.

    Returns:
        Float in [0, 1]. Higher is better.
    """
    if not scenes:
        return 0.0
    accepted = sum(1 for st in scenes if validator.validate(st).accepted)
    return accepted / len(scenes)


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def diversity(scenes: list[SceneTensor]) -> float:
    """Mean pairwise L2 distance between scene position centroids.

    Matches the diversity metric used in run_universal_guidance_sweep.py
    for consistency. Uses normalized-space positions (presence-filtered).

    Args:
        scenes: SceneTensors (normalized or physical; only positions used).

    Returns:
        Float >= 0. Higher = more diverse layout across the batch.
    """
    centroids: list[torch.Tensor] = []
    for st in scenes:
        pres = st.presence.bool()
        if pres.any():
            centroids.append(st.poses[pres, :3].mean(dim=0))
        else:
            centroids.append(torch.zeros(3))

    if len(centroids) < 2:
        return 0.0

    ct = torch.stack(centroids)  # (K, 3)
    K = ct.shape[0]
    total = 0.0
    count = 0
    for i in range(K):
        for j in range(i + 1, K):
            total += (ct[i] - ct[j]).norm().item()
            count += 1
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Prompt following (VLM-as-judge with sidecar cache)
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    scene_hash: str
    prompt_hash: str
    judge_hash: str
    score: float
    reasoning: str


def _scene_hash(scene: SceneTensor) -> str:
    """Stable SHA256 of (presence, types, poses rounded to 4dp, scales)."""
    pres = scene.presence.bool()
    blob = (
        scene.presence.tolist(),
        scene.object_types.tolist(),
        [[round(v, 4) for v in row] for row in scene.poses.tolist()],
        [[round(v, 4) for v in row] for row in scene.scales.tolist()],
    )
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _judge_hash(judge_prompt: str) -> str:
    return hashlib.sha256(judge_prompt.encode()).hexdigest()[:8]


def prompt_following(
    scenes: list[SceneTensor],
    prompts: list[str],
    judge: "SceneJudge",
    cache_path: Path | None = None,
) -> float:
    """Mean VLM prompt-following score across (scene, prompt) pairs.

    Results are cached by (scene_hash, prompt_hash, judge_prompt_hash) to
    ``cache_path`` (a sidecar JSONL file) to avoid re-scoring on reruns.

    Args:
        scenes: SceneTensors (physical space for rendering).
        prompts: One prompt string per scene. Must be same length as scenes.
        judge: A SceneJudge instance from src/eval/judge.py.
        cache_path: Optional path to a .jsonl sidecar cache file.

    Returns:
        Mean score in [1, 5]. Higher = better prompt following.
    """
    if not scenes:
        return 0.0
    if len(scenes) != len(prompts):
        raise ValueError(
            f"scenes ({len(scenes)}) and prompts ({len(prompts)}) must have equal length"
        )

    # Load existing cache
    cache: dict[tuple[str, str, str], float] = {}
    if cache_path is not None and cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                entry = json.loads(line)
                key = (entry["scene_hash"], entry["prompt_hash"], entry["judge_hash"])
                cache[key] = entry["score"]

    j_hash = _judge_hash(getattr(judge, "_judge_prompt", "default"))

    scores: list[float] = []
    uncached_scenes: list[SceneTensor] = []
    uncached_prompts: list[str] = []
    uncached_keys: list[tuple[str, str, str]] = []

    for scene, prompt in zip(scenes, prompts):
        key = (_scene_hash(scene), _prompt_hash(prompt), j_hash)
        if key in cache:
            scores.append(cache[key])
        else:
            uncached_scenes.append(scene)
            uncached_prompts.append(prompt)
            uncached_keys.append(key)

    # Score uncached entries via the judge
    if uncached_scenes:
        new_scores = judge.score_batch(uncached_scenes, uncached_prompts)
        with open(cache_path, "a") if cache_path else _nullcontext() as f:
            for key, scene, prompt, score in zip(
                uncached_keys, uncached_scenes, uncached_prompts, new_scores
            ):
                scores.append(score)
                entry = {
                    "scene_hash": key[0],
                    "prompt_hash": key[1],
                    "judge_hash": key[2],
                    "score": score,
                }
                if f is not None:
                    f.write(json.dumps(entry) + "\n")

    return sum(scores) / max(1, len(scores))


class _nullcontext:
    """Minimal context manager yielding None when no cache path given."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


# ---------------------------------------------------------------------------
# Downstream success
# ---------------------------------------------------------------------------


def downstream_success(
    scenes: list[SceneTensor],
    prompts: list[str],
    validator: "SceneValidator",
) -> "DownstreamResult":
    """RRT planning success rate on (scene, prompt) pairs.

    Delegates to DownstreamEvaluator to avoid circular imports.

    Args:
        scenes: Physical SceneTensors.
        prompts: One prompt string per scene (used for logging only).
        validator: SceneValidator to extract RRT results from.

    Returns:
        DownstreamResult dataclass with success_rate and per-task records.
    """
    from eval.downstream import DownstreamEvaluator

    evaluator = DownstreamEvaluator(validator)
    return evaluator.evaluate(scenes, prompts)
