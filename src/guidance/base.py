"""Sampler protocol for the Demiurge guidance layer.

All samplers (rejection, classifier-guided, unconditional) implement this
interface so the evaluation harness can treat them uniformly.

Usage:
    from guidance.base import Sampler

    def evaluate(sampler: Sampler, prompts: list[str]) -> list[SceneTensor]:
        return sampler.sample(prompts, n_per_prompt=5)
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from scene.schema import SceneTensor


@runtime_checkable
class Sampler(Protocol):
    """Protocol for scene samplers.

    All concrete samplers must implement these three methods.
    The ``@runtime_checkable`` decorator enables ``isinstance(obj, Sampler)``
    checks in tests and the sweep harness.
    """

    def sample(
        self,
        prompts: list[str],
        n_per_prompt: int,
        **kwargs: Any,
    ) -> list[SceneTensor]:
        """Generate ``n_per_prompt`` scenes per prompt.

        Args:
            prompts: List of task description strings. Length P.
            n_per_prompt: Number of scenes to generate per prompt.
            **kwargs: Sampler-specific options (e.g. max_attempts for
                rejection sampler, guidance_scale for classifier-guided).

        Returns:
            List of SceneTensor of length P * n_per_prompt.
            Ordering: [prompt_0_scene_0, ..., prompt_0_scene_N,
                       prompt_1_scene_0, ..., prompt_P_scene_N].
        """
        ...

    def name(self) -> str:
        """Human-readable sampler identifier for logging and artifact naming."""
        ...

    def config(self) -> dict[str, Any]:
        """Return a serialisable dict of sampler hyperparameters for W&B logging."""
        ...
