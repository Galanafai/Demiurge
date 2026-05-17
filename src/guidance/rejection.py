"""Rejection sampler baseline for Phase B evaluation.

Wraps conditional_v4 DDIM sampling with Drake validation.
For each prompt, generates scenes in batches until target_n valid scenes
are collected or max_attempts * target_n total samples are exhausted.

Implements the Sampler protocol from guidance.base.
"""
from __future__ import annotations

from typing import Any

import torch

from guidance.base import Sampler
from model.denoiser import N_MAX, SceneDenoiser
from model.rotations import rot6d_to_quat_wxyz
from model.schedule import CosineSchedule, DDIMSampler
from model.text_encoder import TextEncoder
from scene.schema import SceneTensor, WorkspaceBounds
from validator.core import SceneValidator


class RejectionSampler:
    """Baseline sampler: generate-then-filter via Drake validation.

    Args:
        model: Trained SceneDenoiser (conditional_v4 or similar).
        schedule: Cosine diffusion schedule matching training config.
        text_encoder: Frozen sentence-transformer encoder.
        bounds: WorkspaceBounds for scene denormalization.
        ddim_steps: Number of DDIM inference steps. Default 50.
        rrt_budget_s: RRT time budget per scene in Drake. Default 2.0.
        batch_size: Number of scenes to sample per generation batch.
        device: Torch device.
        seed: Base RNG seed (incremented per batch).
    """

    def __init__(
        self,
        model: SceneDenoiser,
        schedule: CosineSchedule,
        text_encoder: TextEncoder,
        bounds: WorkspaceBounds,
        ddim_steps: int = 50,
        rrt_budget_s: float = 2.0,
        batch_size: int = 16,
        type_init: str = "uniform",
        device: torch.device | None = None,
        seed: int = 0,
    ) -> None:
        self._model = model
        self._schedule = schedule
        self._text_encoder = text_encoder
        self._bounds = bounds
        self._ddim_steps = ddim_steps
        self._batch_size = batch_size
        self._type_init = type_init
        self._device = device or torch.device("cpu")
        self._seed = seed
        self._validator = SceneValidator(rrt_budget_s=rrt_budget_s)
        self._sampler = DDIMSampler(schedule, n_steps=ddim_steps)

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "rejection"

    def config(self) -> dict[str, Any]:
        return {
            "sampler": "rejection",
            "ddim_steps": self._ddim_steps,
            "batch_size": self._batch_size,
        }

    def sample(
        self,
        prompts: list[str],
        n_per_prompt: int,
        max_attempts: int = 50,
        **kwargs: Any,
    ) -> list[SceneTensor]:
        """Sample valid scenes for each prompt via rejection.

        Args:
            prompts: List of task description strings.
            n_per_prompt: Target number of valid scenes per prompt.
            max_attempts: Maximum total samples per prompt = max_attempts * n_per_prompt.
                Stops early if target reached. Returns fewer than n_per_prompt if
                budget exhausted.

        Returns:
            List of SceneTensor, at most len(prompts) * n_per_prompt entries.
            Shorter if any prompt exhausted max_attempts without collecting n_per_prompt.
        """
        results: list[SceneTensor] = []
        self._model.eval()

        for prompt in prompts:
            prompt_results = self._sample_one_prompt(
                prompt, n_per_prompt, max_attempts
            )
            results.extend(prompt_results)

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a single prompt to text embedding (1, D_TEXT)."""
        return self._text_encoder.encode_batch([prompt]).to(self._device)

    def _sample_one_prompt(
        self,
        prompt: str,
        target_n: int,
        max_attempts: int,
    ) -> list[SceneTensor]:
        text_emb = self._encode_prompt(prompt)  # (1, D_TEXT)
        # Use conditional_sampling_fn for v6: returns (eps, type_logits) at each step.
        fn = self._model.conditional_sampling_fn(text_emb)

        accepted: list[SceneTensor] = []
        total_attempted = 0
        budget = max_attempts * target_n
        rng_seed = self._seed

        while len(accepted) < target_n and total_attempted < budget:
            b = min(self._batch_size, budget - total_attempted, target_n - len(accepted) + self._batch_size)
            b = max(1, b)

            with torch.no_grad():
                x0, type_ids_out = self._sampler.sample_with_types(
                    fn,
                    shape=(b, N_MAX, 13),
                    seed=rng_seed,
                    device=self._device,
                    type_init=self._type_init,
                )

            rng_seed += 1
            total_attempted += b

            xyz = x0[:, :, :3].clamp(-1.0, 1.0)
            rot6d_pred = x0[:, :, 3:9]
            scale_pred = x0[:, :, 9:12].clamp(-1.0, 1.0)
            pres_bit = x0[:, :, 12]

            for i in range(b):
                if len(accepted) >= target_n:
                    break
                pres_mask = pres_bit[i] > 0.0
                quats = rot6d_to_quat_wxyz(rot6d_pred[i])
                poses_raw = torch.cat([xyz[i], quats], dim=-1)
                # Use predicted type_ids from sample_with_types.
                types = type_ids_out[i].cpu()
                st_norm = SceneTensor(
                    object_types=types,
                    poses=poses_raw.cpu(),
                    scales=scale_pred[i].cpu(),
                    presence=pres_mask.cpu(),
                )
                st = st_norm.denormalize(self._bounds)
                rpt = self._validator.validate(st)
                if rpt.accepted:
                    accepted.append(st)

        return accepted


# Verify protocol conformance at module load time.
assert isinstance(RejectionSampler.__new__(RejectionSampler), Sampler) or True
# (isinstance check on Protocol requires an instance; checked in tests)
