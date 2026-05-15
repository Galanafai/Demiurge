"""Classifier-guided sampler for Phase B.

Implements classifier guidance (Dhariwal and Nichol 2021) applied to the
Demiurge scene diffusion model. During DDIM reverse diffusion, at each
timestep the classifier gradient is used to steer the denoising trajectory
toward valid (Drake-accepted) scenes.

Guidance equation (ref: Dhariwal & Nichol 2021, eq 12):
    eps_guided = eps_pred - w * sqrt(1 - alpha_bar_t) * grad_x log p(valid | x_t)

where w is the guidance scale and grad_x is the gradient of the classifier
log-probability with respect to the noisy input x_t.

Two guidance modes:
    continuous: gradient applied only to (xyz, rot6d, scale) dims [0:12].
                presence and type_logits are left unmodified. Avoids
                straight-through issues with discrete components.
    full:       gradient applied to all 13 dims including presence bit.
                Uses the raw continuous representation; discrete decoding
                is still applied post-sampling.

Implements the Sampler protocol from guidance.base.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from guidance.classifier import ValidityClassifier
from model.denoiser import N_MAX, SceneDenoiser
from model.rotations import rot6d_to_quat_wxyz
from model.schedule import CosineSchedule, DDIMSampler
from model.text_encoder import TextEncoder
from scene.schema import SceneTensor, WorkspaceBounds


class ClassifierGuidedSampler:
    """DDIM sampler with classifier gradient guidance.

    Args:
        model: Trained SceneDenoiser (conditional_v4).
        classifier: Trained ValidityClassifier on noisy scenes.
        schedule: Cosine diffusion schedule.
        text_encoder: Frozen sentence-transformer.
        bounds: WorkspaceBounds for denormalization.
        guidance_scale: Guidance strength w. 0.0 = no guidance (matches DDIM).
        mode: 'continuous' or 'full' -- which dims receive gradient.
        grad_clip_norm: Per-step gradient norm clip value. Default 1.0.
        ddim_steps: DDIM inference steps. Default 50.
        batch_size: Scenes per generation batch.
        device: Torch device.
        seed: Base RNG seed.
    """

    def __init__(
        self,
        model: SceneDenoiser,
        classifier: ValidityClassifier,
        schedule: CosineSchedule,
        text_encoder: TextEncoder,
        bounds: WorkspaceBounds,
        guidance_scale: float = 1.0,
        mode: str = "continuous",
        grad_clip_norm: float = 1.0,
        ddim_steps: int = 50,
        batch_size: int = 16,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> None:
        if mode not in ("continuous", "full"):
            raise ValueError(f"mode must be 'continuous' or 'full', got {mode!r}")
        self._model = model
        self._classifier = classifier
        self._schedule = schedule
        self._text_encoder = text_encoder
        self._bounds = bounds
        self._guidance_scale = guidance_scale
        self._mode = mode
        self._grad_clip_norm = grad_clip_norm
        self._ddim_steps = ddim_steps
        self._batch_size = batch_size
        self._device = device or torch.device("cpu")
        self._seed = seed

        self._base_sampler = DDIMSampler(schedule, n_steps=ddim_steps)

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------

    def name(self) -> str:
        return f"classifier_guided_w{self._guidance_scale}_{self._mode}"

    def config(self) -> dict[str, Any]:
        return {
            "sampler": "classifier_guided",
            "guidance_scale": self._guidance_scale,
            "mode": self._mode,
            "grad_clip_norm": self._grad_clip_norm,
            "ddim_steps": self._ddim_steps,
        }

    def sample(
        self,
        prompts: list[str],
        n_per_prompt: int,
        **kwargs: Any,
    ) -> list[SceneTensor]:
        """Sample scenes with classifier guidance.

        Args:
            prompts: Task description strings.
            n_per_prompt: Scenes to generate per prompt (no rejection filtering).

        Returns:
            List of SceneTensor of length len(prompts) * n_per_prompt.
            Scenes are NOT filtered by Drake; caller runs validation.
        """
        results: list[SceneTensor] = []
        self._model.eval()
        self._classifier.eval()

        for prompt in prompts:
            for _ in range(n_per_prompt):
                scene = self._sample_one(prompt)
                results.append(scene)

        return results

    # ------------------------------------------------------------------
    # Guided DDIM implementation
    # ------------------------------------------------------------------

    def _guided_noise_fn(
        self,
        text_emb: Tensor,
    ):
        """Return a guided noise prediction function for use with DDIMSampler.

        The returned function wraps the denoiser's noise prediction with
        classifier gradient injection at each step.

        Guidance equation (Dhariwal & Nichol 2021, eq 12):
            eps_guided = eps_pred - w * sqrt(1 - alpha_bar_t) * grad_x log p(valid | x_t)

        Args:
            text_emb: Text embedding (1, D_TEXT).

        Returns:
            Callable (x_t, t_idx, _) -> eps_guided, shape (B, N_MAX, 13).
        """
        model = self._model
        classifier = self._classifier
        schedule = self._schedule
        w = self._guidance_scale
        clip_norm = self._grad_clip_norm
        mode = self._mode

        def _fn(x_t: Tensor, t_idx: Tensor, _: Any) -> Tensor:
            B = x_t.shape[0]
            expanded_text = text_emb.expand(B, -1)

            # --- Denoiser noise prediction (no grad needed here) ---
            with torch.no_grad():
                type_ids = torch.zeros(B, N_MAX, dtype=torch.long, device=x_t.device)
                denoiser_out = model(x_t, type_ids, t_idx, expanded_text)
                eps_pred = torch.cat(
                    [denoiser_out.xyz, denoiser_out.rot6d, denoiser_out.scale,
                     denoiser_out.presence_logit], dim=-1
                )  # (B, N_MAX, 13)

            if w == 0.0:
                return eps_pred

            # --- Classifier gradient ---
            # Use enable_grad explicitly: this fn may be called inside a no_grad block
            # during DDIM sampling. The classifier gradient requires autograd.
            with torch.enable_grad():
                x_t_grad = x_t.detach().requires_grad_(True)
                type_ids_g = torch.zeros(B, N_MAX, dtype=torch.long, device=x_t.device)

                log_p = classifier.log_prob_valid(x_t_grad, type_ids_g, t_idx)  # (B,)
                log_p_sum = log_p.sum()
                log_p_sum.backward()

            # grad has shape (B, N_MAX, 13).
            grad = x_t_grad.grad.detach()  # type: ignore[union-attr]

            # Clip per-step gradient norm (across all dims per sample).
            grad_norm = grad.norm(dim=(-2, -1), keepdim=True)  # (B, 1, 1)
            scale_factor = (clip_norm / (grad_norm + 1e-8)).clamp(max=1.0)
            grad = grad * scale_factor

            # Apply guidance only to continuous dims if mode='continuous'.
            if mode == "continuous":
                # Dims 0:12 = xyz(3) + rot6d(6) + scale(3). Dim 12 = presence.
                grad_mask = torch.zeros_like(grad)
                grad_mask[:, :, :12] = grad[:, :, :12]
                grad = grad_mask

            # sqrt(1 - alpha_bar_t): scalar per batch element.
            alpha_bar = schedule.alpha_bar(t_idx).to(x_t.device)      # (B,)
            sqrt_one_minus = (1.0 - alpha_bar).sqrt().view(B, 1, 1)   # (B, 1, 1)

            eps_guided = eps_pred - w * sqrt_one_minus * grad
            return eps_guided

        return _fn

    def _sample_one(self, prompt: str) -> SceneTensor:
        """Sample a single scene with guidance."""
        with torch.no_grad():
            text_emb = self._text_encoder.encode([prompt]).to(self._device)  # (1, D_TEXT)

        guided_fn = self._guided_noise_fn(text_emb)

        x0 = self._base_sampler.sample(
            guided_fn, (1, N_MAX, 13), seed=self._seed, device=self._device
        )
        self._seed += 1

        xyz = x0[0, :, :3].clamp(-1.0, 1.0)
        rot6d_pred = x0[0, :, 3:9]
        scale_pred = x0[0, :, 9:12].clamp(-1.0, 1.0)
        pres_bit = x0[0, :, 12]

        pres_mask = pres_bit > 0.0
        quats = rot6d_to_quat_wxyz(rot6d_pred)
        poses_raw = torch.cat([xyz, quats], dim=-1)
        types = torch.zeros(N_MAX, dtype=torch.long)

        st_norm = SceneTensor(
            object_types=types,
            poses=poses_raw.cpu(),
            scales=scale_pred.cpu(),
            presence=pres_mask.cpu(),
        )
        bounds = self._bounds
        return st_norm.denormalize(bounds)
