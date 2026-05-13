"""Diffusion noise schedule and DDIM sampler for the Demiurge model.

Implements the cosine beta schedule from Nichol and Dhariwal (2021),
"Improved Denoising Diffusion Probabilistic Models", and a DDIM sampler
(Song et al. 2021) with a configurable number of inference steps.

Conventions:
  - Timestep t is a 0-indexed integer in [0, T-1].
  - alpha_bar_t (cumulative product of (1 - beta)) goes from near 1.0 at
    t=0 to near 0.0 at t=T-1.
  - x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps,
    where eps ~ N(0, I) is the added noise.
  - The model is trained to predict eps (noise prediction parameterisation).
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Cosine schedule
# ---------------------------------------------------------------------------


class CosineSchedule:
    """Cosine noise schedule (Nichol and Dhariwal 2021).

    Computes cumulative noise fractions (alpha_bar) and derived quantities
    needed for the forward diffusion process and DDIM sampling.

    Args:
        T: Total number of diffusion training steps. Default 1000.
        s: Offset preventing alpha_bar from being too large near t=0.
            Default 0.008 (from the paper).
        clip_beta_max: Maximum value for any individual beta_t to avoid
            numerical instability at the end of the schedule. Default 0.999.
    """

    def __init__(
        self,
        T: int = 1000,
        s: float = 0.008,
        clip_beta_max: float = 0.999,
    ) -> None:
        self.T = T
        self.s = s

        # Build alpha_bar for all t in [0, T] (T+1 values; index 0 is
        # the "clean" state before any noise is added).
        t_all = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((t_all / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bar_raw = f / f[0]

        # Derive betas and re-clip, then recompute alpha_bar from clipped betas
        # to keep everything consistent.
        betas = 1.0 - alpha_bar_raw[1:] / alpha_bar_raw[:-1]
        betas = betas.clamp(0.0, clip_beta_max)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)  # shape: (T,)

        # Prepend 1.0 so that alpha_bar_prev[0] = 1.0 (no noise at t=-1).
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])

        # Store as float32 buffers. Register as plain tensors (no nn.Module).
        self._alpha_bar: Tensor = alpha_bar.float()           # (T,)
        self._alpha_bar_prev: Tensor = alpha_bar_prev.float() # (T,)
        self._betas: Tensor = betas.float()                   # (T,)
        self._sqrt_alpha_bar: Tensor = alpha_bar.sqrt().float()
        self._sqrt_one_minus_alpha_bar: Tensor = (1.0 - alpha_bar).sqrt().float()

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    def alpha_bar(self, t: Tensor) -> Tensor:
        """Cumulative noise fraction at integer timestep t.

        Args:
            t: Long or float tensor of timestep indices in [0, T-1]. Shape: (B,).

        Returns:
            alpha_bar_t: Float tensor of shape (B,).
        """
        return self._alpha_bar[t.long()]

    def alpha_bar_prev(self, t: Tensor) -> Tensor:
        """alpha_bar at timestep t-1 (equals 1.0 when t=0).

        Args:
            t: Long tensor of shape (B,).

        Returns:
            Float tensor of shape (B,).
        """
        return self._alpha_bar_prev[t.long()]

    def sqrt_alpha_bar(self, t: Tensor) -> Tensor:
        """sqrt(alpha_bar_t). Shape: (B,)."""
        return self._sqrt_alpha_bar[t.long()]

    def sqrt_one_minus_alpha_bar(self, t: Tensor) -> Tensor:
        """sqrt(1 - alpha_bar_t). Shape: (B,)."""
        return self._sqrt_one_minus_alpha_bar[t.long()]

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def add_noise(self, x0: Tensor, t: Tensor, eps: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Sample x_t from the forward diffusion process q(x_t | x_0).

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x0: Clean sample. Shape: (B, ...).
            t: Integer timestep indices in [0, T-1]. Shape: (B,).
            eps: Optional noise tensor of same shape as x0. If None,
                 noise is sampled from N(0, I).

        Returns:
            (x_t, eps): Noisy sample and the noise added. Both shape (B, ...).
        """
        if eps is None:
            eps = torch.randn_like(x0)

        # Broadcast schedule values over all feature dimensions.
        # x0 may be (B, N, D); t is (B,); we need (B, 1, 1, ...) for broadcast.
        extra_dims = x0.dim() - 1
        view_shape = (-1,) + (1,) * extra_dims

        sqrt_ab = self.sqrt_alpha_bar(t).view(view_shape).to(x0.device)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar(t).view(view_shape).to(x0.device)

        x_t = sqrt_ab * x0 + sqrt_1mab * eps
        return x_t, eps

    def predict_x0(self, x_t: Tensor, t: Tensor, eps_pred: Tensor) -> Tensor:
        """Estimate x_0 from x_t and predicted noise eps_pred.

        x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)

        Args:
            x_t: Noisy sample at timestep t. Shape: (B, ...).
            t: Integer timestep indices. Shape: (B,).
            eps_pred: Predicted noise from denoiser. Shape: (B, ...).

        Returns:
            Estimated x_0. Shape: (B, ...).
        """
        extra_dims = x_t.dim() - 1
        view_shape = (-1,) + (1,) * extra_dims

        sqrt_ab = self.sqrt_alpha_bar(t).view(view_shape).to(x_t.device)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar(t).view(view_shape).to(x_t.device)

        return (x_t - sqrt_1mab * eps_pred) / sqrt_ab.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# DDIM Sampler
# ---------------------------------------------------------------------------


class DDIMSampler:
    """Deterministic DDIM reverse sampler (Song et al. 2021).

    Uses eta=0 (fully deterministic) by default. Supports a configurable
    number of inference steps (``n_steps``) that is much smaller than the
    training T, giving fast sampling at inference time.

    Args:
        schedule: A :class:`CosineSchedule` instance defining the noise
            levels. Must have the same T as was used during training.
        n_steps: Number of denoising steps at inference. Default 50.
        eta: Stochasticity parameter (0 = fully deterministic DDIM,
             1 = DDPM). Default 0.
    """

    def __init__(
        self,
        schedule: CosineSchedule,
        n_steps: int = 50,
        eta: float = 0.0,
    ) -> None:
        self.schedule = schedule
        self.n_steps = n_steps
        self.eta = eta

        T = schedule.T
        # Select n_steps evenly-spaced timesteps from T-1 down to 0.
        # torch.linspace gives floats; round to integers.
        step_indices = torch.linspace(T - 1, 0, n_steps).round().long()
        # Remove duplicates that may arise from rounding, preserve order.
        seen: set[int] = set()
        unique_steps: list[int] = []
        for idx in step_indices.tolist():
            i = int(idx)
            if i not in seen:
                seen.add(i)
                unique_steps.append(i)
        self._timesteps: list[int] = unique_steps  # descending order

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable[[Tensor, Tensor, Tensor | None], Tensor],
        shape: tuple[int, ...],
        text_emb: Tensor | None = None,
        seed: int | None = None,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Generate a batch of samples via DDIM reverse diffusion.

        Args:
            model_fn: Callable ``(x_t, t, text_emb) -> eps_pred``.
                Must accept a float tensor x_t of ``shape``, a long tensor
                t of shape ``(B,)``, and an optional text embedding tensor.
                Returns predicted noise of the same shape as x_t.
            shape: Output shape ``(B, ...)``.
            text_emb: Optional text conditioning tensor. Shape: ``(B, D_text)``.
                Passed directly to ``model_fn``; may be None for unconditional.
            seed: Optional integer seed for the initial Gaussian noise.
            device: Device on which to run sampling.

        Returns:
            Denoised sample x_0. Shape: ``shape``.
        """
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        # Start from pure Gaussian noise at t = T-1.
        x_t = torch.randn(shape, device=device, generator=generator)
        B = shape[0]
        sched = self.schedule

        for i, t_val in enumerate(self._timesteps):
            t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)

            # Predict noise.
            eps_pred = model_fn(x_t, t_tensor, text_emb)

            # DDIM update step.
            ab_t = sched.alpha_bar(t_tensor)        # (B,)
            if i + 1 < len(self._timesteps):
                t_prev_val = self._timesteps[i + 1]
            else:
                t_prev_val = 0
            t_prev = torch.full((B,), t_prev_val, dtype=torch.long, device=device)
            ab_prev = sched.alpha_bar(t_prev)        # (B,)

            extra_dims = x_t.dim() - 1
            view = (-1,) + (1,) * extra_dims

            ab_t_v = ab_t.view(view).to(device)
            ab_prev_v = ab_prev.view(view).to(device)

            # Predicted x_0.
            x0_pred = (x_t - (1.0 - ab_t_v).sqrt() * eps_pred) / ab_t_v.sqrt().clamp(min=1e-8)

            # Direction pointing to x_t (eta=0 term is zero; kept for clarity).
            sigma = (
                self.eta
                * ((1.0 - ab_prev_v) / (1.0 - ab_t_v)).sqrt()
                * (1.0 - ab_t_v / ab_prev_v).sqrt()
            )
            dir_xt = (1.0 - ab_prev_v - sigma ** 2).clamp(min=0.0).sqrt() * eps_pred

            noise = sigma * torch.randn_like(x_t)
            x_t = ab_prev_v.sqrt() * x0_pred + dir_xt + noise

        return x_t
