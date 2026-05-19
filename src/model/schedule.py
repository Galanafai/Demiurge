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
        zero_terminal_snr: bool = False,
    ) -> None:
        self.T = T
        self.s = s
        self.zero_terminal_snr = zero_terminal_snr

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

        if zero_terminal_snr:
            # Lin et al. 2024: rescale sqrt(alpha_bar) so terminal = 0 exactly.
            sqrt_ab = alpha_bar.sqrt()
            sqrt_ab_0 = sqrt_ab[0].clone()
            sqrt_ab_T = sqrt_ab[-1].clone()
            sqrt_ab = sqrt_ab - sqrt_ab_T
            sqrt_ab = sqrt_ab * sqrt_ab_0 / (sqrt_ab_0 - sqrt_ab_T)
            alpha_bar = sqrt_ab ** 2
            alpha_bar_full = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar])
            betas = 1.0 - alpha_bar_full[1:] / alpha_bar_full[:-1]
            betas = betas.clamp(0.0, clip_beta_max)

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
        return self._alpha_bar[t.long().cpu()]

    def alpha_bar_prev(self, t: Tensor) -> Tensor:
        """alpha_bar at timestep t-1 (equals 1.0 when t=0).

        Args:
            t: Long tensor of shape (B,).

        Returns:
            Float tensor of shape (B,).
        """
        return self._alpha_bar_prev[t.long().cpu()]

    def sqrt_alpha_bar(self, t: Tensor) -> Tensor:
        """sqrt(alpha_bar_t). Shape: (B,)."""
        return self._sqrt_alpha_bar[t.long().cpu()]

    def sqrt_one_minus_alpha_bar(self, t: Tensor) -> Tensor:
        """sqrt(1 - alpha_bar_t). Shape: (B,)."""
        return self._sqrt_one_minus_alpha_bar[t.long().cpu()]

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

    def predict_x0_from_v(self, x_t, t, v_pred):
        """x0 = sqrt(ab)*x_t - sqrt(1-ab)*v"""
        ed = x_t.dim() - 1
        vs = (-1,) + (1,) * ed
        sa = self.sqrt_alpha_bar(t).view(vs).to(x_t.device)
        s1 = self.sqrt_one_minus_alpha_bar(t).view(vs).to(x_t.device)
        return sa * x_t - s1 * v_pred

    def predict_eps_from_v(self, x_t, t, v_pred):
        """eps = sqrt(1-ab)*x_t + sqrt(ab)*v"""
        ed = x_t.dim() - 1
        vs = (-1,) + (1,) * ed
        sa = self.sqrt_alpha_bar(t).view(vs).to(x_t.device)
        s1 = self.sqrt_one_minus_alpha_bar(t).view(vs).to(x_t.device)
        return s1 * x_t + sa * v_pred

    def compute_v_target(self, x0, eps, t):
        """v = sqrt(ab)*eps - sqrt(1-ab)*x0"""
        ed = x0.dim() - 1
        vs = (-1,) + (1,) * ed
        sa = self.sqrt_alpha_bar(t).view(vs).to(x0.device)
        s1 = self.sqrt_one_minus_alpha_bar(t).view(vs).to(x0.device)
        return sa * eps - s1 * x0

    def type_corruption_prob(self, t_idx: Tensor) -> Tensor:
        """Corruption probability for discrete type diffusion.

        gamma_t = 1 - sqrt(alpha_bar_t).

        At t=0, gamma=0 (no corruption -- clean type_ids preserved).
        At t=T-1, gamma approx 1 (almost fully random types).
        This mirrors the signal-to-noise schedule of the continuous process,
        ensuring the denoiser sees consistent noise levels across modalities.

        Args:
            t_idx: Integer timestep indices in [0, T-1]. Shape: (B,).

        Returns:
            gamma: Float tensor of corruption probabilities. Shape: (B,).
        """
        ab = self.alpha_bar(t_idx)
        return 1.0 - ab.sqrt()



# ---------------------------------------------------------------------------
# Discrete type diffusion
# ---------------------------------------------------------------------------


def corrupt_type_ids(
    type_ids: Tensor,
    t_idx: Tensor,
    schedule: CosineSchedule,
    n_valid_types: int = 12,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Uniform-corruption discrete diffusion for object type IDs.

    Each present slot is independently replaced with a uniformly random valid
    type with probability gamma_t = 1 - sqrt(alpha_bar_t). PAD slots
    (type_id == n_valid_types) are never corrupted so the model cannot learn
    to predict PAD from noise.

    The corruption schedule is tied to the continuous alpha_bar schedule so
    that at t=0 the types are clean (gamma=0) and at t=T-1 they are almost
    fully random (gamma~=1). This ensures that during DDIM inference with
    sample_with_types(), the type context seen at each reverse step matches
    the distribution the model was trained on.

    Args:
        type_ids: Ground-truth integer type IDs. Shape: (B, N). Long tensor.
        t_idx: Integer timestep indices in [0, T-1]. Shape: (B,).
        schedule: The CosineSchedule instance used for training.
        n_valid_types: Number of valid (non-PAD) object types. Default 12.
            Types in [0, n_valid_types) are valid; type == n_valid_types is PAD.
        generator: Optional RNG generator for reproducibility.

    Returns:
        Corrupted type IDs of the same shape and dtype as type_ids.
        PAD slots retain their original value. Present slots are corrupted
        with probability gamma_t.
    """
    B, N = type_ids.shape
    device = type_ids.device

    # gamma shape: (B,) -- one corruption probability per sample in the batch.
    gamma = schedule.type_corruption_prob(t_idx).to(device)  # (B,)
    gamma_expanded = gamma.unsqueeze(1).expand(B, N)           # (B, N)

    # Bernoulli mask: True where we replace with a random type.
    corrupt_mask = torch.bernoulli(gamma_expanded, generator=generator).bool()

    # Draw replacement types uniformly from [0, n_valid_types).
    random_types = torch.randint(
        0, n_valid_types, (B, N), device=device, generator=generator
    )

    # Never corrupt PAD slots -- they carry structural (mask) information.
    is_pad = type_ids >= n_valid_types
    corrupt_mask = corrupt_mask & ~is_pad

    return torch.where(corrupt_mask, random_types, type_ids)


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
        prediction_type: str = "epsilon",
    ) -> None:
        if prediction_type not in ("epsilon", "v"):
            raise ValueError(f"prediction_type must be epsilon or v, got {prediction_type!r}")
        self.schedule = schedule
        self.n_steps = n_steps
        self.eta = eta
        self.prediction_type = prediction_type

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

            # Predicted x_0 and eps.
            # Branch on prediction_type so v-prediction models are handled
            # correctly. Using epsilon formula on a v-prediction model causes
            # output explosion (v5 failure: std=5.32, 0% Drake validity).
            if self.prediction_type == "v":
                x0_pred = self.schedule.predict_x0_from_v(x_t, t_tensor, eps_pred)
                eps_pred = self.schedule.predict_eps_from_v(x_t, t_tensor, eps_pred)
            else:
                # Epsilon prediction (original formula).
                # Clamp to prevent explosion when ab_t is near zero.
                x0_pred = (x_t - (1.0 - ab_t_v).sqrt() * eps_pred) / ab_t_v.sqrt().clamp(min=1e-8)
            x0_pred = x0_pred.clamp(-10.0, 10.0)

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

    @torch.no_grad()
    def sample_with_types(
        self,
        fn: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
        shape: tuple[int, ...],
        seed: int | None = None,
        device: torch.device | str = "cpu",
        type_init: str = "uniform",
        n_valid_types: int = 12,
    ) -> tuple[Tensor, Tensor]:
        """Generate (x_cont, type_ids) jointly via DDIM reverse diffusion.

        Closes the training/inference gap present in :meth:`sample`. During
        training, the model receives clean ``type_ids`` at every step; during
        inference with the legacy :meth:`sample`, ``type_ids`` were frozen at
        zero (CUBE), causing type collapse. This method maintains ``type_ids``
        state across denoising steps by updating from the model's own
        ``head_type`` logits at each step.

        Args:
            fn: Callable returned by
                :meth:`~model.denoiser.SceneDenoiser.conditional_sampling_fn`.
                Signature: ``(x_t, type_ids, t) -> (eps_pred, type_logits)``
                where ``eps_pred`` has shape ``shape`` and ``type_logits``
                has shape ``(B, N_MAX, N_TYPE)``.
            shape: Output shape ``(B, N_MAX, N_CONT)``. N_CONT must match
                the model's continuous feature dimension (13).
            seed: Optional integer seed for the initial Gaussian noise AND
                the uniform type_ids prior. Both use the same generator so
                the full output is reproducible from a single seed.
            device: Device on which to run sampling.
            type_init: How to initialise ``type_ids`` before the first step.
                ``"uniform"`` samples uniformly from ``[0, n_valid_types)``,
                giving each type an equal prior probability.
                ``"zeros"`` replicates the legacy broken behavior (all CUBE)
                and is provided only for regression comparison.
            n_valid_types: Upper bound (exclusive) for uniform type sampling.
                Should match the number of non-padding vocabulary entries.
                Default 12 (types 0-11; type 12 is the PAD token).

        Returns:
            ``(x_cont, type_ids)`` where:
              - ``x_cont``: Denoised continuous features. Shape: ``shape``.
              - ``type_ids``: Final predicted type per slot. Shape: ``(B, N_MAX)``,
                dtype long, values in ``[0, N_TYPE)``.
        """
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        B = shape[0]
        N = shape[1]

        # Start from pure Gaussian noise.
        x_t = torch.randn(shape, device=device, generator=generator)

        # Initialise type_ids from prior.
        if type_init == "uniform":
            type_ids = torch.randint(
                0, n_valid_types, (B, N), dtype=torch.long, device=device,
                generator=generator,
            )
        elif type_init == "zeros":
            # Replicates the legacy (broken) fixed-zero behavior.
            type_ids = torch.zeros(B, N, dtype=torch.long, device=device)
        else:
            raise ValueError(
                f"Unknown type_init {type_init!r}. Must be 'uniform' or 'zeros'."
            )

        sched = self.schedule

        for i, t_val in enumerate(self._timesteps):
            t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)

            # Forward pass: get both eps and type logits.
            eps_pred, type_logits = fn(x_t, type_ids, t_tensor)

            # Mask out the PAD token (index n_valid_types and above) so argmax
            # never selects vocabulary entries outside [0, n_valid_types).
            # This prevents the sampler from emitting PAD-type objects at inference.
            if n_valid_types < type_logits.shape[-1]:
                type_logits = type_logits.clone()
                type_logits[..., n_valid_types:] = float("-inf")

            # Update type_ids from this step's logits before the continuous update.
            # Greedy argmax: deterministic, consistent with DDIM eta=0 spirit.
            type_ids = type_logits.argmax(dim=-1)  # (B, N_MAX)

            # Standard DDIM continuous update (identical math to sample()).
            ab_t = sched.alpha_bar(t_tensor)        # (B,)
            if i + 1 < len(self._timesteps):
                t_prev_val = self._timesteps[i + 1]
            else:
                t_prev_val = 0
            t_prev = torch.full((B,), t_prev_val, dtype=torch.long, device=device)
            ab_prev = sched.alpha_bar(t_prev)       # (B,)

            extra_dims = x_t.dim() - 1
            view = (-1,) + (1,) * extra_dims

            ab_t_v = ab_t.view(view).to(device)
            ab_prev_v = ab_prev.view(view).to(device)

            # Branch on prediction_type: v-prediction or epsilon.
            # Using epsilon formula on v-prediction model causes output explosion.
            if self.prediction_type == "v":
                x0_pred = self.schedule.predict_x0_from_v(x_t, t_tensor, eps_pred)
                eps_pred = self.schedule.predict_eps_from_v(x_t, t_tensor, eps_pred)
            else:
                x0_pred = (x_t - (1.0 - ab_t_v).sqrt() * eps_pred) / ab_t_v.sqrt().clamp(min=1e-8)
            x0_pred = x0_pred.clamp(-10.0, 10.0)

            sigma = (
                self.eta
                * ((1.0 - ab_prev_v) / (1.0 - ab_t_v)).sqrt()
                * (1.0 - ab_t_v / ab_prev_v).sqrt()
            )
            dir_xt = (1.0 - ab_prev_v - sigma ** 2).clamp(min=0.0).sqrt() * eps_pred
            noise = sigma * torch.randn_like(x_t)
            x_t = ab_prev_v.sqrt() * x0_pred + dir_xt + noise

        return x_t, type_ids

    def sample_with_universal_guidance(
        self,
        fn: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
        shape: tuple[int, ...],
        seed: int | None = None,
        device: torch.device | str = "cpu",
        type_init: str = "uniform",
        n_valid_types: int = 12,
        guidance_scale: float = 0.0,
        guidance_min_step_frac: float = 0.1,
        guidance_max_step_frac: float = 0.9,
        grad_clip: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """Sample using Universal Guidance with analytical energy surrogate.

        Implements Bansal et al. (ICML 2023) with pairwise interpenetration
        energy as the guidance function.

        At each DDIM step:
          1. Compute conditional eps prediction (no grad, fast path).
          2. Project noisy state to clean estimate via Tweedie's formula.
          3. Evaluate pairwise_overlap_energy on the clean estimate.
          4. Backpropagate energy to x_t to get gradient.
          5. Modify eps: eps_guided = eps - guidance_scale * sqrt(1 - ab_t) * grad
          6. Standard DDIM update with eps_guided.

        The guidance is active only between guidance_min_step_frac and
        guidance_max_step_frac of the total DDIM steps. Guiding too early
        (high noise) amplifies noise; guiding too late (low noise) fights
        the final structure.

        Args:
            fn: Callable from denoiser.conditional_sampling_fn.
                Signature: (x_t, type_ids, t) -> (eps_pred, type_logits).
            shape: Output shape (B, N_MAX, N_CONT=13).
            seed: Optional RNG seed.
            device: Compute device.
            type_init: 'uniform' or 'zeros'. See sample_with_types.
            n_valid_types: Number of valid non-PAD types (12).
            guidance_scale: Scale of analytical energy gradient. 0.0 = baseline
                (identical to sample_with_types). Recommended range: 0.5-4.0.
            guidance_min_step_frac: Don't guide before this fraction of steps.
                0.1 = skip first 10% (very high noise levels).
            guidance_max_step_frac: Don't guide after this fraction of steps.
                0.9 = skip last 10% (nearly clean, let model finish).
            grad_clip: Max L-inf norm for the guidance gradient. Prevents
                numerical explosion on early steps with large gradients.

        Returns:
            (x_cont, type_ids): Denoised continuous features and type IDs.
        """
        # Lazy import to avoid circular dependency at module load.
        from guidance.energy import pairwise_overlap_energy

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        B = shape[0]
        N = shape[1]

        x_t = torch.randn(shape, device=device, generator=generator)

        if type_init == "uniform":
            type_ids = torch.randint(
                0, n_valid_types, (B, N), dtype=torch.long, device=device,
                generator=generator,
            )
        elif type_init == "zeros":
            type_ids = torch.zeros(B, N, dtype=torch.long, device=device)
        else:
            raise ValueError(
                f"Unknown type_init {type_init!r}. Must be 'uniform' or 'zeros'."
            )

        sched = self.schedule
        n_steps = len(self._timesteps)
        guide_start = int(guidance_min_step_frac * n_steps)
        guide_end = int(guidance_max_step_frac * n_steps)

        for i, t_val in enumerate(self._timesteps):
            t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)

            # ---------------------------------------------------------------
            # Standard forward pass (no grad for efficiency).
            # ---------------------------------------------------------------
            with torch.no_grad():
                eps_pred, type_logits = fn(x_t, type_ids, t_tensor)

            # Update type_ids from this step's logits (matches sample_with_types).
            if n_valid_types < type_logits.shape[-1]:
                type_logits = type_logits.clone()
                type_logits[..., n_valid_types:] = float("-inf")
            type_ids = type_logits.argmax(dim=-1)

            # ---------------------------------------------------------------
            # Universal Guidance: compute energy gradient on clean estimate.
            # ---------------------------------------------------------------
            if guidance_scale > 0.0 and guide_start <= i < guide_end:
                ab_t = sched.alpha_bar(t_tensor).to(device).view(-1, 1, 1)  # (B,1,1)
                sqrt_ab = ab_t.sqrt()
                sqrt_1mab = (1.0 - ab_t).sqrt()

                with torch.enable_grad():
                    x_t_g = x_t.detach().requires_grad_(True)
                    # Second forward pass through fn to get differentiable eps.
                    eps_g, tl_g = fn(x_t_g, type_ids, t_tensor)

                    # Tweedie's formula: x_0_hat = (x_t - sqrt(1-ab)*eps) / sqrt(ab)
                    x0_hat = (x_t_g - sqrt_1mab * eps_g) / sqrt_ab.clamp(min=1e-8)
                    x0_hat = x0_hat.clamp(-10.0, 10.0)  # match sample_with_types clamp

                    # Type IDs for energy: use current greedy type_ids (already updated).
                    energy = pairwise_overlap_energy(x0_hat, type_ids)
                    energy_sum = energy.sum()

                # Backprop only if energy is non-zero (skip if no overlaps).
                if energy_sum.item() > 0.0:
                    energy_sum.backward()
                    if x_t_g.grad is not None:
                        grad = x_t_g.grad.detach()
                        # Clip gradient to prevent explosion.
                        grad_norm = grad.abs().max()
                        if grad_norm > grad_clip:
                            grad = grad * (grad_clip / grad_norm)
                        # Inject into eps: eps_guided = eps - s * sqrt(1-ab) * grad
                        # (negative because we minimize energy).
                        eps_pred = eps_pred - guidance_scale * sqrt_1mab.detach() * grad

            # ---------------------------------------------------------------
            # Standard DDIM update (identical to sample_with_types).
            # ---------------------------------------------------------------
            ab_t_v = sched.alpha_bar(t_tensor).to(device)
            if i + 1 < len(self._timesteps):
                t_prev_val = self._timesteps[i + 1]
            else:
                t_prev_val = 0
            t_prev = torch.full((B,), t_prev_val, dtype=torch.long, device=device)
            ab_prev = sched.alpha_bar(t_prev).to(device)

            extra_dims = x_t.dim() - 1
            view = (-1,) + (1,) * extra_dims
            ab_t_v = ab_t_v.view(view)
            ab_prev_v = ab_prev.view(view)

            x0_pred = (x_t - (1.0 - ab_t_v).sqrt() * eps_pred) / ab_t_v.sqrt().clamp(min=1e-8)
            x0_pred = x0_pred.clamp(-10.0, 10.0)

            sigma = (
                self.eta
                * ((1.0 - ab_prev_v) / (1.0 - ab_t_v)).sqrt()
                * (1.0 - ab_t_v / ab_prev_v).sqrt()
            )
            dir_xt = (1.0 - ab_prev_v - sigma ** 2).clamp(min=0.0).sqrt() * eps_pred
            noise = sigma * torch.randn_like(x_t)
            x_t = ab_prev_v.sqrt() * x0_pred + dir_xt + noise

        return x_t, type_ids

    @torch.no_grad()
    def sample_cfg(
        self,
        cond_fn: Callable[[Tensor, Tensor, Tensor | None], Tensor],
        uncond_fn: Callable[[Tensor, Tensor, Tensor | None], Tensor],
        shape: tuple[int, ...],
        guidance_scale: float = 3.0,
        seed: int | None = None,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Generate samples via DDIM with classifier-free guidance.

        Requires two forward passes per denoising step:
            eps_pred = uncond_eps + w * (cond_eps - uncond_eps)

        Args:
            cond_fn: Noise prediction callable with text conditioning.
                Signature: ``(x_t, t, text_emb) -> eps_pred``.
            uncond_fn: Noise prediction callable without text conditioning.
                Signature: ``(x_t, t, None) -> eps_pred``.
            shape: Output shape ``(B, ...)``.
            guidance_scale: CFG scale w. 1.0 = no guidance (same as uncond).
                Typical range 2.0-7.5; higher amplifies conditioning signal.
            seed: Optional integer seed for the initial Gaussian noise.
            device: Device on which to run sampling.

        Returns:
            Denoised sample x_0. Shape: ``shape``.
        """
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        x_t = torch.randn(shape, device=device, generator=generator)
        B = shape[0]
        sched = self.schedule

        for i, t_val in enumerate(self._timesteps):
            t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)

            # Two forward passes: conditional and unconditional.
            # Both callables wrap text_emb in their closure; the 3rd argument
            # is ignored by noise_prediction_fn's returned _fn.
            eps_cond = cond_fn(x_t, t_tensor, None)
            eps_uncond = uncond_fn(x_t, t_tensor, None)

            # CFG combination.
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # DDIM update (identical to sample()).
            ab_t = sched.alpha_bar(t_tensor)
            if i + 1 < len(self._timesteps):
                t_prev_val = self._timesteps[i + 1]
            else:
                t_prev_val = 0
            t_prev = torch.full((B,), t_prev_val, dtype=torch.long, device=device)
            ab_prev = sched.alpha_bar(t_prev)

            extra_dims = x_t.dim() - 1
            view = (-1,) + (1,) * extra_dims
            ab_t_v = ab_t.view(view).to(device)
            ab_prev_v = ab_prev.view(view).to(device)

            x0_pred = (x_t - (1.0 - ab_t_v).sqrt() * eps_pred) / ab_t_v.sqrt().clamp(min=1e-8)
            x0_pred = x0_pred.clamp(-10.0, 10.0)

            sigma = (
                self.eta
                * ((1.0 - ab_prev_v) / (1.0 - ab_t_v)).sqrt()
                * (1.0 - ab_t_v / ab_prev_v).sqrt()
            )
            dir_xt = (1.0 - ab_prev_v - sigma ** 2).clamp(min=0.0).sqrt() * eps_pred
            noise = sigma * torch.randn_like(x_t)
            x_t = ab_prev_v.sqrt() * x0_pred + dir_xt + noise

        return x_t
