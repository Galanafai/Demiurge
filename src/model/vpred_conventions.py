"""V-prediction conventions, fully tested and documented.

This module exists because v-prediction has multiple conventions in the
literature with different sign conventions. Getting them wrong causes
output explosion (std 5x too large) which is exactly what killed v5.

THE CANONICAL CONVENTION (Salimans & Ho 2022, Lin et al. 2024):

  v_t = sqrt(alpha_bar_t) * eps - sqrt(1 - alpha_bar_t) * x_0

Recovery formulas (used during sampling):

  x_0 = sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t) * v_t
  eps = sqrt(1 - alpha_bar_t) * x_t + sqrt(alpha_bar_t) * v_t

Sanity check (must hold for any t):
  q_sample: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps
  Therefore: x_t = sqrt(ab) * (sqrt(ab)*x_t - sqrt(1-ab)*v)
                 + sqrt(1-ab) * (sqrt(1-ab)*x_t + sqrt(ab)*v)
           = ab*x_t - sqrt(ab*(1-ab))*v + (1-ab)*x_t + sqrt(ab*(1-ab))*v
           = (ab + 1 - ab) * x_t = x_t  checkmark

If training and sampler use DIFFERENT conventions for the sign of v,
or different recovery formulas, the model output will explode. Use ONLY
the functions in this module.
"""
from __future__ import annotations
import torch


def compute_v_target(
    x_0: torch.Tensor,
    eps: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x_0.

    Used as the training target when prediction_type='v'.
    Model is trained to minimise MSE(model_output, v_target).
    """
    sqrt_ab = alpha_bar.sqrt()
    sqrt_one_minus_ab = (1.0 - alpha_bar).sqrt()
    return sqrt_ab * eps - sqrt_one_minus_ab * x_0


def v_to_x0(
    x_t: torch.Tensor,
    v: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """x_0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v.

    Used during sampling: convert model v prediction to x_0 estimate.
    """
    sqrt_ab = alpha_bar.sqrt()
    sqrt_one_minus_ab = (1.0 - alpha_bar).sqrt()
    return sqrt_ab * x_t - sqrt_one_minus_ab * v


def v_to_eps(
    x_t: torch.Tensor,
    v: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """eps = sqrt(1 - alpha_bar) * x_t + sqrt(alpha_bar) * v.

    Used during sampling: convert model v prediction to eps estimate.
    """
    sqrt_ab = alpha_bar.sqrt()
    sqrt_one_minus_ab = (1.0 - alpha_bar).sqrt()
    return sqrt_one_minus_ab * x_t + sqrt_ab * v


def x0_to_v(
    x_0: torch.Tensor,
    x_t: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """Derive v from x_t and x_0. Used for verification only."""
    sqrt_ab = alpha_bar.sqrt()
    sqrt_one_minus_ab = (1.0 - alpha_bar).sqrt()
    eps = (x_t - sqrt_ab * x_0) / sqrt_one_minus_ab
    return compute_v_target(x_0, eps, alpha_bar)


def eps_to_x0(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """x_0 = (x_t - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar).

    Used during sampling with epsilon prediction.
    """
    sqrt_ab = alpha_bar.sqrt()
    sqrt_one_minus_ab = (1.0 - alpha_bar).sqrt()
    return (x_t - sqrt_one_minus_ab * eps) / sqrt_ab


def q_sample(
    x_0: torch.Tensor,
    eps: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps.

    Forward diffusion: noise the clean data.
    """
    sqrt_ab = alpha_bar.sqrt()
    sqrt_one_minus_ab = (1.0 - alpha_bar).sqrt()
    return sqrt_ab * x_0 + sqrt_one_minus_ab * eps
