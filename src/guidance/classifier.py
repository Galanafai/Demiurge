"""Validity classifier for noise-conditioned Drake validity prediction.

Used during classifier-guided sampling (Phase B). Takes a noisy scene
tensor at timestep t and predicts whether the underlying clean scene
would pass Drake validation.

INVARIANT: This classifier must be trained on NOISED inputs x_t at
varying t, not on clean scenes x_0. A clean-only classifier gives
near-useless gradients at intermediate diffusion steps because the
noisy inputs it sees at inference time are out of distribution.
See: Dhariwal and Nichol (2021), Section 4.

Architecture:
    - Smaller transformer: d_model=128, 4 layers, 4 heads (~3M params)
    - Tokenization: same per-object embedding as SceneDenoiser (type_embed
      + cont_proj), but projected to d_model=128
    - Timestep conditioning: sinusoidal embedding -> MLP, via AdaLN
    - Pool over object tokens -> single scalar logit (valid/invalid)
    - Loss: BCE with logits during training
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from model.denoiser import N_CONT, N_TYPE

_D_CLASSIFIER: int = 128
_N_LAYERS: int = 4
_N_HEADS: int = 4
_FFN_MULT: int = 4


def _sinusoidal_embedding(t: Tensor, d: int) -> Tensor:
    """Sinusoidal timestep embedding, matching SceneDenoiser convention."""
    assert d % 2 == 0
    half = d // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class _ClassifierBlock(nn.Module):
    """Transformer block with AdaLN timestep conditioning (no cross-attention)."""

    def __init__(self, d: int, n_heads: int, ffn_mult: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.adaln = nn.Linear(d, 4 * d, bias=True)
        self.norm_sa = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d, elementwise_affine=False)
        d_ffn = d * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d, d_ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ffn, d)
        )
        self.drop = nn.Dropout(dropout)
        # Zero-init AdaLN so it starts as identity LayerNorm.
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Args:
            x: (B, N_MAX, d)
            t_emb: (B, d)
        Returns:
            (B, N_MAX, d)
        """
        params = self.adaln(t_emb)
        s_sa, h_sa, s_ffn, h_ffn = params.chunk(4, dim=-1)
        s_sa = s_sa.unsqueeze(1)
        h_sa = h_sa.unsqueeze(1)
        s_ffn = s_ffn.unsqueeze(1)
        h_ffn = h_ffn.unsqueeze(1)

        normed = (1.0 + s_sa) * self.norm_sa(x) + h_sa
        attn_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        x = x + self.drop(attn_out)

        normed_ffn = (1.0 + s_ffn) * self.norm_ffn(x) + h_ffn
        x = x + self.drop(self.ffn(normed_ffn))
        return x


class ValidityClassifier(nn.Module):
    """Noise-conditioned Drake validity classifier.

    Args:
        d_model: Embedding dimension. Default 128.
        n_layers: Transformer depth. Default 4.
        n_heads: Attention heads. Default 4.
        ffn_mult: FFN hidden dim multiplier. Default 4.
        dropout: Dropout rate. Default 0.0 (inference is deterministic).
    """

    def __init__(
        self,
        d_model: int = _D_CLASSIFIER,
        n_layers: int = _N_LAYERS,
        n_heads: int = _N_HEADS,
        ffn_mult: int = _FFN_MULT,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Timestep embedding: sinusoidal -> MLP.
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Per-slot tokenization (same structure as SceneDenoiser).
        self.type_embed = nn.Embedding(N_TYPE, d_model)
        self.cont_proj = nn.Linear(N_CONT, d_model)

        # Transformer blocks (no cross-attention; no text conditioning).
        self.blocks = nn.ModuleList([
            _ClassifierBlock(d_model, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Pooling then binary output.
        self.head = nn.Linear(d_model, 1)
        # Zero-init bias so starting logit is 0.0 (neutral prior).
        # Do NOT zero-init weight -- that would zero all gradients through the head.
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x_t: Tensor,
        type_ids: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Predict validity logit for noisy scene.

        Args:
            x_t: Noisy continuous features. Shape: (B, N_MAX, N_CONT=13).
            type_ids: Object type indices. Shape: (B, N_MAX), dtype=long.
                May be noisy/dummy during training; passed through embedding.
            t: Diffusion timestep indices. Shape: (B,), dtype=long.

        Returns:
            Validity logit, shape (B,). Pass through sigmoid for probability.
        """
        t_sin = _sinusoidal_embedding(t, self.d_model)   # (B, d)
        t_emb = self.t_mlp(t_sin)                         # (B, d)

        tokens = self.type_embed(type_ids) + self.cont_proj(x_t)  # (B, N_MAX, d)

        for block in self.blocks:
            tokens = block(tokens, t_emb)

        out = self.final_norm(tokens)                      # (B, N_MAX, d)
        pooled = out.mean(dim=1)                           # (B, d)  mean pooling
        logit = self.head(pooled).squeeze(-1)              # (B,)
        return logit

    def log_prob_valid(self, x_t: Tensor, type_ids: Tensor, t: Tensor) -> Tensor:
        """Return log p(valid | x_t, t) = log sigmoid(logit).

        Convenience method for gradient computation during guided sampling.
        Gradient of this w.r.t. x_t is the guidance signal.
        """
        logit = self.forward(x_t, type_ids, t)
        return torch.nn.functional.logsigmoid(logit)
