"""SceneDenoiser transformer for diffusion-based scene generation.

Architecture (per DenoiserConfig):
  - 12 object tokens (N_MAX=12) + 1 [NOISE_T] timestep token
  - Per-slot input: type embedding (Embedding(13, d_model)) summed with
    continuous feature projection (Linear(13, d_model))
  - No positional encoding (set-structured; permutation-equivariant)
  - AdaLN timestep conditioning per block (scale+shift for SA and FFN norms)
  - Cross-attention to text embedding per block (scene tokens only;
    [NOISE_T] token is excluded from cross-attention)
  - Output heads: type logits, xyz residual, rot6d residual, scale residual,
    presence logit

Continuous slot feature layout (N_CONT=13 dims):
  xyz (3) | rot6d (6) | scale (3) | presence_bit (1)
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

N_MAX: int = 12        # Fixed object slots from Week 2 schema
N_TYPE: int = 13       # 12 active vocab types + 1 PAD
N_CONT: int = 13       # xyz(3) + rot6d(6) + scale(3) + presence(1)
D_TEXT: int = 384      # all-MiniLM-L6-v2 output dimension


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DenoiserConfig:
    """Hyperparameters for SceneDenoiser.

    Attributes:
        n_layers: Number of transformer blocks. Default 6.
        d_model: Token embedding dimension. Default 256 (small/default config).
            Promote to 512 only if unconditional validity rate < 0.4.
        n_heads: Number of attention heads. Default 8.
        ffn_mult: FFN hidden dim multiplier relative to d_model. Default 4.
        dropout: Dropout probability in attention and FFN. Default 0.1.
    """

    n_layers: int = 6
    d_model: int = 256
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class DenoiserOutput:
    """Raw output tensors from SceneDenoiser.

    All tensors have leading batch dimension B and N_MAX slot dimension.
    These are raw model outputs (logits / noise predictions), not yet
    post-processed into valid scene representations.

    Attributes:
        type_logits: Un-normalised logits over N_TYPE classes. (B, N_MAX, 13)
        rot6d: Predicted noise for 6D rotation representation. (B, N_MAX, 6)
        xyz: Predicted noise for xyz position. (B, N_MAX, 3)
        scale: Predicted noise for scale. (B, N_MAX, 3)
        presence_logit: Un-normalised logit for slot occupancy. (B, N_MAX, 1)
    """

    type_logits: Tensor
    rot6d: Tensor
    xyz: Tensor
    scale: Tensor
    presence_logit: Tensor


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


def _sinusoidal_embedding(t: Tensor, d_model: int) -> Tensor:
    """Map integer timestep indices to sinusoidal embeddings.

    Args:
        t: Long tensor of timestep indices. Shape: (B,).
        d_model: Output dimension. Must be even.

    Returns:
        Float tensor of shape (B, d_model).
    """
    assert d_model % 2 == 0
    half = d_model // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, d_model)


class _FFN(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, d_model: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        d_ffn = d_model * ffn_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # type: ignore[no-any-return]


class _DenoiserBlock(nn.Module):
    """Single transformer block with AdaLN and optional cross-attention.

    Self-attention and FFN use adaptive layer norm conditioned on the
    timestep embedding. Cross-attention uses a standard (non-adaptive)
    layer norm and attends to the text embedding.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        # AdaLN: produces (scale_sa, shift_sa, scale_ffn, shift_ffn) from t_emb.
        # elementwise_affine=False: affine transform is replaced by AdaLN.
        self.adaln = nn.Linear(d_model, 4 * d_model, bias=True)
        self.norm_sa = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention (scene tokens attend to text).
        self.norm_ca = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
            kdim=D_TEXT, vdim=D_TEXT,
        )
        self.norm_ffn = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = _FFN(d_model, ffn_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, t_emb: Tensor, text_emb: Tensor | None) -> Tensor:
        """Forward pass for one transformer block.

        Args:
            x: Token sequence including [NOISE_T] at position N_MAX.
               Shape: (B, N_MAX+1, d_model).
            t_emb: Timestep embedding. Shape: (B, d_model).
            text_emb: Optional text context. Shape: (B, D_TEXT).

        Returns:
            Updated token sequence. Shape: (B, N_MAX+1, d_model).
        """
        # Compute AdaLN parameters.
        adaln_params = self.adaln(t_emb)                   # (B, 4*d)
        s_sa, h_sa, s_ffn, h_ffn = adaln_params.chunk(4, dim=-1)  # (B, d) each
        s_sa = s_sa.unsqueeze(1)    # (B, 1, d)
        h_sa = h_sa.unsqueeze(1)
        s_ffn = s_ffn.unsqueeze(1)
        h_ffn = h_ffn.unsqueeze(1)

        # Self-attention with AdaLN (all tokens attend to each other).
        normed = (1.0 + s_sa) * self.norm_sa(x) + h_sa
        attn_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        x = x + self.drop(attn_out)

        # Cross-attention: scene tokens (0:N_MAX) attend to text.
        # [NOISE_T] token (index N_MAX) is excluded.
        if text_emb is not None:
            scene = x[:, :N_MAX]                            # (B, N_MAX, d)
            text_kv = text_emb.unsqueeze(1)                 # (B, 1, D_TEXT)
            ca_in = self.norm_ca(scene)
            ca_out, _ = self.cross_attn(ca_in, text_kv, text_kv, need_weights=False)
            x = torch.cat([scene + self.drop(ca_out), x[:, N_MAX:]], dim=1)

        # FFN with AdaLN.
        normed_ffn = (1.0 + s_ffn) * self.norm_ffn(x) + h_ffn
        x = x + self.drop(self.ffn(normed_ffn))

        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class SceneDenoiser(nn.Module):
    """Transformer denoiser for the Demiurge diffusion model.

    Takes a noisy scene tensor (continuous features + type ids) and a
    diffusion timestep, and predicts the noise that was added.

    Args:
        cfg: DenoiserConfig instance. Defaults to the small (d_model=256) config.
    """

    def __init__(self, cfg: DenoiserConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = DenoiserConfig()
        self.cfg = cfg
        d = cfg.d_model

        # Timestep embedding: sinusoidal -> MLP.
        self.t_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.SiLU(),
            nn.Linear(d * 4, d),
        )

        # Per-slot input encoders.
        self.type_embed = nn.Embedding(N_TYPE, d)
        self.cont_proj = nn.Linear(N_CONT, d)

        # [NOISE_T] token: project timestep embedding to a single token.
        self.t_token_proj = nn.Linear(d, d)

        # Transformer blocks.
        self.blocks = nn.ModuleList([
            _DenoiserBlock(d, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        # Final layer norm before output heads.
        self.final_norm = nn.LayerNorm(d)

        # Output heads (applied to scene token representations only).
        self.head_type = nn.Linear(d, N_TYPE)
        self.head_rot6d = nn.Linear(d, 6)
        self.head_xyz = nn.Linear(d, 3)
        self.head_scale = nn.Linear(d, 3)
        self.head_presence = nn.Linear(d, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Zero-initialise output heads so the model starts near identity."""
        for head in (self.head_rot6d, self.head_xyz, self.head_scale, self.head_presence):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        # AdaLN linear layers: zero-init so scale=0 and shift=0 at start
        # (equivalent to standard LayerNorm on the first forward pass).
        for block in self.blocks:
            if isinstance(block, _DenoiserBlock):
                nn.init.zeros_(block.adaln.weight)
                nn.init.zeros_(block.adaln.bias)

    def forward(
        self,
        x_cont: Tensor,
        type_ids: Tensor,
        t: Tensor,
        text_emb: Tensor | None = None,
    ) -> DenoiserOutput:
        """Predict noise for a noisy scene.

        Args:
            x_cont: Noisy continuous features. Shape: (B, N_MAX, N_CONT=13).
            type_ids: Object type indices (may be noisy at training time).
                Shape: (B, N_MAX), dtype=long, values in [0, N_TYPE).
            t: Diffusion timestep indices. Shape: (B,), dtype=long.
            text_emb: Optional frozen text embedding from sentence-transformer.
                Shape: (B, D_TEXT=384). Pass None for unconditional generation.

        Returns:
            DenoiserOutput with per-slot noise predictions and type logits.
        """
        d = self.cfg.d_model

        # Timestep embedding.
        t_sin = _sinusoidal_embedding(t, d)     # (B, d)
        t_emb = self.t_mlp(t_sin)               # (B, d)

        # Per-slot token encoding: type embedding + continuous projection.
        tok = self.type_embed(type_ids) + self.cont_proj(x_cont)  # (B, N_MAX, d)

        # Append [NOISE_T] token at position N_MAX.
        t_token = self.t_token_proj(t_emb).unsqueeze(1)  # (B, 1, d)
        tokens = torch.cat([tok, t_token], dim=1)         # (B, N_MAX+1, d)

        # Transformer blocks.
        for block in self.blocks:
            tokens = block(tokens, t_emb, text_emb)

        # Extract scene token outputs (exclude [NOISE_T]).
        scene_out = self.final_norm(tokens[:, :N_MAX])    # (B, N_MAX, d)

        return DenoiserOutput(
            type_logits=self.head_type(scene_out),        # (B, N_MAX, 13)
            rot6d=self.head_rot6d(scene_out),             # (B, N_MAX, 6)
            xyz=self.head_xyz(scene_out),                 # (B, N_MAX, 3)
            scale=self.head_scale(scene_out),             # (B, N_MAX, 3)
            presence_logit=self.head_presence(scene_out), # (B, N_MAX, 1)
        )

    def noise_prediction_fn(
        self, text_emb: Tensor | None = None
    ) -> Callable[[Tensor, Tensor, Tensor | None], Tensor]:
        """Return a callable compatible with DDIMSampler.sample().

        The returned function packs the full DenoiserOutput back into a
        single (B, N_MAX, N_CONT) noise-prediction tensor for the DDIM loop.
        Type ids are treated as zeros during sampling (argmax of logits is
        applied post-hoc, not during denoising).

        Args:
            text_emb: Optional text embedding to close over.

        Returns:
            Callable ``(x_t, t, _text_emb) -> eps_pred`` where x_t has
            shape (B, N_MAX, N_CONT) and eps_pred has the same shape.
        """
        def _fn(x_t: Tensor, t_idx: Tensor, _: Tensor | None) -> Tensor:
            # Dummy type ids: zeros (will be replaced post-sampling).
            type_ids = torch.zeros(x_t.shape[0], N_MAX, dtype=torch.long, device=x_t.device)
            out = self.forward(x_t, type_ids, t_idx, text_emb)
            # Pack continuous noise predictions back into (B, N_MAX, 13).
            return torch.cat([out.xyz, out.rot6d, out.scale, out.presence_logit], dim=-1)

        return _fn
