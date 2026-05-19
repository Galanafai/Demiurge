"""Loss module for the Demiurge scene diffusion model.

Computes a weighted sum of per-component losses:
  - pose_xyz:      MSE between predicted and actual noise for XYZ position
  - pose_rot:      MSE between predicted and actual noise for 6D rotation
  - scale:         MSE between predicted and actual noise for scale
  - type_ce:       Cross-entropy on predicted type logits
  - presence_bce:  Binary cross-entropy on presence logit

Continuous losses (pose_xyz, pose_rot, scale) and type_ce are masked to
present slots only. presence_bce is applied to ALL slots because the model
must learn to predict which slots are occupied.

All per-component losses are returned individually for W&B logging.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from model.denoiser import DenoiserOutput

# ---------------------------------------------------------------------------
# Config types
# ---------------------------------------------------------------------------


@dataclass
class LossWeights:
    """Per-component loss weights.

    Defaults are calibrated so presence_bce and type_ce do not dominate
    the noise MSE terms during early training.
    """

    pose_xyz: float = 1.0
    pose_rot: float = 1.0
    scale: float = 0.5
    type_ce: float = 0.1
    presence_bce: float = 0.05
    slot_diversity: float = 0.0
    """Weight for the slot-diversity penalty.

    Penalises high cosine similarity between predicted XYZ noise across slots,
    encouraging the model to learn distinct positions for each object slot
    rather than collapsing all slots to the same spatial mode.

    Recommended value for anti-collapse training: 0.05.
    Default 0.0 disables the penalty (backward compatible).
    """


@dataclass
class LossOutput:
    """Per-component loss values plus total.

    All Tensor values are scalar (0-dim). Use ``as_log_dict()`` to convert
    to a plain dict[str, float] suitable for W&B logging.
    """

    total: Tensor
    pose_xyz: Tensor
    pose_rot: Tensor
    scale: Tensor
    type_ce: Tensor
    presence_bce: Tensor
    slot_diversity: Tensor | None = None

    def as_log_dict(self) -> dict[str, float]:
        """Return all components (including total) as float values."""
        d = {
            "total": float(self.total.detach().item()),
            "pose_xyz": float(self.pose_xyz.detach().item()),
            "pose_rot": float(self.pose_rot.detach().item()),
            "scale": float(self.scale.detach().item()),
            "type_ce": float(self.type_ce.detach().item()),
            "presence_bce": float(self.presence_bce.detach().item()),
        }
        if self.slot_diversity is not None:
            d["slot_diversity"] = float(self.slot_diversity.detach().item())
        return d


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


class SceneDiffusionLoss(torch.nn.Module):
    """Weighted multi-component diffusion loss for scene generation.

    Args:
        weights: LossWeights instance controlling per-component scaling.
            Defaults to LossWeights() if not provided.
        class_weights: Optional 1-D float tensor of shape (N_TYPE,) providing
            per-class weights for the type cross-entropy term. When provided,
            passed as the ``weight`` argument to ``F.cross_entropy``.
            Inverse-frequency weighting mitigates majority-class collapse.
            Default None preserves existing uniform-weight behavior.
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        # Register as buffer so it moves with .to(device) automatically.
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights: torch.Tensor | None = None

    def forward(
        self,
        pred: DenoiserOutput,
        target_xyz: Tensor,
        target_rot6d: Tensor,
        target_scale: Tensor,
        target_presence: Tensor,
        target_type_ids: Tensor,
        presence_mask: Tensor,
    ) -> LossOutput:
        """Compute the weighted loss.

        Args:
            pred: Raw model output from SceneDenoiser.forward().
            target_xyz: Regression target for XYZ (eps or v). Shape: (B, N_MAX, 3).
            target_rot6d: Regression target for 6D rotation. Shape: (B, N_MAX, 6).
            target_scale: Regression target for scale. Shape: (B, N_MAX, 3).
            target_presence: Regression target for presence. Shape: (B, N_MAX, 1).
            target_type_ids: True object type IDs (from clean scene).
                Shape: (B, N_MAX), dtype=long.
            presence_mask: Boolean mask of occupied slots in the CLEAN scene.
                Shape: (B, N_MAX). True = slot is occupied.
                Used to mask continuous and type losses. Presence BCE uses
                all slots.

        Returns:
            LossOutput with total and individual component losses.
        """
        w = self.weights
        mask = presence_mask.float()           # (B, N_MAX) -- 1 for present slots
        n_present = mask.sum().clamp(min=1.0)  # avoid division by zero

        # --- XYZ MSE (present slots only) ---
        xyz_err = ((pred.xyz - target_xyz) ** 2).sum(dim=-1)        # (B, N_MAX)
        loss_xyz = (xyz_err * mask).sum() / n_present

        # --- Rotation 6D MSE (present slots only) ---
        rot_err = ((pred.rot6d - target_rot6d) ** 2).sum(dim=-1)    # (B, N_MAX)
        loss_rot = (rot_err * mask).sum() / n_present

        # --- Scale MSE (present slots only) ---
        scale_err = ((pred.scale - target_scale) ** 2).sum(dim=-1)  # (B, N_MAX)
        loss_scale = (scale_err * mask).sum() / n_present

        # --- Type cross-entropy (present slots only) ---
        B, N_MAX_local, N_TYPE = pred.type_logits.shape
        # Flatten to (B*N_MAX, N_TYPE) for F.cross_entropy, then mask.
        logits_flat = pred.type_logits.reshape(B * N_MAX_local, N_TYPE)
        ids_flat = target_type_ids.reshape(B * N_MAX_local)
        # class_weights: inverse-frequency tensor (N_TYPE,) on same device as logits.
        # None = uniform weights (original behavior).
        cw = self.class_weights  # type: ignore[attr-defined]
        ce_flat = F.cross_entropy(logits_flat, ids_flat, weight=cw, reduction="none")  # (B*N_MAX,)
        ce_2d = ce_flat.reshape(B, N_MAX_local)
        loss_type = (ce_2d * mask).sum() / n_present

        # --- Presence BCE (all slots) ---
        # pred.presence_logit: (B, N_MAX, 1); eps_presence used as target proxy
        # here we use the CLEAN presence mask as the target (not the noisy bit).
        pres_target = presence_mask.float().unsqueeze(-1)                   # (B, N_MAX, 1)
        loss_pres = F.binary_cross_entropy_with_logits(
            pred.presence_logit, pres_target, reduction="mean"
        )

        total = (
            w.pose_xyz * loss_xyz
            + w.pose_rot * loss_rot
            + w.scale * loss_scale
            + w.type_ce * loss_type
            + w.presence_bce * loss_pres
        )

        # --- Slot diversity penalty (optional) ---
        # Penalise high mean pairwise cosine similarity between predicted XYZ
        # noise across slots. This discourages the collapse mode where all
        # slots predict the same position. Only computed when weight > 0.
        loss_div: Tensor | None = None
        if w.slot_diversity > 0.0:
            xyz_pred = pred.xyz                         # (B, N_MAX, 3)
            xyz_norm = F.normalize(xyz_pred, dim=-1)    # unit vectors (B, N_MAX, 3)
            # Gram matrix of cosine similarities: (B, N_MAX, N_MAX)
            gram = torch.bmm(xyz_norm, xyz_norm.transpose(1, 2))
            # Mean off-diagonal similarity (exclude self-similarity on diagonal).
            eye = torch.eye(gram.shape[1], device=gram.device).unsqueeze(0)
            off_diag = gram * (1.0 - eye)
            n_pairs = gram.shape[1] * (gram.shape[1] - 1)
            loss_div = off_diag.sum() / (gram.shape[0] * max(n_pairs, 1))
            total = total + w.slot_diversity * loss_div

        return LossOutput(
            total=total,
            pose_xyz=loss_xyz,
            pose_rot=loss_rot,
            scale=loss_scale,
            type_ce=loss_type,
            presence_bce=loss_pres,
            slot_diversity=loss_div,
        )
