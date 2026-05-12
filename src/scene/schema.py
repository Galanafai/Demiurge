"""Scene tensor schema for the Demiurge diffusion model.

SceneTensor is the fixed-length representation of a tabletop scene. It holds up to N_MAX=16
objects. Inactive slots are masked by the presence field. All numerical fields are stored
as float32 tensors. The dataclass is frozen to prevent accidental in-place mutation; all
transformations return new instances.

Normalization maps the raw physical values into a unit-scaled latent space suitable for
diffusion model training:
  - xyz positions: linearly mapped to [-1, 1] using workspace_bounds.
  - quaternions (wxyz): kept on the 4-sphere via L2 normalization.
  - scales: mapped as (scale - 1.0) / 0.5, so scale=1.0 maps to 0.0 and
    scale in [0.5, 1.5] maps to [-1, 1].
  - object_types and presence: unchanged (integer and bool, not passed to denoiser).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

N_MAX: int = 16
"""Maximum number of objects in a scene. Fixed at 16 per Week 2.5 vocab expansion."""

QUAT_NORM_TOL: float = 1e-6
"""Minimum quaternion norm below which re-projection is skipped to avoid NaN."""


class WorkspaceBounds(NamedTuple):
    """Axis-aligned workspace bounding box in metres.

    All coordinates are in the robot base frame. The z-floor is the table surface.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @classmethod
    def default(cls) -> WorkspaceBounds:
        """Default 1m x 1m x 0.5m workspace centred at the robot base."""
        return cls(x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z_min=0.0, z_max=0.5)


@dataclass(frozen=True)
class SceneTensor:
    """Fixed-length tensor representation of a tabletop scene.

    All tensors have leading dimension N_MAX (=8). Inactive slots are zeroed and
    gated by the presence mask.

    Attributes:
        object_types: Long tensor [N_MAX] of ObjectTypeId values (0-15). Inactive slots
            contain 0 (CUBE) but are ignored when presence is False.
        poses: Float tensor [N_MAX, 7]. Columns 0:3 are xyz position in metres;
            columns 3:7 are quaternion in wxyz convention. Quaternions in present
            slots must have unit norm within QUAT_NORM_TOL.
        scales: Float tensor [N_MAX, 3]. Uniform scale factors per axis. In the
            physical (denormalized) space, values near 1.0 mean canonical size.
        presence: Bool tensor [N_MAX]. True for active objects.
    """

    object_types: torch.Tensor  # shape (N_MAX,) dtype=torch.int64
    poses: torch.Tensor  # shape (N_MAX, 7) dtype=torch.float32
    scales: torch.Tensor  # shape (N_MAX, 3) dtype=torch.float32
    presence: torch.Tensor  # shape (N_MAX,) dtype=torch.bool

    def __post_init__(self) -> None:
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if self.object_types.shape != (N_MAX,):
            raise ValueError(
                f"object_types must have shape ({N_MAX},), got {tuple(self.object_types.shape)}"
            )
        if self.poses.shape != (N_MAX, 7):
            raise ValueError(
                f"poses must have shape ({N_MAX}, 7), got {tuple(self.poses.shape)}"
            )
        if self.scales.shape != (N_MAX, 3):
            raise ValueError(
                f"scales must have shape ({N_MAX}, 3), got {tuple(self.scales.shape)}"
            )
        if self.presence.shape != (N_MAX,):
            raise ValueError(
                f"presence must have shape ({N_MAX},), got {tuple(self.presence.shape)}"
            )
        if self.object_types.dtype != torch.int64:
            raise ValueError(
                f"object_types must be int64, got {self.object_types.dtype}"
            )
        if self.poses.dtype != torch.float32:
            raise ValueError(f"poses must be float32, got {self.poses.dtype}")
        if self.scales.dtype != torch.float32:
            raise ValueError(f"scales must be float32, got {self.scales.dtype}")
        if self.presence.dtype != torch.bool:
            raise ValueError(f"presence must be bool, got {self.presence.dtype}")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls) -> SceneTensor:
        """Return a scene with no active objects (all-zeros, all-False presence)."""
        return cls(
            object_types=torch.zeros(N_MAX, dtype=torch.int64),
            poses=torch.zeros(N_MAX, 7, dtype=torch.float32),
            scales=torch.ones(N_MAX, 3, dtype=torch.float32),
            presence=torch.zeros(N_MAX, dtype=torch.bool),
        )

    # ------------------------------------------------------------------
    # Quaternion utilities
    # ------------------------------------------------------------------

    def project_quaternions(self) -> SceneTensor:
        """Return a new SceneTensor with all quaternion rows normalized to unit L2 norm.

        Only active (present) slots are normalized; inactive slots are left as-is.
        Slots with quaternion norm below QUAT_NORM_TOL are left unchanged to avoid NaN.
        """
        quats = self.poses[:, 3:7].clone()  # (N_MAX, 4)
        norms = quats.norm(dim=1, keepdim=True)  # (N_MAX, 1)
        safe_mask = (norms.squeeze(1) > QUAT_NORM_TOL) & self.presence
        quats[safe_mask] = quats[safe_mask] / norms[safe_mask]
        new_poses = self.poses.clone()
        new_poses[:, 3:7] = quats
        return SceneTensor(
            object_types=self.object_types,
            poses=new_poses,
            scales=self.scales,
            presence=self.presence,
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(self, bounds: WorkspaceBounds) -> SceneTensor:
        """Map physical values into normalized latent space.

        xyz: linearly mapped to [-1, 1] using bounds.
        quaternions: L2 normalized (project_quaternions).
        scales: (scale - 1.0) / 0.5, so scale=1.0 -> 0.0, scale in [0.5, 1.5] -> [-1, 1].

        Returns a new SceneTensor. Does not modify in place.
        """
        lower = torch.tensor(
            [bounds.x_min, bounds.y_min, bounds.z_min], dtype=torch.float32
        )
        upper = torch.tensor(
            [bounds.x_max, bounds.y_max, bounds.z_max], dtype=torch.float32
        )
        extent = upper - lower  # (3,)

        new_poses = self.poses.clone()
        # Normalize xyz positions to [-1, 1].
        new_poses[:, 0:3] = 2.0 * (self.poses[:, 0:3] - lower) / extent - 1.0
        # Quaternions: pass through project_quaternions for normalization.
        # (Already done on the intermediate tensor; just update xyz here.)

        # Normalize scales.
        new_scales = (self.scales - 1.0) / 0.5

        intermediate = SceneTensor(
            object_types=self.object_types,
            poses=new_poses,
            scales=new_scales,
            presence=self.presence,
        )
        return intermediate.project_quaternions()

    def denormalize(self, bounds: WorkspaceBounds) -> SceneTensor:
        """Invert normalize(). Maps latent space back to physical units.

        xyz: mapped from [-1, 1] back to workspace bounds.
        scales: (normalized_scale * 0.5) + 1.0.
        quaternions: project_quaternions (idempotent for unit quaternions).
        """
        lower = torch.tensor(
            [bounds.x_min, bounds.y_min, bounds.z_min], dtype=torch.float32
        )
        upper = torch.tensor(
            [bounds.x_max, bounds.y_max, bounds.z_max], dtype=torch.float32
        )
        extent = upper - lower

        new_poses = self.poses.clone()
        new_poses[:, 0:3] = (self.poses[:, 0:3] + 1.0) / 2.0 * extent + lower

        new_scales = self.scales * 0.5 + 1.0

        intermediate = SceneTensor(
            object_types=self.object_types,
            poses=new_poses,
            scales=new_scales,
            presence=self.presence,
        )
        return intermediate.project_quaternions()

    # ------------------------------------------------------------------
    # SDF generation
    # ------------------------------------------------------------------

    def to_sdf(self, bounds: WorkspaceBounds | None = None) -> str:
        """Generate a multi-model SDF string for all present objects.

        The SDF contains one <model> per active object, each placed at its pose
        using a <pose> element (xyz rpy). Quaternion poses are converted to
        roll-pitch-yaw for the SDF pose element.

        This method operates on the physical (denormalized) SceneTensor. If the
        tensor is in normalized space, call denormalize() first.

        Args:
            bounds: Unused parameter kept for API symmetry with normalize/denormalize.
                Pass None or omit.

        Returns:
            SDF XML string containing a <sdf> root with one <model> per active object.
        """
        from scene.vocab import OBJECT_VOCAB, build_sdf

        models: list[str] = []
        for i in range(N_MAX):
            if not self.presence[i].item():
                continue

            type_id = int(self.object_types[i].item())
            entry = OBJECT_VOCAB[type_id]
            # Use the mean of x-scale as a scalar; SceneTensor stores (sx, sy, sz).
            scale = float(self.scales[i].mean().item())
            model_name = f"obj_{i}_{entry.name}"
            sdf_model = build_sdf(entry, model_name=model_name, scale=scale)

            # Extract pose: xyz in metres, quaternion wxyz -> rpy for SDF.
            xyz = self.poses[i, 0:3]
            quat_wxyz = self.poses[i, 3:7]
            rpy = _quat_wxyz_to_rpy(quat_wxyz)

            pose_str = (
                f"{xyz[0].item():.6f} {xyz[1].item():.6f} {xyz[2].item():.6f} "
                f"{rpy[0].item():.6f} {rpy[1].item():.6f} {rpy[2].item():.6f}"
            )

            # Inject a <pose> element into the model SDF.
            sdf_model = sdf_model.replace(
                f'<model name="{model_name}">',
                f'<model name="{model_name}">\n    <pose>{pose_str}</pose>',
            )
            models.append(sdf_model)

        if not models:
            raise ValueError("to_sdf() called on a SceneTensor with no present objects.")

        # Drake's AddModelsFromString accepts a world-wrapped SDF for multi-model loading.
        # Strip the per-model XML declaration and <sdf> wrapper before embedding.
        def _strip_sdf_wrapper(sdf: str) -> str:
            import re

            sdf = re.sub(r"<\?xml[^?]*\?>\s*", "", sdf)
            sdf = re.sub(r"<sdf[^>]*>", "", sdf)
            sdf = re.sub(r"</sdf>", "", sdf)
            return sdf.strip()

        inner = "\n".join(_strip_sdf_wrapper(m) for m in models)
        return (
            '<?xml version="1.0"?>\n'
            '<sdf version="1.7">\n'
            '  <world name="scene">\n'
            f"{inner}\n"
            "  </world>\n"
            "</sdf>"
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_present(self) -> int:
        """Number of active objects in this scene."""
        return int(self.presence.sum().item())

    def __repr__(self) -> str:
        return (
            f"SceneTensor(n_present={self.n_present}, "
            f"n_max={N_MAX}, "
            f"dtype=float32)"
        )


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _quat_wxyz_to_rpy(q: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion (wxyz) to roll-pitch-yaw (radians) for SDF pose.

    Args:
        q: Float tensor of shape (4,) in wxyz order.

    Returns:
        Float tensor of shape (3,) with [roll, pitch, yaw] in radians.
    """
    import math

    w, x, y, z = q[0].item(), q[1].item(), q[2].item(), q[3].item()

    # Roll (rotation about x-axis).
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation about y-axis).
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))  # clamp for numerical safety
    pitch = math.asin(sinp)

    # Yaw (rotation about z-axis).
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return torch.tensor([roll, pitch, yaw], dtype=torch.float32)
