"""Tests for src/scene/schema.py."""

from __future__ import annotations

import math

import pytest
import torch

from scene.schema import N_MAX, SceneTensor, WorkspaceBounds
from scene.vocab import ObjectTypeId


def _make_random_scene(n_present: int = 4, seed: int = 0) -> SceneTensor:
    """Return a valid physical-space SceneTensor with n_present active objects."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    bounds = WorkspaceBounds.default()

    # Random xyz within workspace.
    xyz = torch.zeros(N_MAX, 3)
    xyz[:n_present, 0] = torch.rand(n_present, generator=rng) * (
        bounds.x_max - bounds.x_min
    ) + bounds.x_min
    xyz[:n_present, 1] = torch.rand(n_present, generator=rng) * (
        bounds.y_max - bounds.y_min
    ) + bounds.y_min
    xyz[:n_present, 2] = torch.rand(n_present, generator=rng) * (
        bounds.z_max - bounds.z_min
    ) + bounds.z_min

    # Random unit quaternions (wxyz) for present slots.
    raw_quats = torch.randn(n_present, 4, generator=rng)
    unit_quats = raw_quats / raw_quats.norm(dim=1, keepdim=True)
    quats = torch.zeros(N_MAX, 4)
    quats[:n_present] = unit_quats
    # Inactive slots get identity quaternion.
    quats[n_present:, 0] = 1.0

    poses = torch.cat([xyz, quats], dim=1).float()

    scales = torch.ones(N_MAX, 3, dtype=torch.float32)
    scales[:n_present] = (
        torch.rand(n_present, 3, generator=rng) * 0.4 + 0.8
    )  # uniform in [0.8, 1.2]

    type_ids = torch.zeros(N_MAX, dtype=torch.int64)
    type_ids[:n_present] = torch.randint(
        0, len(ObjectTypeId), (n_present,), generator=rng
    )

    presence = torch.zeros(N_MAX, dtype=torch.bool)
    presence[:n_present] = True

    return SceneTensor(
        object_types=type_ids,
        poses=poses,
        scales=scales,
        presence=presence,
    )


class TestShapeValidation:
    def test_valid_scene_constructs(self) -> None:
        scene = _make_random_scene()
        assert scene.n_present == 4

    def test_wrong_object_types_shape_raises(self) -> None:
        scene = _make_random_scene()
        with pytest.raises(ValueError, match="object_types"):
            SceneTensor(
                object_types=torch.zeros(5, dtype=torch.int64),
                poses=scene.poses,
                scales=scene.scales,
                presence=scene.presence,
            )

    def test_wrong_poses_shape_raises(self) -> None:
        scene = _make_random_scene()
        with pytest.raises(ValueError, match="poses"):
            SceneTensor(
                object_types=scene.object_types,
                poses=torch.zeros(N_MAX, 6, dtype=torch.float32),
                scales=scene.scales,
                presence=scene.presence,
            )

    def test_wrong_dtype_raises(self) -> None:
        scene = _make_random_scene()
        with pytest.raises(ValueError, match="poses must be float32"):
            SceneTensor(
                object_types=scene.object_types,
                poses=scene.poses.double(),
                scales=scene.scales,
                presence=scene.presence,
            )


class TestNormalizeDenormalize:
    def test_roundtrip_max_error_below_threshold(self) -> None:
        """Normalize then denormalize must recover original values within 1e-5."""
        bounds = WorkspaceBounds.default()
        scene = _make_random_scene(n_present=6, seed=42)
        recovered = scene.normalize(bounds).denormalize(bounds)

        # XYZ positions.
        xyz_err = (scene.poses[:, 0:3] - recovered.poses[:, 0:3]).abs().max().item()
        assert xyz_err < 1e-5, f"xyz round-trip error {xyz_err:.2e} >= 1e-5"

        # Quaternions (compare only present slots to avoid inactive-slot drift).
        mask = scene.presence
        quat_err = (
            (scene.poses[mask, 3:7] - recovered.poses[mask, 3:7]).abs().max().item()
        )
        assert quat_err < 1e-5, f"quat round-trip error {quat_err:.2e} >= 1e-5"

        # Scales (present slots only).
        scale_err = (scene.scales[mask] - recovered.scales[mask]).abs().max().item()
        assert scale_err < 1e-5, f"scale round-trip error {scale_err:.2e} >= 1e-5"

    def test_normalized_xyz_in_unit_range(self) -> None:
        bounds = WorkspaceBounds.default()
        scene = _make_random_scene(n_present=8, seed=7)
        normed = scene.normalize(bounds)
        xyz = normed.poses[scene.presence, 0:3]
        assert xyz.min().item() >= -1.0 - 1e-6
        assert xyz.max().item() <= 1.0 + 1e-6

    def test_normalized_scales_near_zero_for_unit_scale(self) -> None:
        bounds = WorkspaceBounds.default()
        scene = _make_random_scene()
        # Set all scales to canonical 1.0.
        unit_scales = torch.ones(N_MAX, 3, dtype=torch.float32)
        unit_scene = SceneTensor(
            object_types=scene.object_types,
            poses=scene.poses,
            scales=unit_scales,
            presence=scene.presence,
        )
        normed = unit_scene.normalize(bounds)
        scale_err = normed.scales[scene.presence].abs().max().item()
        assert scale_err < 1e-6, f"Unit scale should map to 0.0, got max={scale_err:.2e}"


class TestQuaternionProjection:
    def test_projection_yields_unit_norm(self) -> None:
        scene = _make_random_scene(n_present=8, seed=99)
        # Corrupt quaternion norms.
        bad_poses = scene.poses.clone()
        bad_poses[:, 3:7] *= 2.5
        bad_scene = SceneTensor(
            object_types=scene.object_types,
            poses=bad_poses,
            scales=scene.scales,
            presence=scene.presence,
        )
        projected = bad_scene.project_quaternions()
        quat_norms = projected.poses[scene.presence, 3:7].norm(dim=1)
        max_err = (quat_norms - 1.0).abs().max().item()
        assert max_err < 1e-6, f"Projected quaternion norms deviate from 1: max_err={max_err:.2e}"

    def test_projection_preserves_rotation(self) -> None:
        """Scaling a quaternion should not change the represented rotation."""
        scene = _make_random_scene(n_present=4, seed=3)
        scaled_poses = scene.poses.clone()
        scaled_poses[:, 3:7] *= 3.7
        scaled_scene = SceneTensor(
            object_types=scene.object_types,
            poses=scaled_poses,
            scales=scene.scales,
            presence=scene.presence,
        )
        projected = scaled_scene.project_quaternions()
        original_quats = scene.poses[scene.presence, 3:7]
        projected_quats = projected.poses[scene.presence, 3:7]
        # Dot product of unit quaternions should be close to 1.0 (or -1.0 for antipodal).
        dots = (original_quats * projected_quats).sum(dim=1).abs()
        min_dot = dots.min().item()
        assert min_dot > 1.0 - 1e-5, f"Rotation changed after projection: min |dot|={min_dot:.6f}"

    def test_projection_skips_near_zero_norm(self) -> None:
        """A quaternion with near-zero norm must not produce NaN."""
        scene = SceneTensor.empty()
        bad_poses = scene.poses.clone()
        # Set presence[0] = True with zero quaternion.
        presence = scene.presence.clone()
        presence[0] = True
        bad_poses[0, 3] = 1e-9  # near-zero norm
        near_zero_scene = SceneTensor(
            object_types=scene.object_types,
            poses=bad_poses,
            scales=scene.scales,
            presence=presence,
        )
        projected = near_zero_scene.project_quaternions()
        assert not projected.poses.isnan().any(), "project_quaternions produced NaN"


class TestToSdf:
    def _parse_in_drake(self, sdf: str) -> int:
        """Return number of non-world bodies parsed from the SDF."""
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
        from pydrake.systems.framework import DiagramBuilder

        builder = DiagramBuilder()
        plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        parser = Parser(plant)
        parser.AddModelsFromString(sdf, "sdf")
        plant.Finalize()
        return int(plant.num_bodies()) - 1  # subtract world body

    def test_sdf_parses_for_partial_presence(self) -> None:
        scene = _make_random_scene(n_present=3, seed=11)
        sdf = scene.to_sdf()
        n_bodies = self._parse_in_drake(sdf)
        assert n_bodies == 3, f"Expected 3 bodies, got {n_bodies}"

    def test_sdf_parses_for_full_scene(self) -> None:
        scene = _make_random_scene(n_present=8, seed=22)
        sdf = scene.to_sdf()
        n_bodies = self._parse_in_drake(sdf)
        assert n_bodies == 8

    def test_presence_mask_excludes_inactive_objects(self) -> None:
        scene = _make_random_scene(n_present=2, seed=33)
        sdf = scene.to_sdf()
        n_bodies = self._parse_in_drake(sdf)
        assert n_bodies == 2, (
            f"Inactive objects must not appear in SDF; expected 2, got {n_bodies}"
        )

    def test_to_sdf_raises_on_empty_scene(self) -> None:
        empty = SceneTensor.empty()
        with pytest.raises(ValueError, match="no present objects"):
            empty.to_sdf()

    def test_rpy_conversion_identity_quaternion(self) -> None:
        """Identity quaternion (w=1,x=y=z=0) must yield zero roll-pitch-yaw."""
        from scene.schema import _quat_wxyz_to_rpy

        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        rpy = _quat_wxyz_to_rpy(q)
        assert rpy.abs().max().item() < 1e-6, f"Identity quat -> non-zero rpy: {rpy}"

    def test_rpy_conversion_90deg_yaw(self) -> None:
        """Quaternion for 90-deg yaw: w=cos(pi/4), z=sin(pi/4)."""
        from scene.schema import _quat_wxyz_to_rpy

        angle = math.pi / 2
        q = torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        rpy = _quat_wxyz_to_rpy(q)
        assert abs(rpy[2].item() - angle) < 1e-5, f"Expected yaw={angle:.4f}, got {rpy[2].item():.4f}"
