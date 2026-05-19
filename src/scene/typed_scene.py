"""Typed wrappers to distinguish normalized/physical/whitened scenes.

Prevents the class of bugs where model output (whitened) is passed directly
to Drake (which expects physical units) without the decode chain.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scene.schema import SceneTensor, WorkspaceBounds


@dataclass(frozen=True)
class PhysicalScene:
    """Scene in physical units. Input to SceneValidator and Drake."""
    inner: "SceneTensor"
    def normalize(self, bounds: "WorkspaceBounds") -> "NormalizedScene":
        return NormalizedScene(self.inner.normalize(bounds))
    def validate(self, validator, rrt_seed: int = 0):
        return validator.validate(self.inner, rrt_seed=rrt_seed)


@dataclass(frozen=True)
class NormalizedScene:
    """Scene range-normalized to [-1, 1] per dimension."""
    inner: "SceneTensor"
    def denormalize(self, bounds: "WorkspaceBounds") -> PhysicalScene:
        return PhysicalScene(self.inner.denormalize(bounds))
    def whiten(self, mean_xyz, std_xyz, mean_scale, std_scale) -> "WhitenedScene":
        from scene.schema import SceneTensor
        s = self.inner
        p = s.poses.clone().float()
        p[:, :3] = (p[:, :3] - mean_xyz.to(p)) / std_xyz.to(p)
        sc = (s.scales.float() - mean_scale.to(s.scales)) / std_scale.to(s.scales)
        return WhitenedScene(SceneTensor(
            object_types=s.object_types, poses=p, scales=sc, presence=s.presence))


@dataclass(frozen=True)
class WhitenedScene:
    """Scene whitened to zero-mean unit-variance. Model input/output space."""
    inner: "SceneTensor"
    def unwhiten(self, mean_xyz, std_xyz, mean_scale, std_scale) -> NormalizedScene:
        from scene.schema import SceneTensor
        s = self.inner
        p = s.poses.clone().float()
        p[:, :3] = (p[:, :3] * std_xyz.to(p) + mean_xyz.to(p)).clamp(-1.0, 1.0)
        sc = (s.scales.float() * std_scale.to(s.scales) + mean_scale.to(s.scales)).clamp(-1.0, 1.0)
        return NormalizedScene(SceneTensor(
            object_types=s.object_types, poses=p, scales=sc, presence=s.presence))


def whitened_to_physical(whitened: WhitenedScene, bounds: "WorkspaceBounds",
                         mean_xyz, std_xyz, mean_scale, std_scale) -> PhysicalScene:
    """Single canonical decode path: whitened -> normalized -> physical.
    The ONLY function that should convert model output to validator input."""
    return whitened.unwhiten(mean_xyz, std_xyz, mean_scale, std_scale).denormalize(bounds)
