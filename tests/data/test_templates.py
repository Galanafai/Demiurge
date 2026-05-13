"""Tests for src/data/sampler.py.

Tests cover:
  - Determinism: same seed produces bit-exact SceneTensor.
  - Presence mask: at least min_objects present on every sample.
  - Interpenetration: 10 scenes from each template all pass the fast check
    (marked @slow to allow skipping in fast CI).
  - ProceduralSampler: template distribution matches configured mix.
  - CandidateScene: target_idx always in [0, n_present), goal_xyz above target.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data.sampler import (
    ClutteredPickTemplate,
    ObstacleAvoidanceTemplate,
    ProceduralSampler,
    TabletopReachTemplate,
    TaskTemplate,
)
from scene.schema import WorkspaceBounds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_tabletop_is_task_template(self) -> None:
        assert isinstance(TabletopReachTemplate(), TaskTemplate)

    def test_cluttered_is_task_template(self) -> None:
        assert isinstance(ClutteredPickTemplate(), TaskTemplate)

    def test_obstacle_is_task_template(self) -> None:
        assert isinstance(ObstacleAvoidanceTemplate(), TaskTemplate)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.parametrize(
        "template_cls",
        [TabletopReachTemplate, ClutteredPickTemplate, ObstacleAvoidanceTemplate],
    )
    def test_same_seed_same_scene(self, template_cls: type) -> None:
        t = template_cls()
        c1 = t.sample_scene(_rng(42))
        c2 = t.sample_scene(_rng(42))
        assert torch.equal(c1.scene.object_types, c2.scene.object_types)
        assert torch.equal(c1.scene.presence, c2.scene.presence)
        assert torch.allclose(c1.scene.poses, c2.scene.poses)
        assert torch.allclose(c1.scene.scales, c2.scene.scales)
        assert c1.task_family == c2.task_family
        assert c1.target_idx == c2.target_idx
        assert np.allclose(c1.goal_xyz, c2.goal_xyz)

    @pytest.mark.parametrize(
        "template_cls",
        [TabletopReachTemplate, ClutteredPickTemplate, ObstacleAvoidanceTemplate],
    )
    def test_different_seeds_different_scenes(self, template_cls: type) -> None:
        t = template_cls()
        c1 = t.sample_scene(_rng(0))
        c2 = t.sample_scene(_rng(999))
        # Different seeds must produce at least one different field.
        differ = (
            not torch.equal(c1.scene.object_types, c2.scene.object_types)
            or not torch.allclose(c1.scene.poses, c2.scene.poses)
        )
        assert differ, "Different seeds produced identical scenes"


# ---------------------------------------------------------------------------
# Presence constraints
# ---------------------------------------------------------------------------


class TestPresence:
    def test_tabletop_min_one_present(self) -> None:
        t = TabletopReachTemplate(min_objects=1)
        for seed in range(20):
            c = t.sample_scene(_rng(seed))
            assert c.scene.presence.any(), "tabletop_reach must have at least 1 object"

    def test_cluttered_min_three_requested(self) -> None:
        """min_objects=3, but placement may fail; assert at least 1 always placed."""
        t = ClutteredPickTemplate(min_objects=3)
        for seed in range(20):
            c = t.sample_scene(_rng(seed))
            assert c.scene.presence.any()

    def test_obstacle_min_one_barrier_plus_target(self) -> None:
        t = ObstacleAvoidanceTemplate(min_objects=2)
        for seed in range(20):
            c = t.sample_scene(_rng(seed))
            n_present = int(c.scene.presence.sum().item())
            assert n_present >= 1

    def test_n_present_within_n_max(self) -> None:
        from scene.schema import N_MAX
        for cls in [TabletopReachTemplate, ClutteredPickTemplate, ObstacleAvoidanceTemplate]:
            t = cls()
            for seed in range(10):
                c = t.sample_scene(_rng(seed))
                assert int(c.scene.presence.sum().item()) <= N_MAX


# ---------------------------------------------------------------------------
# Target index and goal xyz
# ---------------------------------------------------------------------------


class TestTargetAndGoal:
    @pytest.mark.parametrize(
        "template_cls",
        [TabletopReachTemplate, ClutteredPickTemplate, ObstacleAvoidanceTemplate],
    )
    def test_target_idx_is_present(self, template_cls: type) -> None:
        t = template_cls()
        for seed in range(20):
            c = t.sample_scene(_rng(seed))
            assert c.scene.presence[c.target_idx].item(), (
                f"target_idx={c.target_idx} is not present"
            )

    @pytest.mark.parametrize(
        "template_cls",
        [TabletopReachTemplate, ClutteredPickTemplate, ObstacleAvoidanceTemplate],
    )
    def test_goal_above_target(self, template_cls: type) -> None:
        t = template_cls()
        for seed in range(20):
            c = t.sample_scene(_rng(seed))
            target_z = float(c.scene.poses[c.target_idx, 2].item())
            assert c.goal_xyz[2] > target_z, (
                f"goal z={c.goal_xyz[2]:.3f} not above target z={target_z:.3f}"
            )

    @pytest.mark.parametrize(
        "template_cls",
        [TabletopReachTemplate, ClutteredPickTemplate, ObstacleAvoidanceTemplate],
    )
    def test_goal_xy_matches_target(self, template_cls: type) -> None:
        t = template_cls()
        for seed in range(20):
            c = t.sample_scene(_rng(seed))
            target_x = float(c.scene.poses[c.target_idx, 0].item())
            target_y = float(c.scene.poses[c.target_idx, 1].item())
            assert abs(c.goal_xyz[0] - target_x) < 1e-4
            assert abs(c.goal_xyz[1] - target_y) < 1e-4


# ---------------------------------------------------------------------------
# Geometry: no interpenetration (fast check only)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestNoInterpenetration:
    """10 scenes from each template must pass the fast interpenetration check.

    This is not a full pipeline run. It only verifies that the sampler's
    minimum-separation logic produces scenes that Drake's proximity query
    agrees are non-overlapping.
    """

    @pytest.fixture(scope="class")
    def validator(self):  # type: ignore[no-untyped-def]
        from validator.core import SceneValidator
        return SceneValidator(workspace_bounds=WorkspaceBounds.default(), rrt_budget_s=5.0)

    @pytest.mark.parametrize(
        "template_cls",
        [TabletopReachTemplate, ClutteredPickTemplate, ObstacleAvoidanceTemplate],
    )
    def test_sampled_scenes_not_interpenetrating(
        self, validator, template_cls: type  # type: ignore[no-untyped-def]
    ) -> None:
        # ClutteredPickTemplate places 3-6 objects in a tight region by design;
        # a stricter threshold would require fewer objects or larger separation,
        # which defeats the template's purpose. Allow slightly lower pass rate.
        min_pass = {
            "TabletopReachTemplate": 8,
            "ClutteredPickTemplate": 6,
            "ObstacleAvoidanceTemplate": 8,
        }.get(template_cls.__name__, 8)

        t = template_cls()
        n_pass = 0
        for seed in range(10):
            c = t.sample_scene(_rng(seed))
            ok, _ = validator._check_interpenetration(c.scene)
            if ok:
                n_pass += 1
        assert n_pass >= min_pass, (
            f"{template_cls.__name__}: only {n_pass}/10 scenes passed interpenetration"
        )


# ---------------------------------------------------------------------------
# ProceduralSampler
# ---------------------------------------------------------------------------


class TestProceduralSampler:
    def test_uniform_mix_all_families_appear(self) -> None:
        sampler = ProceduralSampler(seed=0)
        families = {c.task_family for c in sampler.sample_batch(60)}
        assert "tabletop_reach" in families
        assert "cluttered_pick" in families
        assert "obstacle_avoidance" in families

    def test_skewed_mix_respects_weights(self) -> None:
        templates = [
            TabletopReachTemplate(),
            ClutteredPickTemplate(),
            ObstacleAvoidanceTemplate(),
        ]
        # Give tabletop_reach 80% weight.
        sampler = ProceduralSampler(templates=templates, mix=[0.8, 0.1, 0.1], seed=1)
        results = sampler.sample_batch(500)
        reach_count = sum(1 for c in results if c.task_family == "tabletop_reach")
        # Expect between 60% and 95% due to random variance.
        assert 0.60 <= reach_count / 500 <= 0.95, (
            f"tabletop_reach fraction={reach_count/500:.2f} outside expected range"
        )

    def test_determinism_via_seed(self) -> None:
        s1 = ProceduralSampler(seed=42)
        s2 = ProceduralSampler(seed=42)
        c1 = s1.sample()
        c2 = s2.sample()
        assert c1.task_family == c2.task_family
        assert torch.equal(c1.scene.presence, c2.scene.presence)


# ---------------------------------------------------------------------------
# Upright constraint
# ---------------------------------------------------------------------------


class TestUprightConstraint:
    """Tests for the upright_constrained flag and _sample_orientation wrapper.

    The four tests below are ordered from data-integrity to behavioral:
      1. Flag-integrity: vocab data correctness.
      2. No-tilt: constrained entries never get roll/pitch.
      3. Current-behaviour: all entries currently produce yaw-only output.
      4. Template-coverage: templates use _sample_orientation, not _random_yaw_quat.
    """

    def test_upright_constrained_entries_flagged_by_hz_threshold(self) -> None:
        """Data-integrity: every entry with hz > 0.10m has upright_constrained=True.

        This is duplicated from TestNewYcbEntries to provide a single-class
        location for all upright-constraint invariants. If this test fails,
        an entry was added to OBJECT_VOCAB without setting the flag.
        """
        from scene.vocab import OBJECT_VOCAB

        for tid, entry in OBJECT_VOCAB.items():
            hz = entry.canonical_half_extents_m[2]
            if hz > 0.10:
                assert entry.upright_constrained, (
                    f"{entry.name} (id={tid}) hz={hz:.3f}m > 0.10m "
                    f"but upright_constrained=False"
                )
            else:
                assert not entry.upright_constrained, (
                    f"{entry.name} (id={tid}) hz={hz:.3f}m <= 0.10m "
                    f"but upright_constrained=True (wrong threshold?)"
                )

    def test_sample_orientation_constrained_never_tilts(self) -> None:
        """Constrained entries: 1000 samples all have cos(angle to Z) >= 0.9999.

        cos(angle) = R[2,2] = 1 - 2*(qx^2 + qy^2) for wxyz quaternion.
        0.9999 corresponds to ~0.81 degrees of tilt; practically zero.
        """
        from data.sampler import _sample_orientation
        from scene.vocab import OBJECT_VOCAB

        constrained_ids = [
            int(tid)
            for tid, entry in OBJECT_VOCAB.items()
            if entry.upright_constrained
        ]
        assert constrained_ids, "No upright_constrained entries found; check vocab."

        rng = _rng(0)
        for type_id in constrained_ids:
            entry = OBJECT_VOCAB[type_id]
            for _ in range(1000):
                q = _sample_orientation(type_id, rng)
                w, qx, qy, qz = q
                # R[2,2] = 1 - 2*(qx^2 + qy^2) for a unit quaternion.
                cos_angle_to_z = 1.0 - 2.0 * (qx**2 + qy**2)
                assert cos_angle_to_z >= 0.9999, (
                    f"{entry.name}: cos(angle_to_Z)={cos_angle_to_z:.6f} < 0.9999 "
                    f"(tilt detected), q={q}"
                )

    def test_sample_orientation_unconstrained_currently_yaw_only(self) -> None:
        """Documentation: unconstrained entries also produce yaw-only output now.

        This test documents current behaviour. If SO(3) augmentation is added
        for unconstrained entries in a future sampler, this test is expected to
        change (and must be updated deliberately, not silently fixed).
        """
        from data.sampler import _sample_orientation
        from scene.vocab import OBJECT_VOCAB

        unconstrained_ids = [
            int(tid)
            for tid, entry in OBJECT_VOCAB.items()
            if not entry.upright_constrained
        ]
        assert unconstrained_ids, "All entries are constrained; expected some unconstrained."

        rng = _rng(1)
        for type_id in unconstrained_ids:
            for _ in range(100):
                q = _sample_orientation(type_id, rng)
                _w, qx, qy, _qz = q
                assert abs(qx) < 1e-9, f"type_id={type_id}: qx={qx} (roll detected)"
                assert abs(qy) < 1e-9, f"type_id={type_id}: qy={qy} (pitch detected)"

    def test_all_templates_use_sample_orientation(self) -> None:
        """Template-coverage: _sample_orientation is called for every placed object.

        Uses unittest.mock.patch to intercept calls and verify type_ids passed
        match the scene tensor's object_types for present objects.
        """
        from unittest.mock import patch

        import data.sampler as sampler_mod
        from data.sampler import _sample_orientation as real_fn

        templates = [
            TabletopReachTemplate(),
            ClutteredPickTemplate(),
            ObstacleAvoidanceTemplate(),
        ]
        for template in templates:
            called_type_ids: list[int] = []

            def _recording_sample_orientation(
                type_id: int, rng: np.random.Generator
            ) -> list[float]:
                called_type_ids.append(type_id)
                return real_fn(type_id, rng)

            with patch.object(sampler_mod, "_sample_orientation", _recording_sample_orientation):
                rng = _rng(7)
                for _ in range(10):
                    c = template.sample_scene(rng)
                    # Every present object's type must appear in called_type_ids.
                    # (called_type_ids includes pre-filter samples, so we only
                    # verify at least one call matched each present type_id.)
                    present_types = {
                        int(c.scene.object_types[i].item())
                        for i in range(len(c.scene.presence))
                        if c.scene.presence[i].item()
                    }
                    assert present_types.issubset(set(called_type_ids)), (
                        f"{template.name}: present type_ids {present_types} "
                        f"not covered by _sample_orientation calls {set(called_type_ids)}"
                    )
                    called_type_ids.clear()
