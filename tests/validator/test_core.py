"""Tests for src/validator/core.py.

Tests are separated into fast (interpenetration-only) and slow (full pipeline)
categories. The slow tests run the full four-check pipeline on each fixture.

Run the full suite:
    pytest tests/validator/test_core.py -xvs

Run only fast tests:
    pytest tests/validator/test_core.py -xvs -m "not slow"
"""

from __future__ import annotations

import pytest

from scene.schema import WorkspaceBounds
from tests.validator.fixtures import (
    make_invalid_ik_blocked,
    make_invalid_interpenetrating,
    make_invalid_unstable,
    make_valid_sparse,
    make_valid_tall,
)
from validator.core import SceneValidator, ValidityReport


@pytest.fixture(scope="module")
def validator() -> SceneValidator:
    """Single SceneValidator instance shared across tests in this module."""
    return SceneValidator(workspace_bounds=WorkspaceBounds.default(), rrt_budget_s=5.0)


# ---------------------------------------------------------------------------
# Smoke: ValidityReport construction
# ---------------------------------------------------------------------------


class TestValidityReport:
    def test_default_accepted_false(self) -> None:
        report = ValidityReport()
        assert not report.accepted

    def test_accepted_true_when_all_checks_pass(self) -> None:
        report = ValidityReport(
            no_interpenetration=True,
            stable_rest=True,
            ik_reachable=True,
            rrt_solvable=True,
        )
        assert report.accepted

    def test_accepted_false_if_any_check_fails(self) -> None:
        report = ValidityReport(
            no_interpenetration=True,
            stable_rest=True,
            ik_reachable=True,
            rrt_solvable=False,
        )
        assert not report.accepted


# ---------------------------------------------------------------------------
# Fast: interpenetration check only
# ---------------------------------------------------------------------------


class TestInterpenetration:
    def test_separated_objects_pass(self, validator: SceneValidator) -> None:
        scene = make_valid_sparse()
        ok, min_dist = validator._check_interpenetration(scene)
        assert ok, f"Separated objects should pass interpenetration; min_dist={min_dist:.4f}m"
        assert min_dist > 0.0

    def test_overlapping_objects_fail(self, validator: SceneValidator) -> None:
        scene = make_invalid_interpenetrating()
        ok, min_dist = validator._check_interpenetration(scene)
        assert not ok, f"Overlapping objects must fail interpenetration; min_dist={min_dist:.4f}m"
        assert min_dist < 0.0, f"Expected negative distance for penetrating objects, got {min_dist}"


# ---------------------------------------------------------------------------
# Slow: full pipeline on valid fixtures
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestValidFixtures:
    def test_valid_sparse_passes_all_checks(self, validator: SceneValidator) -> None:
        scene = make_valid_sparse()
        report = validator.validate(scene, rrt_seed=0)
        assert report.no_interpenetration, (
            f"valid_sparse failed no_interpenetration; min_dist={report.min_signed_distance_m}"
        )
        assert report.stable_rest, (
            f"valid_sparse failed stable_rest; max_trans={report.max_pose_drift_trans_m:.4f}m"
        )
        assert report.ik_reachable, (
            f"valid_sparse failed ik_reachable; error={report.error}"
        )
        assert report.rrt_solvable, (
            f"valid_sparse failed rrt_solvable; error={report.error}"
        )
        assert report.accepted

    def test_valid_tall_passes_all_checks(self, validator: SceneValidator) -> None:
        scene = make_valid_tall()
        report = validator.validate(scene, rrt_seed=1)
        assert report.no_interpenetration
        assert report.stable_rest
        assert report.ik_reachable
        assert report.rrt_solvable
        assert report.accepted


# ---------------------------------------------------------------------------
# Slow: full pipeline on invalid fixtures
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestInvalidFixtures:
    def test_interpenetrating_fails_interp_check(
        self, validator: SceneValidator
    ) -> None:
        scene = make_invalid_interpenetrating()
        report = validator.validate(scene, rrt_seed=0)
        assert not report.no_interpenetration, (
            "Interpenetrating scene must fail no_interpenetration"
        )
        assert report.min_signed_distance_m < 0.0

    def test_unstable_fails_stable_rest(self, validator: SceneValidator) -> None:
        scene = make_invalid_unstable()
        report = validator.validate(scene, rrt_seed=0)
        # Interpenetration may or may not pass depending on exact geometry; the key
        # assertion is that stable_rest fails.
        assert not report.stable_rest, (
            f"Precariously stacked scene must fail stable_rest; "
            f"max_trans={report.max_pose_drift_trans_m:.4f}m, "
            f"max_rot={report.max_pose_drift_rot_rad:.4f}rad"
        )

    def test_ik_blocked_fails_ik(self, validator: SceneValidator) -> None:
        scene = make_invalid_ik_blocked()
        report = validator.validate(scene, rrt_seed=0)
        assert not report.ik_reachable, (
            "Scene with sub-floor goal must fail ik_reachable"
        )
        # When ik fails, rrt must also be False (BiRRT skipped).
        assert not report.rrt_solvable


# ---------------------------------------------------------------------------
# Cache consistency
# ---------------------------------------------------------------------------


class TestCache:
    def test_same_scene_returns_same_report(self, validator: SceneValidator) -> None:
        scene = make_valid_sparse()
        r1 = validator.validate(scene, rrt_seed=42)
        r2 = validator.validate(scene, rrt_seed=42)
        assert r1 is r2, "Cache must return the identical report object on second call"

    def test_batch_single_worker_matches_sequential(
        self, validator: SceneValidator
    ) -> None:
        scenes = [make_valid_sparse(), make_invalid_interpenetrating()]
        seeds = [0, 0]
        reports_seq = [validator.validate(s, rrt_seed=sd) for s, sd in zip(scenes, seeds)]
        reports_batch = validator.validate_batch(scenes, seeds=seeds, num_workers=1)
        for r_seq, r_batch in zip(reports_seq, reports_batch):
            assert r_seq.no_interpenetration == r_batch.no_interpenetration
            assert r_seq.accepted == r_batch.accepted
