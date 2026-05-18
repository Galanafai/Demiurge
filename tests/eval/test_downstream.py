"""Unit tests for src/eval/downstream.py.

Uses a mocked SceneValidator to avoid Drake process dependency.
Checks DownstreamResult field correctness and edge cases.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from scene.schema import N_MAX, SceneTensor


def _make_scene(n_present: int = 3) -> SceneTensor:
    poses = torch.zeros(N_MAX, 7)
    poses[:, 3] = 1.0  # unit quaternion
    presence = torch.zeros(N_MAX, dtype=torch.bool)
    presence[:n_present] = True
    return SceneTensor(
        object_types=torch.zeros(N_MAX, dtype=torch.int64),
        poses=poses,
        scales=torch.ones(N_MAX, 3),
        presence=presence,
    )


def _mock_report(
    accepted: bool,
    rrt_solvable: bool,
    rrt_plan_length: float = 0.0,
    ik_reachable: bool = True,
    no_interpenetration: bool = True,
    stable_rest: bool = True,
) -> MagicMock:
    r = MagicMock()
    r.accepted = accepted
    r.rrt_solvable = rrt_solvable
    r.rrt_plan_length = rrt_plan_length
    r.ik_reachable = ik_reachable
    r.no_interpenetration = no_interpenetration
    r.stable_rest = stable_rest
    return r


class TestDownstreamEvaluator:
    def test_all_success(self) -> None:
        from eval.downstream import DownstreamEvaluator

        reports = [_mock_report(True, True, rrt_plan_length=10.0) for _ in range(4)]
        mock_validator = MagicMock()
        mock_validator.validate.side_effect = reports

        scenes = [_make_scene() for _ in range(4)]
        prompts = ["task"] * 4

        evaluator = DownstreamEvaluator(mock_validator)
        result = evaluator.evaluate(scenes, prompts)

        assert result.success_rate == pytest.approx(1.0)
        assert result.mean_plan_length == pytest.approx(10.0)
        assert len(result.records) == 4

    def test_no_success(self) -> None:
        from eval.downstream import DownstreamEvaluator

        mock_validator = MagicMock()
        mock_validator.validate.return_value = _mock_report(False, False)

        scenes = [_make_scene() for _ in range(3)]
        result = DownstreamEvaluator(mock_validator).evaluate(scenes, ["t"] * 3)

        assert result.success_rate == pytest.approx(0.0)
        assert result.mean_plan_length == pytest.approx(0.0)

    def test_partial_success(self) -> None:
        from eval.downstream import DownstreamEvaluator

        mock_validator = MagicMock()
        mock_validator.validate.side_effect = [
            _mock_report(True, True, rrt_plan_length=8.0),
            _mock_report(False, False),
            _mock_report(True, True, rrt_plan_length=12.0),
        ]
        scenes = [_make_scene() for _ in range(3)]
        result = DownstreamEvaluator(mock_validator).evaluate(scenes, ["t"] * 3)

        assert result.success_rate == pytest.approx(2 / 3)
        assert result.mean_plan_length == pytest.approx(10.0)

    def test_empty_input(self) -> None:
        from eval.downstream import DownstreamEvaluator, DownstreamResult

        mock_validator = MagicMock()
        result = DownstreamEvaluator(mock_validator).evaluate([], [])

        assert isinstance(result, DownstreamResult)
        assert result.success_rate == pytest.approx(0.0)
        mock_validator.validate.assert_not_called()

    def test_per_task_records(self) -> None:
        from eval.downstream import DownstreamEvaluator

        mock_validator = MagicMock()
        mock_validator.validate.return_value = _mock_report(True, True, rrt_plan_length=5.0)

        scenes = [_make_scene(n_present=2)]
        result = DownstreamEvaluator(mock_validator).evaluate(scenes, ["pick the cube"])

        assert len(result.records) == 1
        rec = result.records[0]
        assert rec.prompt == "pick the cube"
        assert rec.rrt_solvable is True
        assert rec.rrt_plan_length == pytest.approx(5.0)
        assert rec.n_present == 2
