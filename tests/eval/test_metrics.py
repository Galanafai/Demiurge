"""Unit tests for src/eval/metrics.py.

Tests raw_validity_rate and diversity with toy scenes.
prompt_following is tested with a mocked SceneJudge.
downstream_success delegates to DownstreamEvaluator tested separately.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from scene.schema import N_MAX, SceneTensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_scene(n_present: int, xyz_base: float = 0.0) -> SceneTensor:
    """Minimal valid SceneTensor with n_present active slots."""
    poses = torch.zeros(N_MAX, 7)
    poses[:, 3] = 1.0  # unit quaternion w=1
    poses[:n_present, 0] = xyz_base + torch.arange(n_present, dtype=torch.float32) * 0.2

    presence = torch.zeros(N_MAX, dtype=torch.bool)
    presence[:n_present] = True

    return SceneTensor(
        object_types=torch.zeros(N_MAX, dtype=torch.int64),
        poses=poses,
        scales=torch.ones(N_MAX, 3),
        presence=presence,
    )


# ---------------------------------------------------------------------------
# raw_validity_rate
# ---------------------------------------------------------------------------


class TestRawValidityRate:
    def test_all_accepted(self) -> None:
        from eval.metrics import raw_validity_rate

        mock_validator = MagicMock()
        mock_validator.validate.return_value = MagicMock(accepted=True)
        scenes = [_make_scene(3) for _ in range(10)]
        rate = raw_validity_rate(scenes, mock_validator)
        assert rate == pytest.approx(1.0)
        assert mock_validator.validate.call_count == 10

    def test_none_accepted(self) -> None:
        from eval.metrics import raw_validity_rate

        mock_validator = MagicMock()
        mock_validator.validate.return_value = MagicMock(accepted=False)
        scenes = [_make_scene(3) for _ in range(5)]
        assert raw_validity_rate(scenes, mock_validator) == pytest.approx(0.0)

    def test_partial_acceptance(self) -> None:
        from eval.metrics import raw_validity_rate

        results = [True, True, False, True, False]
        mock_validator = MagicMock()
        mock_validator.validate.side_effect = [MagicMock(accepted=r) for r in results]
        scenes = [_make_scene(2) for _ in range(5)]
        rate = raw_validity_rate(scenes, mock_validator)
        assert rate == pytest.approx(3 / 5)

    def test_empty_list(self) -> None:
        from eval.metrics import raw_validity_rate

        mock_validator = MagicMock()
        assert raw_validity_rate([], mock_validator) == 0.0
        mock_validator.validate.assert_not_called()


# ---------------------------------------------------------------------------
# diversity
# ---------------------------------------------------------------------------


class TestDiversity:
    def test_identical_scenes_zero(self) -> None:
        from eval.metrics import diversity

        scenes = [_make_scene(3, xyz_base=0.0) for _ in range(5)]
        d = diversity(scenes)
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_spread_scenes_nonzero(self) -> None:
        from eval.metrics import diversity

        # Two scenes far apart should produce nonzero diversity.
        s1 = _make_scene(1, xyz_base=0.0)
        s2 = _make_scene(1, xyz_base=1.0)
        d = diversity([s1, s2])
        assert d > 0.5

    def test_single_scene_zero(self) -> None:
        from eval.metrics import diversity

        assert diversity([_make_scene(3)]) == pytest.approx(0.0)

    def test_empty_zero(self) -> None:
        from eval.metrics import diversity

        assert diversity([]) == pytest.approx(0.0)

    def test_symmetry(self) -> None:
        from eval.metrics import diversity

        s1 = _make_scene(2, xyz_base=0.0)
        s2 = _make_scene(2, xyz_base=0.5)
        assert diversity([s1, s2]) == pytest.approx(diversity([s2, s1]), abs=1e-6)


# ---------------------------------------------------------------------------
# prompt_following (mocked judge)
# ---------------------------------------------------------------------------


class TestPromptFollowing:
    def test_mean_score(self, tmp_path: "Path") -> None:
        from eval.metrics import prompt_following

        mock_judge = MagicMock()
        mock_judge.score_batch.return_value = [3.0, 4.0, 5.0]
        mock_judge._judge_prompt = "test"

        scenes = [_make_scene(2) for _ in range(3)]
        prompts = ["task A", "task B", "task C"]
        score = prompt_following(scenes, prompts, mock_judge, cache_path=None)
        assert score == pytest.approx(4.0)

    def test_cache_avoids_rescore(self, tmp_path: "Path") -> None:
        from eval.metrics import prompt_following

        mock_judge = MagicMock()
        mock_judge.score_batch.return_value = [4.0]
        mock_judge._judge_prompt = "test"

        scene = _make_scene(2, xyz_base=0.1)
        cache = tmp_path / "cache.jsonl"

        # First call should invoke judge
        prompt_following([scene], ["task A"], mock_judge, cache_path=cache)
        assert mock_judge.score_batch.call_count == 1

        # Second call should use cache
        prompt_following([scene], ["task A"], mock_judge, cache_path=cache)
        assert mock_judge.score_batch.call_count == 1  # not incremented

    def test_length_mismatch_raises(self) -> None:
        from eval.metrics import prompt_following

        mock_judge = MagicMock()
        with pytest.raises(ValueError, match="equal length"):
            prompt_following([_make_scene(2)], ["a", "b"], mock_judge)
