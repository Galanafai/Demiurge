"""Unit tests for SceneJudge.

All tests mock the Anthropic client to avoid API calls and key requirements.
Tests cover: tier selection, scene formatting, response parsing, retry logic,
batch scoring, and error handling.
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic responses
# ---------------------------------------------------------------------------


def _mock_response(score: int, reasoning: str, model: str = "claude-haiku-4-5-20251001",
                   input_tokens: int = 400, output_tokens: int = 40) -> MagicMock:
    resp = MagicMock()
    resp.model = model
    resp.content = [MagicMock()]
    resp.content[0].text = json.dumps({"score": score, "reasoning": reasoning})
    resp.usage = MagicMock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    return resp


def _make_judge(tier_str: str = "claude-haiku-4-5-20251001") -> SceneJudge:  # noqa: F821
    """Build a SceneJudge with a mocked Anthropic client."""
    from eval.judge import JudgeConfig, JudgeTier, SceneJudge

    tier = JudgeTier(tier_str)
    with patch.dict(os.environ, {"CLAUDE_API_KEY": "test-key"}):
        judge = SceneJudge.__new__(SceneJudge)
        judge.cfg = JudgeConfig(tier=tier)
        judge._system_prompt = "test prompt"

        mock_client = MagicMock()
        judge._client = mock_client
        return judge


# ---------------------------------------------------------------------------
# Tests: environment / instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_missing_api_key_raises(self) -> None:
        from eval.judge import SceneJudge

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLAUDE_API_KEY", None)
            with pytest.raises(EnvironmentError, match="CLAUDE_API_KEY"):
                SceneJudge()

    def test_missing_anthropic_raises(self) -> None:
        from eval.judge import SceneJudge

        with patch.dict(os.environ, {"CLAUDE_API_KEY": "test"}):
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                with pytest.raises((RuntimeError, ImportError)):
                    SceneJudge()


# ---------------------------------------------------------------------------
# Tests: scene formatting
# ---------------------------------------------------------------------------


class TestSceneFormatting:
    def test_dict_passthrough(self) -> None:
        judge = _make_judge()
        d = {"n_objects": 1, "objects": [{"type": "YCB_MUG"}]}
        assert judge._format_scene(d) is d

    def test_scene_tensor_formatting(self) -> None:
        """SceneTensor with 2 present objects formats correctly."""

        judge = _make_judge()

        # Build a minimal SceneTensor-like object.
        N_MAX = 12
        presence = torch.zeros(N_MAX, dtype=torch.bool)
        presence[0] = True
        presence[3] = True
        object_types = torch.zeros(N_MAX, dtype=torch.long)
        object_types[0] = 9   # YCB_MUG
        object_types[3] = 0   # YCB_CRACKER_BOX
        poses = torch.zeros(N_MAX, 7)
        poses[0, :3] = torch.tensor([0.1, 0.2, 0.05])
        poses[3, :3] = torch.tensor([-0.1, 0.3, 0.05])
        scales = torch.ones(N_MAX, 3)

        scene = MagicMock()
        scene.presence = presence
        scene.object_types = object_types
        scene.poses = poses
        scene.scales = scales

        result = judge._format_scene(scene)
        assert result["n_objects"] == 2
        objs = result["objects"]
        assert len(objs) == 2
        assert objs[0]["type"] == "YCB_MUG"
        assert objs[1]["type"] == "YCB_CRACKER_BOX"
        assert objs[0]["position_m"] == [0.1, 0.2, 0.05]

    def test_empty_scene(self) -> None:
        """Scene with no present objects returns n_objects=0."""
        judge = _make_judge()
        N_MAX = 12
        scene = MagicMock()
        scene.presence = torch.zeros(N_MAX, dtype=torch.bool)
        scene.object_types = torch.zeros(N_MAX, dtype=torch.long)
        scene.poses = torch.zeros(N_MAX, 7)
        scene.scales = torch.ones(N_MAX, 3)
        result = judge._format_scene(scene)
        assert result["n_objects"] == 0
        assert result["objects"] == []


# ---------------------------------------------------------------------------
# Tests: response parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_valid_json_response(self) -> None:
        judge = _make_judge()
        resp = _mock_response(4, "Objects match task well.")
        result = judge._parse_response(
            json.dumps({"score": 4, "reasoning": "Objects match task well."}),
            latency=0.5,
            response=resp,
        )
        assert result.score == 4
        assert result.reasoning == "Objects match task well."
        assert not result.parse_error
        assert result.latency_s == 0.5

    def test_score_0(self) -> None:
        judge = _make_judge()
        resp = _mock_response(0, "No match.")
        result = judge._parse_response(
            json.dumps({"score": 0, "reasoning": "No match."}),
            latency=0.1,
            response=resp,
        )
        assert result.score == 0
        assert not result.parse_error

    def test_score_5(self) -> None:
        judge = _make_judge()
        resp = _mock_response(5, "Perfect.")
        result = judge._parse_response(
            json.dumps({"score": 5, "reasoning": "Perfect."}),
            latency=0.1,
            response=resp,
        )
        assert result.score == 5

    def test_invalid_json_returns_parse_error(self) -> None:
        judge = _make_judge()
        resp = MagicMock()
        resp.model = "claude-haiku-4-5-20251001"
        result = judge._parse_response("not json at all", latency=0.1, response=resp)
        assert result.parse_error
        assert result.score == -1

    def test_score_out_of_range_returns_parse_error(self) -> None:
        judge = _make_judge()
        resp = MagicMock()
        resp.model = "claude-haiku-4-5-20251001"
        result = judge._parse_response(
            json.dumps({"score": 7, "reasoning": "Bad"}),
            latency=0.1,
            response=resp,
        )
        assert result.parse_error

    def test_markdown_fences_stripped(self) -> None:
        """Model sometimes wraps JSON in ```json ... ``` despite the prompt."""
        judge = _make_judge()
        resp = _mock_response(3, "Partial.")
        raw = "```json\n" + json.dumps({"score": 3, "reasoning": "Partial."}) + "\n```"
        result = judge._parse_response(raw, latency=0.2, response=resp)
        assert result.score == 3
        assert not result.parse_error


# ---------------------------------------------------------------------------
# Tests: score() (single call)
# ---------------------------------------------------------------------------


class TestScore:
    def test_score_returns_result(self) -> None:
        judge = _make_judge()
        judge._client.messages.create.return_value = _mock_response(3, "Decent match.")
        result = judge.score("place the mug on the table", {"n_objects": 1, "objects": []})
        assert result.score == 3
        assert result.reasoning == "Decent match."
        judge._client.messages.create.assert_called_once()

    def test_score_uses_correct_model(self) -> None:
        judge = _make_judge()
        judge._client.messages.create.return_value = _mock_response(2, "Weak.")
        judge.score("test task", {})
        call_kwargs = judge._client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_score_sonnet_tier(self) -> None:
        judge = _make_judge("claude-sonnet-4-6")
        judge._client.messages.create.return_value = _mock_response(
            4, "Good.", model="claude-sonnet-4-6"
        )
        judge.score("test", {})
        call_kwargs = judge._client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Tests: retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_retries_on_rate_limit(self) -> None:
        import anthropic

        judge = _make_judge()
        good_resp = _mock_response(4, "Good.")
        judge._client.messages.create.side_effect = [
            anthropic.RateLimitError("rate limit", response=MagicMock(), body={}),
            good_resp,
        ]
        # Override delay to 0 for test speed.
        judge.cfg = judge.cfg.__class__(retry_delay_s=0.0)

        with patch("time.sleep"):
            result = judge.score("test", {})
        assert result.score == 4
        assert judge._client.messages.create.call_count == 2

    def test_raises_after_max_retries(self) -> None:
        import anthropic

        judge = _make_judge()
        judge.cfg = judge.cfg.__class__(max_retries=2, retry_delay_s=0.0)
        judge._client.messages.create.side_effect = anthropic.RateLimitError(
            "rate limit", response=MagicMock(), body={}
        )

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="failed after"):
                judge.score("test", {})

        assert judge._client.messages.create.call_count == 3  # initial + 2 retries


# ---------------------------------------------------------------------------
# Tests: score_batch()
# ---------------------------------------------------------------------------


class TestScoreBatch:
    def test_batch_returns_correct_length(self) -> None:
        judge = _make_judge()
        judge._client.messages.create.return_value = _mock_response(3, "Ok.")
        pairs = [("task A", {}), ("task B", {}), ("task C", {})]
        results = judge.score_batch(pairs, max_workers=2, show_progress=False)
        assert len(results) == 3

    def test_batch_all_scores_valid(self) -> None:
        judge = _make_judge()
        judge._client.messages.create.return_value = _mock_response(4, "Good.")
        pairs = [("task", {}) for _ in range(10)]
        results = judge.score_batch(pairs, max_workers=4, show_progress=False)
        assert all(r.score == 4 for r in results)

    def test_batch_empty_returns_empty(self) -> None:
        judge = _make_judge()
        results = judge.score_batch([], show_progress=False)
        assert results == []

    def test_batch_handles_single_failure(self) -> None:
        """One failing call should produce a parse_error result, not crash."""
        judge = _make_judge()
        good = _mock_response(3, "Ok.")
        bad = MagicMock()
        bad.content = [MagicMock()]
        bad.content[0].text = "INVALID_JSON"
        bad.model = "claude-haiku-4-5-20251001"
        bad.usage = MagicMock()
        bad.usage.input_tokens = 0
        bad.usage.output_tokens = 0
        judge._client.messages.create.side_effect = [good, bad, good]
        pairs = [("A", {}), ("B", {}), ("C", {})]
        results = judge.score_batch(pairs, max_workers=1, show_progress=False)
        scores = [r.score for r in results]
        assert -1 in scores   # the failed one
        assert scores.count(3) == 2
