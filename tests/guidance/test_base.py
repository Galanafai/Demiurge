"""Tests for src/guidance/base.py -- Sampler protocol."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from guidance.base import Sampler  # noqa: E402
from scene.schema import SceneTensor  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal concrete sampler for testing
# ---------------------------------------------------------------------------


class _DummySampler:
    """Minimal Sampler-compliant class for protocol testing."""

    def sample(
        self, prompts: list[str], n_per_prompt: int, **kwargs: Any
    ) -> list[SceneTensor]:
        return []

    def name(self) -> str:
        return "dummy"

    def config(self) -> dict[str, Any]:
        return {"sampler": "dummy"}


class _IncompleteSampler:
    """Missing config() -- should NOT satisfy Sampler protocol."""

    def sample(self, prompts: list[str], n_per_prompt: int, **kwargs: Any) -> list[SceneTensor]:
        return []

    def name(self) -> str:
        return "incomplete"
    # config() is intentionally absent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sampler_protocol_runtime_check() -> None:
    """@runtime_checkable Sampler: complete impl passes, incomplete fails."""
    ok = _DummySampler()
    bad = _IncompleteSampler()

    assert isinstance(ok, Sampler), (
        "_DummySampler implements all three methods and should pass isinstance check."
    )
    # Protocol only checks method existence at runtime, not signatures.
    # _IncompleteSampler missing config() should NOT be an instance.
    assert not isinstance(bad, Sampler), (
        "_IncompleteSampler missing config() should fail isinstance check."
    )


def test_sampler_protocol_method_names() -> None:
    """Sampler protocol must require sample, name, and config methods."""
    # Verify via __protocol_attrs__ (Python 3.12+) or manual check.
    required = {"sample", "name", "config"}
    s = _DummySampler()
    for method in required:
        assert hasattr(s, method), f"Sampler must have {method!r} method."
        assert callable(getattr(s, method)), f"{method!r} must be callable."


def test_sampler_config_is_dict() -> None:
    """config() must return a dict."""
    s = _DummySampler()
    cfg = s.config()
    assert isinstance(cfg, dict), f"config() must return dict, got {type(cfg)}"


def test_sampler_name_is_str() -> None:
    """name() must return a string."""
    s = _DummySampler()
    assert isinstance(s.name(), str), f"name() must return str, got {type(s.name())}"
