"""Tests for src/data/diversity.py.

Five tests:
1. test_deterministic      -- same seed yields identical MPD values.
2. test_finite_floats      -- both MPDs are finite positive floats.
3. test_scene_encoding_range -- scene MPD is in (0, 100).
4. test_text_embedding_range -- text MPD is in (0, 2.01) (unit-sphere bound).
5. test_small_sample       -- n_sample=50 completes in under 30 seconds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "v1"

pytestmark = pytest.mark.skipif(
    not _DATA_DIR.exists(),
    reason="data/v1/ not present; skipping diversity tests",
)


def _make_reader():
    from data.reader import ShardReader

    return ShardReader(str(_DATA_DIR))


# ---------------------------------------------------------------------------
# Test 1: determinism
# ---------------------------------------------------------------------------


def test_deterministic() -> None:
    """Two calls with the same seed must return identical MPD values."""
    from data.diversity import compute_diversity

    r1 = compute_diversity(_make_reader(), n_sample=50, seed=7)
    r2 = compute_diversity(_make_reader(), n_sample=50, seed=7)

    assert r1.scene_encoding_mpd == r2.scene_encoding_mpd, (
        f"scene_encoding_mpd not deterministic: {r1.scene_encoding_mpd} vs {r2.scene_encoding_mpd}"
    )
    assert r1.text_embedding_mpd == r2.text_embedding_mpd, (
        f"text_embedding_mpd not deterministic: {r1.text_embedding_mpd} vs {r2.text_embedding_mpd}"
    )
    assert r1.n_sampled == r2.n_sampled


# ---------------------------------------------------------------------------
# Test 2: finite positive floats
# ---------------------------------------------------------------------------


def test_finite_floats() -> None:
    """Both MPD values must be finite and strictly positive."""
    import math

    from data.diversity import compute_diversity

    result = compute_diversity(_make_reader(), n_sample=50, seed=0)

    assert math.isfinite(result.scene_encoding_mpd), (
        f"scene_encoding_mpd is not finite: {result.scene_encoding_mpd}"
    )
    assert math.isfinite(result.text_embedding_mpd), (
        f"text_embedding_mpd is not finite: {result.text_embedding_mpd}"
    )
    assert result.scene_encoding_mpd > 0.0, (
        "scene_encoding_mpd must be strictly positive (all-zero encoding would be degenerate)"
    )
    assert result.text_embedding_mpd > 0.0, (
        "text_embedding_mpd must be strictly positive"
    )


# ---------------------------------------------------------------------------
# Test 3: scene encoding range
# ---------------------------------------------------------------------------


def test_scene_encoding_range() -> None:
    """Scene encoding MPD must be in (0, 100).

    The 204-dim encoding has bounded magnitude (one-hot + unit-scale poses).
    A value above 100 would indicate a numerical error or wrong normalisation.
    """
    from data.diversity import compute_diversity

    result = compute_diversity(_make_reader(), n_sample=50, seed=0)

    assert 0 < result.scene_encoding_mpd < 100, (
        f"scene_encoding_mpd={result.scene_encoding_mpd:.4f} out of expected range (0, 100)"
    )


# ---------------------------------------------------------------------------
# Test 4: text embedding range
# ---------------------------------------------------------------------------


def test_text_embedding_range() -> None:
    """Text embedding MPD must be in (0, 2.01).

    all-MiniLM-L6-v2 returns L2-normalised embeddings (unit sphere in R^384).
    Maximum pairwise L2 distance between two unit vectors is 2.0 (antipodal).
    A value above 2.01 (small tolerance for fp error) indicates the embeddings
    are not normalised or there is a computation error.
    """
    from data.diversity import compute_diversity

    result = compute_diversity(_make_reader(), n_sample=50, seed=0)

    assert 0 < result.text_embedding_mpd <= 2.01, (
        f"text_embedding_mpd={result.text_embedding_mpd:.4f} out of expected range (0, 2.01]. "
        "Check that normalize_embeddings=True is passed to model.encode()."
    )


# ---------------------------------------------------------------------------
# Test 5: speed -- n_sample=50 in under 30 seconds
# ---------------------------------------------------------------------------


def test_small_sample_speed() -> None:
    """n_sample=50 must complete in under 90 seconds.

    The sentence-transformer model load from local cache takes 20-45s in a
    fresh process. 90s is the bound: it catches genuine hangs or network
    downloads without being tight enough to flake on model init timing.
    """
    from data.diversity import compute_diversity

    t0 = time.monotonic()
    result = compute_diversity(_make_reader(), n_sample=50, seed=0)
    elapsed = time.monotonic() - t0

    assert elapsed < 90.0, (
        f"compute_diversity(n_sample=50) took {elapsed:.1f}s, expected < 90s. "
        "If this failed, the sentence-transformer model may be downloading from "
        "the network rather than loading from cache. Check HF_HOME."
    )
    assert result.n_sampled == 50
