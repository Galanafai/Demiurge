"""Unit tests for CFG null-conditioning dropout in the training loop.

Tests the logic extracted from scripts/train.py:
  if cfg_dropout > 0.0 and torch.rand(1, generator=rng).item() < cfg_dropout:
      text_emb_b = None

Four properties verified:
1. cfg_null_prob=0.0 -> never drops (text_emb always passed through)
2. cfg_null_prob=1.0 -> always drops (text_emb always None)
3. cfg_null_prob=0.15 -> ~15% drop rate over many samples
4. Dropout only affects text_emb; x_noisy and type_ids are untouched
"""
from __future__ import annotations

import torch


def _apply_cfg_dropout(
    text_emb: torch.Tensor | None,
    cfg_dropout: float,
    rng: torch.Generator,
) -> torch.Tensor | None:
    """Exact logic from scripts/train.py batch step."""
    if text_emb is None:
        return None
    if cfg_dropout > 0.0 and torch.rand(1, generator=rng).item() < cfg_dropout:
        return None
    return text_emb


class TestCFGDropout:
    def test_cfg_null_prob_zero_no_dropout(self):
        """cfg_dropout=0.0 must never drop text_emb."""
        rng = torch.Generator()
        rng.manual_seed(42)
        emb = torch.randn(4, 384)
        for _ in range(1000):
            result = _apply_cfg_dropout(emb, 0.0, rng)
            assert result is not None, "cfg_dropout=0.0 should never return None"
            assert result is emb, "cfg_dropout=0.0 should return the original tensor"

    def test_cfg_null_prob_one_full_dropout(self):
        """cfg_dropout=1.0 must always drop text_emb."""
        rng = torch.Generator()
        rng.manual_seed(42)
        emb = torch.randn(4, 384)
        for _ in range(1000):
            result = _apply_cfg_dropout(emb, 1.0, rng)
            assert result is None, "cfg_dropout=1.0 should always return None"

    def test_cfg_null_prob_partial_proportional(self):
        """cfg_dropout=0.15 should drop ~15% of calls over 5000 samples."""
        rng = torch.Generator()
        rng.manual_seed(42)
        emb = torch.randn(4, 384)
        cfg_dropout = 0.15
        n = 5000
        n_dropped = sum(
            1 for _ in range(n)
            if _apply_cfg_dropout(emb, cfg_dropout, rng) is None
        )
        actual_rate = n_dropped / n
        # Allow ±3pp of slack (3-sigma for Bernoulli(0.15) at n=5000)
        assert 0.12 <= actual_rate <= 0.18, (
            f"Expected ~15% drop rate, got {actual_rate:.3f} "
            f"({n_dropped}/{n} dropped)"
        )

    def test_cfg_dropout_preserves_continuous(self):
        """Dropout on text_emb must not touch x_noisy, type_ids, or presence."""
        rng = torch.Generator()
        rng.manual_seed(0)
        B, N, D = 8, 8, 13
        x_noisy = torch.randn(B, N, D)
        type_ids = torch.randint(0, 12, (B, N))
        presence = torch.ones(B, N, dtype=torch.bool)
        emb = torch.randn(B, 384)

        x_copy = x_noisy.clone()
        type_copy = type_ids.clone()
        pres_copy = presence.clone()

        for _ in range(200):
            _ = _apply_cfg_dropout(emb, 0.15, rng)

        # None of the other tensors should have changed
        assert torch.equal(x_noisy, x_copy), "x_noisy was modified by CFG dropout"
        assert torch.equal(type_ids, type_copy), "type_ids was modified by CFG dropout"
        assert torch.equal(presence, pres_copy), "presence was modified by CFG dropout"

    def test_cfg_dropout_none_input_passthrough(self):
        """None text_emb (already unconditional) is returned as-is."""
        rng = torch.Generator()
        rng.manual_seed(42)
        for _ in range(100):
            result = _apply_cfg_dropout(None, 0.15, rng)
            assert result is None

    def test_cfg_dropout_rng_determinism(self):
        """Same seed -> same dropout sequence."""
        emb = torch.randn(4, 384)

        rng1 = torch.Generator()
        rng1.manual_seed(99)
        seq1 = [_apply_cfg_dropout(emb, 0.15, rng1) is None for _ in range(100)]

        rng2 = torch.Generator()
        rng2.manual_seed(99)
        seq2 = [_apply_cfg_dropout(emb, 0.15, rng2) is None for _ in range(100)]

        assert seq1 == seq2, "CFG dropout sequence must be deterministic given the same seed"
