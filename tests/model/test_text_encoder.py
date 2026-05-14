"""Unit tests for TextEncoder.

These tests mock the sentence_transformers model to avoid downloading
weights in CI. The encoder API and cache logic are tested independently
of the actual model weights.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_st(d_text: int = 384) -> MagicMock:
    """Return a mock SentenceTransformer that returns deterministic embeddings."""
    mock = MagicMock()

    def _encode(texts: list[str], **kwargs) -> torch.Tensor:
        # Deterministic: hash the concatenated text to a seed.
        seed = int(hashlib.md5("".join(texts).encode()).hexdigest()[:8], 16)
        gen = torch.Generator().manual_seed(seed)
        emb = torch.randn(len(texts), d_text, generator=gen)
        # Normalise to unit norm (encoder uses normalize_embeddings=True).
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    mock.encode.side_effect = _encode
    mock.parameters.return_value = iter([])  # no params to freeze
    return mock


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_encoder(tmp_path: Path, mock_st: MagicMock | None = None) -> TextEncoder:  # noqa: F821
    """Construct a TextEncoder with a mocked SentenceTransformer."""
    from model.text_encoder import TextEncoder

    if mock_st is None:
        mock_st = _make_mock_st()

    with patch("model.text_encoder.TextEncoder.__init__") as mock_init:
        mock_init.return_value = None
        enc = object.__new__(TextEncoder)

    # Manually init internal state.
    enc._device = torch.device("cpu")
    enc._cache: dict = {}
    enc._cache_path = tmp_path / "test_embeddings.pt"
    enc._model = mock_st
    return enc


# ---------------------------------------------------------------------------
# Tests: description_hash
# ---------------------------------------------------------------------------


class TestDescriptionHash:
    def test_deterministic(self) -> None:
        from model.text_encoder import description_hash

        h1 = description_hash("place the mug on the table")
        h2 = description_hash("place the mug on the table")
        assert h1 == h2

    def test_length_64(self) -> None:
        from model.text_encoder import description_hash

        h = description_hash("test")
        assert len(h) == 64

    def test_different_texts_differ(self) -> None:
        from model.text_encoder import description_hash

        h1 = description_hash("place the mug")
        h2 = description_hash("place the cup")
        assert h1 != h2

    def test_is_hex(self) -> None:
        from model.text_encoder import description_hash

        h = description_hash("demiurge")
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Tests: encode (single)
# ---------------------------------------------------------------------------


class TestEncode:
    def test_shape(self, tmp_path: Path) -> None:
        enc = _make_encoder(tmp_path)
        emb = enc.encode("place the mug on the table")
        assert emb.shape == (384,)

    def test_dtype_float32(self, tmp_path: Path) -> None:
        enc = _make_encoder(tmp_path)
        emb = enc.encode("some description")
        assert emb.dtype == torch.float32

    def test_cache_hit_on_second_call(self, tmp_path: Path) -> None:
        mock_st = _make_mock_st()
        enc = _make_encoder(tmp_path, mock_st)
        text = "place the mug on the table"
        enc.encode(text)
        enc.encode(text)
        # Model should only be called once.
        assert mock_st.encode.call_count == 1

    def test_different_texts_different_embeddings(self, tmp_path: Path) -> None:
        enc = _make_encoder(tmp_path)
        e1 = enc.encode("place the mug on the table")
        e2 = enc.encode("stack the can on the box")
        # With MD5-seeded random, different texts produce different embeddings.
        assert not torch.allclose(e1, e2)


# ---------------------------------------------------------------------------
# Tests: encode_batch
# ---------------------------------------------------------------------------


class TestEncodeBatch:
    def test_shape(self, tmp_path: Path) -> None:
        enc = _make_encoder(tmp_path)
        texts = ["text one", "text two", "text three"]
        embs = enc.encode_batch(texts)
        assert embs.shape == (3, 384)

    def test_empty_batch(self, tmp_path: Path) -> None:
        enc = _make_encoder(tmp_path)
        embs = enc.encode_batch([])
        assert embs.shape == (0, 384)

    def test_partial_cache_hit(self, tmp_path: Path) -> None:
        mock_st = _make_mock_st()
        enc = _make_encoder(tmp_path, mock_st)
        texts = ["text A", "text B", "text C"]
        # Pre-warm cache with text A.
        enc.encode("text A")
        assert mock_st.encode.call_count == 1
        # encode_batch should only encode B and C.
        enc.encode_batch(texts)
        # The model was called twice total (1 for A, 1 for B+C batch).
        assert mock_st.encode.call_count == 2
        last_call_texts = mock_st.encode.call_args[0][0]
        assert "text A" not in last_call_texts
        assert "text B" in last_call_texts
        assert "text C" in last_call_texts

    def test_all_cache_hit_no_model_call(self, tmp_path: Path) -> None:
        mock_st = _make_mock_st()
        enc = _make_encoder(tmp_path, mock_st)
        texts = ["text X", "text Y"]
        enc.encode_batch(texts)
        call_count = mock_st.encode.call_count
        enc.encode_batch(texts)
        assert mock_st.encode.call_count == call_count  # no new calls

    def test_single_text_batch(self, tmp_path: Path) -> None:
        enc = _make_encoder(tmp_path)
        embs = enc.encode_batch(["only one text"])
        assert embs.shape == (1, 384)


# ---------------------------------------------------------------------------
# Tests: cache persistence
# ---------------------------------------------------------------------------


class TestCachePersistence:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        enc = _make_encoder(tmp_path)
        enc.encode("test text")
        enc.save_cache()
        assert enc._cache_path.exists()

    def test_saved_format_is_dict_str_tensor(self, tmp_path: Path) -> None:
        enc = _make_encoder(tmp_path)
        enc.encode("test text")
        enc.save_cache()
        loaded = torch.load(enc._cache_path, map_location="cpu", weights_only=True)
        assert isinstance(loaded, dict)
        for k, v in loaded.items():
            assert isinstance(k, str) and len(k) == 64
            assert isinstance(v, torch.Tensor)
            assert v.shape == (384,)
            assert v.dtype == torch.float32

    def test_cache_round_trip(self, tmp_path: Path) -> None:
        from model.text_encoder import description_hash

        enc1 = _make_encoder(tmp_path)
        text = "place the mug on the table"
        emb1 = enc1.encode(text)
        enc1.save_cache()

        # Simulate loading on restart: new encoder loads from same path.
        enc2 = _make_encoder(tmp_path)
        # Manually load cache (mimicking constructor behaviour).
        loaded = torch.load(enc1._cache_path, map_location="cpu", weights_only=True)
        enc2._cache = loaded
        emb2 = enc2.lookup(description_hash(text))
        assert torch.allclose(emb1, emb2)

    def test_save_requires_path(self, tmp_path: Path) -> None:
        enc = _make_encoder(tmp_path)
        enc._cache_path = None
        with pytest.raises(ValueError, match="No cache path configured"):
            enc.save_cache()

    def test_save_to_override_path(self, tmp_path: Path) -> None:
        enc = _make_encoder(tmp_path)
        enc.encode("hello")
        alt_path = tmp_path / "alt_cache.pt"
        enc.save_cache(path=alt_path)
        assert alt_path.exists()


# ---------------------------------------------------------------------------
# Tests: lookup
# ---------------------------------------------------------------------------


class TestLookup:
    def test_lookup_after_encode(self, tmp_path: Path) -> None:
        from model.text_encoder import description_hash

        enc = _make_encoder(tmp_path)
        text = "stack the box"
        emb = enc.encode(text)
        assert torch.allclose(enc.lookup(description_hash(text)), emb)

    def test_lookup_missing_key_raises(self, tmp_path: Path) -> None:
        enc = _make_encoder(tmp_path)
        with pytest.raises(KeyError):
            enc.lookup("a" * 64)


# ---------------------------------------------------------------------------
# Tests: cache_size property
# ---------------------------------------------------------------------------


class TestCacheSize:
    def test_empty_initially(self, tmp_path: Path) -> None:
        enc = _make_encoder(tmp_path)
        assert enc.cache_size == 0

    def test_grows_with_new_entries(self, tmp_path: Path) -> None:
        enc = _make_encoder(tmp_path)
        enc.encode("text one")
        enc.encode("text two")
        assert enc.cache_size == 2

    def test_duplicate_does_not_grow(self, tmp_path: Path) -> None:
        enc = _make_encoder(tmp_path)
        enc.encode("same text")
        enc.encode("same text")
        assert enc.cache_size == 1
