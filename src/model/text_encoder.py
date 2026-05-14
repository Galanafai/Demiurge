"""Frozen text encoder for Demiurge conditional diffusion.

Wraps ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim outputs) behind a
thin interface that matches the cross-attention shape expected by SceneDenoiser.

Design decisions
----------------
* **Frozen weights**: The encoder is loaded once and never back-propped through.
  Conditional training only updates the denoiser's cross-attention layers.
* **SHA-256 hash cache**: Embeddings are keyed by ``sha256(description)`` so the
  cache is deterministic across Python sessions (unlike ``hash()`` which is
  randomised since CPython 3.3).
* **Batched encoding**: ``encode_batch`` processes a list of strings in one
  forward pass; the cache is checked per-entry to avoid redundant GPU calls.
* **Device awareness**: The encoder moves to ``device`` on construction. The
  cache stores CPU tensors; callers move to device as needed.

Cache format
------------
``data/v1/text_embeddings.pt`` is a ``dict[str, Tensor]`` where:
  - key: 64-character hex string (SHA-256 of the UTF-8 description)
  - value: float32 Tensor of shape (384,) on CPU

The cache is append-only: entries are never deleted. Loading is safe to call
multiple times; the file is only written when new entries are added.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Output dimension for all-MiniLM-L6-v2. Must match D_TEXT in denoiser.py.
D_TEXT: int = 384
_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"


def description_hash(text: str) -> str:
    """SHA-256 hex digest of a UTF-8 encoded description string.

    Used as the cache key. Deterministic across Python sessions and machines.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TextEncoder:
    """Frozen sentence-transformer wrapper for conditioning the denoiser.

    Args:
        device: Torch device to run the encoder on. Embeddings are cached on
            CPU regardless.
        model_name: HuggingFace model identifier. Override in tests only.
        cache_path: Path to the persistent embedding cache file. Pass None to
            disable persistence (useful in unit tests).

    Raises:
        RuntimeError: If the sentence_transformers package is not installed.
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        model_name: str = _MODEL_NAME,
        cache_path: Path | str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence_transformers is required for conditional training. "
                "Install with: uv pip install sentence-transformers"
            ) from e

        self._device = torch.device(device)
        self._cache: dict[str, Tensor] = {}
        self._cache_path: Path | None = Path(cache_path) if cache_path else None

        # Load from disk if it exists.
        if self._cache_path and self._cache_path.exists():
            loaded = torch.load(self._cache_path, map_location="cpu", weights_only=True)
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"Expected dict[str, Tensor] in {self._cache_path}, "
                    f"got {type(loaded).__name__}"
                )
            self._cache = loaded
            log.info("Loaded %d cached embeddings from %s", len(self._cache), self._cache_path)

        log.info("Loading text encoder %s on %s", model_name, self._device)
        self._model = SentenceTransformer(model_name, device=str(self._device))
        # Freeze: no grad tracking needed.
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> Tensor:
        """Encode a single description string.

        Returns a CPU float32 tensor of shape (D_TEXT,).
        Cache hit is O(1); cache miss runs one forward pass.
        """
        key = description_hash(text)
        if key not in self._cache:
            self._cache[key] = self._encode_raw([text])[0]
        return self._cache[key]

    def encode_batch(self, texts: list[str]) -> Tensor:
        """Encode a batch of description strings.

        Checks the cache per-entry; only missing entries are encoded.
        Returns a CPU float32 tensor of shape (len(texts), D_TEXT).
        """
        keys = [description_hash(t) for t in texts]
        missing_indices = [i for i, k in enumerate(keys) if k not in self._cache]

        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            new_embs = self._encode_raw(missing_texts)  # (len(missing), D_TEXT)
            for idx, emb in zip(missing_indices, new_embs.unbind(0)):
                self._cache[keys[idx]] = emb

        if not texts:
            return torch.zeros(0, D_TEXT, dtype=torch.float32)
        return torch.stack([self._cache[k] for k in keys])  # (B, D_TEXT)

    def lookup(self, sha256_key: str) -> Tensor:
        """Direct cache lookup by pre-computed SHA-256 key.

        Args:
            sha256_key: 64-char hex string returned by ``description_hash()``.

        Raises:
            KeyError: If the key is not in the cache. Call ``encode()`` first.
        """
        return self._cache[sha256_key]

    def save_cache(self, path: Path | str | None = None) -> None:
        """Persist the embedding cache to disk.

        Args:
            path: Override path. Defaults to ``self._cache_path``.

        Raises:
            ValueError: If neither ``path`` nor the constructor ``cache_path``
                was provided.
        """
        target = Path(path) if path else self._cache_path
        if target is None:
            raise ValueError(
                "No cache path configured. Pass path= or set cache_path in constructor."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._cache, target)
        log.info("Saved %d embeddings to %s", len(self._cache), target)

    @property
    def cache_size(self) -> int:
        """Number of entries in the in-memory cache."""
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode_raw(self, texts: list[str]) -> Tensor:
        """Run the encoder on a raw list; return CPU float32 (N, D_TEXT)."""
        with torch.no_grad():
            embs = self._model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        # sentence_transformers returns on the model device; move to CPU.
        return embs.float().cpu()  # type: ignore[union-attr]
