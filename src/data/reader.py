"""WebDataset shard reader for the Demiurge dataset.

Returns an iterable of (SceneTensor, description, report_dict) tuples.
All four keys written by writer.py are decoded:

  .pt   -> SceneTensor  (bit-exact torch.load; same bytes as writer.write)
  .txt  -> str          (UTF-8)
  .json -> dict         (JSON; float values exactly match what was written)
  .sdf  -> str          (UTF-8; not decoded further)

The reader is a thin wrapper around webdataset.WebDataset. It does not shuffle
by default so that round-trip tests can assert positional equality.
"""

from __future__ import annotations

import io
import json
import pathlib
from collections.abc import Iterator
from typing import Any

import torch
import webdataset as wds

from data.writer import _scene_tensor_from_dict
from scene.schema import SceneTensor

# Sentinel returned when a shard directory has no tar files.
_EMPTY: list[str] = []


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def _decode_pt(value: bytes) -> SceneTensor:
    """Decode .pt bytes back to a SceneTensor (bit-exact)."""
    d: dict[str, torch.Tensor] = torch.load(  # type: ignore[assignment]
        io.BytesIO(value), weights_only=True
    )
    return _scene_tensor_from_dict(d)


def _decode_txt(value: bytes) -> str:
    return value.decode("utf-8")


def _decode_json(value: bytes) -> dict[str, Any]:
    return json.loads(value.decode("utf-8"))  # type: ignore[no-any-return]


def _decode_sdf(value: bytes) -> str:
    return value.decode("utf-8")


def _sample_decoder(sample: dict[str, bytes]) -> dict[str, Any]:
    """Decode all four keys in a WebDataset sample dict."""
    return {
        "__key__": sample["__key__"],
        "pt": _decode_pt(sample["pt"]),
        "txt": _decode_txt(sample["txt"]),
        "json": _decode_json(sample["json"]),
        "sdf": _decode_sdf(sample["sdf"]),
    }


# ---------------------------------------------------------------------------
# Public reader
# ---------------------------------------------------------------------------


class ShardReader:
    """Iterate over sharded WebDataset tars as (SceneTensor, str, dict) tuples.

    Args:
        shard_dir: Directory containing .tar shard files.
        shuffle_buffer: Number of samples to buffer for shuffling. Set to 0
            to disable shuffling (required for round-trip equality tests).
        shardshuffle: If True, shuffle shard order (use for training, not eval).
    """

    def __init__(
        self,
        shard_dir: pathlib.Path | str,
        shuffle_buffer: int = 0,
        shardshuffle: bool = False,
    ) -> None:
        self._shard_dir = pathlib.Path(shard_dir)
        self._shuffle_buffer = shuffle_buffer
        self._shardshuffle = shardshuffle

    def _shard_urls(self) -> list[str]:
        urls = sorted(str(p) for p in self._shard_dir.glob("*.tar"))
        return urls

    def __iter__(self) -> Iterator[tuple[SceneTensor, str, dict[str, Any], str]]:
        urls = self._shard_urls()
        if not urls:
            return

        ds: wds.WebDataset = wds.WebDataset(
            urls,
            shardshuffle=self._shardshuffle,
            nodesplitter=wds.shardlists.single_node_only,
        )

        if self._shuffle_buffer > 0:
            ds = ds.shuffle(self._shuffle_buffer)

        for sample in ds:
            decoded = _sample_decoder(sample)
            yield (
                decoded["pt"],
                decoded["txt"],
                decoded["json"],
                decoded["sdf"],
            )

    def iter_scenes(self) -> Iterator[tuple[SceneTensor, str, dict[str, Any], str]]:
        """Alias for __iter__ with an explicit name for clarity in scripts."""
        return iter(self)
