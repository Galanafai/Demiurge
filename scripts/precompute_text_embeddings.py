"""Pre-compute and cache text embeddings for all dataset descriptions.

Reads every scene from data/v1, collects unique descriptions, encodes them
with the frozen all-MiniLM-L6-v2 encoder, and writes the cache to
data/v1/text_embeddings.pt (dict[sha256_hex, Tensor]).

This is a one-off offline step run before conditional training. The cache
is then used by train.py's collate function during the conditional run.

Usage
-----
    uv run python scripts/precompute_text_embeddings.py \\
        --data data/v1 \\
        --output data/v1/text_embeddings.pt \\
        --device cuda          # or cpu; cuda is ~10x faster for large datasets

The script is idempotent: if the cache already exists, only missing
descriptions are encoded and the file is updated in-place.

Output
------
A ``dict[str, Tensor]`` saved with ``torch.save``:
  - key: 64-char SHA-256 hex of the UTF-8 description
  - value: float32 Tensor of shape (384,) on CPU

A summary is printed on completion showing unique/total descriptions
and the final cache size.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/v1", help="Path to v1 dataset directory")
    p.add_argument(
        "--output",
        default="data/v1/text_embeddings.pt",
        help="Output cache path (default: data/v1/text_embeddings.pt)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device for the encoder (cpu or cuda). Default: cpu",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size (default: 256)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from data.reader import ShardReader
    from model.text_encoder import TextEncoder, description_hash

    data_path = Path(args.data)
    output_path = Path(args.output)
    device = torch.device(args.device)

    # ── Collect unique descriptions ─────────────────────────────────────────
    log.info("Reading dataset from %s ...", data_path)
    t0 = time.monotonic()
    descriptions: list[str] = []
    total_scenes = 0
    for _scene, desc, _report, _sdf in ShardReader(str(data_path)):
        descriptions.append(desc)
        total_scenes += 1

    unique_descs = list(dict.fromkeys(descriptions))  # preserve order, deduplicate
    elapsed = time.monotonic() - t0
    log.info(
        "Read %d scenes in %.1fs | %d unique descriptions (%.1f%% unique)",
        total_scenes,
        elapsed,
        len(unique_descs),
        100 * len(unique_descs) / max(1, total_scenes),
    )

    # ── Load encoder + existing cache ───────────────────────────────────────
    encoder = TextEncoder(device=device, cache_path=output_path)
    already_cached = sum(
        1 for d in unique_descs if description_hash(d) in encoder._cache
    )
    to_encode = [d for d in unique_descs if description_hash(d) not in encoder._cache]
    log.info(
        "Cache: %d existing | %d to encode",
        already_cached,
        len(to_encode),
    )

    # ── Encode missing descriptions in batches ───────────────────────────────
    if to_encode:
        t1 = time.monotonic()
        bs = args.batch_size
        n_batches = (len(to_encode) + bs - 1) // bs
        for i in range(n_batches):
            batch = to_encode[i * bs : (i + 1) * bs]
            encoder.encode_batch(batch)
            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                pct = 100 * (i + 1) / n_batches
                log.info("  batch %d/%d (%.0f%%)", i + 1, n_batches, pct)
        elapsed2 = time.monotonic() - t1
        log.info(
            "Encoded %d descriptions in %.1fs (%.0f desc/sec)",
            len(to_encode),
            elapsed2,
            len(to_encode) / max(0.001, elapsed2),
        )
    else:
        log.info("All descriptions already cached -- nothing to encode.")

    # ── Save ─────────────────────────────────────────────────────────────────
    encoder.save_cache(output_path)
    log.info(
        "Cache saved: %d entries | path: %s",
        encoder.cache_size,
        output_path,
    )

    # ── Verification ─────────────────────────────────────────────────────────
    log.info("Verifying cache integrity ...")
    cache = torch.load(output_path, map_location="cpu", weights_only=True)
    shapes = {v.shape for v in cache.values()}
    assert shapes == {torch.Size([384])}, f"Unexpected embedding shapes: {shapes}"
    dtypes = {v.dtype for v in cache.values()}
    assert dtypes == {torch.float32}, f"Unexpected dtypes: {dtypes}"
    log.info(
        "OK -- %d entries, all shape=(384,) float32",
        len(cache),
    )

    # Coverage: what fraction of dataset descriptions are cached?
    covered = sum(1 for d in descriptions if description_hash(d) in cache)
    log.info(
        "Coverage: %d/%d scenes (%.1f%%)",
        covered,
        total_scenes,
        100 * covered / max(1, total_scenes),
    )


if __name__ == "__main__":
    main()
