"""WebDataset shard writer for the Demiurge dataset.

Writes accepted scenes to sharded tar files using webdataset.ShardWriter.
Each sample contains four keys:

  <key>.pt   -- SceneTensor state dict (torch.save to BytesIO; bit-exact round-trip)
  <key>.txt  -- Task description (UTF-8)
  <key>.json -- ValidityReport metrics dict (JSON; exact round-trip for all finite floats)
  <key>.sdf  -- Per-object SDF strings (newline-joined; reconstructable from SceneTensor)

Manifest:
  A manifest.json file is maintained alongside the shards. It records the list of
  completed shard filenames and the total accepted count. On startup, ShardWriter reads
  the manifest to resume. Any .tar file in the output directory that is NOT listed in the
  manifest is treated as a partial shard (truncated due to a crash or interruption) and is
  DELETED before writing resumes. Partial shards are never read; they are always restarted.

  This avoids the failure mode of tarfile seeking past truncated entries, which raises
  ReadError or silently skips data depending on the tarfile implementation.

Usage:
  with ShardWriter(output_dir) as w:
      for scene, description, report_dict in stream:
          w.write(scene, description, report_dict)
"""

from __future__ import annotations

import io
import json
import logging
import pathlib
from typing import Any

import torch
import webdataset as wds

from scene.schema import N_MAX, SceneTensor
from scene.vocab import OBJECT_VOCAB, build_sdf

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "manifest.json"
_SHARD_GLOB = "*.tar"


# ---------------------------------------------------------------------------
# SceneTensor serialisation helpers
# ---------------------------------------------------------------------------


def _scene_tensor_to_dict(scene: SceneTensor) -> dict[str, torch.Tensor]:
    """Pack SceneTensor fields into a plain dict for torch.save.

    The dict is keyed by field name so that it can be unpacked back into a
    SceneTensor without ambiguity.
    """
    return {
        "object_types": scene.object_types,
        "poses": scene.poses,
        "scales": scene.scales,
        "presence": scene.presence,
    }


def _scene_tensor_from_dict(d: dict[str, torch.Tensor]) -> SceneTensor:
    """Reconstruct a SceneTensor from a dict produced by _scene_tensor_to_dict."""
    return SceneTensor(
        object_types=d["object_types"],
        poses=d["poses"],
        scales=d["scales"],
        presence=d["presence"],
    )


# ---------------------------------------------------------------------------
# SDF helper
# ---------------------------------------------------------------------------


def _scene_to_sdf(scene: SceneTensor) -> str:
    """Generate SDF strings for all present objects, newline-joined.

    Each line is the SDF for one present object. Consumers can split on newline
    to recover individual SDFs. The order matches the slot order in the SceneTensor.
    Objects in inactive slots are omitted.
    """
    parts: list[str] = []
    for i in range(N_MAX):
        if not scene.presence[i].item():
            continue
        type_id = int(scene.object_types[i].item())
        entry = OBJECT_VOCAB[type_id]
        scale = float(scene.scales[i].mean().item())
        model_name = f"obj_{i}_{entry.name}"
        parts.append(build_sdf(entry, model_name=model_name, scale=scale))
    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _load_manifest(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed_shards": [], "accepted_count": 0, "version": 1}
    with path.open() as f:
        return json.load(f)


def _write_manifest(path: pathlib.Path, manifest: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# ShardWriter
# ---------------------------------------------------------------------------


class ShardWriter:
    """Write accepted scenes to sharded WebDataset tars with resume support.

    Args:
        output_dir: Directory to write shards and manifest into.
        pattern: Shard filename pattern with a %05d placeholder.
        shard_size_mb: Maximum shard size in megabytes.
        resume: If True, read manifest on startup and resume from last checkpoint.
            Partial shards (tars present but not in the manifest) are DELETED.
            If False, start fresh (existing manifest and shards are left in place
            but not consulted -- use only for tests that manage their own dir).
    """

    def __init__(
        self,
        output_dir: pathlib.Path | str,
        pattern: str = "v1-%05d.tar",
        shard_size_mb: float = 500.0,
        resume: bool = True,
    ) -> None:
        self._output_dir = pathlib.Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._pattern = pattern
        self._shard_size_bytes = int(shard_size_mb * 1024 * 1024)
        self._manifest_path = self._output_dir / _MANIFEST_FILENAME

        # Load or initialise manifest.
        if resume:
            manifest = _load_manifest(self._manifest_path)
        else:
            manifest = {"completed_shards": [], "accepted_count": 0, "version": 1}

        self._completed_shards: list[str] = manifest["completed_shards"]
        self._accepted_count: int = manifest["accepted_count"]

        if resume:
            self._delete_partial_shards()

        # Determine which shard index to start on.
        start_shard = len(self._completed_shards)

        # Open the underlying webdataset.ShardWriter.
        full_pattern = str(self._output_dir / pattern)
        self._writer = wds.ShardWriter(
            full_pattern,
            maxsize=self._shard_size_bytes,
            start_shard=start_shard,
            post=self._on_shard_complete,
            verbose=False,
        )

        logger.info(
            "ShardWriter opened: output_dir=%s, start_shard=%d, "
            "already_accepted=%d, completed_shards=%d",
            self._output_dir,
            start_shard,
            self._accepted_count,
            len(self._completed_shards),
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        scene: SceneTensor,
        description: str,
        report_dict: dict[str, Any],
    ) -> None:
        """Write one accepted scene to the current shard.

        Args:
            scene: Physical-space SceneTensor.
            description: Task description string.
            report_dict: Serialisable dict from ValidityReport (float values only;
                bool fields included as int for JSON compactness).
        """
        key = f"{self._accepted_count:08d}"

        # Encode SceneTensor to bytes via torch.save (bit-exact round-trip).
        pt_buf = io.BytesIO()
        torch.save(_scene_tensor_to_dict(scene), pt_buf)
        pt_bytes = pt_buf.getvalue()

        # SDF strings for all present objects.
        sdf_str = _scene_to_sdf(scene)

        sample: dict[str, Any] = {
            "__key__": key,
            "pt": pt_bytes,
            "txt": description.encode("utf-8"),
            "json": json.dumps(report_dict).encode("utf-8"),
            "sdf": sdf_str.encode("utf-8"),
        }
        self._writer.write(sample)
        self._accepted_count += 1

    # ------------------------------------------------------------------
    # Shard completion callback
    # ------------------------------------------------------------------

    def _on_shard_complete(self, shard_path: str) -> None:
        """Called by webdataset after each shard tar is closed."""
        shard_name = pathlib.Path(shard_path).name
        self._completed_shards.append(shard_name)
        manifest: dict[str, Any] = {
            "completed_shards": self._completed_shards,
            "accepted_count": self._accepted_count,
            "version": 1,
        }
        _write_manifest(self._manifest_path, manifest)
        logger.info("Shard complete: %s (total accepted=%d)", shard_name, self._accepted_count)

    # ------------------------------------------------------------------
    # Resume: delete partial shards
    # ------------------------------------------------------------------

    def _delete_partial_shards(self) -> None:
        """Delete any .tar file in output_dir that is not in the manifest.

        Partial shards arise from crashes mid-write. They may be truncated
        and cannot be safely read. Deleting and restarting is the only safe
        recovery strategy.

        Never attempts to read or seek past truncated tar entries.
        """
        completed_set = set(self._completed_shards)
        for tar_path in self._output_dir.glob(_SHARD_GLOB):
            if tar_path.name not in completed_set:
                logger.warning(
                    "Deleting partial shard %s (not in manifest)", tar_path.name
                )
                tar_path.unlink()

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the underlying writer. Writes the final manifest."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None  # type: ignore[assignment]

        # Write final manifest (accepted_count reflects all written samples,
        # including any in the currently-open-but-not-yet-completed shard).
        manifest: dict[str, Any] = {
            "completed_shards": self._completed_shards,
            "accepted_count": self._accepted_count,
            "version": 1,
        }
        _write_manifest(self._manifest_path, manifest)
        logger.info("ShardWriter closed. Final accepted_count=%d", self._accepted_count)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def accepted_count(self) -> int:
        """Number of scenes written since this ShardWriter was created (plus resumed)."""
        return self._accepted_count
