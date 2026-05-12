"""Tests for src/data/writer.py and src/data/reader.py.

Round-trip tests use STRICT equality:
  - SceneTensor fields: torch.equal (bit-exact; torch.save/load is deterministic)
  - Description string: == (exact string equality)
  - Report dict floats: == (exact; Python json round-trips finite floats exactly)

Resume / partial-shard tests verify:
  - A partial .tar (not in manifest) is deleted on startup, not read past.
  - Writing resumes from the last manifest checkpoint.
  - Accepted count in the resumed manifest equals total written across both runs.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch

from data.reader import ShardReader
from data.sampler import TabletopReachTemplate
from data.writer import ShardWriter, _scene_tensor_from_dict, _scene_tensor_to_dict
from scene.schema import SceneTensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sample_scene(seed: int = 0) -> SceneTensor:
    t = TabletopReachTemplate()
    c = t.sample_scene(_rng(seed))
    return c.scene


def _fake_report(seed: int = 0) -> dict:
    """Return a minimal serialisable ValidityReport-like dict."""
    return {
        "no_interpenetration": True,
        "stable_rest": True,
        "ik_reachable": True,
        "rrt_solvable": True,
        "accepted": True,
        "min_signed_distance_m": float(0.001 + seed * 0.0001),
        "max_pose_drift_trans_m": float(1e-5 + seed * 1e-6),
        "elapsed_s": float(1.2 + seed * 0.01),
    }


def _write_n(
    output_dir: pathlib.Path,
    n: int,
    start_seed: int = 0,
    **kwargs,
) -> list[tuple]:
    """Write n samples to output_dir. Returns list of (scene, desc, report_dict)."""
    written: list[tuple] = []
    with ShardWriter(output_dir, **kwargs) as w:
        for i in range(n):
            scene = _sample_scene(start_seed + i)
            desc = f"Sample {start_seed + i} description."
            report = _fake_report(start_seed + i)
            w.write(scene, desc, report)
            written.append((scene, desc, report))
    return written


# ---------------------------------------------------------------------------
# SceneTensor serialisation (unit)
# ---------------------------------------------------------------------------


class TestSceneTensorSerde:
    """Verify the _scene_tensor_to_dict / _from_dict helpers are bit-exact."""

    def test_round_trip_object_types(self) -> None:
        scene = _sample_scene(0)
        d = _scene_tensor_to_dict(scene)
        recovered = _scene_tensor_from_dict(d)
        assert torch.equal(scene.object_types, recovered.object_types), (
            "object_types must be bit-exact after dict round-trip"
        )

    def test_round_trip_poses(self) -> None:
        scene = _sample_scene(0)
        d = _scene_tensor_to_dict(scene)
        recovered = _scene_tensor_from_dict(d)
        assert torch.equal(scene.poses, recovered.poses), (
            "poses must be bit-exact after dict round-trip"
        )

    def test_round_trip_scales(self) -> None:
        scene = _sample_scene(0)
        d = _scene_tensor_to_dict(scene)
        recovered = _scene_tensor_from_dict(d)
        assert torch.equal(scene.scales, recovered.scales), (
            "scales must be bit-exact after dict round-trip"
        )

    def test_round_trip_presence(self) -> None:
        scene = _sample_scene(0)
        d = _scene_tensor_to_dict(scene)
        recovered = _scene_tensor_from_dict(d)
        assert torch.equal(scene.presence, recovered.presence), (
            "presence must be bit-exact after dict round-trip"
        )


# ---------------------------------------------------------------------------
# Full write-read round-trip (100 samples)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """100-example write-read cycle with strict (non-allclose) assertions."""

    N = 100

    @pytest.fixture()
    def shard_dir(self, tmp_path: pathlib.Path) -> pathlib.Path:
        return tmp_path / "shards"

    def test_count(self, shard_dir: pathlib.Path) -> None:
        _write_n(shard_dir, self.N, shard_size_mb=1.0)
        read = list(ShardReader(shard_dir))
        assert len(read) == self.N, (
            f"Expected {self.N} samples back, got {len(read)}"
        )

    def test_scene_tensor_bit_exact(self, shard_dir: pathlib.Path) -> None:
        """All SceneTensor fields must be bit-exact (torch.equal, not allclose)."""
        written = _write_n(shard_dir, self.N, shard_size_mb=1.0)
        for i, (scene_r, _desc_r, _rep_r, _sdf_r) in enumerate(ShardReader(shard_dir)):
            scene_w = written[i][0]
            assert torch.equal(scene_w.object_types, scene_r.object_types), (
                f"Sample {i}: object_types mismatch"
            )
            assert torch.equal(scene_w.poses, scene_r.poses), (
                f"Sample {i}: poses mismatch (no allclose tolerance allowed)"
            )
            assert torch.equal(scene_w.scales, scene_r.scales), (
                f"Sample {i}: scales mismatch"
            )
            assert torch.equal(scene_w.presence, scene_r.presence), (
                f"Sample {i}: presence mismatch"
            )

    def test_description_exact(self, shard_dir: pathlib.Path) -> None:
        written = _write_n(shard_dir, self.N, shard_size_mb=1.0)
        for i, (_scene_r, desc_r, _rep_r, _sdf_r) in enumerate(ShardReader(shard_dir)):
            assert written[i][1] == desc_r, (
                f"Sample {i}: description mismatch: {written[i][1]!r} != {desc_r!r}"
            )

    def test_report_dict_exact(self, shard_dir: pathlib.Path) -> None:
        """Report dict float values must round-trip exactly through JSON (not allclose)."""
        written = _write_n(shard_dir, self.N, shard_size_mb=1.0)
        for i, (_scene_r, _desc_r, rep_r, _sdf_r) in enumerate(ShardReader(shard_dir)):
            rep_w = written[i][2]
            for key, val_w in rep_w.items():
                val_r = rep_r[key]
                assert val_w == val_r, (
                    f"Sample {i}, key={key!r}: {val_w!r} != {val_r!r} "
                    "(strict equality required -- no allclose)"
                )

    def test_sdf_non_empty(self, shard_dir: pathlib.Path) -> None:
        _write_n(shard_dir, self.N, shard_size_mb=1.0)
        for i, (_scene_r, _desc_r, _rep_r, sdf_r) in enumerate(ShardReader(shard_dir)):
            assert len(sdf_r) > 0, f"Sample {i}: sdf string is empty"


# ---------------------------------------------------------------------------
# Manifest correctness
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_exists_after_write(self, tmp_path: pathlib.Path) -> None:
        shard_dir = tmp_path / "shards"
        _write_n(shard_dir, 10, shard_size_mb=0.001)
        manifest_path = shard_dir / "manifest.json"
        assert manifest_path.exists(), "manifest.json must be created by ShardWriter"

    def test_manifest_accepted_count_matches(self, tmp_path: pathlib.Path) -> None:
        shard_dir = tmp_path / "shards"
        _write_n(shard_dir, 50, shard_size_mb=0.001)
        with (shard_dir / "manifest.json").open() as f:
            manifest = json.load(f)
        assert manifest["accepted_count"] == 50, (
            f"manifest.accepted_count={manifest['accepted_count']} != 50"
        )

    def test_manifest_completed_shards_non_empty(self, tmp_path: pathlib.Path) -> None:
        shard_dir = tmp_path / "shards"
        _write_n(shard_dir, 10, shard_size_mb=0.001)
        with (shard_dir / "manifest.json").open() as f:
            manifest = json.load(f)
        assert len(manifest["completed_shards"]) >= 1


# ---------------------------------------------------------------------------
# Resume: partial shard deletion
# ---------------------------------------------------------------------------


class TestResumePartialShard:
    """Verify that partial shards are deleted, not read past.

    A partial shard is a .tar file present in the directory but NOT listed in
    manifest.json. It may be truncated. ShardWriter must delete it on startup
    rather than attempting to iterate past truncated entries.
    """

    def test_partial_shard_deleted_on_resume(self, tmp_path: pathlib.Path) -> None:
        """Write 5 samples, then inject a fake partial shard, resume, verify it is gone."""
        shard_dir = tmp_path / "shards"

        # First run: write 5 samples, close cleanly.
        _write_n(shard_dir, 5, shard_size_mb=0.001)

        # Inject a fake partial shard (truncated tar, not in manifest).
        partial_path = shard_dir / "v1-99999.tar"
        partial_path.write_bytes(b"PK\x03\x04" + b"\x00" * 128)  # garbage bytes

        assert partial_path.exists(), "Partial shard should exist before resume"

        # Second run: ShardWriter must delete the partial shard before writing.
        _write_n(shard_dir, 5, start_seed=5, shard_size_mb=0.001, resume=True)

        assert not partial_path.exists(), (
            "Partial shard must be deleted on resume, not kept"
        )

    def test_partial_shard_not_in_read_output(self, tmp_path: pathlib.Path) -> None:
        """After delete-then-restart, all readable samples come from clean shards."""
        shard_dir = tmp_path / "shards"

        # First run: write 5 samples cleanly.
        _write_n(shard_dir, 5, shard_size_mb=0.001)

        # Inject a partial shard.
        (shard_dir / "v1-99999.tar").write_bytes(b"\x00" * 512)

        # Second run: write 5 more (partial is deleted, then 5 written).
        _write_n(shard_dir, 5, start_seed=5, shard_size_mb=0.001, resume=True)

        # Read all samples: should be exactly 10 (5 from run 1 + 5 from run 2).
        samples = list(ShardReader(shard_dir))
        assert len(samples) == 10, (
            f"Expected 10 samples (5+5), got {len(samples)}. "
            "Partial shard should have been deleted, not read."
        )

    def test_resume_accepted_count_accumulates(self, tmp_path: pathlib.Path) -> None:
        """Manifest accepted_count after two runs must equal the sum of both."""
        shard_dir = tmp_path / "shards"

        _write_n(shard_dir, 7, shard_size_mb=0.001)
        _write_n(shard_dir, 3, start_seed=7, shard_size_mb=0.001, resume=True)

        with (shard_dir / "manifest.json").open() as f:
            manifest = json.load(f)
        assert manifest["accepted_count"] == 10, (
            f"Expected accepted_count=10, got {manifest['accepted_count']}"
        )

    def test_no_partial_shard_on_clean_resume(self, tmp_path: pathlib.Path) -> None:
        """Resume with no partial shards must not delete any completed shard."""
        shard_dir = tmp_path / "shards"
        _write_n(shard_dir, 5, shard_size_mb=0.001)

        # Collect completed shard names from manifest.
        with (shard_dir / "manifest.json").open() as f:
            manifest_before = json.load(f)
        shards_before = set(manifest_before["completed_shards"])

        # Resume cleanly.
        _write_n(shard_dir, 3, start_seed=5, shard_size_mb=0.001, resume=True)

        # All original completed shards must still exist.
        for name in shards_before:
            assert (shard_dir / name).exists(), (
                f"Completed shard {name} was incorrectly deleted on clean resume"
            )


# ---------------------------------------------------------------------------
# Empty directory edge case
# ---------------------------------------------------------------------------


class TestEmptyDirectory:
    def test_reader_empty_dir_returns_no_samples(self, tmp_path: pathlib.Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        samples = list(ShardReader(empty_dir))
        assert samples == [], "Empty shard dir must yield no samples"

    def test_writer_creates_dir(self, tmp_path: pathlib.Path) -> None:
        new_dir = tmp_path / "new" / "nested" / "dir"
        assert not new_dir.exists()
        _write_n(new_dir, 1, shard_size_mb=1.0)
        assert new_dir.exists()
