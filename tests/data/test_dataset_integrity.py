"""Dataset integrity tests for data/v1/.

Tests:
1. test_streaming_read_1000 -- stream 1000 examples from data/v1/; assert
   all fields present, non-empty descriptions, correct tensor shapes.
2. test_tensor_round_trip   -- write 5 scenes via ShardWriter, read back via
   ShardReader, assert bit-exact tensor match and string-exact description.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "v1"

pytestmark = pytest.mark.skipif(
    not _DATA_DIR.exists(),
    reason="data/v1/ not present; skipping dataset integrity tests",
)


# ---------------------------------------------------------------------------
# Test 1: streaming read of 1000 examples
# ---------------------------------------------------------------------------


def test_streaming_read_1000() -> None:
    """Stream 1000 examples from data/v1/ and assert all fields are valid.

    Checks:
    - scene.object_types has shape [12] (N_MAX for 12-vocab dataset)
    - scene.poses has shape [12, 7]
    - scene.presence has shape [12]
    - description is a non-empty string
    - report dict contains required validity keys and accepted=True for all
    - no exception during iteration
    """
    from data.reader import ShardReader

    reader = ShardReader(str(_DATA_DIR))
    count = 0

    for scene, desc, report, sdf in reader:
        # Tensor shapes
        assert scene.object_types.shape == torch.Size([12]), (
            f"Expected object_types shape [12], got {scene.object_types.shape}"
        )
        assert scene.poses.shape == torch.Size([12, 7]), (
            f"Expected poses shape [12, 7], got {scene.poses.shape}"
        )
        assert scene.presence.shape == torch.Size([12]), (
            f"Expected presence shape [12], got {scene.presence.shape}"
        )

        # Non-empty description
        assert isinstance(desc, str) and len(desc) > 0, (
            f"Empty or non-string description at example {count}"
        )

        # Required report keys
        required_keys = {
            "accepted", "no_interpenetration", "stable_rest",
            "ik_reachable", "rrt_solvable",
        }
        missing = required_keys - set(report.keys())
        assert not missing, f"Report missing keys {missing} at example {count}"

        # All stored scenes must be accepted
        assert report["accepted"] is True, (
            f"Stored scene at index {count} has accepted=False (only valid scenes should be stored)"
        )

        count += 1
        if count >= 1000:
            break

    assert count == 1000, f"Reader exhausted before 1000 examples: got {count}"


# ---------------------------------------------------------------------------
# Test 2: tensor round-trip via ShardWriter -> ShardReader
# ---------------------------------------------------------------------------


def test_tensor_round_trip(tmp_path: Path) -> None:
    """Write 5 scenes via ShardWriter, read back via ShardReader.

    Asserts:
    - scene.object_types matches bit-exactly (LongTensor equality)
    - scene.poses matches bit-exactly (FloatTensor equality)
    - description matches string-exactly
    """
    import numpy as np

    from data.reader import ShardReader
    from data.sampler import ProceduralSampler
    from data.writer import ShardWriter
    from validator.core import SceneValidator

    output_dir = tmp_path / "shards"
    seed = 42
    n = 5

    sampler = ProceduralSampler(seed=seed)
    validator = SceneValidator(rrt_budget_s=0.5)
    rng = np.random.default_rng(seed)
    desc_rng = np.random.default_rng(seed + 1)

    written: list[tuple] = []

    with ShardWriter(output_dir, shard_size_mb=1.0, resume=False) as writer:
        attempts = 0
        while writer.accepted_count < n and attempts < 500:
            candidate = sampler.sample()
            rrt_seed = int(rng.integers(0, 2**31))
            report = validator.validate(candidate.scene, rrt_seed=rrt_seed)
            attempts += 1
            if report.accepted:
                from data.descriptions import generate_description
                desc = generate_description(candidate, desc_rng)
                report_dict: dict = {
                    "accepted": True,
                    "no_interpenetration": report.no_interpenetration,
                    "stable_rest": report.stable_rest,
                    "ik_reachable": report.ik_reachable,
                    "rrt_solvable": report.rrt_solvable,
                    "task_family": candidate.task_family,
                }
                writer.write(candidate.scene, desc, report_dict)
                written.append((candidate.scene, desc))

    assert len(written) == n, f"Only produced {len(written)}/{n} valid scenes in {attempts} attempts"

    reader = ShardReader(str(output_dir))
    read_back: list[tuple] = []
    for scene_r, desc_r, _report_r, _sdf_r in reader:
        read_back.append((scene_r, desc_r))

    assert len(read_back) == n, f"Expected {n} scenes from reader, got {len(read_back)}"

    for i, ((scene_w, desc_w), (scene_r, desc_r)) in enumerate(zip(written, read_back)):
        assert torch.equal(scene_w.object_types, scene_r.object_types), (
            f"object_types mismatch at index {i}"
        )
        assert torch.equal(scene_w.poses, scene_r.poses), (
            f"poses mismatch at index {i}"
        )
        assert desc_w == desc_r, (
            f"Description mismatch at index {i}: {desc_w!r} vs {desc_r!r}"
        )
