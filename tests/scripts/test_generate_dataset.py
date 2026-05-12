"""Tests for scripts/generate_dataset.py.

Tests cover the importable logic units:
  - Config loading and override parsing.
  - _report_to_dict: all fields, NaN/inf, ik_solution as list.
  - _config_hash: different configs produce different hashes.
  - Integration smoke (num_workers=1): generate 5 accepted scenes end-to-end,
    verify the output directory contains valid shards that the reader can open.
    Marked @pytest.mark.slow to allow skipping in fast CI.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest
import torch

# generate_dataset.py lives in scripts/, not in src/ or a package.
# Add the scripts directory to sys.path for the import.
_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import generate_dataset as gd  # noqa: E402

from data.reader import ShardReader  # noqa: E402
from validator.core import ValidityReport  # noqa: E402

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path: pathlib.Path) -> None:
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("target_n: 100\nnum_workers: 2\n")
        cfg = gd._load_config(str(cfg_file), [])
        assert cfg["target_n"] == 100
        assert cfg["num_workers"] == 2

    def test_override_int(self, tmp_path: pathlib.Path) -> None:
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("target_n: 100\n")
        cfg = gd._load_config(str(cfg_file), ["target_n=999"])
        assert cfg["target_n"] == 999

    def test_override_float(self, tmp_path: pathlib.Path) -> None:
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("rrt_budget_s: 5.0\n")
        cfg = gd._load_config(str(cfg_file), ["rrt_budget_s=2.0"])
        assert cfg["rrt_budget_s"] == 2.0

    def test_override_bool_false(self, tmp_path: pathlib.Path) -> None:
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("wandb:\n  enabled: true\n")
        cfg = gd._load_config(str(cfg_file), ["enabled=false"])
        assert cfg["enabled"] is False

    def test_override_invalid_format_raises(self, tmp_path: pathlib.Path) -> None:
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("target_n: 100\n")
        with pytest.raises(ValueError, match="key=value"):
            gd._load_config(str(cfg_file), ["bad_override_no_equals"])


# ---------------------------------------------------------------------------
# _config_hash
# ---------------------------------------------------------------------------


class TestConfigHash:
    def test_same_config_same_hash(self) -> None:
        cfg = {"target_n": 100, "rrt_budget_s": 5.0}
        assert gd._config_hash(cfg) == gd._config_hash(cfg)

    def test_different_config_different_hash(self) -> None:
        cfg1 = {"target_n": 100}
        cfg2 = {"target_n": 999}
        assert gd._config_hash(cfg1) != gd._config_hash(cfg2)

    def test_hash_is_12_chars(self) -> None:
        assert len(gd._config_hash({"x": 1})) == 12


# ---------------------------------------------------------------------------
# _report_to_dict
# ---------------------------------------------------------------------------


class TestReportToDict:
    def _make_report(self, **kwargs) -> ValidityReport:
        defaults = dict(
            no_interpenetration=True,
            stable_rest=True,
            ik_reachable=True,
            rrt_solvable=True,
            min_signed_distance_m=0.05,
            max_pose_drift_trans_m=1e-5,
            max_pose_drift_rot_rad=1e-4,
            ik_solution=np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]),
            rrt_seed=7,
            error="",
            elapsed_s=1.23,
        )
        defaults.update(kwargs)
        return ValidityReport(**defaults)

    def test_accepted_is_bool(self) -> None:
        r = self._make_report()
        d = gd._report_to_dict(r)
        assert d["accepted"] is True

    def test_ik_solution_as_list(self) -> None:
        r = self._make_report()
        d = gd._report_to_dict(r)
        assert isinstance(d["ik_solution"], list)
        assert len(d["ik_solution"]) == 6

    def test_ik_solution_none(self) -> None:
        r = self._make_report(ik_solution=None)
        d = gd._report_to_dict(r)
        assert d["ik_solution"] is None

    def test_nan_encoded_as_string(self) -> None:
        r = self._make_report(min_signed_distance_m=float("nan"))
        d = gd._report_to_dict(r)
        assert d["min_signed_distance_m"] == "nan"

    def test_inf_encoded_as_string(self) -> None:
        r = self._make_report(min_signed_distance_m=float("inf"))
        d = gd._report_to_dict(r)
        assert d["min_signed_distance_m"] == "inf"

    def test_finite_floats_exact(self) -> None:
        v = 0.12345678901234567
        r = self._make_report(elapsed_s=v)
        d = gd._report_to_dict(r)
        # Must survive JSON round-trip exactly.
        d2 = json.loads(json.dumps(d))
        assert d2["elapsed_s"] == v

    def test_json_serialisable(self) -> None:
        r = self._make_report()
        d = gd._report_to_dict(r)
        # Must not raise.
        json.dumps(d)


# ---------------------------------------------------------------------------
# Integration smoke (slow): 5 accepted scenes written and readable
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGenerationIntegration:
    """End-to-end generation with num_workers=1 and target_n=5.

    Verifies that the harness produces a readable shard directory when called
    directly (bypassing argparse). Uses generate_dataset internals rather than
    subprocess to stay within pytest coverage.
    """

    def test_generates_five_scenes(self, tmp_path: pathlib.Path) -> None:
        """5 accepted scenes are written and readable with strict field equality."""
        from data.sampler import ProceduralSampler
        from data.writer import ShardWriter
        from scene.schema import WorkspaceBounds
        from validator.core import SceneValidator

        seed = 0
        rng = np.random.default_rng(seed + 2)
        desc_rng = np.random.default_rng(seed + 1)
        sampler = ProceduralSampler(seed=seed)
        validator = SceneValidator(workspace_bounds=WorkspaceBounds.default(), rrt_budget_s=5.0)

        output_dir = tmp_path / "shards"
        target_n = 5
        written: list[tuple] = []

        with ShardWriter(output_dir, shard_size_mb=1.0, resume=False) as writer:
            attempts = 0
            while writer.accepted_count < target_n and attempts < 300:
                candidate = sampler.sample()
                rrt_seed = int(rng.integers(0, 2**31))
                report = validator.validate(candidate.scene, rrt_seed=rrt_seed)
                attempts += 1
                if report.accepted:
                    desc = generate_description(candidate, desc_rng)
                    d = gd._report_to_dict(report)
                    writer.write(candidate.scene, desc, d)
                    written.append((candidate.scene, desc, d))

        assert writer.accepted_count == target_n, (
            f"Expected {target_n} accepted, got {writer.accepted_count} in {attempts} attempts"
        )

        samples = list(ShardReader(output_dir))
        assert len(samples) == target_n

        for i, (scene_r, desc_r, rep_r, _sdf_r) in enumerate(samples):
            scene_w, desc_w, rep_w = written[i]
            assert torch.equal(scene_w.object_types, scene_r.object_types)
            assert torch.equal(scene_w.poses, scene_r.poses)
            assert desc_w == desc_r


def generate_description(candidate, rng):
    from data.descriptions import generate_description as _gd
    return _gd(candidate, rng)
