"""Tests for src/scene/vocab.py."""

import pytest

from scene.vocab import OBJECT_VOCAB, ObjectTypeId, build_sdf


class TestVocabLength:
    def test_vocab_has_eight_entries(self) -> None:
        assert len(OBJECT_VOCAB) == 8

    def test_all_type_ids_present(self) -> None:
        for tid in ObjectTypeId:
            assert int(tid) in OBJECT_VOCAB, f"Missing ObjectTypeId {tid!r}"


class TestBoundingRadii:
    def test_all_bounding_radii_positive(self) -> None:
        for tid, entry in OBJECT_VOCAB.items():
            assert entry.bounding_radius_m > 0, (
                f"Entry {entry.name} (id={tid}) has non-positive bounding_radius_m"
            )

    def test_all_canonical_half_extents_positive(self) -> None:
        for tid, entry in OBJECT_VOCAB.items():
            for i, v in enumerate(entry.canonical_half_extents_m):
                assert v > 0, (
                    f"Entry {entry.name} (id={tid}) canonical_half_extents_m[{i}]={v} <= 0"
                )


class TestSdfRoundTrip:
    """Verify that every entry's SDF parses cleanly in Drake without exceptions."""

    def _parse_sdf_in_drake(self, sdf_string: str) -> int:
        """Return num_bodies after parsing; raises on any Drake error."""
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
        from pydrake.systems.framework import DiagramBuilder

        builder = DiagramBuilder()
        plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        parser = Parser(plant)
        parser.AddModelsFromString(sdf_string, "sdf")
        plant.Finalize()
        return int(plant.num_bodies())

    @pytest.mark.parametrize("type_id", list(ObjectTypeId))
    def test_build_sdf_parses_in_drake(self, type_id: ObjectTypeId) -> None:
        entry = OBJECT_VOCAB[int(type_id)]
        sdf = build_sdf(entry, model_name=f"test_{entry.name}", scale=1.0, mass=0.5)
        num_bodies = self._parse_sdf_in_drake(sdf)
        # World body + 1 link = 2
        assert num_bodies == 2, (
            f"Expected 2 bodies (world + link) for {entry.name}, got {num_bodies}"
        )

    @pytest.mark.parametrize("type_id", list(ObjectTypeId))
    def test_build_sdf_at_non_unit_scale(self, type_id: ObjectTypeId) -> None:
        entry = OBJECT_VOCAB[int(type_id)]
        for scale in (0.5, 1.5, 2.0):
            sdf = build_sdf(entry, model_name=f"test_{entry.name}_s{scale}", scale=scale)
            self._parse_sdf_in_drake(sdf)  # must not raise


class TestSdfKind:
    def test_sdf_kind_values_are_valid(self) -> None:
        valid_kinds = {"box", "sphere", "cylinder"}
        for tid, entry in OBJECT_VOCAB.items():
            assert entry.sdf_kind in valid_kinds, (
                f"Entry {entry.name} (id={tid}) has unknown sdf_kind '{entry.sdf_kind}'"
            )

    def test_primitive_entries_have_correct_kinds(self) -> None:
        assert OBJECT_VOCAB[ObjectTypeId.CUBE].sdf_kind == "box"
        assert OBJECT_VOCAB[ObjectTypeId.SPHERE].sdf_kind == "sphere"
        assert OBJECT_VOCAB[ObjectTypeId.CYLINDER].sdf_kind == "cylinder"
        assert OBJECT_VOCAB[ObjectTypeId.BOX_TALL].sdf_kind == "box"
        assert OBJECT_VOCAB[ObjectTypeId.BOX_FLAT].sdf_kind == "box"
