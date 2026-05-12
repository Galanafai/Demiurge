"""Tests for src/scene/vocab.py."""

import pytest

from scene.vocab import OBJECT_VOCAB, ObjectTypeId, build_sdf


class TestVocabLength:
    def test_vocab_has_sixteen_entries(self) -> None:
        assert len(OBJECT_VOCAB) == 16

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


# ---------------------------------------------------------------------------
# Week 2.5 additions: new YCB entry validation
# ---------------------------------------------------------------------------

_NEW_ENTRY_IDS = [
    ObjectTypeId.BLEACH_CLEANSER,
    ObjectTypeId.BANANA,
    ObjectTypeId.MASTER_CHEF_CAN,
    ObjectTypeId.GELATIN_BOX,
    ObjectTypeId.PUDDING_BOX,
    ObjectTypeId.CRACKER_BOX,
    ObjectTypeId.POTTED_MEAT_CAN,
    ObjectTypeId.POWER_DRILL,
]


class TestNewYcbEntries:
    """Per-entry validation for the 8 new Week 2.5 vocab additions."""

    @pytest.mark.parametrize("type_id", _NEW_ENTRY_IDS)
    def test_bounding_radius_covers_max_half_extent(self, type_id: ObjectTypeId) -> None:
        """bounding_radius_m must be >= the maximum half-extent of the geometry.

        For a cylinder entry, the effective bounding radius is sqrt(r^2 + (h/2)^2).
        For a box entry, it is the half-diagonal sqrt(hx^2 + hy^2 + hz^2).
        This test verifies the stored bounding_radius_m is at least as large as
        the maximum single half-extent (a weaker but fast algebraic check).
        """
        entry = OBJECT_VOCAB[int(type_id)]
        max_half = max(entry.canonical_half_extents_m)
        assert entry.bounding_radius_m >= max_half, (
            f"{entry.name}: bounding_radius_m={entry.bounding_radius_m:.4f} < "
            f"max_half_extent={max_half:.4f}"
        )

    @pytest.mark.parametrize("type_id", _NEW_ENTRY_IDS)
    def test_bounding_radius_equals_geometric_formula(self, type_id: ObjectTypeId) -> None:
        """bounding_radius_m must match the expected geometric formula within 1mm."""
        import math
        entry = OBJECT_VOCAB[int(type_id)]
        hx, hy, hz = entry.canonical_half_extents_m
        if entry.sdf_kind == "box":
            expected = math.sqrt(hx**2 + hy**2 + hz**2)
        elif entry.sdf_kind == "cylinder":
            # hx == hy == r; hz == h/2
            r, half_h = hx, hz
            expected = math.sqrt(r**2 + half_h**2)
        else:
            expected = math.sqrt(hx**2 + hy**2 + hz**2)
        assert abs(entry.bounding_radius_m - expected) < 0.001, (
            f"{entry.name}: bounding_radius_m={entry.bounding_radius_m:.4f}, "
            f"geometric formula={expected:.4f} (diff > 1mm)"
        )

    @pytest.mark.parametrize("type_id", _NEW_ENTRY_IDS)
    def test_sdf_kind_is_box_or_cylinder(self, type_id: ObjectTypeId) -> None:
        """New YCB entries use only box or cylinder primitives (no sphere)."""
        entry = OBJECT_VOCAB[int(type_id)]
        assert entry.sdf_kind in {"box", "cylinder"}, (
            f"{entry.name}: unexpected sdf_kind={entry.sdf_kind!r}"
        )

    @pytest.mark.parametrize("type_id", _NEW_ENTRY_IDS)
    def test_name_matches_enum_member(self, type_id: ObjectTypeId) -> None:
        """Entry name must match the enum member name (lowercase, underscores)."""
        entry = OBJECT_VOCAB[int(type_id)]
        expected_name = type_id.name.lower()
        assert entry.name == expected_name, (
            f"Entry name {entry.name!r} != enum name {expected_name!r} for id={type_id}"
        )


class TestColorMapCoverage:
    """Verify _TYPE_COLOR covers all 16 vocab entries with distinct values."""

    def test_color_map_covers_all_type_ids(self) -> None:
        from data.descriptions import _TYPE_COLOR
        for tid in ObjectTypeId:
            assert int(tid) in _TYPE_COLOR, (
                f"ObjectTypeId {tid!r} (id={int(tid)}) missing from _TYPE_COLOR"
            )

    def test_all_colors_distinct(self) -> None:
        from data.descriptions import _TYPE_COLOR
        colors = list(_TYPE_COLOR.values())
        assert len(colors) == len(set(colors)), (
            f"Duplicate colors in _TYPE_COLOR: {[c for c in colors if colors.count(c) > 1]}"
        )
