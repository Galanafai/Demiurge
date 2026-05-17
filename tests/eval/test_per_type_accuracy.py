"""Unit tests for compute_per_type_accuracy.py.

Tests:
1. Keyword extraction from prompts
2. Exact/partial/miss classification logic
3. No-named-type handling
4. Vocab boundary conditions (out-of-vocab terms)
"""
from __future__ import annotations

import pytest

from scripts.compute_per_type_accuracy import (
    compute_accuracy,
    extract_type_ids_from_prompt,
)

# ---------------------------------------------------------------------------
# extract_type_ids_from_prompt tests
# ---------------------------------------------------------------------------


class TestExtractTypeIds:
    def test_cube(self) -> None:
        assert extract_type_ids_from_prompt("place a cube on the table") == [0]

    def test_sphere(self) -> None:
        assert extract_type_ids_from_prompt("put a sphere here") == [1]

    def test_mustard_bottle_full_phrase(self) -> None:
        # "mustard bottle" should map to type 5
        assert extract_type_ids_from_prompt("place a mustard bottle") == [5]

    def test_mustard_word_only(self) -> None:
        # "mustard" alone also maps to type 5
        assert extract_type_ids_from_prompt("retrieve the mustard") == [5]

    def test_multi_type_prompt(self) -> None:
        # "arrange a sugar box and a banana" -> types 6 and 9
        ids = extract_type_ids_from_prompt("arrange a sugar box and a banana")
        assert 6 in ids and 9 in ids

    def test_no_type(self) -> None:
        # Generic prompt without named object
        assert extract_type_ids_from_prompt("pick an object at 30cm forward") == []

    def test_mug_not_in_vocab(self) -> None:
        # "mug" is not in the vocabulary (0-11 only)
        assert extract_type_ids_from_prompt("place a mug on the table") == []

    def test_tomato_soup_can(self) -> None:
        # Long phrase -- should match type 7
        ids = extract_type_ids_from_prompt("set a tomato soup can at 20cm")
        assert ids == [7]

    def test_tomato_alone(self) -> None:
        # "tomato" alone should map to type 7
        ids = extract_type_ids_from_prompt("grab the tomato")
        assert 7 in ids

    def test_gelatin_box(self) -> None:
        ids = extract_type_ids_from_prompt("place a gelatin box")
        assert ids == [11]

    def test_banana(self) -> None:
        ids = extract_type_ids_from_prompt("pick the banana from the area")
        assert ids == [9]

    def test_bleach_cleanser(self) -> None:
        ids = extract_type_ids_from_prompt("retrieve the bleach cleanser")
        assert ids == [8]

    def test_master_chef_can(self) -> None:
        ids = extract_type_ids_from_prompt("place a master chef can at center")
        assert ids == [10]


# ---------------------------------------------------------------------------
# compute_accuracy tests
# ---------------------------------------------------------------------------


def _make_record(description: str, type_ids: list[int]) -> dict:
    return {
        "description": description,
        "type_ids": type_ids,
        "vlm_score": None,
        "drake_accepted": False,
    }


class TestComputeAccuracy:
    def test_exact_match(self) -> None:
        records = [
            _make_record("place a cube", [0, 3]),  # cube=0 present
        ]
        result = compute_accuracy(records)
        assert result["exact_match"] == 1
        assert result["partial_match"] == 0
        assert result["miss"] == 0

    def test_partial_match(self) -> None:
        # Prompt mentions cube AND sphere, scene only has cube
        records = [
            _make_record("place a cube and sphere", [0, 5]),
        ]
        result = compute_accuracy(records)
        assert result["partial_match"] == 1
        assert result["exact_match"] == 0
        assert result["miss"] == 0

    def test_miss(self) -> None:
        # Prompt mentions cube, scene has only mustard bottle
        records = [
            _make_record("place a cube", [5, 11]),
        ]
        result = compute_accuracy(records)
        assert result["miss"] == 1
        assert result["exact_match"] == 0

    def test_no_named_type(self) -> None:
        records = [
            _make_record("pick an object at 30cm forward", [0, 1]),
        ]
        result = compute_accuracy(records)
        assert result["n_no_named_type"] == 1
        assert result["n_scorable"] == 0

    def test_exact_rate_calculation(self) -> None:
        records = [
            _make_record("place a cube", [0]),          # exact
            _make_record("place a sphere", [5]),         # miss
            _make_record("pick an object", [0]),         # no_named_type
        ]
        result = compute_accuracy(records)
        assert result["n_scorable"] == 2
        assert result["exact_match"] == 1
        assert result["miss"] == 1
        assert result["exact_rate"] == pytest.approx(0.5)

    def test_per_type_recall(self) -> None:
        records = [
            _make_record("place a banana", [9]),   # tp for banana
            _make_record("place a banana", [5]),   # fn for banana
        ]
        result = compute_accuracy(records)
        recall = result["per_type_recall"]["BANANA"]
        assert recall == pytest.approx(0.5)

    def test_empty_records(self) -> None:
        result = compute_accuracy([])
        assert result["n_records"] == 0
        assert result["exact_rate"] == 0.0
