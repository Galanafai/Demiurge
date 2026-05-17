"""compute_per_type_accuracy.py -- Gate 3 measured correctly.

For each (prompt, scene) record in the VLM probe output:
- Parse the prompt for object-type keywords using the vocab name list
- Check if the generated scene's type_ids contain the expected type(s)
- Aggregate: exact_match (all mentioned types present), partial_match,
  miss (no mentioned type present), and no_named_type (prompt has no
  vocab keyword -- skip from accuracy calc)

This avoids the averaging-artifact failure that caused the Phase 4
type-collapse false alarm.

Usage:
    python3 scripts/compute_per_type_accuracy.py \
        --probe-output artifacts/conditional_v7_vlm_probe.json \
        --out artifacts/v7_per_type_accuracy.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Vocabulary: (type_id, canonical_name, keyword_aliases)
# Aliases are words/phrases that appear in natural language prompts
# and unambiguously refer to this type. Multi-word phrases are checked
# as substrings after lowercasing. Single words are matched as whole words.
VOCAB: list[tuple[int, str, list[str]]] = [
    (0,  "CUBE",            ["cube"]),
    (1,  "SPHERE",          ["sphere", "ball"]),
    (2,  "CYLINDER",        ["cylinder"]),
    (3,  "BOX_TALL",        ["tall box", "cracker box", "cracker_box"]),
    (4,  "BOX_FLAT",        ["flat box", "box_flat"]),
    (5,  "MUSTARD_BOTTLE",  ["mustard bottle", "mustard", "mustard_bottle"]),
    (6,  "SUGAR_BOX",       ["sugar box", "sugar_box", "sugar"]),
    (7,  "TOMATO_SOUP_CAN", ["tomato soup can", "tomato soup", "tomato_soup", "tomato"]),
    (8,  "BLEACH_CLEANSER", ["bleach cleanser", "bleach", "cleanser", "bleach_cleanser"]),
    (9,  "BANANA",          ["banana"]),
    (10, "MASTER_CHEF_CAN", ["master chef can", "master chef", "master_chef"]),
    (11, "GELATIN_BOX",     ["gelatin box", "gelatin_box", "gelatin"]),
]

# Build lookup: (keyword -> type_id), longest-match wins
_KEYWORD_MAP: list[tuple[str, int]] = []
for _tid, _name, _aliases in VOCAB:
    for _alias in _aliases:
        _KEYWORD_MAP.append((_alias, _tid))
# Sort by length descending so multi-word phrases match before subwords
_KEYWORD_MAP.sort(key=lambda x: -len(x[0]))


def extract_type_ids_from_prompt(prompt: str) -> list[int]:
    """Return type IDs whose keyword appears in the prompt.

    Uses longest-match to avoid "mustard" matching before "mustard bottle".
    Each type ID is reported at most once per prompt.
    """
    low = prompt.lower()
    found: set[int] = set()
    consumed_spans: list[tuple[int, int]] = []

    for keyword, tid in _KEYWORD_MAP:
        # Find all non-overlapping occurrences
        for m in re.finditer(re.escape(keyword), low):
            span = (m.start(), m.end())
            # Check not already covered by a longer match
            if any(s <= span[0] and span[1] <= e for s, e in consumed_spans):
                continue
            consumed_spans.append(span)
            found.add(tid)

    return sorted(found)


def compute_accuracy(records: list[dict]) -> dict:
    """Compute per-type match statistics from probe records."""
    exact_match = 0
    partial_match = 0
    miss = 0
    no_named_type = 0

    per_type_tp: Counter[int] = Counter()
    per_type_fn: Counter[int] = Counter()  # expected but not generated

    for rec in records:
        desc = rec.get("description", "")
        generated_tids: set[int] = set(rec.get("type_ids", []))
        expected_tids = extract_type_ids_from_prompt(desc)

        if not expected_tids:
            no_named_type += 1
            continue

        expected_set = set(expected_tids)
        hits = expected_set & generated_tids
        misses = expected_set - generated_tids

        for t in hits:
            per_type_tp[t] += 1
        for t in misses:
            per_type_fn[t] += 1

        if hits == expected_set:
            exact_match += 1
        elif hits:
            partial_match += 1
        else:
            miss += 1

    scorable = exact_match + partial_match + miss
    exact_rate = exact_match / max(1, scorable)
    partial_rate = partial_match / max(1, scorable)
    miss_rate = miss / max(1, scorable)

    # Per-type recall
    per_type_recall: dict[str, float] = {}
    for tid, name, _ in VOCAB:
        tp = per_type_tp[tid]
        fn = per_type_fn[tid]
        total = tp + fn
        per_type_recall[name] = tp / max(1, total) if total > 0 else None  # type: ignore[assignment]

    return {
        "n_records": len(records),
        "n_scorable": scorable,
        "n_no_named_type": no_named_type,
        "exact_match": exact_match,
        "partial_match": partial_match,
        "miss": miss,
        "exact_rate": exact_rate,
        "partial_rate": partial_rate,
        "miss_rate": miss_rate,
        "per_type_recall": per_type_recall,
        "per_type_tp": dict(per_type_tp),
        "per_type_fn": dict(per_type_fn),
    }


def print_report(result: dict) -> None:
    print("\n=== Per-Type Accuracy Report ===")
    print(f"  Total records:       {result['n_records']}")
    print(f"  Scorable (has named type): {result['n_scorable']}")
    print(f"  No named type:       {result['n_no_named_type']}")
    print()
    print(f"  Exact match:   {result['exact_match']:4d}  ({result['exact_rate']*100:5.1f}%)")
    print(f"  Partial match: {result['partial_match']:4d}  ({result['partial_rate']*100:5.1f}%)")
    print(f"  Miss:          {result['miss']:4d}  ({result['miss_rate']*100:5.1f}%)")
    print()
    print("  Per-type recall (generated | expected):")
    print(f"  {'Type':>20}  {'Recall':>8}  {'TP':>5}  {'FN':>5}")
    for tid, name, _ in VOCAB:
        recall = result["per_type_recall"].get(name)
        tp = result["per_type_tp"].get(tid, 0)
        fn = result["per_type_fn"].get(tid, 0)
        if recall is None:
            print(f"  {name:>20}: {'N/A':>8}  {tp:>5}  {fn:>5}  (not in any prompt)")
        else:
            status = "PASS" if recall >= 0.4 else "fail"
            print(f"  {name:>20}: {recall*100:>7.1f}%  {tp:>5}  {fn:>5}  [{status}]")
    print()
    gate_pass = result["exact_rate"] >= 0.40
    print(f"  Gate 3 (exact_rate >= 40%): {'PASS' if gate_pass else 'FAIL'}")
    print(f"  (exact_rate = {result['exact_rate']*100:.1f}%, threshold = 40%)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probe-output", required=True, help="JSONL or JSON from probe_vlm.py")
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    probe_path = Path(args.probe_output)
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe output not found: {probe_path}")

    # probe_vlm outputs a JSON with a "records" list
    raw = json.loads(probe_path.read_text())
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and "records" in raw:
        records = raw["records"]
    else:
        raise ValueError(f"Unrecognized format in {probe_path}: top-level keys = {list(raw.keys())[:5]}")

    print(f"Loaded {len(records)} records from {probe_path.name}")
    result = compute_accuracy(records)
    print_report(result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
