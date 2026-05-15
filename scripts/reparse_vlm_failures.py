"""Re-parse failed VLM samples from an existing probe_vlm.py artifact.

Reads the artifact JSON, finds records where vlm_score is None and
vlm_raw_response is non-empty, and re-applies the new fence-stripping
parser. Writes an updated artifact in-place (or to --output).

This recovers parse failures without re-calling the API.

Usage:
    python3 scripts/reparse_vlm_failures.py \\
        --input artifacts/conditional_v6_vlm_probe.json \\
        --output artifacts/conditional_v6_vlm_probe_reparsed.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _strip_fences(text: str) -> str:
    """Strip markdown code fences."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _try_parse(raw: str) -> tuple[int, str] | None:
    """Attempt to parse a score/reasoning from raw VLM response text.

    Tries fence-stripped then raw. Returns (score, reasoning) or None.
    """
    for candidate in (_strip_fences(raw), raw):
        m = re.search(r'\{.*?"score".*?\}', candidate, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                score = int(parsed["score"])
                reasoning = str(parsed.get("reasoning", ""))
                return score, reasoning
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to existing probe_vlm artifact JSON")
    p.add_argument("--output", default="", help="Output path (default: overwrite input)")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path

    print(f"Loading {in_path} ...")
    data = json.loads(in_path.read_text())

    records: list[dict] = data.get("records", [])
    print(f"Records: {len(records)}")

    n_failed_before = sum(1 for r in records if r.get("vlm_score") is None)
    n_raw_available = sum(
        1 for r in records
        if r.get("vlm_score") is None and r.get("vlm_raw_response")
    )
    print(f"  vlm_score=None: {n_failed_before}")
    print(f"  with raw_response available: {n_raw_available}")

    recovered = 0
    still_failed = 0
    new_scores: list[int] = []

    for r in records:
        if r.get("vlm_score") is not None:
            new_scores.append(r["vlm_score"])
            continue

        raw = r.get("vlm_raw_response", "")
        if not raw:
            still_failed += 1
            continue

        result = _try_parse(raw)
        if result is not None:
            score, reasoning = result
            r["vlm_score"] = score
            r["vlm_reasoning"] = reasoning
            r["vlm_reparsed"] = True
            new_scores.append(score)
            recovered += 1
        else:
            still_failed += 1

    print(f"\nRecovered: {recovered}")
    print(f"Still failed (no parse): {still_failed}")
    print(f"Still failed (no raw): {n_failed_before - n_raw_available}")

    # Recompute gate metrics from updated records.
    all_scores = [r["vlm_score"] for r in records if r.get("vlm_score") is not None]
    joint_ok = [
        r for r in records
        if r.get("vlm_score") is not None and r.get("vlm_score", 0) >= 4 and r.get("drake_accepted")
    ]

    gate1_mean = sum(all_scores) / len(all_scores) if all_scores else None
    gate2_frac = len(joint_ok) / len(records) if records else None

    print("\n=== UPDATED GATE METRICS ===")
    if gate1_mean is not None:
        g1 = "PASS" if gate1_mean > 2.0 else "FAIL"
        print(f"  Gate 1 VLM mean:   {gate1_mean:.3f} (threshold >2.0) [{g1}]")
    if gate2_frac is not None:
        g2 = "PASS" if gate2_frac > 0.05 else "FAIL"
        print(f"  Gate 2 Joint ok:   {gate2_frac:.1%} (threshold >5%) [{g2}]")
    gate3_pass = data.get("gate3_pass", False)
    print(f"  Gate 3 Type div:   {'PASS' if gate3_pass else 'FAIL'} (unchanged)")

    from collections import Counter
    dist = Counter(all_scores)
    print(f"  Score distribution: {dict(sorted(dist.items()))}")
    print(f"  n scored: {len(all_scores)}/{len(records)}")

    # Update artifact.
    data["gate1_vlm_mean"] = gate1_mean
    data["gate1_pass"] = gate1_mean is not None and gate1_mean > 2.0
    data["gate2_joint_frac"] = gate2_frac
    data["gate2_pass"] = gate2_frac is not None and gate2_frac > 0.05
    data["vlm_n_scored"] = len(all_scores)
    data["records"] = records

    out_path.write_text(json.dumps(data, indent=2))
    print(f"\nUpdated artifact written to {out_path}")


if __name__ == "__main__":
    main()
