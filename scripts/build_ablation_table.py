"""Build the Phase A-prime ablation table from five probe_drake.py runs.

Reads artifacts/ablation_v6_{mode}.json for each text mode and produces
a markdown table + JSON summary.

Usage:
    python3 scripts/build_ablation_table.py \\
        --out artifacts/v6_ablation_table.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

MODES = [
    ("none",          "A: Unconditional"),
    ("type-only",     "B: Type-only"),
    ("position-only", "C: Position-only"),
    ("full-cond",     "D: Full conditioning"),
    ("held-out",      "E: Held-out descriptions"),
]

DIAGNOSTIC_CASES = """
## Diagnostic Case Identification

Based on ablation validity rates, identify the failure mode:

| Case | Pattern | Diagnosis | v7 Remedy |
|------|---------|-----------|-----------|
| 1 | A≈B≈C≈D >> E | Held-out prompts OOD | Augment training descriptions |
| 2 | A≈D, B >> E | Type signal broken | Increase type_ce weight, separate type head |
| 3 | A≈D, C >> E | Position signal broken | Explicit position loss term |
| 4 | A≈D, B≈C≈E (all low) | Conditioning broken entirely | Cross-attention architecture fix |
| 5 | A >> B≈C≈D≈E | Conditioning hurts generative quality | Classifier-free guidance (CFG) training |
| 6 | A≈B≈C≈D≈E (all high) | No conditioning failure | Phase B directly |
"""


def load_result(mode: str) -> dict | None:
    path = _ROOT / "artifacts" / f"ablation_v6_{mode}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_result_from(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def classify_case(rates: dict[str, float | None]) -> str:
    """Heuristic case classification from the five validity rates."""
    a = rates.get("none")
    b = rates.get("type-only")
    c = rates.get("position-only")
    d = rates.get("full-cond")
    e = rates.get("held-out")

    if any(v is None for v in [a, b, c, d, e]):
        return "INCOMPLETE - not all regimes have finished"

    cond_rates = [b, c, d, e]
    max_cond = max(cond_rates)  # type: ignore[type-var]
    min_cond = min(cond_rates)  # type: ignore[type-var]

    # Case 6: all high
    if min_cond > 0.08:
        return "Case 6: All conditioning regimes high -- proceed to Phase B"

    # Case 5: unconditional >> all conditioned
    if a is not None and a > 0.08 and max_cond < 0.04:
        return "Case 5: Conditioning degrades quality -- CFG training needed for v7"

    # Case 4: all conditioned low, unconditional also low
    if a is not None and max_cond < 0.04 and a < 0.04:
        return "Case 4: Everything low -- cross-attention architecture broken"

    # Case 2: type-only high, position-only low, full low
    if b is not None and c is not None and d is not None:
        if b > 0.06 and c < 0.04 and d < 0.04:
            return "Case 2: Type signal works, position broken -- add position loss"
        if c > 0.06 and b < 0.04 and d < 0.04:
            return "Case 3: Position signal works, type broken -- increase type_ce weight"

    # Case 1: held-out specifically low
    if e is not None and e < 0.03 and d is not None and d > 0.05:
        return "Case 1: Full-cond OK but held-out prompts OOD -- augment descriptions"

    return "MIXED -- review rates manually"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--version", default="v6",
        help="Model version tag (default: v6). Sets input file prefix to ablation_{version}_.",
    )
    p.add_argument(
        "--out", default=str(_ROOT / "artifacts" / "v6_ablation_table.md"),
    )
    p.add_argument("--json-out", default=None, help="Also write JSON summary")
    args = p.parse_args()
    version = args.version

    rows: list[dict] = []
    rates: dict[str, float | None] = {}

    for mode, label in MODES:
        path = _ROOT / "artifacts" / f"ablation_{version}_{mode}.json"
        r = load_result_from(path)
        if r is None:
            rows.append({"label": label, "mode": mode, "status": "MISSING"})
            rates[mode] = None
            continue

        vr = r["validity_rate"]
        n = r["n_scenes"]
        accepted = r["accepted"]
        rej = r["rejection_reasons"]
        top_rej = max(rej, key=lambda k: rej[k]) if sum(rej.values()) > 0 else "-"
        top_pct = 100 * rej.get(top_rej, 0) / max(1, n - accepted)

        rows.append({
            "label": label,
            "mode": mode,
            "n_scenes": n,
            "accepted": accepted,
            "validity_pct": f"{100 * vr:.1f}%",
            "validity_rate": vr,
            "top_rejection": top_rej,
            "top_rej_pct": f"{top_pct:.0f}%",
            "elapsed_s": r.get("total_elapsed_s", r.get("elapsed_s", "?")),
            "status": "OK",
        })
        rates[mode] = vr

    # ── Markdown table ─────────────────────────────────────────────────────────
    lines = [
        "# Phase A-prime Ablation Table — conditional_v6",
        "",
        "| Regime | Mode | N | Accepted | Validity | Top Rejection | Status |",
        "|--------|------|---|----------|----------|---------------|--------|",
    ]
    for row in rows:
        if row.get("status") == "MISSING":
            lines.append(
                f"| {row['label']} | `{row['mode']}` | - | - | - | - | **MISSING** |"
            )
        else:
            lines.append(
                f"| {row['label']} | `{row['mode']}` | {row['n_scenes']} "
                f"| {row['accepted']} | **{row['validity_pct']}** "
                f"| {row['top_rejection']} ({row['top_rej_pct']}) | OK |"
            )

    lines += ["", "## Rejection Breakdown Detail", ""]
    for row in rows:
        if row.get("status") == "MISSING":
            continue
        r = load_result_from(_ROOT / "artifacts" / f"ablation_{version}_{row['mode']}.json")
        if r is None:
            continue
        lines.append(f"### {row['label']}")
        rej = r["rejection_reasons"]
        total_rej = sum(rej.values())
        for reason, count in sorted(rej.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100 * count / max(1, total_rej)
                lines.append(f"- {reason}: {count} ({pct:.1f}%)")
        lines.append("")

    # ── Diagnostic case ────────────────────────────────────────────────────────
    case = classify_case(rates)
    lines += [
        "## Diagnostic Case",
        "",
        f"**Identified: {case}**",
        "",
    ]
    lines.append(DIAGNOSTIC_CASES)

    # ── Rate comparison ────────────────────────────────────────────────────────
    lines += ["## Rate Summary", ""]
    for mode, label in MODES:
        vr = rates.get(mode)
        if vr is not None:
            lines.append(f"- **{label}**: {100 * vr:.1f}%")
        else:
            lines.append(f"- **{label}**: MISSING")

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Ablation table written to {out_path}")
    print(f"\nDiagnostic: {case}")
    for mode, label in MODES:
        vr = rates.get(mode)
        vr_str = f"{100 * vr:.1f}%" if vr is not None else "MISSING"
        print(f"  {label}: {vr_str}")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.write_text(json.dumps({
            "rates": rates,
            "case": case,
            "rows": [r for r in rows if r.get("status") == "OK"],
        }, indent=2))
        print(f"JSON summary written to {json_path}")


if __name__ == "__main__":
    main()
