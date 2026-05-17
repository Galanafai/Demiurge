"""Diagnostic 2: Training data text-type correlation.

Audits whether specific type indices have disproportionate text
co-occurrence, which would explain learned type bias.

Reports:
- Type frequency distribution across 5k sampled scenes
- Top-5 words by lift (P(word|type) / P(type)) for each type
- Identification of whether type 5 (tomato_soup_can) is
  over-represented in text vs other types
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
from data.reader import ShardReader

word_type_counts: dict[str, Counter[int]] = defaultdict(Counter)
type_totals: Counter[int] = Counter()
n_scenes = 0

print("Scanning training data (up to 5000 scenes)...")
for scene, desc, *_ in ShardReader("data/v1"):
    words = set(desc.lower().split()) if desc else set()
    for i in range(len(scene.object_types)):
        if scene.presence[i]:
            tid = int(scene.object_types[i])
            type_totals[tid] += 1
            for w in words:
                word_type_counts[w][tid] += 1
    n_scenes += 1
    if n_scenes >= 5000:
        break

print(f"Scanned {n_scenes} scenes\n")

total_objs = sum(type_totals.values())
print("Type frequency distribution:")
print(f"  {'type':>6}  {'count':>7}  {'pct':>6}  bar")
for tid in sorted(type_totals.keys()):
    pct = 100 * type_totals[tid] / total_objs
    bar = "#" * int(pct / 2)
    print(f"  type {tid:2d}: {type_totals[tid]:7d}  {pct:5.1f}%  {bar}")

print()
print("Most predictive words per type (lift = P(word|type) / baseline_type_frac):")
for tid in sorted(type_totals.keys()):
    if type_totals[tid] == 0:
        continue
    type_frac = type_totals[tid] / total_objs
    lifts: dict[str, float] = {}
    for word, cnt in word_type_counts.items():
        # Only consider words that appear with this type at least 5 times
        if cnt[tid] < 5:
            continue
        p_word_given_type = cnt[tid] / type_totals[tid]
        lifts[word] = p_word_given_type / type_frac
    top5 = sorted(lifts.items(), key=lambda x: -x[1])[:5]
    words_str = "  ".join(f"{w}={l:.1f}x" for w, l in top5)
    print(f"  type {tid:2d}: {words_str}")

# Check if type 5 has excess text coverage
print()
print("--- Type 5 (tomato_soup_can) vs type 6 (mustard_bottle) text coverage ---")
for tid, label in [(5, "tomato_soup_can"), (6, "mustard_bottle")]:
    # Count scenes where type appears in text
    name_words = label.replace("_", " ").split()
    text_hits = sum(
        1 for w in name_words
        for cnt in [word_type_counts[w]]
        if cnt.get(tid, 0) > 0
    )
    print(f"  type {tid} ({label}): {type_totals[tid]} objects, "
          f"text references for '{name_words[0]}' = "
          f"{word_type_counts[name_words[0]][tid]}")

print("\nDiagnostic 2 complete.")
