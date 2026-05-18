# Antigravity Master Prompt — Phase A: Data Preparation (v9)

You are continuing the Demiurge project for Phase A of the v9 build. Week 5
shipped Universal Guidance + evaluation harness on the 9M v7 model. v7 was
retrained from scratch (uncond 0.8%, cond 3.0% — different from original).
v2 dataset (184k scenes) was generated and uploaded to W&B.

Phase A: Combine v1 (50k) + v2 (184k) = 234k Drake-validated scenes into
unified training corpus with locked train/val/held-out splits.

# Pre-Flight: Branch Setup and Pod Verification

## Step 0.1: Create Phase A branch off main

ssh -tt -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ServerAliveInterval=30 f0xpg6tkynmbk9-64411a32@ssh.runpod.io 'bash --noprofile --norc'

In session:

cd /workspace/Demiurge
git fetch origin
git checkout main
git pull origin main
git log --oneline -3
git checkout -b week6/v9-phase-a

Verify:
- main has Week 5 merge at HEAD
- Local working tree on week6/v9-phase-a
- Clean status

## Step 0.2: Pod state verification

echo "=== Pod resumed state ==="
date
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
nproc
free -h
df -h /workspace

echo ""
echo "=== Critical artifacts preserved ==="
ls -lh checkpoints/conditional_v7/latest.pt 2>/dev/null && echo "v7 on disk: OK" || echo "v7 on disk: MISSING"
ls /workspace/Demiurge/data/v1/ | wc -l
du -sh /workspace/Demiurge/data/v1/

echo ""
echo "=== W&B connectivity and key artifacts ==="
python3 - << 'PYEOF'
import wandb
api = wandb.Api()
checks = [
    "galanafai-self/demiurge/conditional_v7_checkpoint:latest",
    "galanafai-self/demiurge/dataset_v1_tars:latest",
    "galanafai-self/demiurge/dataset_v2_tars:latest",
]
for path in checks:
    try:
        art = api.artifact(path)
        print(f"OK: {path}")
        print(f"  Size: {art.size / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"MISSING: {path} -- {e}")
PYEOF

echo ""
echo "=== Environment variables ==="
[ -n "$WANDB_API_KEY" ] && echo "WANDB_API_KEY: SET" || echo "WANDB_API_KEY: MISSING - source ~/.bashrc"

echo ""
echo "=== Free disk for combined dataset ==="
echo "Need ~6 GB free for v1 + v2 combined download"
df -h /workspace | tail -1

Pass criteria:
- WANDB_API_KEY set
- All 3 W&B artifacts accessible
- v7 checkpoint on disk OR confirmed on W&B
- Free disk > 10 GB
- 64 CPUs available

Halt conditions:
- Any W&B artifact missing: HALT
- Free disk < 6 GB: HALT (cleanup needed first)
- WANDB_API_KEY unset: source ~/.bashrc, retry

# Before Doing Anything Else

1. Read AGENTS.md and .agents/skills/demiurge/SKILL.md in full.
2. Read artifacts/week5_results.md (Week 5 final state).
3. Read artifacts/dataset_v1_card.md (v1 dataset statistics).
4. State in Plan Artifact:
   - v1 stats: 50,000 scenes, ~600 MB, 4 shards
   - v2 stats: 184,720 scenes, ~2.2 GB, 2 shards
   - Combined target: 234,720 scenes
   - Split: 220k/7k/7k (train/val/held-out)
   - Phase A success criterion: combined dataset loads + locked splits saved

# Phase A Execution

## Step A.1: Download v1 and v2 datasets from W&B

Create combined data directory:

mkdir -p /workspace/Demiurge/data/combined

Download v1 (if not already on disk):

python3 - << 'PYEOF'
import wandb
import os

api = wandb.Api()

# Check if v1 already on disk
v1_path = "/workspace/Demiurge/data/v1"
if os.path.exists(v1_path) and len(os.listdir(v1_path)) >= 4:
    print(f"v1 already on disk: {len(os.listdir(v1_path))} files")
else:
    print("Downloading v1 from W&B...")
    art = api.artifact("galanafai-self/demiurge/dataset_v1_tars:latest")
    art.download(root=v1_path)
    print(f"v1 downloaded to {v1_path}")

# Download v2
v2_path = "/workspace/Demiurge/data/v2"
if os.path.exists(v2_path) and len(os.listdir(v2_path)) >= 2:
    print(f"v2 already on disk: {len(os.listdir(v2_path))} files")
else:
    print("Downloading v2 from W&B...")
    art = api.artifact("galanafai-self/demiurge/dataset_v2_tars:latest")
    art.download(root=v2_path)
    print(f"v2 downloaded to {v2_path}")

print("\nDownload complete.")
PYEOF

Verify downloads:
ls -lh /workspace/Demiurge/data/v1/
ls -lh /workspace/Demiurge/data/v2/
du -sh /workspace/Demiurge/data/v1/ /workspace/Demiurge/data/v2/

## Step A.2: Combined corpus verification

python3 - << 'PYEOF'
import sys
sys.path.insert(0, "src")
from data.reader import ShardReader
from collections import Counter
import os

# Check v1
v1_files = sorted([f for f in os.listdir("/workspace/Demiurge/data/v1") if f.endswith(".tar")])
print(f"v1 shards: {v1_files}")

# Check v2
v2_files = sorted([f for f in os.listdir("/workspace/Demiurge/data/v2") if f.endswith(".tar")])
print(f"v2 shards: {v2_files}")

# Sample 100 random scenes from each, verify tensor shape consistency
print("\nVerifying v1 tensor format...")
v1_reader = ShardReader("/workspace/Demiurge/data/v1")
v1_count = 0
v1_shapes = Counter()
v1_template_counts = Counter()
for scene_tuple in v1_reader:
    scene = scene_tuple[0] if isinstance(scene_tuple, tuple) else scene_tuple
    v1_shapes[tuple(scene.positions.shape)] += 1
    if hasattr(scene, "template_name"):
        v1_template_counts[scene.template_name] += 1
    v1_count += 1
    if v1_count >= 200:
        break

print(f"v1 sample: {v1_count} scenes")
print(f"v1 shapes: {dict(v1_shapes)}")
print(f"v1 templates: {dict(v1_template_counts)}")

print("\nVerifying v2 tensor format...")
v2_reader = ShardReader("/workspace/Demiurge/data/v2")
v2_count = 0
v2_shapes = Counter()
v2_template_counts = Counter()
for scene_tuple in v2_reader:
    scene = scene_tuple[0] if isinstance(scene_tuple, tuple) else scene_tuple
    v2_shapes[tuple(scene.positions.shape)] += 1
    if hasattr(scene, "template_name"):
        v2_template_counts[scene.template_name] += 1
    v2_count += 1
    if v2_count >= 200:
        break

print(f"v2 sample: {v2_count} scenes")
print(f"v2 shapes: {dict(v2_shapes)}")
print(f"v2 templates: {dict(v2_template_counts)}")

# Verify tensor formats match
if v1_shapes != v2_shapes:
    print("\nFAIL: Tensor shapes differ between v1 and v2")
    print("v1:", v1_shapes)
    print("v2:", v2_shapes)
    sys.exit(1)
else:
    print(f"\nPASS: Tensor shapes consistent")
PYEOF

Pass criteria:
- v1 has 4 shards
- v2 has 2 shards
- Tensor shapes identical between datasets
- Both datasets have valid scene objects

Halt if:
- Shape mismatch (would corrupt training)
- Either dataset fails to load

## Step A.3: Build combined corpus directory

Decision: don't physically merge shards (wastes disk). Use a manifest approach.

mkdir -p /workspace/Demiurge/data/combined

python3 - << 'PYEOF'
import json
import os
from pathlib import Path

# Build manifest pointing at both source dirs
v1_shards = sorted(Path("/workspace/Demiurge/data/v1").glob("*.tar"))
v2_shards = sorted(Path("/workspace/Demiurge/data/v2").glob("*.tar"))

manifest = {
    "version": "combined_v1v2",
    "total_target_scenes": 234720,
    "v1_scenes": 50000,
    "v2_scenes": 184720,
    "shards": [
        {"path": str(s), "source": "v1", "estimated_scenes": 12500} for s in v1_shards
    ] + [
        {"path": str(s), "source": "v2", "estimated_scenes": 92360} for s in v2_shards
    ],
    "rrt_budget_s": 1.0,
    "validator_version": "rrt_birrt_drake",
    "n_max": 12,
    "n_types": 12,
}

with open("/workspace/Demiurge/data/combined/shard_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Manifest created with {len(manifest['shards'])} shards")
print(f"Total scenes: {manifest['total_target_scenes']}")
PYEOF

cat /workspace/Demiurge/data/combined/shard_manifest.json

## Step A.4: Train/val/held-out split

This is the locked benchmark. Once set, do not regenerate.

python3 - << 'PYEOF'
import sys
sys.path.insert(0, "src")
import json
import hashlib
from data.reader import ShardReader

# Iterate through all shards, compute deterministic split based on scene hash
reader = ShardReader("/workspace/Demiurge/data/combined", manifest="shard_manifest.json")

train_indices = []
val_indices = []
heldout_indices = []

# Deterministic split: hash of scene index modulo 100
# 94% train (0-93), 3% val (94-96), 3% heldout (97-99)
print("Building train/val/heldout split...")
scene_idx = 0
for scene_tuple in reader:
    bucket = scene_idx % 100
    if bucket < 94:
        train_indices.append(scene_idx)
    elif bucket < 97:
        val_indices.append(scene_idx)
    else:
        heldout_indices.append(scene_idx)
    scene_idx += 1

    if scene_idx % 50000 == 0:
        print(f"  Processed {scene_idx} scenes")

print(f"\nFinal split:")
print(f"  Total: {scene_idx}")
print(f"  Train: {len(train_indices)} ({100*len(train_indices)/scene_idx:.1f}%)")
print(f"  Val:   {len(val_indices)} ({100*len(val_indices)/scene_idx:.1f}%)")
print(f"  Heldout: {len(heldout_indices)} ({100*len(heldout_indices)/scene_idx:.1f}%)")

split_manifest = {
    "version": "split_v1v2_locked",
    "total_scenes": scene_idx,
    "train_count": len(train_indices),
    "val_count": len(val_indices),
    "heldout_count": len(heldout_indices),
    "split_method": "scene_idx % 100",
    "train_buckets": list(range(0, 94)),
    "val_buckets": list(range(94, 97)),
    "heldout_buckets": list(range(97, 100)),
    "locked_date": "2026-05-18",
    "note": "DO NOT REGENERATE - this is the locked benchmark for v7 vs v9 comparison",
}

with open("/workspace/Demiurge/data/combined/split_manifest.json", "w") as f:
    json.dump(split_manifest, f, indent=2)

# Also save explicit index lists for reproducibility
with open("/workspace/Demiurge/data/combined/train_indices.json", "w") as f:
    json.dump(train_indices, f)
with open("/workspace/Demiurge/data/combined/val_indices.json", "w") as f:
    json.dump(val_indices, f)
with open("/workspace/Demiurge/data/combined/heldout_indices.json", "w") as f:
    json.dump(heldout_indices, f)

print(f"\nSplit manifest saved.")
print(f"Index lists saved (train: {len(train_indices)}, val: {len(val_indices)}, heldout: {len(heldout_indices)})")
PYEOF

Pass criteria:
- Total > 230,000 scenes
- Train: ~220k
- Val: ~7k
- Heldout: ~7k

## Step A.5: Text embeddings cache

Compute MiniLM-L6-v2 embeddings for all descriptions.

# Check existing v1 embeddings cache
ls -lh /workspace/Demiurge/data/v1/text_embeddings.pt 2>/dev/null && echo "v1 embeddings cached: OK" || echo "v1 embeddings need recompute"

python3 - << 'PYEOF'
import sys
sys.path.insert(0, "src")
import torch
from pathlib import Path
from model.text_encoder import TextEncoder
from data.reader import ShardReader

# Initialize encoder
print("Loading TextEncoder (MiniLM-L6-v2)...")
encoder = TextEncoder()
encoder.eval()

# Iterate all scenes, collect descriptions
print("Collecting descriptions from combined corpus...")
reader = ShardReader("/workspace/Demiurge/data/combined", manifest="shard_manifest.json")

descriptions = []
scene_idx = 0
for scene_tuple in reader:
    if isinstance(scene_tuple, tuple) and len(scene_tuple) > 1:
        desc = scene_tuple[1] if isinstance(scene_tuple[1], str) else None
        if desc:
            descriptions.append(desc)
    scene_idx += 1
    if scene_idx % 50000 == 0:
        print(f"  Collected {len(descriptions)} descriptions from {scene_idx} scenes")

print(f"\nTotal descriptions: {len(descriptions)}")

# Batch encode
print("Encoding descriptions in batches...")
batch_size = 256
all_embeddings = []
for i in range(0, len(descriptions), batch_size):
    batch = descriptions[i:i+batch_size]
    with torch.no_grad():
        emb = encoder.encode(batch)
    all_embeddings.append(emb.cpu())
    if (i // batch_size) % 50 == 0:
        print(f"  {i + len(batch)}/{len(descriptions)}")

embeddings = torch.cat(all_embeddings, dim=0)
print(f"\nFinal embedding tensor: {embeddings.shape}")

# Save
out_path = "/workspace/Demiurge/data/combined/text_embeddings.pt"
torch.save({
    "embeddings": embeddings,
    "descriptions": descriptions,
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dim": embeddings.shape[1],
}, out_path)

print(f"Cached to {out_path}")
print(f"File size: {Path(out_path).stat().st_size / 1024 / 1024:.1f} MB")
PYEOF

## Step A.6: Per-template balance audit

python3 - << 'PYEOF'
import sys
sys.path.insert(0, "src")
import json
from collections import Counter
from data.reader import ShardReader

reader = ShardReader("/workspace/Demiurge/data/combined", manifest="shard_manifest.json")

template_counts = Counter()
type_counts = Counter()
scene_count = 0

print("Auditing combined corpus...")
for scene_tuple in reader:
    scene = scene_tuple[0] if isinstance(scene_tuple, tuple) else scene_tuple
    if hasattr(scene, "template_name"):
        template_counts[scene.template_name] += 1
    for i in range(len(scene.object_types)):
        if scene.presence[i]:
            type_counts[int(scene.object_types[i])] += 1
    scene_count += 1
    if scene_count % 50000 == 0:
        print(f"  {scene_count}")

print(f"\nTotal scenes: {scene_count}")
print(f"\nTemplate distribution:")
for tmpl, count in template_counts.most_common():
    pct = 100 * count / scene_count
    print(f"  {tmpl}: {count} ({pct:.1f}%)")

print(f"\nType distribution (across all objects):")
total_objects = sum(type_counts.values())
for tid, count in sorted(type_counts.items()):
    pct = 100 * count / total_objects
    print(f"  type_id={tid}: {count} ({pct:.1f}%)")

# Save audit
audit = {
    "total_scenes": scene_count,
    "templates": dict(template_counts),
    "object_types": {str(k): v for k, v in type_counts.items()},
    "total_objects": total_objects,
}
with open("/workspace/Demiurge/artifacts/v9_data_audit.json", "w") as f:
    json.dump(audit, f, indent=2)

# Check balance
min_pct = min(100*v/total_objects for v in type_counts.values())
max_pct = max(100*v/total_objects for v in type_counts.values())
print(f"\nType balance: min={min_pct:.1f}%, max={max_pct:.1f}%, ratio={max_pct/min_pct:.1f}x")

if max_pct / min_pct > 5.0:
    print("WARNING: Severe type imbalance detected. May need class-balanced sampler in Phase B.")
else:
    print("OK: Type distribution within acceptable balance (<5x ratio)")
PYEOF

## Step A.7: Dataset card

python3 - << 'PYEOF'
import json
from pathlib import Path
from datetime import datetime

with open("/workspace/Demiurge/data/combined/split_manifest.json") as f:
    split = json.load(f)
with open("/workspace/Demiurge/artifacts/v9_data_audit.json") as f:
    audit = json.load(f)

card = f"""# Combined v1+v2 Dataset Card (v9 Training Corpus)

Generated: {datetime.now().strftime('%Y-%m-%d')}
Total scenes: {split['total_scenes']:,}
Status: LOCKED for v9 training

## Composition

- v1 (Week 2 baseline): 50,000 scenes (~21.3%)
- v2 (recent generation): {split['total_scenes'] - 50000:,} scenes (~{100*(split['total_scenes']-50000)/split['total_scenes']:.1f}%)

## Splits (LOCKED)

| Split | Count | Percent | Buckets |
|---|---|---|---|
| Train | {split['train_count']:,} | {100*split['train_count']/split['total_scenes']:.1f}% | 0-93 |
| Val | {split['val_count']:,} | {100*split['val_count']/split['total_scenes']:.1f}% | 94-96 |
| Heldout | {split['heldout_count']:,} | {100*split['heldout_count']/split['total_scenes']:.1f}% | 97-99 |

Split method: scene_idx % 100 (deterministic, reproducible)

## Template Distribution

"""

for tmpl, count in audit["templates"].items():
    pct = 100 * count / audit["total_scenes"]
    card += f"- {tmpl}: {count:,} ({pct:.1f}%)\n"

card += "\n## Object Type Distribution\n\n"
total_objs = audit["total_objects"]
for tid, count in sorted(audit["object_types"].items(), key=lambda x: int(x[0])):
    pct = 100 * count / total_objs
    card += f"- type_id={tid}: {count:,} ({pct:.1f}%)\n"

card += f"""
## Validation Parameters

- RRT budget: 1.0s
- Validator: Drake BiRRT + IK + collision check + stable_rest
- N_MAX: 12 objects per scene
- N types: 12

## Files

- shard_manifest.json: which shards belong to combined corpus
- split_manifest.json: train/val/heldout bucket assignment
- train_indices.json: explicit train index list
- val_indices.json: explicit val index list  
- heldout_indices.json: explicit heldout index list
- text_embeddings.pt: cached MiniLM-L6-v2 embeddings

## Reproducibility

Re-running this split code on the same shards will produce identical splits.
The split is locked and MUST NOT be regenerated to maintain v7 vs v9 comparability.

Source artifacts:
- W&B: dataset_v1_tars:latest
- W&B: dataset_v2_tars:latest
"""

out = "/workspace/Demiurge/artifacts/dataset_combined_v1v2_card.md"
with open(out, "w") as f:
    f.write(card)

print(f"Dataset card saved: {out}")
print(f"\n{card[:500]}...")
PYEOF

## Step A.8: Sanity check - load training split

Quick validation that the splits are usable for training:

python3 - << 'PYEOF'
import sys
sys.path.insert(0, "src")
import json
import time
from data.reader import ShardReader

with open("/workspace/Demiurge/data/combined/train_indices.json") as f:
    train_indices = set(json.load(f))

reader = ShardReader("/workspace/Demiurge/data/combined", manifest="shard_manifest.json")

print(f"Sampling 1000 scenes from train split (out of {len(train_indices)})...")
t0 = time.time()
loaded = 0
for idx, scene_tuple in enumerate(reader):
    if idx not in train_indices:
        continue
    loaded += 1
    if loaded >= 1000:
        break

elapsed = time.time() - t0
print(f"Loaded 1000 train scenes in {elapsed:.1f}s ({1000/elapsed:.1f} scenes/sec)")

if loaded < 1000:
    print("FAIL: Couldn't load 1000 train scenes")
    sys.exit(1)
else:
    print("PASS: Train split iterable")
PYEOF

## Step A.9: Verification pass

cd /workspace/Demiurge

echo "=== ruff ==="
ruff check . 2>&1 | tail -5

echo ""
echo "=== pytest (quick) ==="
pytest tests/data/ -q --tb=no 2>&1 | tail -10

# Phase A Decision Gate

Required to proceed to Phase B:

- [ ] data/combined/shard_manifest.json exists
- [ ] data/combined/split_manifest.json exists with locked splits
- [ ] data/combined/text_embeddings.pt cached (~80 MB for 234k descriptions)
- [ ] Total scenes >= 230,000
- [ ] Train >= 215,000
- [ ] Val >= 6,000
- [ ] Heldout >= 6,000
- [ ] No critical type imbalance (max/min ratio < 10x)
- [ ] artifacts/dataset_combined_v1v2_card.md exists
- [ ] tests/data/ pass

If all gates pass: commit and proceed to Phase B.
If any fail: surface, do not proceed.

## Step A.10: Commit

cd /workspace/Demiurge
git add data/combined/shard_manifest.json
git add data/combined/split_manifest.json
git add artifacts/dataset_combined_v1v2_card.md
git add artifacts/v9_data_audit.json
git commit -m "data: prepare combined v1+v2 corpus for v9 training

- 234k Drake-validated scenes total (v1: 50k + v2: 184k)
- Locked train/val/heldout split (220k/7k/7k)
- MiniLM-L6-v2 text embeddings cached
- Per-template and per-type audit complete

Phase A complete. Ready for Phase B (v9 unconditional training)."

git push origin week6/v9-phase-a 2>&1 || echo "push failed (no creds on pod, expected)"

# Final Status

echo ""
echo "=================================================="
echo "PHASE A COMPLETE - DATA PREPARATION"
echo "=================================================="
echo ""
echo "Deliverables:"
echo "  - data/combined/ with manifests and embeddings"
echo "  - Locked train/val/heldout splits"
echo "  - Dataset card and audit"
echo ""
echo "Combined corpus:"
python3 -c "
import json
with open('/workspace/Demiurge/data/combined/split_manifest.json') as f:
    s = json.load(f)
print(f'  Total: {s[\"total_scenes\"]:,}')
print(f'  Train: {s[\"train_count\"]:,}')
print(f'  Val: {s[\"val_count\"]:,}')
print(f'  Heldout: {s[\"heldout_count\"]:,}')
"
echo ""
echo "Disk usage:"
du -sh /workspace/Demiurge/data/combined/
du -sh /workspace/Demiurge/data/v1/
du -sh /workspace/Demiurge/data/v2/
df -h /workspace | tail -1
echo ""
echo "Ready for Phase B: v9 unconditional training (49M params, 234k data)"
echo "=================================================="

exit

# Constraints

- python3 -u for all training-related commands
- DO NOT regenerate splits after Step A.4 (locked)
- DO NOT modify /workspace/Demiurge/data/v1/ or /workspace/Demiurge/data/v2/
- DO NOT delete W&B artifacts

# Halt Conditions

- W&B artifact missing: halt
- Tensor shape mismatch between v1 and v2: halt
- Total scenes < 230,000: halt
- Severe type imbalance (max/min > 10x): halt, may need class balancing in Phase B
- Free disk < 5 GB during processing: halt

# Cost Estimate

- Pod time: ~4 hours × $0.74/hr = ~$3
- VLM API: $0 (no scoring in Phase A)
- Total: ~$3

Begin with Step 0.1: branch setup and pod verification.
