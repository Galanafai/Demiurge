# Combined v1+v2 Dataset Card (v9 Training Corpus)

Generated: 2026-05-18
Total scenes: 234,720
Status: LOCKED for v9 training

## Composition

- v1 (Week 2 baseline): ~50,000 scenes (~21.3%)
- v2 (May 2026 run): ~184,720 scenes (~78.7%)

## Splits (LOCKED — DO NOT REGENERATE)

| Split | Count | Percent |
|---|---|---|
| Train | 220,638 | 94.0% |
| Val | 7,041 | 3.0% |
| Heldout | 7,041 | 3.0% |

Split method: `scene_idx % 100` (deterministic, reproducible)
Locked date: 2026-05-18

## Object Type Distribution

| type_id | count | % |
|---|---|---|
| 0 | 67,084 | 10.0% |
| 1 | 68,277 | 10.2% |
| 2 | 65,957 | 9.8% |
| 3 | 55,257 | 8.2% |
| 4 | 58,858 | 8.8% |
| 5 | 54,960 | 8.2% |
| 6 | 51,398 | 7.7% |
| 7 | 62,678 | 9.3% |
| 8 | 40,838 | 6.1% |
| 9 | 49,832 | 7.4% |
| 10 | 50,702 | 7.5% |
| 11 | 45,728 | 6.8% |

Balance ratio: 1.7x (min=6.1%, max=10.2%)

Note: template_name not stored in shard metadata (written to generation log only).
All scenes are tabletop_reach / cluttered_pick / obstacle_avoidance variants.

## Validation Parameters

- RRT budget: 1.0s (BiRRT)
- Drake: pydrake latest stable
- Validator checks: collision, stable_rest, IK, RRT plan
- N_MAX: 12 objects per scene
- Vocabulary: 12 object types

## Files

| File | Size | Description |
|---|---|---|
| shard_manifest.json | ~1 KB | Shard paths for v1 + v2 |
| split_manifest.json | ~1 KB | Locked train/val/heldout assignment |
| train_indices.json | ~1 MB | Explicit train index list |
| val_indices.json | ~30 KB | Explicit val index list |
| heldout_indices.json | ~30 KB | Explicit heldout index list |
| text_embeddings.pt | 362 MB | MiniLM-L6-v2 (384-dim) for all 234k scenes |

## Reproducibility

Split is deterministic: `scene_idx % 100`, iterating shards in manifest order.
DO NOT regenerate — this is the locked benchmark for v7 vs v9 comparison.

W&B sources:
- `galanafai-self/demiurge/dataset_v1_tars:latest`
- `galanafai-self/demiurge/dataset_v2_tars:latest`
