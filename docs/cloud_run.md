# Cloud Run: Demiurge 50k Dataset Generation

**Target instance:** Hetzner CCX43 (16 dedicated AMD cores, 64 GB RAM, NVMe SSD, ~$0.21/hr)
**Fallback:** Hetzner CCX33 (8 dedicated AMD cores, 32 GB RAM) if CCX43 is unavailable
**OS:** Ubuntu 24.04 LTS (Hetzner image)
**Expected runtime:** ~8.1h (2s RRT budget, 15 workers, 9.8% acceptance rate)
**Estimated cost:** ~$1.70-3.50 total (8-17h including setup and teardown)

> [!IMPORTANT]
> Do not provision the instance until the re-profile results in `artifacts/validator_profile.md`
> have been reviewed and the gate override has been explicitly approved.
> Gate override for the 12-vocab run was approved by Galanafai on 2026-05-12.

---

## Prerequisites

- Hetzner Cloud account with billing set up
- SSH key uploaded to Hetzner Cloud (Settings > SSH Keys)
- `hcloud` CLI installed locally (optional; web UI path is the default below)
- `rsync` available on your local machine

---

## Step 1: Provision the Instance

### Option A: Web UI (recommended for first-time use)

1. Log in to [console.hetzner.cloud](https://console.hetzner.cloud)
2. Select your project (or create one named `demiurge`)
3. Click **Add Server**
4. Configure:
   - **Location:** Falkenstein (FSN1) or Nuremberg (NBG1) -- pick whichever is available
   - **Image:** Ubuntu 24.04
   - **Type:** Dedicated vCPU > **CCX43** (16 vCPU, 64 GB, NVMe)
     - If CCX43 is sold out: fall back to **CCX33** (8 vCPU, 32 GB) and reduce `num_workers` to 7 in `configs/dataset/v1.yaml`
   - **SSH Keys:** select your uploaded key
   - **Name:** `demiurge-gen`
5. Click **Create & Buy Now**
6. Wait for the instance to show **Running** (typically <60 seconds)
7. Note the public IPv4 address

### Option B: hcloud CLI (reproducible / automatable)

```bash
# Create server (adjust ssh-key-name to match your uploaded key)
hcloud server create \
    --name demiurge-gen \
    --type ccx43 \
    --image ubuntu-24.04 \
    --location fsn1 \
    --ssh-key <your-ssh-key-name>

# Get the IP
hcloud server describe demiurge-gen | grep "Public Net"
```

> [!NOTE]
> CCX43 Hetzner type string is `ccx43`. CCX33 fallback is `ccx33`.
> Run `hcloud server-type list` to confirm availability in your chosen location.

---

## Step 2: SSH and Bootstrap

```bash
# Replace <INSTANCE_IP> with the actual IPv4 from Step 1.
ssh root@<INSTANCE_IP>
```

Once connected, run the bootstrap script. The recommended approach is to paste
the script directly (avoids git auth for a private repo) or use scp:

```bash
# On your LOCAL machine:
scp scripts/cloud_setup.sh root@<INSTANCE_IP>:/root/cloud_setup.sh

# On the INSTANCE:
bash /root/cloud_setup.sh
```

The bootstrap script:
- Installs system deps, uv, clones the repo, runs `uv sync`
- Runs a Drake smoke test and a Demiurge schema smoke test (asserts N_MAX=12, vocab=12)
- Creates `logs/` and `data/v1/`, warns if `data/v1/` is non-empty
- Prints the production run command on exit

Expected bootstrap time: 5-10 minutes (dominated by `uv sync` pulling pydrake).

> [!IMPORTANT]
> Before starting generation, verify `data/v1/` is empty or absent:
> ```bash
> ls /root/Demiurge/data/v1/
> ```
> If it contains any `.tar` shards from a previous run, remove them:
> ```bash
> rm -rf /root/Demiurge/data/v1 && mkdir -p /root/Demiurge/data/v1
> ```
> Starting with stale shards will corrupt the manifest.

---

## Step 3: Start Generation in tmux

Always run the generation inside `tmux` so it survives SSH disconnection.

```bash
# On the INSTANCE:
cd /root/Demiurge

# Start a tmux session
tmux new -s gen

# Inside tmux, start the run
nohup uv run python scripts/generate_dataset.py \
    --config configs/dataset/v1.yaml \
    --seed 42 \
    > logs/generate_v1.log 2>&1 &

PID=$!
echo "Generation PID: $PID"
echo $PID > logs/generate_v1.pid
```

Detach from tmux: `Ctrl-b d`

Reattach later: `tmux attach -t gen`

### Monitor progress

```bash
# Follow the log (from any SSH session)
tail -f /root/Demiurge/logs/generate_v1.log

# System resource check
htop

# Accepted scene count (from manifest)
python3 -c "
import json, pathlib
m = pathlib.Path('/root/Demiurge/data/v1/manifest.json')
if m.exists():
    d = json.loads(m.read_text())
    print(f\"Shards: {d.get('n_shards', 0)}, Scenes: {d.get('n_total', 0)}\")
else:
    print('No manifest yet.')
"
```

### Expected checkpoints

| Time | Expected scenes accepted |
|---|---|
| +1h | ~6,200 |
| +4h | ~24,800 |
| +8h | ~49,600 (near completion) |
| +8.5h | 50,000 (done) |

These are estimates based on 1.718 acc/s on CCX43 at 2s budget. Actual rate
may vary by +/-20% depending on AMD core performance and memory bandwidth.

---

## Step 4: Verify and Retrieve Data

When the log shows `Generation complete` or `target_n reached`:

### Verify the manifest on the instance

```bash
python3 -c "
import json, pathlib, hashlib

manifest_path = pathlib.Path('/root/Demiurge/data/v1/manifest.json')
d = json.loads(manifest_path.read_text())

print(f'Scenes accepted: {d[\"n_total\"]}')
print(f'Shards: {d[\"n_shards\"]}')
print(f'Dataset hash: {d.get(\"dataset_hash\", \"N/A\")}')

# Verify shard files exist
missing = []
for shard in d.get('shards', []):
    p = manifest_path.parent / shard['filename']
    if not p.exists():
        missing.append(shard['filename'])
if missing:
    print(f'MISSING SHARDS: {missing}')
else:
    print('All shards present.')
"
```

### rsync to local machine

```bash
# On your LOCAL machine:
rsync -avz --progress \
    root@<INSTANCE_IP>:/root/Demiurge/data/v1/ \
    ./data/v1/

# Also retrieve the log
rsync -avz root@<INSTANCE_IP>:/root/Demiurge/logs/generate_v1.log ./logs/
```

Verify the local copy matches the remote manifest before destroying the instance:

```bash
# On LOCAL:
python3 -c "
import json, pathlib

manifest = json.loads(pathlib.Path('data/v1/manifest.json').read_text())
n_shards = manifest['n_shards']
local_shards = list(pathlib.Path('data/v1').glob('*.tar'))
print(f'Manifest shards: {n_shards}, Local shards: {len(local_shards)}')
assert len(local_shards) == n_shards, 'Shard count mismatch -- do not destroy instance'
print('Shard count OK. Safe to destroy.')
"
```

---

## Step 5: Destroy the Instance (Kill Switch)

> [!CAUTION]
> Only destroy the instance AFTER verifying the local shard count matches the manifest.
> Instance destruction is irreversible. There is no Hetzner undo.

### Option A: Web UI

1. Go to [console.hetzner.cloud](https://console.hetzner.cloud) > Your Project > Servers
2. Click `demiurge-gen` > Actions > **Delete**
3. Confirm deletion

### Option B: hcloud CLI

```bash
hcloud server delete demiurge-gen
```

### Verify deletion

```bash
hcloud server list | grep demiurge-gen
# Should return nothing.
```

---

## CCX33 Fallback Procedure

If CCX43 is unavailable in your chosen datacenter:

1. Provision CCX33 instead (8 dedicated AMD cores, 32 GB RAM, ~$0.10/hr)
2. Update `configs/dataset/v1.yaml`:
   ```yaml
   num_workers: 7   # CCX33: 8 cores minus 1 for OS/writer
   ```
3. The CCX33 50k projection at 2s budget is ~15.0h (vs 8.1h on CCX43).
   Cost: ~$1.50. Still within the approved 20h wall-clock gate.
4. Run the same `cloud_setup.sh` bootstrap and generation command unchanged.

> [!NOTE]
> The CCX43 -> CCX33 worker reduction (15 -> 7) is the only required change.
> `target_n`, `rrt_budget_s`, and `seed` remain unchanged.

---

## Decision Record

| Parameter | Value | Rationale |
|---|---|---|
| Instance | CCX43 (16 cores, 64 GB) | Dedicated cores, no noisy-neighbor throttling |
| Workers | 15 | 16 cores minus 1 for OS + writer; writer bottleneck caps at ~13x single-worker |
| RRT budget | 2s | 9.8% acceptance; 8.1h CCX43 projection passes <12h gate |
| Target | 50,000 scenes | Full production dataset for Week 3 training |
| Vocab | 12 objects (IDs 0-11) | 4 large-volume entries dropped after 16-vocab profiling |
| Seed | 42 | Fixed; logged in W&B run for reproducibility |
| Cost estimate | ~$1.70-3.50 | CCX43 at $0.21/hr for 8-17h including setup/teardown |

Gate override: per-template <20% acceptance rate override approved 2026-05-12.
See `artifacts/validator_profile.md` for full profiling record and rationale.
