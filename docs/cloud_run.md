# Cloud Run: Demiurge 50k Dataset Generation

**Primary provider:** Hetzner CCX33 (8 dedicated AMD cores, 32 GB RAM, ~$0.10/hr)
**Upgrade path:** Hetzner CCX43 (16 cores, 64 GB) if CCX33 proves too slow or CCX43 opens
**OS:** Ubuntu 24.04 LTS (Hetzner image)
**Expected runtime:** ~15h (2s RRT budget, 7 workers on CCX33)
**Estimated cost:** ~$1.50 total

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
   - **Location:** Ashburn (ASH, US East) -- preferred for rsync speed from Oakland.
     Hillsboro (HIL, US West) as second choice. EU locations (Falkenstein FSN1,
     Nuremberg NBG1) as last resort -- transatlantic rsync adds ~30-60 min.
   - **Image:** Ubuntu 24.04
   - **Type:** Dedicated vCPU > **CCX33** (8 vCPU, 32 GB RAM, NVMe)
     - If CCX43 (16 vCPU, 64 GB) is now available: use it instead and set
       `num_workers: 15` in `configs/dataset/v1.yaml`. Wall-clock drops to ~8h.
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
    --type ccx33 \
    --image ubuntu-24.04 \
    --location ash \
    --ssh-key <your-ssh-key-name>

# Get the IP
hcloud server describe demiurge-gen | grep "Public Net"
```

> [!NOTE]
> Type strings: `ccx33` (8 vCPU, primary), `ccx43` (16 vCPU, upgrade).
> US locations: `ash` (Ashburn, VA), `hil` (Hillsboro, OR).
> EU fallbacks: `fsn1` (Falkenstein), `nbg1` (Nuremberg).
> Run `hcloud server-type list` and `hcloud location list` to confirm availability.

---

## GitHub Authentication

The repository is **public** on GitHub. `cloud_setup.sh` uses HTTPS:

```
REPO_URL="https://github.com/Galanafai/Demiurge.git"
```

No credentials or SSH keys are required on the instance. `git clone` works without
any auth setup.

**If the repo is ever made private:** switch `REPO_URL` to
`git@github.com:Galanafai/Demiurge.git` and add before the clone step:

```bash
# scp ~/.ssh/demiurge_deploy_key root@<INSTANCE_IP>:~/.ssh/id_ed25519  (run locally first)
chmod 600 ~/.ssh/id_ed25519
ssh-keyscan github.com >> ~/.ssh/known_hosts
```

---

## Step 2: SSH and Bootstrap

```bash
# Replace <INSTANCE_IP> with the actual IPv4 from Step 1.
ssh root@<INSTANCE_IP>
```

Copy the bootstrap script and run it:

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

Detach from tmux: `Ctrl-b d` -- Reattach: `tmux attach -t gen`

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

### Expected checkpoints (CCX33, 7 workers)

| Time | Expected scenes accepted |
|---|---|
| +2h | ~6,800 |
| +7h | ~23,800 |
| +14h | ~47,700 |
| +15h | 50,000 (done) |

Estimates based on 0.925 acc/s on CCX33 at 2s budget (from profiling projection).
Actual rate may vary +/-20% depending on AMD core turbo and memory bandwidth.

---

## Step 4: Verify and Retrieve Data

When the log shows `Generation complete` or `target_n reached`:

### Verify the manifest on the instance

```bash
python3 -c "
import json, pathlib

manifest_path = pathlib.Path('/root/Demiurge/data/v1/manifest.json')
d = json.loads(manifest_path.read_text())

print(f'Scenes accepted: {d[\"n_total\"]}')
print(f'Shards: {d[\"n_shards\"]}')
print(f'Dataset hash: {d.get(\"dataset_hash\", \"N/A\")}')

missing = []
for shard in d.get('shards', []):
    p = manifest_path.parent / shard['filename']
    if not p.exists():
        missing.append(shard['filename'])
print('MISSING SHARDS:' if missing else 'All shards present.', missing or '')
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

Verify the local copy before destroying the instance:

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

Verify: `hcloud server list | grep demiurge-gen` -- should return nothing.

---

## Decision Record

| Parameter | Value | Rationale |
|---|---|---|
| Provider | Hetzner CCX33 | CCX43 unavailable; RunPod EU-RO-1 capacity failure after 1h wait |
| Instance | CCX33 (8 cores, 32 GB) | Available now; CCX43 is upgrade path if it opens |
| Workers | 7 | 8 cores minus 1 for OS + writer |
| RRT budget | 2s | 9.8% acceptance; ~15h CCX33 projection within 20h gate |
| Target | 50,000 scenes | Full production dataset for Week 3 training |
| Vocab | 12 objects (IDs 0-11) | 4 large-volume entries dropped after 16-vocab profiling |
| Seed | 42 | Fixed; logged in W&B run for reproducibility |
| Cost estimate | ~$1.50 | CCX33 at $0.10/hr for ~15h |

Gate override: per-template <20% acceptance rate override approved 2026-05-12.
See `artifacts/validator_profile.md` for full profiling record and rationale.

---

## Appendix: CCX43 Upgrade Path

If CCX43 (16 vCPU, 64 GB) becomes available after provisioning CCX33, or for
a future re-run:

1. Provision CCX43 instead: `--type ccx43` in the hcloud CLI command above
2. Update `configs/dataset/v1.yaml`:
   ```yaml
   num_workers: 15   # CCX43: 16 cores minus 1
   ```
3. Wall-clock drops from ~15h to ~8.1h. Cost increases to ~$1.70 (~$0.21/hr).
4. All other config and bootstrap steps unchanged.

---

## Appendix: RunPod (Week 3 GPU only)

RunPod remains the target for Week 3 GPU training runs. It is **not** used for
the Week 2 dataset generation due to a capacity failure: the EU-RO-1 16-vCPU
Compute pod returned "not enough free vcpu" after a 1-hour wait.

For GPU work in Week 3, see the Week 3 runbook (to be written). RunPod GPU pod
provisioning differs significantly from CPU pod provisioning and will be
documented separately.
