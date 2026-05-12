# Cloud Run: Demiurge 50k Dataset Generation

**Primary provider:** RunPod CPU pod
**Fallback:** Hetzner CCX43 / CCX33 (see Appendix)
**OS:** Ubuntu 24.04 (select during pod creation)
**Expected runtime:** ~8.1h (2s RRT budget, 15 workers on 16 vCPU pod)
**Estimated cost:** <$5 total at RunPod per-minute CPU rates

> [!IMPORTANT]
> Do not provision the pod until the re-profile results in `artifacts/validator_profile.md`
> have been reviewed and the gate override has been explicitly approved.
> Gate override for the 12-vocab run was approved by Galanafai on 2026-05-12.

---

## Prerequisites

- RunPod account with billing set up (runpod.io)
- SSH public key ready to paste during pod creation
- `rsync` available on your local machine

---

## Step 1: Provision the RunPod CPU Pod

1. Sign in to [runpod.io](https://www.runpod.io) and click **Pods** in the left sidebar
2. Click **+ Deploy**
3. At the top of the deployment page, switch the pod type to **CPU** (not GPU)
4. Select a **Community Cloud** CPU pod. Look for pods with:
   - **vCPU count:** 16 vCPU (preferred) or 8 vCPU (fallback)
   - **RAM:** ~32 GB or more
   - **Disk:** 50 GB minimum (dataset output will be ~20-30 GB)
   - **Template:** Ubuntu 22.04 or 24.04 -- select whichever is available;
     the bootstrap script targets Ubuntu 24.04 but is compatible with 22.04
5. Under **SSH Terminal Access**, paste your SSH public key
6. Click **Deploy**
7. Wait for pod status to show **Running** (typically <60 seconds)

> [!NOTE]
> If no 16 vCPU CPU pods are available in Community Cloud, use 8 vCPU and
> update `configs/dataset/v1.yaml`:
> ```yaml
> num_workers: 7   # 8 vCPU minus 1 for OS/writer
> ```
> The 8 vCPU projection at 2s budget is ~15h -- within the approved 20h gate.

---

## GitHub Authentication

The repository is **public** on GitHub. `cloud_setup.sh` uses HTTPS:

```
REPO_URL="https://github.com/Galanafai/Demiurge.git"
```

No credentials or SSH keys are required on the pod. `git clone` works without
any auth setup.

**If the repo is ever made private:** switch to SSH (`git@github.com:Galanafai/Demiurge.git`)
and add before the clone step:

```bash
# scp ~/.ssh/demiurge_deploy_key root@<POD_HOST>:~/.ssh/id_ed25519  (run locally first)
chmod 600 ~/.ssh/id_ed25519
ssh-keyscan github.com >> ~/.ssh/known_hosts
```

---

## Step 2: SSH into the Pod

RunPod exposes SSH via a non-standard port on their gateway host rather than
directly on the pod IP. After the pod is running:

1. In the RunPod console, click **Connect** on your pod
2. Copy the **SSH over exposed TCP** connection string. It looks like:
   ```
   ssh root@ssh.runpod.io -p <PORT>
   ```
   or for newer pods:
   ```
   ssh root@<POD_ID>-22.proxy.runpod.net
   ```
3. Connect:
   ```bash
   ssh root@ssh.runpod.io -p <PORT>
   # or whichever string RunPod gives you
   ```

> [!NOTE]
> The connection string changes each time you redeploy. Always copy it fresh
> from the RunPod console **Connect** button rather than reusing a saved string.

---

## Step 3: Bootstrap the Pod

Copy the setup script and run it:

```bash
# On your LOCAL machine (use the RunPod connection string from Step 2):
scp -P <PORT> scripts/cloud_setup.sh root@ssh.runpod.io:/root/cloud_setup.sh
```

For the newer proxy URL pattern:
```bash
scp scripts/cloud_setup.sh root@<POD_ID>-22.proxy.runpod.net:/root/cloud_setup.sh
```

Then on the pod:
```bash
bash /root/cloud_setup.sh
```

The bootstrap script:
- Installs system deps, uv, clones the repo, runs `uv sync`
- Runs a Drake smoke test and a Demiurge schema smoke test (asserts N_MAX=12, vocab=12)
- Creates `logs/` and `data/v1/`, warns if `data/v1/` is non-empty
- Prints the production run command on exit

Expected bootstrap time: 5-10 minutes (dominated by `uv sync` pulling pydrake).

> [!IMPORTANT]
> Before starting generation, verify `data/v1/` is empty:
> ```bash
> ls /root/Demiurge/data/v1/
> ```
> If it contains `.tar` shards from a previous attempt, clear them:
> ```bash
> rm -rf /root/Demiurge/data/v1 && mkdir -p /root/Demiurge/data/v1
> ```

---

## Step 4: Start Generation in tmux

Always run inside `tmux` so the job survives SSH disconnection.

```bash
# On the pod:
cd /root/Demiurge
tmux new -s gen

# Inside tmux:
nohup uv run python scripts/generate_dataset.py \
    --config configs/dataset/v1.yaml \
    --seed 42 \
    > logs/generate_v1.log 2>&1 &

PID=$!
echo "Generation PID: $PID"
echo $PID > logs/generate_v1.pid
```

Detach: `Ctrl-b d` -- Reattach: `tmux attach -t gen`

### Monitor progress

```bash
# Follow the log
tail -f /root/Demiurge/logs/generate_v1.log

# System check
htop

# Scene count from manifest
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

### Expected checkpoints (16 vCPU pod, 15 workers)

| Time | Expected scenes |
|---|---|
| +1h | ~6,200 |
| +4h | ~24,800 |
| +8h | ~49,600 |
| +8.5h | 50,000 (done) |

Estimates based on 1.718 acc/s at 2s budget. Actual rate may vary +/-20%.

> [!NOTE]
> RunPod CPU pods bill per-minute. If you want to sanity-check throughput
> before committing to the full run, let it run for 15-20 minutes, check the
> manifest scene count, and extrapolate. Cost for a 20-minute test: <$0.10.

---

## Step 5: Verify and Retrieve Data

When the log shows generation complete:

### Verify the manifest on the pod

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
# On your LOCAL machine (use your RunPod connection details):
rsync -avz --progress \
    -e "ssh -p <PORT>" \
    root@ssh.runpod.io:/root/Demiurge/data/v1/ \
    ./data/v1/

# Also retrieve the log
rsync -avz -e "ssh -p <PORT>" \
    root@ssh.runpod.io:/root/Demiurge/logs/generate_v1.log \
    ./logs/
```

For the proxy URL pattern:
```bash
rsync -avz --progress \
    root@<POD_ID>-22.proxy.runpod.net:/root/Demiurge/data/v1/ \
    ./data/v1/
```

Verify the local copy before destroying the pod:

```bash
# On LOCAL:
python3 -c "
import json, pathlib

manifest = json.loads(pathlib.Path('data/v1/manifest.json').read_text())
n_shards = manifest['n_shards']
local_shards = list(pathlib.Path('data/v1').glob('*.tar'))
print(f'Manifest shards: {n_shards}, Local shards: {len(local_shards)}')
assert len(local_shards) == n_shards, 'Shard count mismatch -- do not terminate pod'
print('Shard count OK. Safe to terminate.')
"
```

---

## Step 6: Terminate the Pod (Kill Switch)

> [!CAUTION]
> Only terminate AFTER verifying the local shard count matches the manifest.
> Pod termination is irreversible. RunPod does not have an undo.

1. Go to [runpod.io](https://www.runpod.io) > **Pods**
2. Find `demiurge-gen` (or whatever you named it)
3. Click the **...** menu > **Terminate**
4. Confirm termination

Billing stops immediately on termination.

---

## Decision Record

| Parameter | Value | Rationale |
|---|---|---|
| Provider | RunPod CPU pod | Hetzner CCX43 unavailable across all DCs; RunPod is also Week 3 GPU provider |
| Pod size | 16 vCPU, ~32 GB RAM | Matches CCX43 core count; fallback: 8 vCPU |
| Workers | 15 (or vCPU - 1) | Writer bottleneck caps throughput at ~(cores-1) |
| RRT budget | 2s | 9.8% acceptance; ~8.1h projection on 16 vCPU |
| Target | 50,000 scenes | Full production dataset for Week 3 training |
| Vocab | 12 objects (IDs 0-11) | 4 large-volume entries dropped after 16-vocab profiling |
| Seed | 42 | Fixed; logged in W&B run for reproducibility |

Gate override: per-template <20% acceptance rate override approved 2026-05-12.
See `artifacts/validator_profile.md` for full profiling record and rationale.

---

## Appendix: Hetzner CCX43 (Fallback)

If RunPod CPU pods are unavailable or unsuitable, the original Hetzner provisioning
instructions follow. These were the primary target before CCX43 was found unavailable.

### Provision

**Option A: Web UI**

1. Sign in to [console.hetzner.cloud](https://console.hetzner.cloud)
2. **Add Server** > Location: **Ashburn (ASH)** or **Hillsboro (HIL)**
   (EU fallbacks: Falkenstein FSN1, Nuremberg NBG1 -- add ~30-60 min rsync time)
3. Image: Ubuntu 24.04 | Type: Dedicated vCPU > **CCX43** (16 vCPU, 64 GB)
4. SSH Key: select your uploaded key | Name: `demiurge-gen`
5. **Create & Buy Now** -- note the IPv4 address

**Option B: hcloud CLI**

```bash
hcloud server create \
    --name demiurge-gen \
    --type ccx43 \
    --image ubuntu-24.04 \
    --location ash \
    --ssh-key <your-ssh-key-name>

hcloud server describe demiurge-gen | grep "Public Net"
```

> US locations: `ash` (Ashburn, VA), `hil` (Hillsboro, OR).
> EU fallbacks: `fsn1`, `nbg1`. CCX33 fallback type: `ccx33` (8 vCPU, 32 GB, set `num_workers: 7`).

### SSH and Bootstrap

```bash
ssh root@<INSTANCE_IP>
scp scripts/cloud_setup.sh root@<INSTANCE_IP>:/root/cloud_setup.sh
# On the instance:
bash /root/cloud_setup.sh
```

Steps 4-6 (generation, rsync, destroy) are identical to the RunPod instructions above,
substituting `ssh root@<INSTANCE_IP>` for the RunPod connection string and
`hcloud server delete demiurge-gen` for the RunPod terminate step.
