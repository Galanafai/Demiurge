# Cloud GPU Run Playbook

**Last updated:** 2026-05-13 (Week 3 unconditional_v1 benchmark run)

## Pod Spec

| Field | Value |
|---|---|
| Provider | RunPod community cloud |
| GPU | RTX 4090 (24GB VRAM) |
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Container disk | 30GB |
| Volume disk | 50GB (mounted at `/workspace`) |
| SSH | `ssh root@<PUBLIC_IP> -p <PORT> -i ~/.ssh/id_ed25519` |

## Bootstrap Procedure

### 1. Verify GPU

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Both must succeed before proceeding.

### 2. Install tooling

```bash
# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# tmux + rsync
apt-get update -qq && apt-get install -y tmux rsync
```

### 3. Clone repo and checkout working branch

```bash
cd /workspace
git clone https://github.com/Galanafai/Demiurge.git Demiurge
cd Demiurge
git checkout week3/diffusion-model
git log --oneline -5  # verify HEAD is at expected commit
```

### 4. uv sync

```bash
uv sync --no-dev 2>&1 | tail -10
# Note: uv will resolve and install torch from PyPI (2.7.x+cu126), not
# the system torch (2.4.x). Both work; uv venv is used for all runs.
```

### 5. Set WANDB_API_KEY

The key is NOT set automatically in the pod env. Must be set manually:

```bash
# Key is in ~/.netrc on the laptop
export WANDB_API_KEY='<key>'
echo "export WANDB_API_KEY='$WANDB_API_KEY'" >> ~/.bashrc
echo "export WANDB_API_KEY='$WANDB_API_KEY'" >> ~/.profile

# Verify
uv run wandb login "$WANDB_API_KEY" 2>&1 | head -3
```

### 6. Transfer dataset

From the laptop:

```bash
# Ensure rsync is installed on pod first (step 2 above)
rsync -avz --progress -e "ssh -i ~/.ssh/id_ed25519 -p <PORT>" \
  /home/lap/Demiurge/data/v1/ \
  root@<IP>:/workspace/Demiurge/data/v1/
```

Note: `/workspace` is NFS-mounted; `chown` errors are benign. File data transfers correctly.

Verify on pod:

```bash
cd /workspace/Demiurge
sha256sum data/v1/v1-00001.tar  # must match: af0a8ff60f75eae3b...
uv run python -c "
import sys; sys.path.insert(0, 'src')
from data.reader import ShardReader
n = sum(1 for _ in ShardReader('data/v1'))
print(f'{n} scenes')
"  # must print: 50000 scenes
```

### 7. Verify CUDA via uv

```bash
uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

### 8. Launch training in tmux

```bash
tmux new-session -d -s <session_name> -x 220 -y 50
tmux send-keys -t <session_name> 'source ~/.bashrc; cd /workspace/Demiurge' Enter
sleep 1
tmux send-keys -t <session_name> 'uv run python scripts/train.py --config configs/train/baseline.yaml --seed 42 2>&1 | tee /workspace/<logfile>.log' Enter
```

Monitor:
```bash
tail -f /workspace/<logfile>.log
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
```

## Known Gotchas

| Issue | Fix |
|---|---|
| `rsync: command not found` | `apt-get update && apt-get install -y rsync` |
| `tmux: command not found` | `apt-get install -y tmux` |
| `WANDB_API_KEY` not in pod env | Must be set manually in `~/.bashrc` (see step 5) |
| `RuntimeError: indices should be on cpu or same device` | Fixed in `ea44cc9` -- schedule buffer lookups use `.cpu()` before indexing |
| chown errors during rsync to /workspace | Benign (NFS mount denies ownership ops, but data transfers) |
| uv installs torch 2.7.x not 2.4.x | OK -- uv resolves latest compatible. Both work with CUDA 12.4+ |

## Sync Results Back to Laptop

```bash
# From laptop
rsync -avz --progress -e "ssh -i ~/.ssh/id_ed25519 -p <PORT>" \
  root@<IP>:/workspace/Demiurge/checkpoints/ \
  /home/lap/Demiurge/checkpoints/

rsync -avz --progress -e "ssh -i ~/.ssh/id_ed25519 -p <PORT>" \
  root@<IP>:/workspace/Demiurge/wandb/ \
  /home/lap/Demiurge/wandb/
```

## Cost Reference

| Config | Steps | Time | Cost (@$0.70/hr) |
|---|---|---|---|
| Benchmark (5 epochs) | 1,940 | ~21 sec | ~$0.004 |
| Full unconditional (d_model=256) | 100,000 | ~18 min | ~$0.21 |
| Full conditional (second pass) | 100,000 | ~18 min | ~$0.21 |
| Typical bootstrap time | -- | ~29 min | ~$0.34 |
| **Full Week 3 total** | -- | **~65 min** | **~$0.76** |

Throughput confirmed: **92.5 steps/sec** on RTX 4090 with d_model=256, batch=128, 50k dataset loaded in memory.
