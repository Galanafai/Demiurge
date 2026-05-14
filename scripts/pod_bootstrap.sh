#!/usr/bin/env bash
# pod_bootstrap.sh -- Full Demiurge pod setup on a fresh RunPod GPU instance.
#
# Run this INSIDE the pod after cloning the repo:
#
#   git clone https://github.com/Galanafai/Demiurge.git /workspace/Demiurge
#   cd /workspace/Demiurge
#   bash scripts/pod_bootstrap.sh
#
# What it does:
#   1. Installs uv if not present
#   2. Runs uv sync (installs pydrake, torch, all deps)
#   3. Sets WANDB_API_KEY in .bashrc (reads from env or prompts)
#   4. Downloads all dataset artifacts from W&B:
#        v1_shard_00000, v1_shard_00001, v1_text_embeddings, v1_manifest
#   5. Downloads the unconditional_v3 checkpoint from W&B
#   6. Verifies checkpoint integrity
#   7. Prints a ready-to-train command
#
# Prerequisites:
#   - WANDB_API_KEY must be set in the pod's env or passed via:
#       WANDB_API_KEY=<key> bash scripts/pod_bootstrap.sh
#   - Internet access (W&B API)
#
# Future pod runs: just run this script again. It is idempotent.
# Dataset shards skip download if already present (sha256 check).

set -euo pipefail

REPO_DIR="/workspace/Demiurge"
DATA_DIR="${REPO_DIR}/data/v1"
CKPT_DIR="${REPO_DIR}/checkpoints"
ENTITY="galanafai-self"
PROJECT="demiurge"

log() { echo "[bootstrap] $*"; }
die() { echo "[bootstrap] ERROR: $*" >&2; exit 1; }

# ── 1. uv ────────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:$PATH"
    echo 'export PATH="/root/.local/bin:$PATH"' >> /root/.bashrc
fi
log "uv: $(uv --version)"

# ── 2. Python deps ────────────────────────────────────────────────────────────
cd "$REPO_DIR"
log "Running uv sync (pydrake + torch + all deps)..."
uv sync
log "uv sync complete."

# ── 3. W&B key ───────────────────────────────────────────────────────────────
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    die "WANDB_API_KEY is not set. Export it before running this script."
fi
if ! grep -q "WANDB_API_KEY" /root/.bashrc; then
    echo "export WANDB_API_KEY=${WANDB_API_KEY}" >> /root/.bashrc
    log "WANDB_API_KEY persisted to .bashrc"
fi
# Also write .netrc so wandb CLI works without env var
mkdir -p /root
echo "machine api.wandb.ai" > /root/.netrc
echo "  login user" >> /root/.netrc
echo "  password ${WANDB_API_KEY}" >> /root/.netrc
chmod 600 /root/.netrc
log "W&B credentials configured."

# ── 4. Dataset ────────────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR"

download_artifact() {
    local artifact_name="$1"
    local dest_dir="$2"
    local filename="$3"
    local dest="${dest_dir}/${filename}"

    if [[ -f "$dest" ]]; then
        log "  ${filename} already present, skipping."
        return
    fi
    log "  Downloading ${artifact_name}:latest -> ${dest}..."
    uv run python - <<PYEOF
import wandb, pathlib, shutil
api = wandb.Api()
art = api.artifact('${ENTITY}/${PROJECT}/${artifact_name}:latest')
dl = art.download(root='/tmp/wandb_dl/${artifact_name}')
src = pathlib.Path(dl) / '${filename}'
dst = pathlib.Path('${dest}')
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src, dst)
print(f"  Saved {dst.name}  {dst.stat().st_size/1e6:.1f} MB")
PYEOF
}

log "Downloading dataset artifacts..."
download_artifact "v1_shard_00000"     "$DATA_DIR" "v1-00000.tar"
download_artifact "v1_shard_00001"     "$DATA_DIR" "v1-00001.tar"
download_artifact "v1_text_embeddings" "$DATA_DIR" "text_embeddings.pt"
download_artifact "v1_manifest"        "$DATA_DIR" "manifest.json"

log "Dataset ready: $(du -sh "$DATA_DIR" | cut -f1)"

# ── 5+6. unconditional_v3 checkpoint ─────────────────────────────────────────
V3_CKPT="${CKPT_DIR}/unconditional_v3_pres1.0/latest.pt"
if [[ -f "$V3_CKPT" ]]; then
    log "unconditional_v3 checkpoint already present, skipping."
else
    log "Downloading unconditional_v3_checkpoint:best..."
    uv run python - <<PYEOF
import wandb, pathlib, shutil, torch, sys
api = wandb.Api()
art = api.artifact('${ENTITY}/${PROJECT}/unconditional_v3_checkpoint:best')
dl = art.download(root='/tmp/wandb_dl/v3_ckpt')
src = pathlib.Path(dl) / 'checkpoint.pt'
dst = pathlib.Path('${V3_CKPT}')
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src, dst)
ckpt = torch.load(dst, map_location='cpu', weights_only=False)
assert 'model_state' in ckpt, "model_state missing from checkpoint"
assert ckpt['step'] == 100000, f"unexpected step {ckpt['step']}"
print(f"  Checkpoint OK: step={ckpt['step']}  {dst.stat().st_size/1e6:.1f} MB")
PYEOF
fi

# ── 7. Summary ────────────────────────────────────────────────────────────────
log ""
log "Bootstrap complete. Pod is ready for training."
log ""
log "To launch conditional_v2 training:"
log "  tmux new -s cond_v2"
log "  cd /workspace/Demiurge"
log "  uv run python scripts/train.py --config configs/train/conditional_v2.yaml --seed 42 2>&1 | tee logs/cond_v2.log"
log ""
log "Monitor GPU:"
log "  watch -n2 nvidia-smi"
