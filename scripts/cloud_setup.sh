#!/usr/bin/env bash
# scripts/cloud_setup.sh -- Idempotent bootstrap for Demiurge on Ubuntu 24.04.
#
# Target: Hetzner CCX43 (16 dedicated AMD cores, 64 GB RAM, Ubuntu 24.04 LTS).
# Fallback: Hetzner CCX33 (8 dedicated AMD cores, 32 GB RAM).
#
# Usage (run as root or with sudo on the fresh instance):
#   bash cloud_setup.sh
#
# Idempotency: every step is guarded by a condition check. Safe to re-run
# after a partial failure or system reboot. All installs are pinned or
# version-gated where behaviour would differ across runs.
#
# What this script does:
#   1. System deps: git, curl, build-essential, python3.11, rsync, tmux, htop.
#   2. Drake system deps (libsdformat, libglib, etc.) via pip-wheel prerequisites.
#   3. uv (Python package manager) at the pinned version.
#   4. Clone/update the Demiurge repository.
#   5. uv sync inside the repo (installs pydrake + all deps into the venv).
#   6. Smoke-test: python -c "import pydrake; print(pydrake.__file__)".
#   7. Create logs/ and data/ directories.
#   8. Print the production run command and exit instructions.
#
# This script does NOT start the generation run. That is a deliberate gate
# requiring human review of the profiling results before execution.
#
# Do NOT run this script until the cloud instance is provisioned and you have
# SSH access. See docs/cloud_run.md for the full procedure.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration -- edit these before running if you fork this repo.
# ---------------------------------------------------------------------------

REPO_URL="https://github.com/Galanafai/Demiurge.git"   # Public HTTPS -- no auth required
                                                        # Repo is public; SSH key setup unnecessary
REPO_BRANCH="week2/model-arch"
REPO_DIR="/root/Demiurge"
UV_VERSION="0.4.29"   # Pin uv to a known-good version.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[$(date -u '+%H:%M:%S')] $*"; }
step() { echo; echo "=== $* ==="; }

# ---------------------------------------------------------------------------
# Step 1: System packages
# ---------------------------------------------------------------------------

step "System packages"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq

# Ubuntu 24.04 ships Python 3.12 by default. Demiurge requires Python 3.11
# (pyproject.toml: requires-python = ">=3.11,<3.12"). Add the deadsnakes PPA
# so python3.11 is resolvable before the main install loop.
if ! dpkg -s python3.11 &>/dev/null 2>&1; then
    log "Adding deadsnakes PPA for Python 3.11"
    apt-get install -y -qq software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
fi

PACKAGES=(
    git
    curl
    build-essential
    python3.11
    python3.11-dev
    python3.11-venv
    python3-pip
    rsync
    tmux
    htop
    libglib2.0-0
    libgl1
    libgomp1
)

# Filter out packages already installed.
TO_INSTALL=()
for pkg in "${PACKAGES[@]}"; do
    if ! dpkg -s "$pkg" &>/dev/null 2>&1; then
        TO_INSTALL+=("$pkg")
    fi
done

if [[ ${#TO_INSTALL[@]} -gt 0 ]]; then
    log "Installing: ${TO_INSTALL[*]}"
    apt-get install -y -qq "${TO_INSTALL[@]}"
else
    log "All system packages already installed."
fi

# libsdformat-dev is optional (not in all Ubuntu 24.04 mirrors); non-fatal if absent.
apt-get install -y -qq libsdformat-dev 2>/dev/null \
    || log "libsdformat-dev unavailable; skipping (non-fatal)."

# ---------------------------------------------------------------------------
# Step 2: uv
# ---------------------------------------------------------------------------

step "uv package manager"

UV_BIN="/root/.local/bin/uv"
if [[ -x "$UV_BIN" ]]; then
    INSTALLED_UV=$("$UV_BIN" --version 2>/dev/null | awk '{print $2}' || echo "unknown")
    log "uv already installed: $INSTALLED_UV"
else
    log "Installing uv $UV_VERSION"
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
fi

# Modern astral.sh installer (canonical since uv 0.2+) puts the binary in
# ~/.local/bin, not ~/.cargo/bin. The cargo path was for older releases.
export PATH="/root/.local/bin:$PATH"
uv --version
log "uv binary: $(which uv)"

# ---------------------------------------------------------------------------
# Step 3: Clone or update repository
# ---------------------------------------------------------------------------

step "Repository"

if [[ -d "$REPO_DIR/.git" ]]; then
    log "Repository exists at $REPO_DIR -- pulling latest."
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$REPO_BRANCH"
    git -C "$REPO_DIR" pull --ff-only origin "$REPO_BRANCH"
else
    log "Cloning $REPO_URL (branch: $REPO_BRANCH)"
    git clone --branch "$REPO_BRANCH" --depth 1 "$REPO_URL" "$REPO_DIR"
fi

log "HEAD: $(git -C "$REPO_DIR" rev-parse --short HEAD)"

# ---------------------------------------------------------------------------
# Step 4: Python dependencies via uv sync
# ---------------------------------------------------------------------------

step "Python dependencies (uv sync)"

cd "$REPO_DIR"

# uv sync reads pyproject.toml and creates/updates .venv. Idempotent.
uv sync --frozen

# ---------------------------------------------------------------------------
# Step 5: Drake smoke test
# ---------------------------------------------------------------------------

step "Drake smoke test"

uv run python -c "
import pydrake
import pydrake.multibody.plant
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
builder = DiagramBuilder()
plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
plant.Finalize()
print('Drake OK:', pydrake.__file__)
"

# ---------------------------------------------------------------------------
# Step 6: Scene schema smoke test
# ---------------------------------------------------------------------------

step "Demiurge import smoke test"

uv run python -c "
import sys
sys.path.insert(0, 'src')
from scene.schema import N_MAX, SceneTensor
from scene.vocab import OBJECT_VOCAB
assert N_MAX == 12, f'Expected N_MAX=12, got {N_MAX}'
assert len(OBJECT_VOCAB) == 12, f'Expected 12 vocab entries, got {len(OBJECT_VOCAB)}'
print(f'Scene schema OK: N_MAX={N_MAX}, vocab size={len(OBJECT_VOCAB)}')
"

# ---------------------------------------------------------------------------
# Step 7: Create working directories
# ---------------------------------------------------------------------------

step "Working directories"

mkdir -p "$REPO_DIR/logs"
mkdir -p "$REPO_DIR/data/v1"
log "logs/ and data/v1/ ready."

# Verify data/v1/ is empty before generation.
SHARD_COUNT=$(find "$REPO_DIR/data/v1" -name "*.tar" 2>/dev/null | wc -l)
if [[ "$SHARD_COUNT" -gt 0 ]]; then
    echo
    echo "WARNING: data/v1/ already contains $SHARD_COUNT .tar shard(s)."
    echo "This bootstrap expects a clean output directory."
    echo "If you intend to start fresh: rm -rf $REPO_DIR/data/v1 && mkdir -p $REPO_DIR/data/v1"
    echo "If you intend to resume a partial run, ignore this warning."
    echo
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

step "Bootstrap complete"

GIT_SHA=$(git -C "$REPO_DIR" rev-parse --short HEAD)

# Use a quoted heredoc so bash does not expand < or $ inside it.
cat <<'RUNEOF'

Bootstrap complete. Instance is ready for the Demiurge generation run.

---- PRODUCTION RUN (review first, then execute in tmux) ----

  cd /root/Demiurge
  tmux new -s gen
  nohup uv run python scripts/generate_dataset.py \
      --config configs/dataset/v1.yaml \
      --seed 42 \
      > logs/generate_v1.log 2>&1 &
  echo "PID: $!"

---- MONITOR ----

  tail -f /root/Demiurge/logs/generate_v1.log
  htop

---- KILL SWITCH: destroy instance after rsync ----

  # On your LOCAL machine (replace INSTANCE_IP):
  rsync -avz --progress root@INSTANCE_IP:/root/Demiurge/data/v1/ ./data/v1/
  # Verify shard count matches manifest, then destroy:
  hcloud server delete demiurge-gen

See docs/cloud_run.md for the full procedure including rsync verification
and manifest integrity check before destroying the instance.

RUNEOF

# Print the variable-expanded values separately.
echo "Repository: $REPO_DIR"
echo "Branch:     $REPO_BRANCH"
echo "Git SHA:    $GIT_SHA"
