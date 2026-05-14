# Restore and Demo Guide -- Week 3 Models

## Models Available

| Model | W&B Artifact | Validity | Notes |
|---|---|---|---|
| unconditional_v3 | `galanafai-self/demiurge/unconditional_v3_checkpoint:best` | 5.4% | Pareto baseline |
| conditional_v1 | `galanafai-self/demiurge/conditional_v1_checkpoint:best` | 1.0% | Case 3, CFG missing |

---

## Git Branch

```
week3/diffusion-model
```

All code, configs, and markdown artifacts are in this branch.

---

## Restore on a Fresh Pod

### 1. Clone and install

```bash
git clone git@github.com:Galanafai/Demiurge.git /workspace/Demiurge
cd /workspace/Demiurge
git checkout week3/diffusion-model
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
```

### 2. Download checkpoints from W&B Artifacts

```bash
# unconditional_v3 (best unconditional baseline)
uv run python - <<'EOF'
import wandb
api = wandb.Api()
art = api.artifact('galanafai-self/demiurge/unconditional_v3_checkpoint:best')
art.download(root='checkpoints/unconditional_v3_pres1.0/')
print('Downloaded unconditional_v3')
EOF

# conditional_v1
uv run python - <<'EOF'
import wandb
api = wandb.Api()
art = api.artifact('galanafai-self/demiurge/conditional_v1_checkpoint:best')
art.download(root='checkpoints/conditional_v1/')
print('Downloaded conditional_v1')
EOF
```

> [!NOTE]
> The artifact `add_file()` stores the file as `checkpoint.pt` inside the artifact.
> After download, rename: `mv checkpoints/conditional_v1/checkpoint.pt checkpoints/conditional_v1/latest.pt`

### 3. Alternative: restore from /workspace tarball (if same RunPod network FS)

If the `/workspace` network filesystem is still attached:

```bash
tar -xzf /workspace/conditional_v1_checkpoints.tar.gz -C /workspace/Demiurge/
tar -xzf /workspace/unconditional_v3_checkpoints.tar.gz -C /workspace/Demiurge/
```

### 4. Restore text embedding cache

```bash
uv run python scripts/precompute_text_embeddings.py \
    --data data/v1 \
    --output data/v1/text_embeddings.pt \
    --device cuda --batch-size 512
# Takes ~2 min on RTX 4090. Idempotent.
```

---

## Verify Checkpoint Loads

```bash
uv run python - <<'EOF'
import sys, torch
sys.path.insert(0, 'src')
from model.denoiser import SceneDenoiser, DenoiserConfig

ckpt = torch.load('checkpoints/unconditional_v3_pres1.0/latest.pt',
                  map_location='cpu', weights_only=False)
arch = ckpt['arch']
cfg = DenoiserConfig(**arch)
model = SceneDenoiser(cfg)
model.load_state_dict(ckpt['model_state'])
print(f"unconditional_v3: step={ckpt['step']} "
      f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M OK")

ckpt2 = torch.load('checkpoints/conditional_v1/latest.pt',
                   map_location='cpu', weights_only=False)
cfg2 = DenoiserConfig(**ckpt2['arch'])
model2 = SceneDenoiser(cfg2)
model2.load_state_dict(ckpt2['model_state'])
print(f"conditional_v1:   step={ckpt2['step']} "
      f"params={sum(p.numel() for p in model2.parameters())/1e6:.2f}M OK")
EOF
```

Expected output:
```
unconditional_v3: step=100000 params=8.89M OK
conditional_v1:   step=100000 params=8.89M OK
```

---

## Sample 10 Scenes (Unconditional v3)

```bash
uv run python - <<'EOF'
import sys, torch
sys.path.insert(0, 'src')
from model.denoiser import SceneDenoiser, DenoiserConfig, N_MAX
from model.rotations import rot6d_to_quat_wxyz
from model.schedule import CosineSchedule, DDIMSampler
from scene.schema import WorkspaceBounds, SceneTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('checkpoints/unconditional_v3_pres1.0/latest.pt',
                  map_location='cpu', weights_only=False)
model = SceneDenoiser(DenoiserConfig(**ckpt['arch'])).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()

schedule = CosineSchedule(T=1000)
sampler = DDIMSampler(schedule, n_steps=50)
bounds = WorkspaceBounds.default()

x0 = sampler.sample(model.noise_prediction_fn(None), (10, N_MAX, 13), seed=42, device=device)
for i in range(10):
    pm = x0[i,:,12] > 0.0
    print(f'  scene {i}: {int(pm.sum())} objects')
EOF
```

---

## Infrastructure Notes

- **Local laptop:** Controls/prompts, inspects outputs, monitors W&B. Does NOT run Drake validation (no pydrake installed).
- **Cloud pod (RTX 4090):** Full generation, Drake validation, training. Requires RunPod instance.
- **`/workspace`:** RunPod network filesystem. Persists across pod **pause/resume**. Destroyed on pod **deletion**.
- **W&B Artifacts:** Primary checkpoint preservation path. Survives pod deletion.

---

## Expected Checkpoint Paths After Restore

```
checkpoints/
  unconditional_v3_pres1.0/
    latest.pt          # 121 MB -- restored from W&B or tarball
  conditional_v1/
    latest.pt          # 136 MB -- restored from W&B or tarball
data/v1/
  text_embeddings.pt   # 28 MB -- re-generate with precompute script
```
