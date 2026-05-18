#!/usr/bin/env python3
"""Upload v9 unconditional checkpoint to W&B with verification."""
import wandb, os, sys, time

CHECKPOINT = "/workspace/Demiurge/checkpoints/conditional_v9_uncond/latest.pt"
ARTIFACT_NAME = "conditional_v9_uncond_checkpoint"

print(f"Auto-upload v9 unconditional checkpoint")
print(f"Source: {CHECKPOINT}")
time.sleep(60)  # let checkpoint finalize

if not os.path.exists(CHECKPOINT):
    print(f"FAIL: {CHECKPOINT} not found"); sys.exit(1)

size_mb = os.path.getsize(CHECKPOINT) / 1024 / 1024
print(f"Checkpoint size: {size_mb:.1f} MB")

run = wandb.init(project="demiurge", entity="galanafai-self",
    job_type="checkpoint_upload", name=f"v9_uncond_upload_{int(time.time())}")

artifact = wandb.Artifact(ARTIFACT_NAME, type="model",
    description=f"v9 unconditional: 48.66M params, 234k data, 150k steps, from scratch. Trained {time.strftime('%Y-%m-%d')}.",
    metadata={"params_M": 48.66, "data_scenes": 234720, "max_steps": 150000,
              "batch_size": 64, "lr": 3e-4, "from_scratch": True})
artifact.add_file(CHECKPOINT, name="checkpoint.pt")
run.log_artifact(artifact)
run.finish()

print("Waiting 60s for W&B commit...")
time.sleep(60)

api = wandb.Api()
try:
    a = api.artifact(f"galanafai-self/demiurge/{ARTIFACT_NAME}:latest")
    print(f"VERIFIED: {a.name}  {a.size/1024/1024:.1f} MB  {a.created_at}")
    print("UPLOAD SUCCESS")
except Exception as e:
    print(f"VERIFICATION FAILED: {e}"); sys.exit(1)
