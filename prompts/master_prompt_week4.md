# Antigravity Master Prompt — Week 4: Type Collapse Fix + Classifier Guidance

You are continuing the Demiurge project. Week 3 shipped a conditional 
DDPM that generates 6-DoF tabletop scenes at 17.0% Drake validity, but 
revealed a critical limitation: the model collapses to type_id=0 (cube) 
in 99% of generations, producing a joint Drake-valid + prompt-matching 
rate of ~0%.

Week 4 has two phases:
- Phase A (priority 0): Fix type collapse so prompt-following becomes 
  measurable
- Phase B (priority 1): Implement classifier-guided sampling to push 
  validity toward the Pareto frontier

## Pod Status: STILL RUNNING from Week 3

The same pod from Week 3 is alive and ready. Use it directly. No 
re-provisioning needed.

SSH (proxy, requires -tt for command execution):
ssh -tt -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no f0xpg6tkynmbk9-64411a32@ssh.runpod.io

Pod has:
- GPU: RTX 5080 (16 GB VRAM, Blackwell sm_120)
- PyTorch: 2.12.0.dev20260407+cu128 (nightly, sm_120 compatible)
- CUDA: 12.8, Python 3.11.10
- All dependencies installed: wandb, sentence-transformers, webdataset, 
  hydra-core, einops, omegaconf, pydrake, anthropic, etc.
- Repo at /workspace/Demiurge on branch week3/diffusion-model
- Dataset at /workspace/Demiurge/data/v1/ (50k validated scenes, 601 MB)
- Checkpoints: conditional_v2/latest.pt, conditional_v3/latest.pt, 
  unconditional_v3_pres1.0/latest.pt
- WANDB_API_KEY set in /root/.bashrc
- ANTHROPIC_API_KEY set in /root/.bashrc

Verify pod state at start of Phase 1:

ssh -tt ... 'cd /workspace/Demiurge && \
  nvidia-smi --query-gpu=name,memory.used --format=csv,noheader && \
  python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())" && \
  git status && \
  ls -lh checkpoints/conditional_v2/latest.pt && \
  exit'

If anything is broken (wrong torch, no checkpoint, no data), HALT and 
surface for re-provisioning decision before any code changes.

## Before Doing Anything Else

1. Read AGENTS.md and .agents/skills/demiurge/SKILL.md in full.
2. Read artifacts/week3_results.md (already contains type collapse 
   finding).
3. Read artifacts/conditional_vlm_eval.md (VLM eval details).
4. Read artifacts/week4_preview.md (preliminary Week 4 plan with type 
   collapse remediation).
5. Checkout the week4/classifier-guidance branch (created locally in 
   Week 3 closeout).
6. State in Plan Artifact:
   - Current baseline metrics: 17.0% Drake validity (uncond probe), 
     9.8% text-conditioned, 1.09/5 VLM score, 0% joint Drake+VLM>=4
   - Phase A success criteria (one of three gates must pass)
   - Phase B hypothesis: classifier guidance improves Drake validity 
     by 5-15pp
   - Whether Phase A and Phase B share architecture

## Mission Scope

### Phase A: Type Collapse Fix (target: 2-3 days, ~$3 GPU)

Address majority-class collapse in type_id prediction. Train a new 
conditional model variant that genuinely learns object identity.

### Phase B: Classifier Guidance (target: 3-4 days, ~$5 GPU)

Build rejection sampling baseline, train noise-conditioned validity 
classifier, implement classifier-guided sampling, sweep guidance 
scales, produce Pareto curve.

## Step-by-Step Execution Plan

### Phase A: Type Collapse Fix

1. **Class distribution audit.** Compute actual type_id distribution 
   across all 50k training scenes:
   
   ssh ... 'cd /workspace/Demiurge && python3 -c "
   from data.reader import ShardReader
   import collections
   import sys
   sys.path.insert(0, \"src\")
   from data.reader import ShardReader
   from collections import Counter
   counts = Counter()
   for scene, *_ in ShardReader(\"data/v1\"):
       for i in range(len(scene.object_types)):
           if scene.presence[i]:
               counts[int(scene.object_types[i])] += 1
   total = sum(counts.values())
   for tid, c in sorted(counts.items()):
       print(f\"type_id={tid}: {c} ({100*c/total:.1f}%)\")
   "'
   
   Save the output to artifacts/class_distribution.md. Compute 
   inverse-frequency class weights = 1 / (freq + epsilon).

2. **Class-balanced sampler.** Implement src/data/balanced_sampler.py:
   - WeightedRandomSampler over the dataset
   - Sample weight per scene = mean(1/freq) across present objects 
     (or max(1/freq) — choose with sanity test)
   - Wrap existing ShardReader so it still streams from disk
   - Smoke test: sample 5000 batches, verify each type appears 
     proportional to 1/freq (not 1/N)
   
3. **Loss reweighting.** Modify src/model/loss.py:
   - Add class_weights argument to type_ce computation
   - Use inverse-frequency weights computed in task 1
   - Default: pass None (existing behavior), accept tensor of shape 
     (num_classes,)
   - In configs/train/conditional_v4.yaml, set type_ce weight to 1.0 
     (up from 0.1) and pass class_weights
   - Verify backward pass: gradient on type head is non-trivial 
     (>1e-5 norm) when class_weights active

4. **conditional_v4 training.** Create configs/train/conditional_v4.yaml:
   - model: identical to v2 (d_model=256, 6 layers, 8 heads)
   - training.cfg_dropout: 0.15
   - training.warm_init_from: checkpoints/conditional_v2/latest.pt 
     (start from v2, NOT random init)
   - training.warm_init_keys: model_state
   - training.batch_size: 128 (smaller, since class-balanced sampler 
     may produce harder batches)
   - training.lr: 1e-4 (lower than v2's 3e-4 since warm-init)
   - training.max_steps: 50000 (half of v2 since starting hot)
   - training.val_every_epochs: 30
   - training.val_n_scenes: 50
   - training.num_workers: 0
   - training.use_class_balanced_sampler: true
   - training.class_weights: [computed from task 1]
   - loss.type_ce: 1.0 (CRITICAL: up from 0.1)
   - All other loss weights: same as v2
   
   Launch in tmux:
   
   ssh ... 'tmux new-session -d -s cond_v4 \
     "cd /workspace/Demiurge && PYTHONUNBUFFERED=1 \
      PYTHONPATH=/workspace/Demiurge/src python3 -u \
      scripts/train.py --config configs/train/conditional_v4.yaml \
      --seed 42 2>&1 | tee logs/cond_v4.log"'
   
   Expected throughput on RTX 5080: ~10-15 steps/sec real (faster 
   than v3 since smaller batch, lower val frequency).
   Expected wall-clock: ~90-120 min for 50k steps.

5. **Phase A mid-run monitoring.**
   At step 10k: verify type_id predictions are diverse (not all 0)
   At step 25k: report mean VLM score on 100-sample probe
   HALT if mean VLM score still < 1.5 at step 25k (fix not working)

6. **Phase A final evaluation.** After training completes:
   
   - Run 500-scene unconditional Drake probe (same as Week 3) on 
     both model_state and ema_state
   - Run text-conditioned generation: 200 held-out descriptions × 8 
     samples (same setup as Task 9)
   - Run Drake validation on all 1600 text-conditioned samples
   - Run Haiku 4.5 VLM judging on all 1600 samples (~$1.10 API cost)
   - Compute per-class type accuracy: of generated scenes claiming 
     to contain a specific object type, what % actually have the 
     right type_id?

7. **Phase A success gate evaluation.** Compare to v2 baseline:
   
   | Metric | v2 (Week 3) | v4 (Phase A) | Gate |
   |---|---|---|---|
   | Drake validity (text-cond) | 9.8% | ? | informational |
   | VLM mean score | 1.09 | ? | >2.0 = PASS |
   | Joint Drake+VLM>=4 | 0.0% | ? | >5% = PASS |
   | Per-type accuracy | ~16% (1/6) | ? | >40% = PASS |
   
   Phase A passes if ANY one of the three gates is met. Document all 
   three metrics regardless.

8. **If Phase A FAILS all gates:** HALT. Do not proceed to Phase B. 
   Surface the failure with diagnostic data:
   - Type prediction distribution from final model
   - Loss curves for type_ce
   - Suspect root causes (loss weight still too low, sampler bug, 
     architectural bottleneck in type embedding)
   
   Propose v5 fix and wait for human review.

9. **If Phase A PASSES at least one gate:** Document v4 results in 
   artifacts/conditional_v4_phase_a_results.md. Upload v4 checkpoint 
   to W&B as conditional_v4_checkpoint with alias 
   "type-collapse-fixed". Proceed to Phase B.

### Phase B: Classifier Guidance

10. **Sampler interface.** Implement src/guidance/base.py with 
    Sampler protocol:
    
    class Sampler(Protocol):
        def sample(self, prompts: list[str], n_per_prompt: int, 
                   **kwargs) -> list[SceneTensor]: ...
        def name(self) -> str: ...
        def config(self) -> dict: ...
    
    Test: protocol runtime check via @runtime_checkable.

11. **Rejection sampler.** Implement src/guidance/rejection.py.
    - Wrap conditional_v4 model
    - For each prompt, sample batches, run Drake validation
    - Keep accepted up to target N, stop after max_attempts × N 
      total samples
    - Log: accept_rate, mean_attempts_per_accept, wall_clock
    
    Test: on 100 prompts, rejection sampler with max_attempts=50 hits 
    validity = 1.0 by construction.

12. **Classifier training data generation.** Implement 
    scripts/generate_classifier_data.py:
    - For each scene in v1 training set: sample t ~ Uniform(0, T)
    - Apply forward noise: x_t = sqrt(alpha_bar_t) * x_0 + 
      sqrt(1-alpha_bar_t) * noise
    - Label valid=1
    - Generate corrupted variant: pick random object, perturb pose 
      by ±0.15m (enough to cause interpenetration), validate to 
      confirm now invalid, apply forward noise at same t, label 
      valid=0
    - If corruption doesn't produce invalid scene (rare), discard
    - Target: 200k labeled noisy scenes (~100k valid + ~100k invalid)
    - Save to data/classifier_v1/ as shards
    
    Expected wall-clock: 30-45 min, ~$0.30 pod cost.

13. **Validity classifier.** Implement src/guidance/classifier.py:
    - Smaller transformer: d_model=128, 4 layers, 4 heads (~3M params)
    - Input: noisy scene (B, N, 13) + timestep t
    - Tokenization: same per-object embedding as denoiser
    - Output: single logit (valid/invalid)
    - Loss: BCE with logits
    
    Train on data from task 12 for one epoch, batch_size=256, 
    lr=3e-4, warmup 500 steps. Log as classifier_v1.
    Acceptance: held-out AUC > 0.85.

14. **Classifier-guided sampler.** Implement 
    src/guidance/classifier_guided.py.
    - During reverse diffusion at each step:
      - Compute classifier gradient: grad_x = autograd(log_p_valid, x_t)
      - Modify noise prediction: eps_guided = eps_uncond + w * grad_x
      - Where w is guidance scale
    - Reference: Dhariwal and Nichol 2021 ("Diffusion Models Beat 
      GANs on Image Synthesis")
    - Gradient clipping: clip grad_x norm to 1.0 per step
    - Two modes:
      a) Continuous-only: gradient on (xyz, rot6d, scale) only, leave 
         (type_logits, presence_logit) alone
      b) Full gradient: straight-through estimator for discrete 
         components

15. **Guidance scale sweep.** Run classifier-guided sampler across:
    w ∈ {0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0} × {continuous, full}
    = 14 sampler configs × 3 seeds = 42 sweep runs
    
    Each run: 200 held-out prompts × 5 scenes = 1000 scenes per run
    
    For each scene measure:
    - Drake validity (accepted/rejected + rejection reason)
    - Pairwise diversity vs other scenes in same prompt (Week 2 
      encoding)
    
    Plus VLM judging on all 1000 scenes per config (subset to 100 to 
    save API cost? agent's call based on budget).

16. **Pareto plot.** Generate the headline figure 
    artifacts/pareto_v1.png:
    - X-axis: mean pairwise diversity
    - Y-axis: validity rate
    - One point per sampler config + 3 seed error bars
    - Annotate Pareto frontier
    - Include: unconditional baseline (5.4%), conditional baseline 
      (9.8%), rejection (1.0 by construction), classifier-guided 
      sweep (14 configs)
    - Save as both .png (300 DPI) and .pdf (vector)

17. **Honesty check.** If classifier guidance does NOT Pareto-
    dominate rejection sampling, do not hide it. Write 
    artifacts/week4_results.md with:
    - Actual numbers from sweep
    - Clear statement: "Classifier guidance [does/does not] 
      Pareto-dominate rejection sampling by X pp validity at Y 
      diversity"
    - If negative: propose alternatives for Week 5 (energy-based 
      guidance, differentiable Drake surrogate, distilled validity 
      rejector)

### Final

18. **Gallery rendering.** From best classifier-guided config, 
    render 30 generated scenes:
    - Use Drake offscreen rendering OR stick with text 
      serialization
    - Include diverse prompts across all 3 templates
    - Save to artifacts/gallery/

19. **Verification pass.** Run on pod:
    - ruff check .
    - pyright (logic errors only)
    - pytest tests/
    
    Acceptance: ruff 0 errors, pyright 0 logic errors, pytest all 
    pass or document expected failures.

20. **Final artifacts.** Update:
    - artifacts/week4_results.md (Pareto results + classifier 
      ablation)
    - artifacts/week5_preview.md (sampler configs advancing to 
      downstream eval)
    - artifacts/conditional_v4_phase_a_results.md (Phase A 
      complete writeup)

21. **Git commit and merge to main.**
    On pod:
    git add artifacts/ src/ scripts/ configs/ tests/
    git commit -m "feat: Week 4 complete — type collapse fixed + 
                   classifier guidance Pareto sweep"
    git push origin week4/classifier-guidance
    
    On laptop:
    cd /home/lap/Demiurge && git pull origin week4/classifier-guidance
    git checkout main && git merge --no-ff week4/classifier-guidance
    git push origin main

## Ground Rules

- Plan first, wait for approval
- Phase A must hit one of three success gates before starting Phase B
- The classifier must be trained on NOISED scenes at varying t, not 
  clean scenes. A classifier trained only on clean data does not 
  give useful gradients in the middle of reverse diffusion. Stop and 
  re-read this if you find yourself writing the clean-only version.
- Use 3 seeds per sampler configuration in Phase B. One-seed numbers 
  are not a result.
- Architecture changes must be resume-safe (warm-init from v2 
  checkpoint, not random init)
- All training in tmux sessions
- No em dashes or double hyphens in prose

## Compute Budget

GPU budget: $15 total for Week 4
- Phase A (50k steps): ~$2.50 expected
- Classifier data generation: ~$0.30
- Classifier training: ~$0.80
- Phase B sweep (42 runs): ~$4 expected
- Probes and evaluation: ~$2
- Buffer: ~$5

VLM API budget: $10 total
- Phase A evaluation (~1600 samples Haiku): ~$1.10
- Phase B sweep (sampled subset): ~$3
- Cross-validation Sonnet: ~$1
- Buffer: ~$5

## Halt Conditions

- Phase A fails all 3 success gates after 50k steps: HALT, surface
- Classifier AUC < 0.85 after one epoch: HALT, surface  
- Pod becomes unre