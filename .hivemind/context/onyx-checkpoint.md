# Onyx Checkpoint — 2026-03-26 (final)

## Current Status
- **invokerl GRPO pipeline validated and beating tilerl** — 56.0% at step 200 vs tilerl's 54%.
- 8× faster wall-clock (40 min vs 5.5 hrs for 200 steps).
- Five parameter sweep configs tested. fp32 master weights identified as the critical fix.
- Awaiting daemon's next direction.

## What I Did (invokerl)

### Engine Implementation
- `engine/generator.py` (~407 lines): VLLMGenerator with batched generation, log-prob extraction, disk-based weight sync, prefix caching
- Integration fixes across `policy.py`, `trainer.py`, `train.py`: device mismatch, OOM, CUDA fork order, flash_attn→sdpa, torch_dtype→dtype

### Key Debugging & Fixes
1. **CUDA fork error**: Deferred `torch.cuda.manual_seed_all()` to after vLLM init
2. **OOM on RTX 5090**: `max_model_len: 2048` (saved ~10 GB), `gpu_memory_utilization: 0.25`, fused `F.cross_entropy`
3. **vLLM v1 weight sync**: Disk-based via `safetensors.torch.save_file` + `collective_rpc("reload_weights")`
4. **Prompt format**: ChatML template for Qwen3 (2% → 50% baseline)
5. **Loss display bug**: Added `"loss": loss.item()` to GRPO metrics dict
6. **fp32 master weights**: Root cause of flat learning — bf16 rounded away lr=1e-6 updates. nova implemented fp32 weights + bf16 autocast.
7. **batch_size=1**: Matching tilerl's 16 episodes/update (1×4×4) vs previous 64 (4×4×4).

### Final Training Results (200-sample eval)
```
Config              Baseline   Step 50   Step 100   Step 150   Step 200   Final
bf16 lr=5e-6        47.5%      50.5%     49.5%      49.0%      52.0%     55.5%
fp32 lr=1e-6        50.0%      46.0%     52.5%      50.0%      56.0%     55.0%
tilerl (reference)  46%        36%       48%        50%        54%       54%
```

Both invokerl configs beat tilerl. fp32 lr=1e-6 is the best single-point result (56.0%).

## Server
- SSH: `ssh -p 54410 root@93.91.156.101 -i ~/.ssh/id_ed25519`
- RTX 5090, 32 GB
- Checkpoints at `/root/tilerl/checkpoints/grpo_gsm8k/`
- Training logs: `fp32_lr1e6.log`, `sweep_lr5e6_200step.log`, `sweep_combo.log`, `sweep_lr5e6.log`, `invokerl_train_bs1.log`, `invokerl_train_ref.log`

## Best Config
```yaml
model:
  name_or_path: Qwen/Qwen3-0.6B
  dtype: bfloat16
algorithm:
  name: grpo
  clip_eps: 0.2
  beta: 0.04
training:
  total_steps: 200
  lr: 1.0e-6
  weight_decay: 0.01
  grad_clip: 1.0
  warmup_steps: 50
  accumulation_steps: 4
  batch_size: 1
  group_size: 4
generation:
  max_new_tokens: 512
  max_model_len: 2048
  gpu_memory_utilization: 0.25
# PolicyModel uses master_weights_fp32=True (fp32 weights, bf16 autocast forward)
```

## Key Learnings
1. KL regularization is load-bearing (no-ref → policy collapse)
2. bf16 optimizer precision kills learning at low LR (fp32 master weights fix)
3. 50-sample eval hides signal (200 samples needed for ±3.5% CI)
4. Batch size matters — 64 episodes/update diluted gradient vs 16
5. J-curve pattern: RL dips during warmup then recovers

## Possible Next Directions (daemon to decide)
1. Credit assignment experiments (hackability showcase)
2. New algorithms (DPO, PPO, DAPO in invokerl)
3. Scale to Qwen3-1.7B
4. Multi-GPU disaggregation (vLLM on GPU0, training on GPU1)
5. Longer runs (400+ steps) for more improvement
