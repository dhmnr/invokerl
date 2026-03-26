# invokerl

Hackable RL post-training for LLMs. Production-grade generation (vLLM), research-grade algorithms (PyTorch).

**46.5% → 62.5% on GSM8K** with Qwen3-0.6B in 2.5 hours on a single RTX 5090.

## Why

RL post-training research is bottlenecked by infrastructure, not ideas. Most of your time is spent fighting OOM errors, debugging weight sync, and waiting for generation — not experimenting with credit assignment or loss functions.

invokerl separates the hackable part (algorithms) from the infrastructure (generation, training loop, weight sync). You write ~100 lines of loss function. Everything else is handled.

## Architecture

```
Algorithm Layer (you hack this)
  compute_advantages(batch) → per-token advantages
  compute_loss(new_log_probs, batch, advantages) → loss
  ─────────────────────────────────────────────────────
Trainer (orchestrator — don't touch)
  rollout: vLLM generate → reward → ref log-probs → RolloutBatch
  train:   advantages → policy forward → loss → backward → optimizer
  sync:    updated weights → vLLM engine
  ─────────────────────────────────────────────────────
Infrastructure
  vLLM     — generation at 5,400 tok/s (PagedAttention, prefix caching)
  PyTorch  — model forward/backward with autograd
  HuggingFace — model loading, tokenization
```

## Quick Start

```bash
pip install vllm transformers datasets pyyaml

# Train GRPO on GSM8K
python -m invokerl --config invokerl/configs/grpo_gsm8k.yaml

# Smoke test (2 steps, fast)
python -m invokerl --config invokerl/configs/grpo_gsm8k.yaml \
    --max-steps 2 --eval-samples 5 --verbose

# Eval only
python -m invokerl --config invokerl/configs/grpo_gsm8k.yaml \
    --eval-only --resume ./checkpoints/grpo_gsm8k/step_200
```

## Writing a New Algorithm

Every algorithm implements two methods:

```python
from invokerl.algorithms.base import BaseAlgorithm, RolloutBatch

class MyAlgorithm(BaseAlgorithm):
    def compute_advantages(self, batch: RolloutBatch) -> Tensor:
        """Turn rewards into per-token learning signals.

        Override this for credit assignment experiments:
        group normalization, GAE, token-level shaping, PRM scores, etc.
        """
        rewards = batch.rewards  # [B] per-sequence
        # ... your credit assignment logic ...
        return advantages  # [B, T] per-token

    def compute_loss(self, new_log_probs, batch, advantages):
        """The policy optimization objective.

        new_log_probs: [B, T] from current policy (has gradients)
        batch: old_log_probs, ref_log_probs, masks, rewards
        advantages: [B, T] from compute_advantages()
        """
        ratio = (new_log_probs - batch.old_log_probs).exp()
        # ... your loss function ...
        return loss, {"reward": ..., "kl": ...}
```

Register it in `train.py`:
```python
ALGORITHMS = {
    "grpo": ("invokerl.algorithms.grpo", "GRPO"),
    "my_algo": ("invokerl.algorithms.my_algo", "MyAlgorithm"),
}
```

Run it:
```yaml
algorithm:
  name: my_algo
  my_hyperparam: 0.1
```

That's it. The trainer handles generation, reward scoring, model forward/backward, optimizer, weight sync, evaluation, and checkpointing.

## RolloutBatch

The data contract between the trainer and your algorithm:

| Field | Shape | Description |
|-------|-------|-------------|
| `token_ids` | `[B, T]` | Prompt + completion token IDs |
| `response_mask` | `[B, T]` | True for generated tokens |
| `rewards` | `[B]` | Per-sequence scalar rewards |
| `token_rewards` | `[B, T]` | Per-token rewards (optional, for credit assignment) |
| `old_log_probs` | `[B, T]` | Log-probs from policy at generation time |
| `ref_log_probs` | `[B, T]` | Log-probs from frozen reference model |
| `group_ids` | `[B]` | Which prompt each completion belongs to |
| `group_size` | `int` | Completions per prompt |

## Performance

Profiled on RTX 5090, Qwen3-0.6B:

| Phase | Time | % Step | MFU |
|-------|------|--------|-----|
| Generation (vLLM) | 5.7s | 70% | 0.7% (memory-bound) |
| Weight sync | 1.6s | 19.5% | — (disk I/O) |
| Backward | 0.4s | 5.1% | 79% |
| Ref forward | 0.2s | 3.0% | 67% |
| Policy forward | 0.2s | 2.1% | 94% |
| Optimizer | 0.02s | 0.3% | — |

Training phases (forward + backward) run at 79-94% of peak BF16 TFLOPS. Generation is the bottleneck — inherent to autoregressive decoding, mitigated by vLLM's optimized inference.

**Total: 8.1s/step, 443 steps/hour.** 1000 GRPO steps in ~2.5 hours on a single GPU.

## Results

GSM8K accuracy with GRPO (Qwen3-0.6B, single RTX 5090):

| Steps | Accuracy | Wall-clock |
|-------|----------|------------|
| 0 | 46.5% | — |
| 200 | 57.5% | 28 min |
| 500 | 52.0% | 70 min |
| 650 | 60.0% | 90 min |
| 1000 | 62.5% | 2.5 hrs |

+16 percentage points from baseline. Model was still improving at step 1000 with no sign of plateau.

## Project Structure

```
invokerl/
├── algorithms/           # The hackable part
│   ├── base.py           # BaseAlgorithm + RolloutBatch
│   └── grpo.py           # GRPO (148 lines)
├── engine/               # Infrastructure (don't modify)
│   ├── generator.py      # vLLM wrapper
│   ├── policy.py         # HuggingFace model wrapper
│   └── trainer.py        # Training loop
├── rewards/
│   ├── base.py           # Reward interface
│   └── rule.py           # Rule-based (exact match)
├── data/
│   ├── base.py           # Dataset interface
│   └── gsm8k.py          # GSM8K loader
├── configs/              # YAML training configs
│   └── grpo_gsm8k.yaml
└── train.py              # Entry point
```

## Configuration

All hyperparameters in YAML:

```yaml
model:
  name_or_path: Qwen/Qwen3-0.6B
  dtype: bfloat16

algorithm:
  name: grpo
  clip_eps: 0.2
  beta: 0.04

training:
  total_steps: 1000
  lr: 1.0e-5
  warmup_steps: 50
  batch_size: 1
  group_size: 4
  accumulation_steps: 4

generation:
  max_new_tokens: 512
  temperature: 0.9
  max_model_len: 2048
  gpu_memory_utilization: 0.3
```

CLI overrides for quick experiments:
```bash
--max-steps 50          # short run
--eval-samples 200      # more reliable eval
--no-ref                # skip reference model (saves 1.2 GB)
--resume ./ckpt/step_N  # resume from checkpoint
--eval-only             # evaluate without training
--verbose               # debug logging
```

## Memory Budget (Single GPU)

| Component | Qwen3-0.6B | Qwen3-1.7B |
|-----------|-----------|-----------|
| vLLM (0.3 GPU) | ~10 GB | ~10 GB |
| Policy + optimizer (bf16) | ~6 GB | ~16 GB |
| Ref model (bf16, frozen) | ~1.2 GB | ~3.4 GB |
| **Total** | **~17 GB** | **~29 GB** |

Fits on a 24 GB card for 0.6B, 32 GB for 1.7B. Use `--no-ref` to drop the reference model if tight on memory.

## Key Design Decisions

- **Credit assignment is first-class.** `compute_advantages()` is separate from `compute_loss()` and independently overridable. Experiment with group normalization, GAE, token-level shaping, or process reward models without touching the loss function.
- **vLLM in-process, not server.** No HTTP overhead, direct weight sync. Prefix caching reuses prompt KV cache across group completions.
- **No autograd for the hackable part.** Algorithms receive pre-computed tensors and return a scalar loss. PyTorch autograd handles the rest.
- **KL regularization is load-bearing.** Empirically validated: without reference model KL penalty, policy collapses (50% → 38%). Don't skip it.
- **bf16 needs higher learning rates.** At lr=1e-6, bf16 rounds away gradient updates. Use lr=5e-6 to 1e-5 with bf16, or enable fp32 master weights for low learning rates.

## License

MIT
