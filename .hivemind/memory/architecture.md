# Architecture Decisions

## Project: invokerl (formerly tilerl)
Hackable post-training RL research framework for LLMs.
**PyTorch + vLLM. Hackable algorithms, production infrastructure.**

## Pivot (2026-03-25)
Abandoned cuTile/CuPy custom kernels. Switched to PyTorch + vLLM.
Reason: Profiling showed generation was 97.8% of training time at 24.5% GPU
utilization. Custom inference can't compete with vLLM. Research value is in
the RL algorithms and credit assignment, not the infrastructure.

## tilerl Results (preserved)
- Proved GRPO works end-to-end: 46% → 54% on GSM8K with Qwen3-0.6B
- Pure cuTile/CuPy, no PyTorch, single RTX 5090
- Old code preserved in `tilerl/` directory

## Design Principles
1. **Hackable where it matters** — algorithm files are pure loss functions (~50-100 lines)
2. **Production where it doesn't** — vLLM for generation, PyTorch+autograd for training
3. **Credit assignment is first-class** — compute_advantages() is separate and overridable
4. **One algorithm = one file** — same principle as tilerl, but cleaner

## Architecture

```
Algorithm Layer (hackable)
  ↓ compute_loss(new_log_probs, batch, advantages) → loss
  ↓ compute_advantages(batch) → per-token advantages
Trainer (orchestrator)
  ↓ generates, scores rewards, runs forward/backward
Infrastructure Layer
  ├── vLLM (generation + log_probs)
  └── PyTorch (model forward/backward + optimizer)
```

## File Structure
```
invokerl/
├── algorithms/          # ★ THE HACKABLE PART ★
│   ├── base.py         # BaseAlgorithm + RolloutBatch
│   ├── grpo.py         # GRPO loss + group advantage
│   ├── dpo.py          # DPO preference loss
│   ├── simpo.py        # SimPO reference-free
│   ├── ppo.py          # PPO + GAE + value
│   └── dapo.py         # DAPO + clip-higher
├── engine/             # Infrastructure (don't hack)
│   ├── generator.py    # vLLM wrapper
│   ├── policy.py       # PyTorch model wrapper
│   └── trainer.py      # Main training loop
├── rewards/            # Pluggable reward functions
│   ├── base.py         # RewardFunction interface
│   └── rule.py         # Rule-based (GSM8K, etc.)
├── data/               # Dataset loading
│   ├── base.py         # BaseDataset
│   └── gsm8k.py        # GSM8K
└── configs/            # YAML configs
    └── grpo_gsm8k.yaml
```

## Key Interface: RolloutBatch
The contract between infrastructure and algorithms:
- token_ids, masks (prompt, response, attention)
- rewards [B] + optional token_rewards [B, T] (for credit assignment)
- old_log_probs [B, T] from generation
- ref_log_probs [B, T] from reference model
- group_ids + group_size (for GRPO-style algorithms)
- extras dict for algorithm-specific data

## Key Interface: BaseAlgorithm
Two methods to override:
- `compute_advantages(batch)` → [B, T] per-token advantages
- `compute_loss(new_log_probs, batch, advantages)` → (loss, metrics)

## Algorithm Overview
| Algorithm | Type | Key Idea |
|-----------|------|----------|
| GRPO | RL | Group sampling, normalize rewards within group, no critic |
| DPO | Preference | Implicit reward from preference pairs, needs reference model |
| SimPO | Preference | Avg log-prob as reward, no reference model |
| PPO | RL | Clipped surrogate + value critic + GAE |
| DAPO | RL | GRPO + clip-higher + dynamic sampling + token-level loss |

## Dependencies
- torch >= 2.4
- transformers >= 4.45
- vllm >= 0.6
- datasets
- pyyaml

## Entry Point
```bash
python -m invokerl --config invokerl/configs/grpo_gsm8k.yaml
python -m invokerl --config ... --max-steps 2 --eval-samples 5 --no-ref --verbose  # smoke test
python -m invokerl --config ... --eval-only --resume ./checkpoints/step_100        # eval only
```

Registry-based component instantiation from YAML config:
- `train.py` → `build_algorithm()`, `build_generator()`, `build_policy()`, `build_reward()`, `build_dataset()`
- Adding a new algorithm: add entry to `ALGORITHMS` dict + algorithm file

## Memory Budget (Single GPU, Qwen3-0.6B bf16)
| Component | Memory |
|-----------|--------|
| vLLM engine (gpu_memory_utilization=0.5) | ~16 GB |
| PolicyModel (bf16 + optimizer states) | ~6 GB |
| Reference PolicyModel (bf16, frozen) | ~1.2 GB |
| **Total** | **~23 GB** |

Fits 32 GB RTX 5090. For 4B+ models, use `--no-ref` or shared-ref optimization
(use `generator.compute_log_probs()` for ref instead of separate PolicyModel).

## Scale Path
Single GPU first (vLLM + PyTorch on same GPU, alternating).
Multi-GPU later (vLLM tensor-parallel, FSDP for training).

## Validated Results (2026-03-26)
- **fp32 lr=1e-6: 56% at step 200 (+6pp over baseline)** — beats tilerl's 54%
- **Wall-clock: 40 min for 200 steps** (vs 5.5 hours with cuTile) — 8× faster
- KL regularization is load-bearing (no-ref run collapsed 50→38%)
- bf16 optimizer precision kills learning at low LR — fp32 master weights required
- 50-sample eval is too noisy (±7%) — use 200+ samples (±3.5%)
- batch_size=1 with group_size=4 and accumulation_steps=4 matches tilerl config

## Critical Config: fp32 Master Weights
PolicyModel loads weights in fp32, uses `torch.autocast("cuda", dtype=bfloat16)` for forward.
Optimizer states (m, v) stay fp32 → small lr=1e-6 updates are preserved.
Without this, bf16 rounds away gradient updates (KL stays at 0.0003 = policy frozen).

## Known Future Improvements
1. **Shared reference model** — use vLLM's `compute_log_probs()` instead of separate PolicyModel (~1.2 GB savings)
2. **Multi-GPU disaggregation** — vLLM generation on GPU 1, training on GPU 2, pipeline overlap
3. **Large dataset lazy loading** — GSM8K loads all items into memory (fine for 8.5K, not for millions)
