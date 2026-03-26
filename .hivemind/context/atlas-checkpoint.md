# Atlas Progress Checkpoint (2026-03-25, invokerl phase)

## Current Status
**All invokerl components implemented and reviewed. Waiting on integration smoke test on server.**

## Architecture Pivot
Abandoned cuTile/CuPy (generation was 97.8% of training time at 24.5% GPU util).
New framework: **invokerl** — PyTorch + vLLM, hackable RL algorithms, production infrastructure.

## tilerl Results (preserved)
- GRPO training: 46% → 54% on GSM8K with Qwen3-0.6B
- Pure cuTile/CuPy, no PyTorch, single RTX 5090

## invokerl Implementation Status
| Component | File | Owner | Status |
|-----------|------|-------|--------|
| Base algorithm interface | `algorithms/base.py` | atlas | ✅ Done |
| GRPO algorithm | `algorithms/grpo.py` | cipher | ✅ Reviewed |
| vLLM generator | `engine/generator.py` | onyx | ✅ Reviewed |
| Policy model wrapper | `engine/policy.py` | atlas | ✅ Done |
| Trainer loop | `engine/trainer.py` | nova | ✅ Reviewed |
| Reward interface + rule | `rewards/base.py`, `rewards/rule.py` | nova | ✅ Reviewed |
| Dataset interface + GSM8K | `data/base.py`, `data/gsm8k.py` | nova | ✅ Reviewed |
| Entry point | `train.py`, `__main__.py` | nova | ✅ Reviewed |
| YAML config | `configs/grpo_gsm8k.yaml` | atlas | ✅ Done |

## What's Next
1. **onyx**: Deploy to server, install deps (vllm, transformers, datasets, pyyaml), run smoke test
   - Task ID: 48c6594b2234
   - Smoke test: `python -m invokerl --config invokerl/configs/grpo_gsm8k.yaml --max-steps 2 --eval-samples 5 --no-ref --verbose`
2. If smoke test passes, run full 200-step GRPO training
3. Compare invokerl results vs tilerl results (both should reach ~54% on GSM8K)

## Server Details
- SSH: `ssh -p 54410 root@93.91.156.101`
- Model: Qwen/Qwen3-0.6B (cached at `/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/...`)
- GPU: RTX 5090 32GB, CUDA 13.1

## Key Design Decisions
- **RolloutBatch** is the contract between trainer and algorithms
- **compute_advantages()** separated from **compute_loss()** for credit assignment hackability
- **token_rewards** field in RolloutBatch supports per-token reward signals
- **vLLM in-process** (not server) with prefix caching for GRPO groups
- **Weight sync** via direct parameter copy after each optimizer step
- **Memory budget**: ~23 GB for 0.6B (fits 32 GB), future optimization: shared ref model via vLLM compute_log_probs()
