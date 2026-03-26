# Decision Log

## 2026-03-24: V2 Architecture Review Complete (7/8 approved)

### Approved Files
- **qwen3.py** (888 lines, onyx): Full Qwen3 transformer with cuTile kernels + NumPy fallback. Causal attention, GQA, QK-Norm, RoPE, SwiGLU FFN. Forward + backward with manual gradient computation. Validated with numerical gradient checks (< 0.6% relative error).
- **optim.py** (314 lines, nova): AdamW with cuTile kernel. Decoupled weight decay, bias-corrected moments, gradient clipping utility.
- **sft.py** (455 lines, nova): Fused log-softmax+gather kernels, masked NLL loss. Clean forward→backward chain.
- **simpo.py** (472 lines, cipher): Reference-free preference optimization. Length-normalized avg log-probs.
- **grpo.py** (461 lines, cipher): Group advantage normalization, clipped surrogate, Schulman KL.
- **ppo.py** (499 lines, cipher): GAE reverse scan, clipped policy loss, clipped value loss, entropy. All three loss components backpropagated.
- **dapo.py** (472 lines, cipher): clip-higher, dynamic sampling, token-level loss, overlong penalty.

### Fixed (was pending)
- **dpo.py** (498 lines, nova): Activation cache mismatch fixed — forward/backward interleaved correctly. Rejected backward runs while rejected cache is active, then chosen re-forward restores cache before chosen backward. 3 policy forwards (optimal).

### Key Design Decisions
1. **Causal masking**: cuTile kernel uses `ct.arange` for row/col indices + `ct.where(col > row, -inf, 0.0)`. NumPy fallback uses `np.triu(..., k=1)`.
2. **No autograd**: All backward passes are explicit. Model caches activations in `self._cache` during forward, consumed during backward. This means forward/backward must be called in matching pairs.
3. **Self-contained files with shared model**: Each algorithm file is standalone except for importing `qwen3.py` (model) and `optim.py` (optimizer). Kernel code (e.g., log_softmax_gather) is duplicated across files by design.
4. **ConstInt × 1000 pattern**: Float hyperparameters passed to cuTile kernels as `int(val * 1000)`, decoded in kernel as `val / 1000.0`. Limits precision to 3 decimal places.
5. **PPO simplification**: Value head is a separate linear layer with its own optimizer step; entropy gradient added directly to policy logits gradient.

## 2026-03-25: Architecture Pivot — invokerl (Production RL Framework)

### Decision
Abandoned cuTile educational approach. New direction: production RL research platform with hackable algorithms, fast generation (vLLM), PyTorch backend. Named `invokerl`.

### Design Principles
- **Hackable algorithms, production infrastructure**: Researchers only write `compute_advantages()` + `compute_loss()`. Everything else (generation, forward/backward, optimization, weight sync) is handled by the trainer.
- **Credit assignment is first-class**: `compute_advantages()` is overridable for experimenting with group normalization, GAE, token-level rewards, PRM.
- **vLLM in-process**: Uses `vllm.LLM` directly (not server) for generation with PagedAttention. Weight sync via direct tensor copy.
- **Single GPU scale** (for now). Qwen3-0.6B fits: ~23 GB total (vLLM 16 GB + policy 6 GB + ref 1.2 GB).

### File Structure (invokerl/)
```
invokerl/
├── __init__.py
├── __main__.py              # python -m invokerl entry
├── train.py                 # CLI entry point, component wiring
├── algorithms/
│   ├── base.py              # RolloutBatch + BaseAlgorithm ABC
│   └── grpo.py              # GRPO: group-normalized advantages, clipped surrogate + KL
├── engine/
│   ├── trainer.py           # Training loop: rollout → train_step → optimizer_step → eval
│   ├── generator.py         # BaseGenerator ABC + VLLMGenerator
│   └── policy.py            # PolicyModel (HF CausalLM wrapper)
├── data/
│   ├── base.py              # PromptItem + BaseDataset ABC
│   └── gsm8k.py             # GSM8K dataset from HuggingFace
├── rewards/
│   ├── base.py              # BaseReward ABC
│   └── rule.py              # ExactMatchReward (answer extraction + numeric matching)
└── configs/
    └── grpo_gsm8k.yaml      # Qwen3-0.6B + GRPO on GSM8K
```

### Component Ownership
- **atlas**: Architecture, base interfaces (base.py files), review
- **cipher**: algorithms/grpo.py (148 lines)
- **onyx**: engine/generator.py (VLLMGenerator, 392 lines), engine/policy.py
- **nova**: engine/trainer.py (470 lines), data/gsm8k.py, rewards/rule.py, train.py entry point

### All Components Reviewed and Approved (atlas)
- algorithms/grpo.py ✅
- engine/trainer.py ✅
- engine/generator.py ✅
- data/gsm8k.py ✅
- rewards/rule.py ✅
- rewards/base.py signature fix (ground_truths param) ✅

### Status: Ready for integration test on server
```
python -m invokerl --config invokerl/configs/grpo_gsm8k.yaml --max-steps 2 --eval-samples 5 --verbose
```

