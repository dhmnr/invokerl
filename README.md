# invokerl

Hackable RL post-training for LLMs. Production-grade generation (vLLM), research-grade algorithms (PyTorch), library-first API.

## Why

RL post-training research is bottlenecked by infrastructure, not ideas. Most of your time is spent fighting OOM errors, debugging weight sync, and waiting for generation — not experimenting with credit assignment or loss functions.

invokerl is a **library**, not a framework. You write a Python script that composes objects (`Trainer`, `Policy`, `VLLMGenerator`, an algorithm, a dataset, a reward). No YAML, no CLI, no config monster. To extend — clone the repo and edit.

## Install

```bash
uv sync
```

## Quick start — single GPU

```python
import invokerl as rl

MODEL = "Qwen/Qwen3-0.6B"

generator = rl.VLLMGenerator(MODEL, gpu_memory_utilization=0.3, max_model_len=2048)
policy = rl.Policy(MODEL)
ref_policy = rl.Policy(MODEL)  # frozen ref for KL
ref_policy.model.eval()
for p in ref_policy.model.parameters():
    p.requires_grad = False

trainer = rl.Trainer(
    config=rl.TrainerConfig(
        model_name_or_path=MODEL, total_steps=200, lr=5e-6,
        batch_size=1, group_size=4, accumulation_steps=4,
    ),
    algorithm=rl.GRPO(clip_eps=0.2, beta=0.04),
    generator=generator, policy=policy, ref_policy=ref_policy,
    reward_fn=rl.ExactMatch(),
    dataset=rl.GSM8K("train"),
    eval_dataset=rl.GSM8K("test"),
)
trainer.train()
```

Full runnable: [examples/train_grpo_gsm8k.py](examples/train_grpo_gsm8k.py)

## Multi-GPU is seamless

Same `trainer.train()` — pass different objects:

```python
# Disagg (generation on cuda:0, training on cuda:1)
pipeline = rl.DisaggPipeline(...)
trainer.train(pipeline=pipeline)

# FSDP (launch with torchrun)
policy = rl.Policy(MODEL).fsdp()     # auto-inits torch.distributed
trainer.train(pipeline=pipeline)     # FSDP auto-detected from the policy
```

Full runnable: [examples/train_disagg.py](examples/train_disagg.py), [examples/train_fsdp.py](examples/train_fsdp.py)

## Profiling is first-class

```python
with rl.profile() as p:
    trainer.step()

p.summary()                   # wall / CPU / CUDA / unaccounted + per-phase
p.export_trace("trace.json")  # open at ui.perfetto.dev
```

Also works with nsys — the NVTX markers are emitted unconditionally, no extra flag needed:

```bash
nsys profile --trace=cuda,nvtx python examples/train_grpo_gsm8k.py
```

Full runnable: [examples/profile_step.py](examples/profile_step.py)

## Writing a new algorithm

Every algorithm implements two methods:

```python
from invokerl import BaseAlgorithm, RolloutBatch

class MyAlgorithm(BaseAlgorithm):
    def compute_advantages(self, batch: RolloutBatch) -> Tensor:
        """Turn rewards into per-token learning signals. The credit
        assignment hook — override for group normalization, GAE,
        token-level shaping, PRM scores, etc."""
        ...

    def compute_loss(self, new_log_probs, batch, advantages):
        """The policy objective. Return (loss, metrics)."""
        ...
```

Pass it to `Trainer`:

```python
trainer = rl.Trainer(..., algorithm=MyAlgorithm(...))
```

No registry, no configuration, no plugin system. It's just a class. Five algorithms already exist as reference: [`GRPO`](invokerl/algorithms/grpo.py), [`DPO`](invokerl/algorithms/dpo.py), [`PPO`](invokerl/algorithms/ppo.py), [`SimPO`](invokerl/algorithms/simpo.py), [`DAPO`](invokerl/algorithms/dapo.py).

## RolloutBatch

The data contract between the trainer and your algorithm:

| Field | Shape | Description |
|-------|-------|-------------|
| `token_ids` | `[B, T]` | Prompt + completion token IDs |
| `response_mask` | `[B, T]` | True for generated tokens |
| `rewards` | `[B]` | Per-sequence scalar rewards |
| `token_rewards` | `[B, T]` | Optional per-token rewards |
| `old_log_probs` | `[B, T]` | Log-probs from policy at generation time |
| `ref_log_probs` | `[B, T]` | Log-probs from frozen reference model |
| `group_ids` | `[B]` | Which prompt each completion belongs to |
| `group_size` | `int` | Completions per prompt |

## Project structure

```
invokerl/
├── __init__.py       # public API (rl.Trainer, rl.Policy, rl.GRPO, ...)
├── trainer.py        # Trainer: train() dispatches to internal standard/disagg/FSDP paths
├── policy.py         # PolicyModel + .fsdp() for distributed
├── generator.py      # VLLMGenerator
├── pipeline.py       # DisaggPipeline (optional, for 2-GPU async)
├── distributed.py    # FSDP init helpers
├── profiling.py      # rl.profile() context manager
├── algorithms/       # base + GRPO, DPO, PPO, SimPO, DAPO
├── data/             # base + GSM8K
└── rewards/          # base + rule-based exact match

examples/
├── train_grpo_gsm8k.py     # single GPU
├── train_disagg.py         # 2 GPUs async
├── train_fsdp.py           # FSDP multi-GPU
├── profile_step.py         # profiling
└── sweep_grpo_lr.py        # hyperparameter sweep
```

## Design principles

- **Library, not framework.** You write Python. To extend, you edit the library.
- **Composition over configuration.** `trainer.train()` is the only entry point. Mode (single-GPU vs disagg vs FSDP) comes from what you pass in, not from flags.
- **Credit assignment is a first-class hook.** `compute_advantages()` is separate from `compute_loss()` so group-norm, GAE, PRM experiments touch 10 lines.
- **Profiling is a context manager.** Opt-in, one line. NVTX markers are always emitted (free when no one's watching).
- **vLLM in-process.** No HTTP, direct GPU→GPU weight sync. Shared-weights mode makes updates zero-copy.
- **Clone-to-hack.** Don't layer abstractions to accommodate extensions; fork the repo and change the code. That's how the reference algorithms are written.

## License

MIT
