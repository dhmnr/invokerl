"""invokerl — hackable RL post-training for LLMs.

Library-first. Compose a training run by passing objects to Trainer:

    import invokerl as rl
    from invokerl.algorithms import GRPO
    from invokerl.datasets import GSM8K
    from invokerl.rewards import ExactMatch

    policy = rl.Policy("Qwen/Qwen3-0.6B")
    trainer = rl.Trainer(
        policy=policy,
        ref_policy=rl.Policy("Qwen/Qwen3-0.6B").freeze(),
        generator=rl.VLLMGenerator("Qwen/Qwen3-0.6B"),
        algorithm=GRPO(clip_eps=0.2, beta=0.04),
        dataset=GSM8K("train"),
        eval_dataset=GSM8K("test"),
        reward_fn=ExactMatch(),
        lr=5e-6, total_steps=200, batch_size=1, group_size=4, accumulation_steps=4,
    )
    trainer.train()

Multi-GPU is seamless — pass different objects:

    # Disaggregated (gen on GPU 0, train on GPU 1):
    trainer.train(pipeline=rl.DisaggPipeline(...))

    # FSDP (with torchrun):
    policy = rl.Policy("Qwen/...").fsdp()   # auto-init torch.distributed
    trainer.train(pipeline=...)             # FSDP auto-detected

Profiling is opt-in:

    with rl.profile() as p:
        trainer.step()
    p.summary()
    p.export_trace("trace.json")   # open at ui.perfetto.dev

## Namespace layout

    rl.Trainer, rl.Policy, rl.VLLMGenerator,
    rl.DisaggPipeline, rl.profile     — framework core (always needed)

    rl.BaseAlgorithm, rl.BaseDataset,
    rl.BaseReward, rl.RolloutBatch,
    rl.PromptItem                     — base contracts (for type hints / subclasses)

    rl.algorithms.GRPO/DPO/PPO/SimPO/DAPO   — concrete algorithm implementations
    rl.datasets.GSM8K                       — concrete datasets
    rl.rewards.ExactMatch                   — concrete reward functions
"""

from __future__ import annotations

# Framework core
from invokerl.trainer import Trainer, TrainerConfig
from invokerl.policy import PolicyModel as Policy
from invokerl.generator import VLLMGenerator, GenerationConfig
from invokerl.pipeline import DisaggPipeline, PipelineConfig

# Base contracts — exposed at top level for type hints and subclassing
from invokerl.algorithms.base import BaseAlgorithm, RolloutBatch
from invokerl.datasets.base import BaseDataset, PromptItem
from invokerl.rewards.base import BaseReward

# Profiling — small surface, cross-cutting
from invokerl.profiling import profile, annotate, Profile

# Concrete implementations live in submodules:
#   from invokerl.algorithms import GRPO        (or rl.algorithms.GRPO)
#   from invokerl.datasets import GSM8K
#   from invokerl.rewards import ExactMatch
from invokerl import algorithms, datasets, rewards

__all__ = [
    # Core
    "Trainer", "TrainerConfig",
    "Policy",
    "VLLMGenerator", "GenerationConfig",
    "DisaggPipeline", "PipelineConfig",
    # Base contracts
    "BaseAlgorithm", "RolloutBatch",
    "BaseDataset", "PromptItem",
    "BaseReward",
    # Profiling
    "profile", "annotate", "Profile",
    # Submodules
    "algorithms", "datasets", "rewards",
]
