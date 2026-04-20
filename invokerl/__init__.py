"""invokerl — hackable RL post-training for LLMs.

Library-first. Compose a training run by passing objects to Trainer:

    import invokerl as rl

    policy = rl.Policy("Qwen/Qwen3-0.6B")
    trainer = rl.Trainer(
        policy=policy,
        ref_policy=rl.Policy("Qwen/Qwen3-0.6B", frozen=True),
        generator=rl.VLLMGenerator("Qwen/Qwen3-0.6B"),
        algorithm=rl.GRPO(clip_eps=0.2, beta=0.04),
        dataset=rl.GSM8K("train"),
        eval_dataset=rl.GSM8K("test"),
        reward=rl.ExactMatch(),
        lr=5e-6, total_steps=200, batch_size=1, group_size=4, accumulation_steps=4,
    )
    trainer.train()

Multi-GPU is seamless — just pass different objects:

    # Disaggregated (gen on GPU 0, train on GPU 1):
    policy = rl.Policy("Qwen/...", device="cuda:1")
    gen = rl.VLLMGenerator("Qwen/...", device="cuda:0")
    pipeline = rl.DisaggPipeline(...)
    trainer.train(pipeline=pipeline)

    # FSDP (with torchrun):
    policy = rl.Policy("Qwen/...").fsdp()  # auto-init torch.distributed
    trainer.train(pipeline=pipeline)       # FSDP auto-detected

Profiling is opt-in:

    with rl.profile() as p:
        trainer.step()      # or trainer.train() for a full run
    p.summary()
    p.export_trace("trace.json")   # open at ui.perfetto.dev
"""

from __future__ import annotations

# Core
from invokerl.trainer import Trainer, TrainerConfig
from invokerl.policy import PolicyModel as Policy
from invokerl.generator import VLLMGenerator, GenerationConfig
from invokerl.pipeline import DisaggPipeline, PipelineConfig

# Algorithms
from invokerl.algorithms.base import BaseAlgorithm, RolloutBatch
from invokerl.algorithms.grpo import GRPO
from invokerl.algorithms.dpo import DPO
from invokerl.algorithms.ppo import PPO
from invokerl.algorithms.simpo import SimPO
from invokerl.algorithms.dapo import DAPO

# Data
from invokerl.data.base import BaseDataset, PromptItem
from invokerl.data.gsm8k import GSM8KDataset as GSM8K

# Rewards
from invokerl.rewards.base import BaseReward
from invokerl.rewards.rule import ExactMatchReward as ExactMatch

# Profiling
from invokerl.profiling import profile, annotate, Profile

__all__ = [
    # Core
    "Trainer", "TrainerConfig",
    "Policy",
    "VLLMGenerator", "GenerationConfig",
    "DisaggPipeline", "PipelineConfig",
    # Algorithms
    "BaseAlgorithm", "RolloutBatch",
    "GRPO", "DPO", "PPO", "SimPO", "DAPO",
    # Data
    "BaseDataset", "PromptItem",
    "GSM8K",
    # Rewards
    "BaseReward",
    "ExactMatch",
    # Profiling
    "profile", "annotate", "Profile",
]
