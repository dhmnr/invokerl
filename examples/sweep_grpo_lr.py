"""Sweep GRPO learning rate — Python-native, no YAML.

Run:
    uv run python examples/sweep_grpo_lr.py

Each value of `lr` kicks off a full training run. Results land in
separate output directories so you can compare accuracy curves.

For parallel sweeps (multiple machines / GPUs): launch this script
per configuration with different LRs via your preferred scheduler
(slurm, ray, or a bash for-loop over machines).
"""

from __future__ import annotations

import random

import torch

import invokerl as rl

LRS = [1e-6, 5e-6, 1e-5, 2e-5]
MODEL = "Qwen/Qwen3-0.6B"
TOTAL_STEPS = 100


def run_one(lr: float) -> None:
    print(f"\n{'=' * 70}\n  SWEEP: lr={lr:.0e}\n{'=' * 70}\n")

    random.seed(42)
    torch.manual_seed(42)

    generator = rl.VLLMGenerator(
        model_name_or_path=MODEL,
        gpu_memory_utilization=0.3,
        max_model_len=2048,
        dtype="bfloat16",
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    policy = rl.Policy(MODEL, dtype=torch.bfloat16)
    ref_policy = rl.Policy(MODEL, dtype=torch.bfloat16).freeze()

    trainer = rl.Trainer(
        config=rl.TrainerConfig(
            model_name_or_path=MODEL,
            total_steps=TOTAL_STEPS,
            lr=lr,
            warmup_steps=20,
            batch_size=1,
            group_size=4,
            accumulation_steps=4,
            max_new_tokens=512,
            eval_every=50,
            save_every=0,
            output_dir=f"./checkpoints/sweep_lr_{lr:.0e}",
        ),
        algorithm=rl.algorithms.GRPO(clip_eps=0.2, beta=0.04),
        generator=generator,
        policy=policy,
        ref_policy=ref_policy,
        reward_fn=rl.rewards.ExactMatch(),
        dataset=rl.datasets.GSM8K("train"),
        eval_dataset=rl.datasets.GSM8K("test"),
    )

    history = trainer.train()
    final_reward = history[-1].get("reward", 0) if history else 0
    print(f"\nSWEEP lr={lr:.0e}: final_reward={final_reward:.3f}")


def main():
    rl.setup_logging()

    # NOTE: sequential loop in one process. vLLM doesn't reload weights
    # cleanly between runs in the same process — prefer launching each
    # LR as its own subprocess if you hit issues:
    #
    #   for lr in 1e-6 5e-6 1e-5 2e-5; do
    #     uv run python examples/sweep_grpo_lr.py --lr $lr
    #   done
    for lr in LRS:
        run_one(lr)


if __name__ == "__main__":
    main()
