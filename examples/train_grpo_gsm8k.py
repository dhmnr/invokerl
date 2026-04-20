"""GRPO on GSM8K — single GPU. The simplest recipe.

Run:
    uv run python examples/train_grpo_gsm8k.py
"""

from __future__ import annotations

import logging
import random

import torch

import invokerl as rl


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    random.seed(42)
    torch.manual_seed(42)

    MODEL = "Qwen/Qwen3-0.6B"

    # Generator goes up first — CUDA initialization must happen inside vLLM.
    generator = rl.VLLMGenerator(
        model_name_or_path=MODEL,
        gpu_memory_utilization=0.3,  # leave room for policy + ref model
        max_model_len=2048,
        dtype="bfloat16",
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    policy = rl.Policy(MODEL, dtype=torch.bfloat16)

    # Reference policy (frozen, used for KL penalty)
    ref_policy = rl.Policy(MODEL, dtype=torch.bfloat16).freeze()

    trainer = rl.Trainer(
        config=rl.TrainerConfig(
            model_name_or_path=MODEL,
            algorithm="grpo",
            total_steps=200,
            lr=5e-6,
            warmup_steps=50,
            lr_schedule="cosine",
            lr_end=1e-6,
            batch_size=1,
            group_size=4,
            accumulation_steps=4,
            max_new_tokens=512,
            temperature=0.9,
            top_k=50,
            log_every=10,
            eval_every=100,
            save_every=500,
            output_dir="./checkpoints/grpo_gsm8k",
        ),
        algorithm=rl.algorithms.GRPO(clip_eps=0.2, beta=0.04),
        generator=generator,
        policy=policy,
        ref_policy=ref_policy,
        reward_fn=rl.rewards.ExactMatch(),
        dataset=rl.datasets.GSM8K("train"),
        eval_dataset=rl.datasets.GSM8K("test"),
    )

    trainer.train()


if __name__ == "__main__":
    main()
