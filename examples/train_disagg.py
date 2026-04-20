"""Disaggregated GRPO — generation on cuda:0, training on cuda:1.

Run:
    uv run python examples/train_disagg.py

The async DisaggPipeline runs vLLM generation on a background thread on
GPU 0 while the training step runs on GPU 1. Weights sync every
`sync_every` optimizer steps (pipeline handles this asynchronously).
"""

from __future__ import annotations

import logging
import random

import torch

import invokerl as rl


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    random.seed(42)
    torch.manual_seed(42)

    MODEL = "Qwen/Qwen3-0.6B"
    GEN_DEVICE = "cuda:0"
    TRAIN_DEVICE = "cuda:1"

    # Generator lives on GPU 0
    torch.cuda.set_device(0)
    generator = rl.VLLMGenerator(
        model_name_or_path=MODEL,
        gpu_memory_utilization=0.5,  # more memory available without policy on this GPU
        max_model_len=2048,
        dtype="bfloat16",
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Policy + ref policy live on GPU 1
    policy = rl.Policy(MODEL, device=TRAIN_DEVICE, dtype=torch.bfloat16)
    # Ref policy lives on gen GPU (frozen, used for KL penalty)
    ref_policy = rl.Policy(MODEL, device=GEN_DEVICE, dtype=torch.bfloat16).freeze()

    trainer = rl.Trainer(
        config=rl.TrainerConfig(
            model_name_or_path=MODEL,
            total_steps=200,
            lr=5e-6,
            warmup_steps=50,
            batch_size=1,
            group_size=4,
            accumulation_steps=4,
            max_new_tokens=512,
            temperature=0.9,
            output_dir="./checkpoints/grpo_gsm8k_disagg",
        ),
        algorithm=rl.algorithms.GRPO(clip_eps=0.2, beta=0.04),
        generator=generator,
        policy=policy,
        ref_policy=ref_policy,
        reward_fn=rl.rewards.ExactMatch(),
        dataset=rl.datasets.GSM8K("train"),
        eval_dataset=rl.datasets.GSM8K("test"),
    )

    # Assemble the async pipeline
    pipeline = rl.DisaggPipeline(
        config=rl.PipelineConfig(
            gen_device=GEN_DEVICE,
            train_device=TRAIN_DEVICE,
            sync_every=1,  # sync weights every optimizer step
            buffer_size=2,  # double-buffered rollouts
            max_staleness=0,  # 0 = no staleness limit
        ),
        generator=generator,
        ref_policy=ref_policy,
        reward_fn=rl.rewards.ExactMatch(),
        dataset=rl.datasets.GSM8K("train"),
        gen_config=rl.GenerationConfig(
            max_new_tokens=512,
            temperature=0.9,
            top_k=50,
        ),
        batch_size=1,
        group_size=4,
    )

    # Trainer.train() auto-detects disagg mode from the pipeline argument
    trainer.train(pipeline=pipeline)


if __name__ == "__main__":
    main()
