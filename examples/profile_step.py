"""Profile one GRPO training step + export a Perfetto trace.

Run:
    uv run python examples/profile_step.py

Produces:
    results/profile/trace.json  — open at ui.perfetto.dev
    stdout                       — wall / CPU / CUDA / per-phase breakdown

Also compatible with nsys: the same NVTX markers are emitted automatically.
    nsys profile --trace=cuda,nvtx python examples/profile_step.py
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

    generator = rl.VLLMGenerator(
        model_name_or_path=MODEL,
        gpu_memory_utilization=0.3,
        max_model_len=2048,
        dtype="bfloat16",
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    policy = rl.Policy(MODEL, dtype=torch.bfloat16)

    trainer = rl.Trainer(
        config=rl.TrainerConfig(
            model_name_or_path=MODEL,
            total_steps=10,
            lr=5e-6,
            batch_size=1,
            group_size=4,
            accumulation_steps=4,
            max_new_tokens=512,
            eval_every=0,  # skip eval during profiling
            save_every=0,  # skip checkpoints
        ),
        algorithm=rl.algorithms.GRPO(clip_eps=0.2, beta=0.04),
        generator=generator,
        policy=policy,
        ref_policy=None,  # saves memory for this profiling run
        reward_fn=rl.rewards.ExactMatch(),
        dataset=rl.datasets.GSM8K("train"),
    )

    # Warmup step (not profiled — amortizes CUDA init, JIT, allocator caches)
    print("\n[warmup] Running 1 step...")
    prompts = trainer.dataset.sample(trainer.config.batch_size)
    batch = trainer.rollout(prompts)
    trainer.train_step(batch)
    trainer.optimizer_step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Profiled step
    print("\n[profile] Running 1 profiled step...")
    with rl.profile() as p:
        prompts = trainer.dataset.sample(trainer.config.batch_size)
        for _ in range(trainer.config.accumulation_steps):
            batch = trainer.rollout(prompts)
            trainer.train_step(batch)
        trainer.optimizer_step()

    p.summary()
    p.export_trace("results/profile/trace.json")


if __name__ == "__main__":
    main()
