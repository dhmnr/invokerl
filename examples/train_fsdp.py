"""FSDP GRPO — generation on cuda:0, FSDP-sharded training across cuda:1..N.

Requires torchrun:
    torchrun --nproc_per_node=2 examples/train_fsdp.py

With --nproc_per_node=2: rank 0 runs on cuda:1, rank 1 runs on cuda:2.
vLLM generation stays on cuda:0 and is only driven by rank 0.
"""

from __future__ import annotations

import os
import random

import torch

import invokerl as rl


def main():
    rl.setup_logging()
    random.seed(42)
    torch.manual_seed(42)

    MODEL = "Qwen/Qwen3-0.6B"
    GEN_DEVICE = "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Rank 0 → cuda:1; rank 1 → cuda:2; etc.
    train_device_id = 1 + local_rank

    # --- vLLM init (BEFORE torch.distributed) ---------------------------------
    # vLLM must initialize before init_process_group to avoid deadlocks.
    # Strip torchrun env vars so vLLM's internal init doesn't try to join
    # torchrun's rendezvous.
    generator = None
    if local_rank == 0:
        _torchrun_keys = [
            "MASTER_ADDR",
            "MASTER_PORT",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "TORCHELASTIC_RUN_ID",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS",
            "OMP_NUM_THREADS",
        ]
        _saved = {k: os.environ.pop(k) for k in _torchrun_keys if k in os.environ}

        torch.cuda.set_device(0)
        generator = rl.VLLMGenerator(
            model_name_or_path=MODEL,
            gpu_memory_utilization=0.5,
            max_model_len=2048,
            dtype="bfloat16",
        )
        os.environ.update(_saved)  # restore for FSDP init

    # --- FSDP policy (on cuda:train_device_id) --------------------------------
    policy = rl.Policy(MODEL, device=f"cuda:{train_device_id}", dtype=torch.bfloat16)
    policy.fsdp(device_id=train_device_id, sharding="FULL_SHARD")  # auto-inits torch.distributed

    # --- Reference policy (only on rank 0, on gen GPU) ------------------------
    ref_policy = None
    if local_rank == 0:
        ref_policy = rl.Policy(MODEL, device=GEN_DEVICE, dtype=torch.bfloat16).freeze()

    # --- Trainer + Disagg pipeline --------------------------------------------
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
            output_dir="./checkpoints/grpo_gsm8k_fsdp",
        ),
        algorithm=rl.algorithms.GRPO(clip_eps=0.2, beta=0.04),
        generator=generator,
        policy=policy,
        ref_policy=ref_policy,
        reward_fn=rl.rewards.ExactMatch(),
        dataset=rl.datasets.GSM8K("train"),
        eval_dataset=rl.datasets.GSM8K("test") if local_rank == 0 else None,
    )

    pipeline = None
    if local_rank == 0:
        pipeline = rl.DisaggPipeline(
            config=rl.PipelineConfig(
                gen_device=GEN_DEVICE,
                train_device=f"cuda:{train_device_id}",
                sync_every=1,
                buffer_size=2,
                max_staleness=0,
            ),
            generator=generator,
            ref_policy=ref_policy,
            reward_fn=rl.rewards.ExactMatch(),
            dataset=rl.datasets.GSM8K("train"),
            gen_config=rl.GenerationConfig(max_new_tokens=512, temperature=0.9, top_k=50),
            batch_size=1,
            group_size=4,
        )

    # Trainer.train() auto-detects FSDP from the policy and picks the
    # right internal loop. Same method as single-GPU and non-FSDP disagg.
    trainer.train(pipeline=pipeline)

    # Cleanup
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
