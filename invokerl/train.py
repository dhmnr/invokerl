"""Entry point for invokerl training.

Usage:
    python -m invokerl.train --config invokerl/configs/grpo_gsm8k.yaml
    python -m invokerl.train --config invokerl/configs/grpo_gsm8k.yaml --resume ./checkpoints/step_100
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import random
import sys

import torch
import yaml

from invokerl.trainer import Trainer, TrainerConfig

logger = logging.getLogger("invokerl")

# Shared dtype mapping used by policy, ref policy, and generator.
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Registry: config name -> (module_path, class_name)
ALGORITHMS = {
    "grpo": ("invokerl.algorithms.grpo", "GRPO"),
    "dpo": ("invokerl.algorithms.dpo", "DPO"),
    "simpo": ("invokerl.algorithms.simpo", "SimPO"),
    "ppo": ("invokerl.algorithms.ppo", "PPO"),
    "dapo": ("invokerl.algorithms.dapo", "DAPO"),
}
DATASETS = {"gsm8k": ("invokerl.data.gsm8k", "GSM8KDataset")}
REWARDS = {"rule": ("invokerl.rewards.rule", "ExactMatchReward")}


def _import_class(module_path: str, class_name: str):
    """Import a class from a dotted module path."""
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _resolve_registry(name: str, registry: dict, kind: str):
    """Look up a name in a registry and return the class."""
    if name not in registry:
        raise ValueError(f"Unknown {kind}: {name!r}. Available: {list(registry)}")
    module_path, class_name = registry[name]
    return _import_class(module_path, class_name)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_trainer_config(cfg: dict) -> TrainerConfig:
    """Build TrainerConfig from YAML structure."""
    training = cfg.get("training", {})
    generation = cfg.get("generation", {})
    log_cfg = cfg.get("logging", {})

    return TrainerConfig(
        model_name_or_path=cfg["model"]["name_or_path"],
        algorithm=cfg["algorithm"]["name"],
        total_steps=training.get("total_steps", 200),
        lr=float(training.get("lr", 1e-6)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        grad_clip=float(training.get("grad_clip", 1.0)),
        warmup_steps=training.get("warmup_steps", 50),
        lr_schedule=training.get("lr_schedule", "constant"),
        lr_end=float(training.get("lr_end", 0.0)),
        accumulation_steps=training.get("accumulation_steps", 4),
        batch_size=training.get("batch_size", 4),
        group_size=training.get("group_size", 4),
        max_new_tokens=generation.get("max_new_tokens", 384),
        temperature=float(generation.get("temperature", 0.9)),
        top_k=generation.get("top_k", 50),
        log_every=log_cfg.get("log_every", 10),
        eval_every=log_cfg.get("eval_every", 50),
        save_every=log_cfg.get("save_every", 100),
        eval_samples=log_cfg.get("eval_samples", 50),
        max_checkpoints=log_cfg.get("max_checkpoints", 2),
        output_dir=log_cfg.get("output_dir", "./checkpoints"),
    )


# ---------------------------------------------------------------------------
# Component construction
# ---------------------------------------------------------------------------


def build_algorithm(cfg: dict):
    """Instantiate algorithm from config."""
    algo_cfg = cfg["algorithm"]
    cls = _resolve_registry(algo_cfg["name"], ALGORITHMS, "algorithm")
    kwargs = {k: v for k, v in algo_cfg.items() if k != "name"}
    return cls(**kwargs)


def build_dataset(cfg: dict, split: str = "train", max_samples: int = 0):
    """Instantiate dataset from config."""
    name = cfg.get("reward", {}).get("dataset", "gsm8k")
    cls = _resolve_registry(name, DATASETS, "dataset")
    return cls(split=split, max_samples=max_samples)


def build_reward(cfg: dict):
    """Instantiate reward function from config."""
    name = cfg.get("reward", {}).get("type", "rule")
    cls = _resolve_registry(name, REWARDS, "reward")
    return cls()


def build_generator(cfg: dict, trainer_config: TrainerConfig, tensor_parallel_size: int = 1):
    """Instantiate vLLM generation engine."""
    from invokerl.generator import VLLMGenerator

    model_cfg = cfg["model"]
    gen_cfg = cfg.get("generation", {})
    return VLLMGenerator(
        model_name_or_path=model_cfg["name_or_path"],
        gpu_memory_utilization=gen_cfg.get("gpu_memory_utilization", 0.5),
        enforce_eager=gen_cfg.get("enforce_eager", False),
        dtype=model_cfg.get("dtype", "bfloat16"),
        max_model_len=gen_cfg.get("max_model_len", None),
        tensor_parallel_size=tensor_parallel_size,
    )


def build_policy(cfg: dict, device: str = "cuda", frozen: bool = False):
    """Instantiate a policy model.

    Args:
        frozen: If True, returns an eval-mode model with no gradients
                (used as the reference policy).
    """
    from invokerl.policy import PolicyModel

    model_cfg = cfg["model"]
    dtype = DTYPE_MAP.get(model_cfg.get("dtype", "bfloat16"), torch.bfloat16)

    # Reference model doesn't need fp32 master weights -- it's never optimized.
    use_fp32 = model_cfg.get("master_weights_fp32", False) and not frozen

    policy = PolicyModel(
        model_name_or_path=model_cfg["name_or_path"],
        device=device,
        dtype=dtype,
        master_weights_fp32=use_fp32,
    )

    if frozen:
        policy.model.eval()
        for param in policy.model.parameters():
            param.requires_grad = False

    return policy


def build_ref_policy(cfg: dict, device: str = "cuda"):
    """Build frozen reference policy, or None if the algorithm doesn't need one."""
    algo_name = cfg["algorithm"]["name"]
    needs_ref = algo_name in {"grpo", "ppo", "dpo", "dapo"}
    if not needs_ref:
        return None
    return build_policy(cfg, device=device, frozen=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="invokerl -- hackable RL post-training")
    p.add_argument("--config", type=str, required=True, help="YAML config file")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint dir to resume from")
    p.add_argument("--eval-only", action="store_true", help="Evaluate without training")
    p.add_argument("--eval-samples", type=int, default=None, help="Override eval sample count")
    p.add_argument("--max-steps", type=int, default=None, help="Override total_steps")
    p.add_argument("--no-ref", action="store_true", help="Skip reference model (saves memory)")
    p.add_argument("--vllm-ref", action="store_true",
                   help="Use vLLM for ref log-probs instead of a separate model (saves ~1.2 GB)")
    p.add_argument("--max-train-samples", type=int, default=None, help="Limit training dataset size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--verbose", action="store_true", help="Debug logging")

    # Disaggregated pipeline (2+ GPUs).
    p.add_argument("--disagg", action="store_true",
                   help="Run generation and training on separate GPUs")
    p.add_argument("--gen-device", type=str, default="cuda:0",
                   help="First GPU for vLLM generation (default: cuda:0)")
    p.add_argument("--gen-tp", type=int, default=1,
                   help="vLLM tensor parallel size — number of GPUs for generation (default: 1)")
    p.add_argument("--train-device", type=str, default="cuda:1",
                   help="GPU for policy training (default: cuda:1)")
    p.add_argument("--sync-every", type=int, default=1,
                   help="Sync weights to gen every N optimizer steps (default: 1)")
    p.add_argument("--buffer-size", type=int, default=2,
                   help="Max rollout batches in queue (default: 2)")
    p.add_argument("--max-staleness", type=int, default=0,
                   help="Drop batches staler than this (0 = no limit)")

    # FSDP distributed training (requires torchrun launcher).
    p.add_argument("--fsdp", action="store_true",
                   help="Enable FSDP distributed training (requires torchrun)")
    p.add_argument("--train-device-start", type=int, default=1,
                   help="First CUDA device index for training ranks (default: 1)")
    p.add_argument("--fsdp-sharding", type=str, default="FULL_SHARD",
                   choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"],
                   help="FSDP sharding strategy (default: FULL_SHARD)")
    p.add_argument("--fsdp-cpu-offload", action="store_true",
                   help="Offload FSDP params to CPU when not in use")

    # Profiling (overrides the profiling: section of the YAML config).
    p.add_argument("--profile", action="store_true",
                   help="Profile training steps instead of running full training")
    p.add_argument("--profile-num-steps", type=int, default=None,
                   help="Number of profiled steps (overrides config)")
    p.add_argument("--profile-perfetto", action="store_true",
                   help="Also export a Chrome/Perfetto trace for one step")
    p.add_argument("--profile-output-dir", type=str, default=None,
                   help="Output directory for profiling artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # Seed before anything else (CUDA seed deferred until after vLLM init)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    trainer_config = build_trainer_config(cfg)

    if args.eval_samples is not None:
        trainer_config.eval_samples = args.eval_samples
    if args.max_steps is not None:
        trainer_config.total_steps = args.max_steps

    logger.info("Config: %s", args.config)
    logger.info("Algorithm: %s, Model: %s", trainer_config.algorithm, trainer_config.model_name_or_path)
    logger.info("Steps: %d, LR: %.2e, Accum: %d, Group: %d",
                trainer_config.total_steps, trainer_config.lr,
                trainer_config.accumulation_steps, trainer_config.group_size)

    if args.disagg:
        logger.info("Disaggregated mode: gen=%s (TP=%d), train=%s, sync_every=%d, buffer=%d",
                    args.gen_device, args.gen_tp, args.train_device, args.sync_every, args.buffer_size)

    # In disagg mode, vLLM runs on gen_device, policy on train_device.
    gen_device = args.gen_device if args.disagg else "cuda"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # -- vLLM init (BEFORE torch.distributed) ----------------------------------
    # vLLM must initialize before init_process_group because vLLM V1 checks
    # torch.distributed.is_initialized() and tries to coordinate across ranks,
    # causing a deadlock when only rank 0 runs vLLM.
    #
    # When VLLM_ENABLE_V1_MULTIPROCESSING=1, vLLM spawns a child process for
    # EngineCore.  That child inherits env vars — if torchrun's MASTER_ADDR /
    # MASTER_PORT / WORLD_SIZE are present, vLLM's internal init_process_group
    # will try to join torchrun's rendezvous and hang.  We strip them before
    # spawning and restore after.
    gen_tp = args.gen_tp if args.disagg else 1
    if local_rank == 0 and args.disagg:
        gen_device_idx = int(gen_device.split(":")[-1]) if ":" in gen_device else 0
        torch.cuda.set_device(gen_device_idx)

        # Hide torchrun env vars from vLLM's spawned subprocess.
        _torchrun_keys = [
            "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE",
            "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
            "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS", "OMP_NUM_THREADS",
        ]
        _saved_env: dict[str, str] = {}
        for key in _torchrun_keys:
            if key in os.environ:
                _saved_env[key] = os.environ.pop(key)

        logger.info("Initializing vLLM generator (TP=%d) on %s...", gen_tp, gen_device)
        generator = build_generator(cfg, trainer_config, tensor_parallel_size=gen_tp)

        # Restore torchrun env vars for FSDP init_process_group.
        os.environ.update(_saved_env)
    elif not args.disagg:
        generator = build_generator(cfg, trainer_config, tensor_parallel_size=gen_tp)
    else:
        generator = None  # Non-zero ranks don't run generation.

    # -- FSDP distributed setup (AFTER vLLM) -----------------------------------
    fsdp_rank = 0
    fsdp_world = 1
    if args.fsdp:
        from invokerl.distributed import init_distributed
        # Bind each rank to its train GPU (not LOCAL_RANK which may overlap
        # with the gen GPU). E.g., with --train-device-start 1 and 2 ranks:
        #   rank 0 → cuda:1, rank 1 → cuda:2  (cuda:0 is gen)
        train_device_id = args.train_device_start + local_rank
        fsdp_rank = init_distributed(device_id=train_device_id)
        fsdp_world = torch.distributed.get_world_size()
        logger.info("FSDP: rank=%d, world=%d, sharding=%s",
                    fsdp_rank, fsdp_world, args.fsdp_sharding)

    # Build components
    algorithm = build_algorithm(cfg)
    train_dataset = build_dataset(cfg, split="train", max_samples=args.max_train_samples or 0)
    eval_dataset = build_dataset(cfg, split="test")
    reward_fn = build_reward(cfg)

    logger.info("Datasets: %d train, %d eval", len(train_dataset), len(eval_dataset))

    if args.fsdp:
        train_device = f"cuda:{args.train_device_start + fsdp_rank}"
    else:
        train_device = args.train_device if args.disagg else "cuda"

    # Now safe to seed CUDA (vLLM has already forked its engine)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Loading policy model on %s...", train_device)
    policy = build_policy(cfg, device=train_device)

    # Wrap in FSDP before creating optimizer.
    if args.fsdp:
        device_id = args.train_device_start + fsdp_rank
        policy.wrap_fsdp(
            device_id=device_id,
            sharding_strategy=args.fsdp_sharding,
            cpu_offload=args.fsdp_cpu_offload,
        )

    if args.no_ref or args.vllm_ref:
        ref_policy = None
        if args.vllm_ref:
            logger.info("Reference model: using vLLM compute_log_probs (--vllm-ref)")
        else:
            logger.info("Reference model skipped (--no-ref)")
    else:
        # Only rank 0 loads ref policy (it's on the gen device).
        if fsdp_rank == 0:
            ref_policy = build_ref_policy(cfg, device=gen_device if args.disagg else "cuda")
            if ref_policy is None:
                logger.info("Algorithm %s doesn't need a reference model", trainer_config.algorithm)
        else:
            ref_policy = None

    trainer = Trainer(
        config=trainer_config,
        algorithm=algorithm,
        generator=generator,
        policy=policy,
        ref_policy=ref_policy,
        reward_fn=reward_fn,
        dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if args.eval_only:
        if args.resume:
            trainer.load_checkpoint(args.resume)
        if fsdp_rank == 0:
            metrics = trainer.evaluate()
            if metrics:
                logger.info("Eval: %d/%d = %.1f%%",
                            metrics["eval_correct"], metrics["eval_total"],
                            metrics["eval_accuracy"] * 100)
        return

    if args.disagg:
        from invokerl.generator import GenerationConfig
        from invokerl.pipeline import DisaggPipeline, PipelineConfig

        # Only rank 0 creates the pipeline. Other ranks pass None.
        if fsdp_rank == 0:
            pipeline = DisaggPipeline(
                config=PipelineConfig(
                    gen_device=args.gen_device,
                    train_device=train_device,
                    sync_every=args.sync_every,
                    buffer_size=args.buffer_size,
                    max_staleness=args.max_staleness,
                ),
                generator=generator,
                ref_policy=ref_policy,
                reward_fn=reward_fn,
                dataset=train_dataset,
                gen_config=GenerationConfig(
                    max_new_tokens=trainer_config.max_new_tokens,
                    temperature=trainer_config.temperature,
                    top_k=trainer_config.top_k,
                ),
                batch_size=trainer_config.batch_size,
                group_size=trainer_config.group_size,
                use_vllm_ref=args.vllm_ref,
            )
        else:
            pipeline = None

        if args.fsdp:
            history = trainer.train_disagg_distributed(
                pipeline, resume_from=args.resume,
            )
        else:
            history = trainer.train_disagg(pipeline, resume_from=args.resume)
    else:
        history = trainer.train(resume_from=args.resume)

    logger.info("Training complete. %d steps recorded.", len(history))

    # -- Cleanup: explicit teardown to avoid hangs on exit ----------------------
    # Without this, Python's atexit/GC may hang trying to tear down FSDP process
    # groups or vLLM EngineCore subprocesses in the wrong order.
    if args.fsdp and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        logger.info("Distributed process group destroyed.")

    # Shut down vLLM EngineCore subprocess (if multiprocessing mode).
    if generator is not None:
        try:
            if hasattr(generator, 'llm') and hasattr(generator.llm, 'shutdown'):
                generator.llm.shutdown()
            elif hasattr(generator, 'llm') and hasattr(generator.llm, '__del__'):
                del generator.llm
        except Exception:
            pass  # Best-effort cleanup.
        logger.info("Generator shut down.")


if __name__ == "__main__":
    main()
