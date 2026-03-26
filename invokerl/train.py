"""Entry point for invokerl training.

Usage:
    python -m invokerl.train --config invokerl/configs/grpo_gsm8k.yaml
    python -m invokerl.train --config invokerl/configs/grpo_gsm8k.yaml --resume ./checkpoints/grpo_gsm8k/step_100

Loads a YAML config, instantiates all components (algorithm, generator, policy,
reward, dataset), and runs the training loop.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import torch
import yaml

from invokerl.engine.trainer import Trainer, TrainerConfig

logger = logging.getLogger("invokerl")


# ---------------------------------------------------------------------------
# Registry: map config names → classes
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "grpo": ("invokerl.algorithms.grpo", "GRPO"),
}

DATASETS = {
    "gsm8k": ("invokerl.data.gsm8k", "GSM8KDataset"),
}

REWARDS = {
    "rule": ("invokerl.rewards.rule", "ExactMatchReward"),
}


def _import_class(module_path: str, class_name: str):
    """Import a class from a dotted module path."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    """Load YAML config and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_trainer_config(cfg: dict) -> TrainerConfig:
    """Build TrainerConfig from the flat YAML structure."""
    training = cfg.get("training", {})
    generation = cfg.get("generation", {})
    log_cfg = cfg.get("logging", {})

    return TrainerConfig(
        model_name_or_path=cfg["model"]["name_or_path"],
        algorithm=cfg["algorithm"]["name"],
        # Training
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
        # Generation
        max_new_tokens=generation.get("max_new_tokens", 384),
        temperature=float(generation.get("temperature", 0.9)),
        top_k=generation.get("top_k", 50),
        # Logging
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
    """Instantiate the algorithm from config."""
    algo_cfg = cfg["algorithm"]
    name = algo_cfg["name"]

    if name not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {name!r}. Available: {list(ALGORITHMS)}"
        )

    module_path, class_name = ALGORITHMS[name]
    cls = _import_class(module_path, class_name)

    # Pass all algorithm config except 'name' as kwargs
    kwargs = {k: v for k, v in algo_cfg.items() if k != "name"}
    return cls(**kwargs)


def build_dataset(cfg: dict, split: str = "train", max_samples: int = 0):
    """Instantiate a dataset from config."""
    dataset_name = cfg.get("reward", {}).get("dataset", "gsm8k")

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. Available: {list(DATASETS)}"
        )

    module_path, class_name = DATASETS[dataset_name]
    cls = _import_class(module_path, class_name)
    return cls(split=split, max_samples=max_samples)


def build_reward(cfg: dict):
    """Instantiate the reward function from config."""
    reward_type = cfg.get("reward", {}).get("type", "rule")

    if reward_type not in REWARDS:
        raise ValueError(
            f"Unknown reward type: {reward_type!r}. Available: {list(REWARDS)}"
        )

    module_path, class_name = REWARDS[reward_type]
    cls = _import_class(module_path, class_name)
    return cls()


def build_generator(cfg: dict, trainer_config: TrainerConfig):
    """Instantiate the vLLM generation engine."""
    from invokerl.engine.generator import VLLMGenerator

    model_cfg = cfg["model"]
    gen_cfg = cfg.get("generation", {})

    return VLLMGenerator(
        model_name_or_path=model_cfg["name_or_path"],
        gpu_memory_utilization=gen_cfg.get("gpu_memory_utilization", 0.5),
        enforce_eager=gen_cfg.get("enforce_eager", False),
        dtype=model_cfg.get("dtype", "bfloat16"),
        max_model_len=gen_cfg.get("max_model_len", None),
    )


def build_policy(cfg: dict, device: str = "cuda"):
    """Instantiate the policy model for training."""
    from invokerl.engine.policy import PolicyModel

    model_cfg = cfg["model"]
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(model_cfg.get("dtype", "bfloat16"), torch.bfloat16)

    return PolicyModel(
        model_name_or_path=model_cfg["name_or_path"],
        device=device,
        dtype=dtype,
        master_weights_fp32=model_cfg.get("master_weights_fp32", False),
    )


def build_ref_policy(cfg: dict, device: str = "cuda"):
    """Instantiate the frozen reference policy (optional).

    Returns None if algorithm doesn't need a reference model.
    """
    algo_name = cfg["algorithm"]["name"]

    # GRPO uses KL penalty against ref — need a ref model.
    # Some algorithms (SimPO) are reference-free.
    needs_ref = algo_name in {"grpo", "ppo", "dpo", "dapo"}

    if not needs_ref:
        return None

    from invokerl.engine.policy import PolicyModel

    model_cfg = cfg["model"]
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(model_cfg.get("dtype", "bfloat16"), torch.bfloat16)

    policy = PolicyModel(
        model_name_or_path=model_cfg["name_or_path"],
        device=device,
        dtype=dtype,
        master_weights_fp32=False,  # ref is frozen — no optimizer, bf16 is fine
    )
    # Freeze reference model — no gradients needed
    policy.model.eval()
    for param in policy.model.parameters():
        param.requires_grad = False

    return policy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    """Configure logging format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="invokerl — hackable RL post-training for LLMs",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., invokerl/configs/grpo_gsm8k.yaml)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint directory",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Run evaluation only (no training)",
    )
    parser.add_argument(
        "--eval-samples", type=int, default=None,
        help="Override number of eval samples",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override total_steps from config (useful for smoke tests)",
    )
    parser.add_argument(
        "--no-ref", action="store_true",
        help="Skip reference model (saves GPU memory, degrades to no KL penalty)",
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=None,
        help="Limit training dataset size (load fewer items into memory)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # Global random seed for reproducibility
    # NOTE: torch.cuda.manual_seed_all() is deferred until AFTER vLLM init,
    # because it triggers CUDA initialization which prevents vLLM from forking.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    cfg = load_config(args.config)
    trainer_config = build_trainer_config(cfg)

    if args.eval_samples is not None:
        trainer_config.eval_samples = args.eval_samples
    if args.max_steps is not None:
        trainer_config.total_steps = args.max_steps

    logger.info("Config: %s", args.config)
    logger.info("Algorithm: %s", trainer_config.algorithm)
    logger.info("Model: %s", trainer_config.model_name_or_path)
    logger.info("Steps: %d, LR: %.2e, Accum: %d, Group: %d",
                trainer_config.total_steps, trainer_config.lr,
                trainer_config.accumulation_steps, trainer_config.group_size)

    # Build components
    logger.info("Loading algorithm...")
    algorithm = build_algorithm(cfg)

    logger.info("Loading training dataset...")
    train_dataset = build_dataset(
        cfg, split="train", max_samples=args.max_train_samples or 0,
    )
    logger.info("Training dataset: %d items", len(train_dataset))

    logger.info("Loading eval dataset...")
    eval_dataset = build_dataset(cfg, split="test")
    logger.info("Eval dataset: %d items", len(eval_dataset))

    logger.info("Loading reward function...")
    reward_fn = build_reward(cfg)

    logger.info("Initializing vLLM generator...")
    generator = build_generator(cfg, trainer_config)

    # Now safe to initialize CUDA (vLLM has already forked its engine core)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Loading policy model...")
    policy = build_policy(cfg)

    logger.info("Loading reference model...")
    if args.no_ref:
        ref_policy = None
        logger.info("Reference model skipped (--no-ref)")
    else:
        ref_policy = build_ref_policy(cfg)
    if ref_policy is None and not args.no_ref:
        logger.info("No reference model needed for %s", trainer_config.algorithm)

    # Build trainer
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
        logger.info("Running evaluation only...")
        if args.resume:
            trainer.load_checkpoint(args.resume)
        metrics = trainer.evaluate()
        if metrics:
            logger.info(
                "Eval: %d/%d = %.1f%%",
                metrics["eval_correct"],
                metrics["eval_total"],
                metrics["eval_accuracy"] * 100,
            )
        else:
            logger.warning("No eval dataset available")
        return

    # Train
    history = trainer.train(resume_from=args.resume)
    logger.info("Training complete. %d steps recorded.", len(history))


if __name__ == "__main__":
    main()
