"""Entry point for invokerl training.

Usage:
    python -m invokerl.train --config invokerl/configs/grpo_gsm8k.yaml
    python -m invokerl.train --config invokerl/configs/grpo_gsm8k.yaml --resume ./checkpoints/step_100
"""

from __future__ import annotations

import argparse
import importlib
import logging
import random
import sys

import torch
import yaml

from invokerl.engine.trainer import Trainer, TrainerConfig

logger = logging.getLogger("invokerl")

# Shared dtype mapping used by policy, ref policy, and generator.
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Registry: config name -> (module_path, class_name)
ALGORITHMS = {"grpo": ("invokerl.algorithms.grpo", "GRPO")}
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


def build_generator(cfg: dict, trainer_config: TrainerConfig):
    """Instantiate vLLM generation engine."""
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


def build_policy(cfg: dict, device: str = "cuda", frozen: bool = False):
    """Instantiate a policy model.

    Args:
        frozen: If True, returns an eval-mode model with no gradients
                (used as the reference policy).
    """
    from invokerl.engine.policy import PolicyModel

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
    p.add_argument("--max-train-samples", type=int, default=None, help="Limit training dataset size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--verbose", action="store_true", help="Debug logging")
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

    # Build components
    algorithm = build_algorithm(cfg)
    train_dataset = build_dataset(cfg, split="train", max_samples=args.max_train_samples or 0)
    eval_dataset = build_dataset(cfg, split="test")
    reward_fn = build_reward(cfg)

    logger.info("Datasets: %d train, %d eval", len(train_dataset), len(eval_dataset))

    logger.info("Initializing vLLM generator...")
    generator = build_generator(cfg, trainer_config)

    # Now safe to seed CUDA (vLLM has already forked its engine)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Loading policy model...")
    policy = build_policy(cfg)

    if args.no_ref:
        ref_policy = None
        logger.info("Reference model skipped (--no-ref)")
    else:
        ref_policy = build_ref_policy(cfg)
        if ref_policy is None:
            logger.info("Algorithm %s doesn't need a reference model", trainer_config.algorithm)

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
        metrics = trainer.evaluate()
        if metrics:
            logger.info("Eval: %d/%d = %.1f%%",
                        metrics["eval_correct"], metrics["eval_total"],
                        metrics["eval_accuracy"] * 100)
        return

    history = trainer.train(resume_from=args.resume)
    logger.info("Training complete. %d steps recorded.", len(history))


if __name__ == "__main__":
    main()
