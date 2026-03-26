"""Parameter sweep runner for invokerl.

Runs multiple training configs sequentially on a single GPU and collects
results into a summary table for comparison.

Usage:
    python -m invokerl.sweep --configs invokerl/configs/sweep_*.yaml
    python -m invokerl.sweep --configs invokerl/configs/grpo_gsm8k.yaml invokerl/configs/sweep_lr2e6.yaml
    python -m invokerl.sweep --configs invokerl/configs/sweep_*.yaml --max-steps 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

import yaml

logger = logging.getLogger("invokerl.sweep")


def parse_sweep_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="invokerl parameter sweep")
    parser.add_argument(
        "--configs", nargs="+", required=True,
        help="Config YAML files to sweep (supports shell globs)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override total_steps for all runs (useful for quick sweeps)",
    )
    parser.add_argument(
        "--eval-samples", type=int, default=50,
        help="Number of eval samples per evaluation",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (same for all runs)",
    )
    parser.add_argument(
        "--output", type=str, default="./sweep_results.json",
        help="Path to save sweep summary",
    )
    return parser.parse_args()


def extract_sweep_label(config_path: str) -> str:
    """Extract a human-readable label from a config path."""
    name = Path(config_path).stem
    # Load config to get key differing params
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    algo = cfg.get("algorithm", {})
    train = cfg.get("training", {})

    parts = [name]
    parts.append(f"lr={train.get('lr', '?')}")
    parts.append(f"beta={algo.get('beta', '?')}")
    parts.append(f"G={train.get('group_size', '?')}")
    return " | ".join(parts)


def run_single_config(
    config_path: str,
    max_steps: int | None = None,
    eval_samples: int = 50,
    seed: int = 42,
) -> dict:
    """Run a single training config and return results.

    Returns:
        Dict with config info and training results.
    """
    label = extract_sweep_label(config_path)
    logger.info(f"\n{'='*60}")
    logger.info(f"SWEEP: {label}")
    logger.info(f"Config: {config_path}")
    logger.info(f"{'='*60}\n")

    cmd = [
        sys.executable, "-u", "-m", "invokerl",
        "--config", config_path,
        "--eval-samples", str(eval_samples),
        "--seed", str(seed),
    ]
    if max_steps is not None:
        cmd.extend(["--max-steps", str(max_steps)])

    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=False,  # stream output to terminal
        text=True,
        cwd=os.getcwd(),
    )
    elapsed = time.time() - t0

    # Load the training history if it exists
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg.get("logging", {}).get("output_dir", "./checkpoints")
    history_path = os.path.join(output_dir, "training_history.json")

    history = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

    # Extract key results
    run_result = {
        "config": config_path,
        "label": label,
        "return_code": result.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "total_steps": len(history),
    }

    if history:
        # Find eval results (steps with eval_accuracy)
        evals = [h for h in history if "eval_accuracy" in h]
        if evals:
            best_eval = max(evals, key=lambda h: h["eval_accuracy"])
            run_result["best_eval_accuracy"] = best_eval["eval_accuracy"]
            run_result["best_eval_step"] = best_eval["step"]
            run_result["final_eval_accuracy"] = evals[-1]["eval_accuracy"]

        # Last step metrics
        last = history[-1]
        run_result["final_reward"] = last.get("reward", 0)
        run_result["final_kl"] = last.get("kl", 0)
        run_result["final_loss"] = last.get("loss", 0)

        # Reward trajectory
        rewards = [h.get("reward", 0) for h in history]
        run_result["reward_first_10"] = round(sum(rewards[:10]) / max(len(rewards[:10]), 1), 3)
        run_result["reward_last_10"] = round(sum(rewards[-10:]) / max(len(rewards[-10:]), 1), 3)

    return run_result


def print_summary(results: list[dict]) -> None:
    """Print a comparison table of sweep results."""
    logger.info(f"\n{'='*80}")
    logger.info("SWEEP SUMMARY")
    logger.info(f"{'='*80}")

    # Header
    logger.info(
        f"{'Config':<30} {'Best Eval':>10} {'Final Eval':>11} "
        f"{'Reward(end)':>12} {'KL':>8} {'Time':>8}"
    )
    logger.info("-" * 80)

    for r in results:
        name = Path(r["config"]).stem[:28]
        best = f"{r.get('best_eval_accuracy', 0):.1%}" if "best_eval_accuracy" in r else "N/A"
        final = f"{r.get('final_eval_accuracy', 0):.1%}" if "final_eval_accuracy" in r else "N/A"
        reward = f"{r.get('final_reward', 0):.3f}"
        kl = f"{r.get('final_kl', 0):.4f}"
        time_s = f"{r.get('elapsed_seconds', 0):.0f}s"
        status = "✓" if r["return_code"] == 0 else "✗"

        logger.info(
            f"{status} {name:<28} {best:>10} {final:>11} "
            f"{reward:>12} {kl:>8} {time_s:>8}"
        )

    logger.info(f"{'='*80}")


def main() -> None:
    args = parse_sweep_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Expand globs
    config_paths: list[str] = []
    for pattern in args.configs:
        expanded = sorted(glob(pattern))
        if not expanded:
            logger.warning(f"No files matched: {pattern}")
        config_paths.extend(expanded)

    if not config_paths:
        logger.error("No config files found!")
        sys.exit(1)

    logger.info(f"Sweep: {len(config_paths)} configs")
    for p in config_paths:
        logger.info(f"  - {p}")

    results: list[dict] = []
    for i, config_path in enumerate(config_paths, 1):
        logger.info(f"\n[{i}/{len(config_paths)}] Running {config_path}...")
        try:
            result = run_single_config(
                config_path,
                max_steps=args.max_steps,
                eval_samples=args.eval_samples,
                seed=args.seed,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed: {config_path}: {e}")
            results.append({
                "config": config_path,
                "label": config_path,
                "return_code": -1,
                "error": str(e),
            })

    # Print summary table
    print_summary(results)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
