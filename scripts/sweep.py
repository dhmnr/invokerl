"""Parameter sweep runner -- run multiple configs sequentially on one GPU.

Usage:
    python scripts/sweep.py --configs invokerl/configs/sweep_*.yaml
    python scripts/sweep.py --configs invokerl/configs/sweep_*.yaml --max-steps 100
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


def run_config(config_path: str, max_steps: int | None, eval_samples: int, seed: int) -> dict:
    """Run a single training config and return results."""
    logger.info("=" * 60)
    logger.info("SWEEP: %s", Path(config_path).stem)
    logger.info("=" * 60)

    cmd = [sys.executable, "-u", "-m", "invokerl",
           "--config", config_path,
           "--eval-samples", str(eval_samples),
           "--seed", str(seed)]
    if max_steps is not None:
        cmd.extend(["--max-steps", str(max_steps)])

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=os.getcwd())
    elapsed = time.time() - t0

    # Load training history
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    history_path = os.path.join(
        cfg.get("logging", {}).get("output_dir", "./checkpoints"),
        "training_history.json",
    )
    history = json.loads(open(history_path).read()) if os.path.exists(history_path) else []

    run_result = {
        "config": config_path,
        "return_code": result.returncode,
        "elapsed_s": round(elapsed, 1),
        "steps": len(history),
    }

    if history:
        evals = [h for h in history if "eval_accuracy" in h]
        if evals:
            best = max(evals, key=lambda h: h["eval_accuracy"])
            run_result["best_eval"] = best["eval_accuracy"]
            run_result["best_eval_step"] = best["step"]
            run_result["final_eval"] = evals[-1]["eval_accuracy"]
        last = history[-1]
        run_result["final_reward"] = last.get("reward", 0)
        run_result["final_kl"] = last.get("kl", 0)

    return run_result


def print_summary(results: list[dict]) -> None:
    logger.info("\n" + "=" * 80)
    logger.info("SWEEP SUMMARY")
    logger.info("=" * 80)
    logger.info("%-28s %10s %11s %10s %8s %8s",
                "Config", "Best Eval", "Final Eval", "Reward", "KL", "Time")
    logger.info("-" * 80)
    for r in results:
        name = Path(r["config"]).stem[:28]
        best = f"{r['best_eval']:.1%}" if "best_eval" in r else "N/A"
        final = f"{r['final_eval']:.1%}" if "final_eval" in r else "N/A"
        reward = f"{r.get('final_reward', 0):.3f}"
        kl = f"{r.get('final_kl', 0):.4f}"
        ok = "+" if r["return_code"] == 0 else "X"
        logger.info("%s %-27s %10s %11s %10s %8s %7.0fs",
                    ok, name, best, final, reward, kl, r.get("elapsed_s", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="invokerl parameter sweep")
    parser.add_argument("--configs", nargs="+", required=True, help="Config YAML files (supports globs)")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total_steps")
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./sweep_results.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)])

    config_paths = []
    for pattern in args.configs:
        config_paths.extend(sorted(glob(pattern)))

    if not config_paths:
        logger.error("No config files found!")
        sys.exit(1)

    logger.info("Sweep: %d configs", len(config_paths))

    results = []
    for i, path in enumerate(config_paths, 1):
        logger.info("[%d/%d] %s", i, len(config_paths), path)
        try:
            results.append(run_config(path, args.max_steps, args.eval_samples, args.seed))
        except Exception as e:
            logger.error("Failed: %s: %s", path, e)
            results.append({"config": path, "return_code": -1, "error": str(e)})

    print_summary(results)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
