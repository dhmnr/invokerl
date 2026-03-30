"""Plot hyperparameter sweep results.

Usage:
    python -m invokerl.plot_sweeps --dirs checkpoints/sweep_lr5e6 checkpoints/sweep_lr1e5 ...
    python -m invokerl.plot_sweeps --base-dir checkpoints/  # auto-discover all sweep dirs

Reads training_history.json from each checkpoint dir and plots accuracy vs step
across all configs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_history(ckpt_dir: str) -> list[dict] | None:
    """Load training history from a checkpoint directory."""
    path = os.path.join(ckpt_dir, "training_history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_eval_points(history: list[dict]) -> tuple[list[int], list[float]]:
    """Extract (step, eval_accuracy) points from training history."""
    steps = []
    accs = []
    for entry in history:
        if "eval_accuracy" in entry:
            steps.append(entry["step"])
            accs.append(entry["eval_accuracy"] * 100)
    return steps, accs


def extract_metric(
    history: list[dict], metric: str, log_every: int = 10,
) -> tuple[list[int], list[float]]:
    """Extract a training metric over time."""
    steps = []
    vals = []
    for entry in history:
        if metric in entry:
            steps.append(entry["step"])
            vals.append(entry[metric])
    return steps, vals


def plot_accuracy(runs: dict[str, list[dict]], output: str = "sweep_accuracy.png"):
    """Plot eval accuracy vs step for all runs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, history in sorted(runs.items()):
        steps, accs = extract_eval_points(history)
        if steps:
            ax.plot(steps, accs, "o-", label=name, markersize=5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("GSM8K Eval Accuracy (%)")
    ax.set_title("Hyperparameter Sweep — Eval Accuracy")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(30, 70)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved: {output}")


def plot_metrics(
    runs: dict[str, list[dict]],
    output: str = "sweep_metrics.png",
):
    """Plot key training metrics (loss, reward, KL, gnorm) for all runs."""
    metrics = ["reward", "kl", "loss", "grad_norm"]
    titles = ["Mean Reward", "KL Divergence", "Loss", "Gradient Norm"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, metric, title in zip(axes.flat, metrics, titles):
        for name, history in sorted(runs.items()):
            steps, vals = extract_metric(history, metric)
            if steps:
                ax.plot(steps, vals, label=name, alpha=0.7, linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Hyperparameter Sweep — Training Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved: {output}")


def plot_summary_table(runs: dict[str, list[dict]], output: str = "sweep_summary.png"):
    """Create a summary table comparing final metrics across runs."""
    rows = []
    for name, history in sorted(runs.items()):
        steps, accs = extract_eval_points(history)
        if not accs:
            continue

        # Get baseline (first eval) and final
        baseline = accs[0] if accs else 0
        final = accs[-1] if accs else 0
        best = max(accs) if accs else 0
        best_step = steps[accs.index(best)] if accs else 0

        # Get final KL
        _, kls = extract_metric(history, "kl")
        final_kl = kls[-1] if kls else 0

        rows.append({
            "Config": name,
            "Baseline": f"{baseline:.1f}%",
            "Final": f"{final:.1f}%",
            "Best": f"{best:.1f}% (step {best_step})",
            "Delta": f"{final - baseline:+.1f}pp",
            "KL": f"{final_kl:.4f}",
        })

    if not rows:
        print("No eval data found")
        return

    # Print table
    print("\nSweep Summary:")
    print("-" * 90)
    header = f"{'Config':<30} {'Baseline':>10} {'Final':>10} {'Best':>20} {'Delta':>8} {'KL':>8}"
    print(header)
    print("-" * 90)
    for row in rows:
        print(
            f"{row['Config']:<30} {row['Baseline']:>10} {row['Final']:>10} "
            f"{row['Best']:>20} {row['Delta']:>8} {row['KL']:>8}"
        )
    print("-" * 90)


def main():
    parser = argparse.ArgumentParser(description="Plot sweep results")
    parser.add_argument(
        "--dirs", nargs="+", help="Checkpoint directories to compare",
    )
    parser.add_argument(
        "--base-dir", type=str, default=None,
        help="Auto-discover all subdirectories containing training_history.json",
    )
    parser.add_argument(
        "--output-prefix", type=str, default="sweep",
        help="Output filename prefix",
    )
    args = parser.parse_args()

    # Discover runs
    dirs = args.dirs or []
    if args.base_dir:
        base = Path(args.base_dir)
        for d in sorted(base.iterdir()):
            if d.is_dir() and (d / "training_history.json").exists():
                dirs.append(str(d))

    if not dirs:
        print("No checkpoint directories found. Use --dirs or --base-dir.")
        return

    # Load all runs
    runs = {}
    for d in dirs:
        name = os.path.basename(d)
        history = load_history(d)
        if history:
            runs[name] = history
            print(f"Loaded: {name} ({len(history)} steps)")
        else:
            print(f"Skipped: {d} (no training_history.json)")

    if not runs:
        print("No valid runs found.")
        return

    # Generate plots
    plot_accuracy(runs, f"{args.output_prefix}_accuracy.png")
    plot_metrics(runs, f"{args.output_prefix}_metrics.png")
    plot_summary_table(runs)


if __name__ == "__main__":
    main()
