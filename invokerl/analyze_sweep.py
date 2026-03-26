"""Quick analysis of screening sweep results.

Usage:
    # Run on server after sweep completes:
    python -m invokerl.analyze_sweep --log /root/tilerl/sweep_v3.log

    # Or parse from individual logs:
    python -m invokerl.analyze_sweep --logs screen_lr1e5.log screen_lr2e5.log ...
"""
from __future__ import annotations

import argparse
import re
import sys


def parse_sweep_log(log_path: str) -> dict[str, dict]:
    """Parse sweep_v3.log and extract results per config.

    Returns dict mapping config_name → {
        steps: list of (step, metrics_dict),
        eval: (step, accuracy),
        status: "done" | "running" | "failed",
    }
    """
    results: dict[str, dict] = {}
    current_config = None

    with open(log_path) as f:
        for line in f:
            line = line.strip()

            # Detect config start
            m = re.match(r"=== (screen_\w+) START", line)
            if m:
                current_config = m.group(1)
                results[current_config] = {
                    "steps": [],
                    "eval": None,
                    "status": "running",
                }
                continue

            # Detect config done
            m = re.match(r"=== (screen_\w+) DONE.*exit=(\d+)", line)
            if m:
                name = m.group(1)
                exit_code = int(m.group(2))
                if name in results:
                    results[name]["status"] = "done" if exit_code == 0 else "failed"
                continue

            # Parse step metrics
            if current_config and "[step" in line:
                m = re.search(
                    r"\[step\s+(\d+)\].*loss=([-\d.]+).*reward=([-\d.]+).*kl=([-\d.]+).*gnorm=([-\d.]+).*lr=([-\d.e+]+).*time=([-\d.]+)s",
                    line,
                )
                if m and current_config in results:
                    results[current_config]["steps"].append({
                        "step": int(m.group(1)),
                        "loss": float(m.group(2)),
                        "reward": float(m.group(3)),
                        "kl": float(m.group(4)),
                        "gnorm": float(m.group(5)),
                        "lr": float(m.group(6)),
                        "time": float(m.group(7)),
                    })

            # Parse eval
            if current_config:
                m = re.search(
                    r"Eval at step (\d+):\s*(\d+)/(\d+)\s*=\s*([\d.]+)%",
                    line,
                )
                if m and current_config in results:
                    results[current_config]["eval"] = {
                        "step": int(m.group(1)),
                        "correct": int(m.group(2)),
                        "total": int(m.group(3)),
                        "accuracy": float(m.group(4)),
                    }

    return results


def print_summary(results: dict[str, dict], baseline_acc: float = 51.5):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("  HYPERPARAMETER SWEEP RESULTS")
    print("=" * 80)

    # Header
    print(f"\n  {'Config':<20s} {'Eval@50':>10s} {'Delta':>8s} "
          f"{'Final KL':>10s} {'Status':>10s}")
    print(f"  {'-' * 62}")

    # Baseline
    print(f"  {'(baseline lr=5e-6)':<20s} {baseline_acc:>9.1f}% {'---':>8s} "
          f"{'~0.002':>10s} {'done':>10s}")

    # Sort by eval accuracy (descending), missing evals last
    sorted_configs = sorted(
        results.items(),
        key=lambda x: (x[1].get("eval", {}).get("accuracy", -1) if x[1].get("eval") else -1),
        reverse=True,
    )

    best_config = None
    best_acc = baseline_acc

    for name, data in sorted_configs:
        eval_data = data.get("eval")
        status = data["status"]

        if eval_data:
            acc = eval_data["accuracy"]
            delta = acc - baseline_acc
            delta_str = f"{delta:+.1f}pp"

            # Get final KL from last step
            steps = data.get("steps", [])
            final_kl = steps[-1]["kl"] if steps else 0
            kl_str = f"{final_kl:.4f}"

            if acc > best_acc:
                best_acc = acc
                best_config = name

            marker = " <-- BEST" if acc >= best_acc and acc > baseline_acc else ""
            print(f"  {name:<20s} {acc:>9.1f}% {delta_str:>8s} "
                  f"{kl_str:>10s} {status:>10s}{marker}")
        else:
            print(f"  {name:<20s} {'---':>10s} {'---':>8s} "
                  f"{'---':>10s} {status:>10s}")

    print(f"  {'-' * 62}")

    if best_config:
        print(f"\n  Winner: {best_config} ({best_acc:.1f}%)")
    else:
        print(f"\n  No config beat baseline ({baseline_acc:.1f}%)")

    # Detailed training curves
    print(f"\n  Training Curves (reward @ each logged step):")
    print(f"  {'Config':<20s}", end="")
    for step in [0, 10, 20, 30, 40]:
        print(f"  {'s' + str(step):>6s}", end="")
    print()

    for name, data in sorted_configs:
        steps = {s["step"]: s for s in data.get("steps", [])}
        print(f"  {name:<20s}", end="")
        for step in [0, 10, 20, 30, 40]:
            if step in steps:
                print(f"  {steps[step]['reward']:>6.3f}", end="")
            else:
                print(f"  {'---':>6s}", end="")
        print()

    return best_config, best_acc


def main():
    parser = argparse.ArgumentParser(description="Analyze sweep results")
    parser.add_argument("--log", type=str, help="Path to sweep_v3.log")
    args = parser.parse_args()

    if not args.log:
        parser.error("--log required")

    results = parse_sweep_log(args.log)
    if not results:
        print("No results found in log file")
        sys.exit(1)

    best_config, best_acc = print_summary(results)
    print(f"\n  Configs parsed: {len(results)}")
    print(f"  Done: {sum(1 for d in results.values() if d['status'] == 'done')}")
    print(f"  Running: {sum(1 for d in results.values() if d['status'] == 'running')}")
    print(f"  Failed: {sum(1 for d in results.values() if d['status'] == 'failed')}")


if __name__ == "__main__":
    main()
