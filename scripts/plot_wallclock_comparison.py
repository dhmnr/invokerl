"""Plot GSM8K accuracy vs wall clock time: single-GPU vs disagg 2-GPU."""

import json
import matplotlib.pyplot as plt
import numpy as np


def main():
    # --- Single GPU data ---
    # Cumulative wall time from per-step timing in training history
    with open("results/v1/training/training_history_1000step.json") as f:
        single_gpu_data = json.load(f)

    single_wall_min = []
    single_acc = []
    cum_time = 0.0
    for entry in single_gpu_data:
        cum_time += entry["time"]
        if "eval_accuracy" in entry:
            single_wall_min.append(cum_time / 60)
            single_acc.append(entry["eval_accuracy"] * 100)

    # --- Disagg 2-GPU data ---
    # 47 min total for 1000 steps = 2.82s/step
    # Eval every 100 steps
    disagg_step_time = 47.0 / 1000  # min per step
    disagg_steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    disagg_acc = [44.0, 54.0, 52.0, 50.0, 52.0, 62.0, 58.0, 58.0, 60.0, 52.0, 48.0]
    disagg_wall_min = [s * disagg_step_time for s in disagg_steps]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Single GPU
    ax.plot(single_wall_min, single_acc,
            color="#4285F4", linewidth=2, marker="o", markersize=4,
            label="Single GPU (B×G=16, ~8s/step)", alpha=0.9)

    # Disagg
    ax.plot(disagg_wall_min, disagg_acc,
            color="#EA4335", linewidth=2, marker="s", markersize=6,
            label="Disagg 2-GPU (B×G=32, ~2.8s/step)", alpha=0.9)

    # Highlight where disagg hits 62%
    disagg_peak_idx = np.argmax(disagg_acc)
    ax.annotate(f"62% at {disagg_wall_min[disagg_peak_idx]:.0f} min",
                xy=(disagg_wall_min[disagg_peak_idx], disagg_acc[disagg_peak_idx]),
                xytext=(disagg_wall_min[disagg_peak_idx] + 8, disagg_acc[disagg_peak_idx] + 4),
                fontsize=10, color="#EA4335", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#EA4335", lw=1.5))

    # Highlight where single GPU hits 62.5%
    single_peak_idx = np.argmax(single_acc)
    ax.annotate(f"62.5% at {single_wall_min[single_peak_idx]:.0f} min",
                xy=(single_wall_min[single_peak_idx], single_acc[single_peak_idx]),
                xytext=(single_wall_min[single_peak_idx] - 35, single_acc[single_peak_idx] + 3),
                fontsize=10, color="#4285F4", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#4285F4", lw=1.5))

    # Vertical lines at peak times
    ax.axvline(disagg_wall_min[disagg_peak_idx], color="#EA4335", linestyle=":",
               linewidth=1, alpha=0.4)
    ax.axvline(single_wall_min[single_peak_idx], color="#4285F4", linestyle=":",
               linewidth=1, alpha=0.4)

    # Speedup annotation
    speedup = single_wall_min[single_peak_idx] / disagg_wall_min[disagg_peak_idx]
    mid_x = (disagg_wall_min[disagg_peak_idx] + single_wall_min[single_peak_idx]) / 2
    ax.annotate("", xy=(disagg_wall_min[disagg_peak_idx], 38),
                xytext=(single_wall_min[single_peak_idx], 38),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5))
    ax.text(mid_x, 36, f"{speedup:.1f}× faster", ha="center", fontsize=10,
            fontweight="bold", color="#333333")

    # Error band for disagg
    disagg_arr = np.array(disagg_acc)
    ax.fill_between(disagg_wall_min, disagg_arr - 7, disagg_arr + 7,
                     color="#EA4335", alpha=0.08)

    # Formatting
    ax.set_xlabel("Wall Clock Time (minutes)", fontsize=12)
    ax.set_ylabel("GSM8K Accuracy (%)", fontsize=12)
    ax.set_title("GSM8K Training: Accuracy vs Wall Clock Time\n"
                 "GRPO on Qwen3-0.6B — 2× RTX PRO 4500 Blackwell",
                 fontsize=13, fontweight="bold")

    ax.set_xlim(-2, 145)
    ax.set_ylim(30, 75)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    # Config box
    config_text = (
        "Single GPU: lr=1e-5, B=4, G=4, accum=4, 1000 steps, ~134 min\n"
        "Disagg 2-GPU: lr=1e-5, B=8, G=4, 512tok, 1000 steps, 47 min"
    )
    ax.text(0.02, 0.02, config_text, transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#CCCCCC"))

    plt.tight_layout()
    out = "results/gsm8k_wallclock_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
