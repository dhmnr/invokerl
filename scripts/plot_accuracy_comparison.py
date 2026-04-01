"""Plot GSM8K accuracy vs step: single-GPU vs disagg 2-GPU pipeline."""

import json
import matplotlib.pyplot as plt
import numpy as np


def main():
    # --- Single GPU data (from training_history_1000step.json) ---
    # lr=1e-5, B=4, G=4, accum=4 (effective batch=16), 200-sample eval every 50 steps
    with open("results/v1/training/training_history_1000step.json") as f:
        single_gpu_data = json.load(f)

    single_steps = []
    single_acc = []
    for entry in single_gpu_data:
        if "eval_accuracy" in entry:
            single_steps.append(entry["step"] + 1)  # 0-indexed steps → 1-indexed
            single_acc.append(entry["eval_accuracy"] * 100)

    # --- Disagg 2-GPU data (from onyx's 1000-step CUDA graphs run) ---
    # lr=1e-5, B=8, G=4 (B×G=32), 512tok, ref model, 50-sample eval every 100 steps
    disagg_steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    disagg_acc =   [44.0, 54.0, 52.0, 50.0, 52.0, 62.0, 58.0, 58.0, 60.0, 52.0, 48.0]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Single GPU
    ax.plot(single_steps, single_acc,
            color="#4285F4", linewidth=2, marker="o", markersize=4,
            label="Single GPU (B×G=16, 200-sample eval)", alpha=0.9)

    # Disagg
    ax.plot(disagg_steps, disagg_acc,
            color="#EA4335", linewidth=2, marker="s", markersize=6,
            label="Disagg 2-GPU (B×G=32, 50-sample eval)", alpha=0.9)

    # Highlight peaks
    single_peak_idx = np.argmax(single_acc)
    ax.annotate(f"Peak: {single_acc[single_peak_idx]:.1f}%",
                xy=(single_steps[single_peak_idx], single_acc[single_peak_idx]),
                xytext=(single_steps[single_peak_idx] - 150, single_acc[single_peak_idx] + 3),
                fontsize=9, color="#4285F4", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#4285F4", lw=1.2))

    disagg_peak_idx = np.argmax(disagg_acc)
    ax.annotate(f"Peak: {disagg_acc[disagg_peak_idx]:.1f}%",
                xy=(disagg_steps[disagg_peak_idx], disagg_acc[disagg_peak_idx]),
                xytext=(disagg_steps[disagg_peak_idx] + 50, disagg_acc[disagg_peak_idx] + 3),
                fontsize=9, color="#EA4335", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#EA4335", lw=1.2))

    # Reference line at 62.5% baseline
    ax.axhline(62.5, color="#999999", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(1010, 62.5, "62.5% baseline", fontsize=8, color="#999999",
            va="bottom", ha="left")

    # Error band for disagg (±7% from 50-sample eval noise)
    disagg_arr = np.array(disagg_acc)
    ax.fill_between(disagg_steps, disagg_arr - 7, disagg_arr + 7,
                     color="#EA4335", alpha=0.08, label="Disagg ±7% eval noise (50 samples)")

    # Formatting
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("GSM8K Accuracy (%)", fontsize=11)
    ax.set_title("GSM8K Training: Single GPU vs Disagg 2-GPU Pipeline\n"
                 "GRPO on Qwen3-0.6B — 2× RTX PRO 4500 Blackwell",
                 fontsize=13, fontweight="bold")

    ax.set_xlim(-20, 1050)
    ax.set_ylim(30, 75)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Add config info box
    config_text = (
        "Single GPU: lr=1e-5, B=4, G=4, accum=4, eval=200 samples\n"
        "Disagg 2-GPU: lr=1e-5, B=8, G=4, 512tok, eval=50 samples\n"
        "Wall time: Single ~180 min, Disagg 47 min (3.8× faster)"
    )
    ax.text(0.02, 0.02, config_text, transform=ax.transAxes,
            fontsize=7.5, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#CCCCCC"))

    plt.tight_layout()
    out = "results/gsm8k_accuracy_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
