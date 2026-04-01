"""Plot GSM8K accuracy at 512tok: all configurations compared.

Two panels:
  Left:  Accuracy vs training step
  Right: Accuracy vs wall clock time

Configs:
  1. Single GPU (constant lr, B×G=16, ~8s/step)
  2. Disagg B×G=32 constant lr (~2.8s/step) — collapses after step 500
  3. Disagg B×G=32 cosine lr_end=1e-6 (~2.8s/step) — stable 58-59%
  4. Disagg B×G=56 cosine lr_end=1e-6 (~4.0s/step) — 100-step test
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def main():
    # ---------------------------------------------------------------
    # 1. Single GPU — constant lr, B×G=16, eval every 50 steps (200 samples)
    # ---------------------------------------------------------------
    with open("results/v1/training/training_history_1000step.json") as f:
        single_gpu_data = json.load(f)

    sg_steps, sg_acc, sg_wall = [], [], []
    cum_time = 0.0
    for entry in single_gpu_data:
        cum_time += entry["time"]
        if "eval_accuracy" in entry:
            sg_steps.append(entry["step"] + 1)
            sg_acc.append(entry["eval_accuracy"] * 100)
            sg_wall.append(cum_time / 60)

    # ---------------------------------------------------------------
    # 2. Disagg B×G=32, constant lr — collapses (50-sample eval)
    # From the original disagg run logs
    # ---------------------------------------------------------------
    d32c_steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    d32c_acc   = [44.0, 54.0, 52.0, 50.0, 52.0, 62.0, 58.0, 58.0, 60.0, 52.0, 48.0]
    d32c_sps = 2.8  # seconds per step
    d32c_wall = [s * d32c_sps / 60 for s in d32c_steps]

    # ---------------------------------------------------------------
    # 3. Disagg B×G=32, cosine lr_end=1e-6 — stable (200-sample eval)
    # From the cosine validation run
    # ---------------------------------------------------------------
    d32cos_steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    d32cos_acc   = [54.0, 52.0, 52.5, 55.5, 58.5, 58.0, 58.0, 59.0, 58.5, 59.0]
    d32cos_sps = 2.8
    d32cos_wall = [s * d32cos_sps / 60 for s in d32cos_steps]

    # ---------------------------------------------------------------
    # 4. Disagg B×G=56, cosine lr_end=1e-6 — 100-step test (50-sample eval)
    # From the chunked lm_head test run
    # ---------------------------------------------------------------
    d56_steps = [50, 100]
    d56_acc   = [50.0, 42.0]
    d56_sps = 4.0
    d56_wall = [s * d56_sps / 60 for s in d56_steps]

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    colors = {
        "single": "#4285F4",
        "d32c":   "#EA4335",
        "d32cos": "#34A853",
        "d56":    "#FBBC04",
    }

    # --- Left panel: Accuracy vs Step ---
    ax1.plot(sg_steps, sg_acc, color=colors["single"], linewidth=2,
             marker="o", markersize=3.5, label="Single GPU (B×G=16, constant lr, ~8s/step)",
             alpha=0.85)

    ax1.plot(d32c_steps, d32c_acc, color=colors["d32c"], linewidth=2,
             marker="s", markersize=5, label="Disagg B×G=32 constant lr (~2.8s/step)",
             alpha=0.85)

    ax1.plot(d32cos_steps, d32cos_acc, color=colors["d32cos"], linewidth=2.5,
             marker="^", markersize=6, label="Disagg B×G=32 cosine lr_end=1e-6 (~2.8s/step)",
             alpha=0.9)

    ax1.plot(d56_steps, d56_acc, color=colors["d56"], linewidth=2.5,
             marker="D", markersize=7, label="Disagg B×G=56 cosine lr_end=1e-6 (~4.0s/step)",
             alpha=0.9, linestyle="--")

    # Annotate key events
    ax1.annotate("Collapse\n(overtraining)", xy=(900, 52), xytext=(750, 42),
                 fontsize=8, color=colors["d32c"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=colors["d32c"], lw=1.2))

    ax1.annotate("Stable 59%", xy=(1000, 59), xytext=(850, 65),
                 fontsize=9, color=colors["d32cos"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=colors["d32cos"], lw=1.2))

    ax1.annotate("62.5%", xy=(1000, 62.5), xytext=(1010, 62.5),
                 fontsize=8, color=colors["single"], fontweight="bold")

    # Confidence bands
    d32c_arr = np.array(d32c_acc)
    ax1.fill_between(d32c_steps, d32c_arr - 7, d32c_arr + 7,
                     color=colors["d32c"], alpha=0.06)

    d32cos_arr = np.array(d32cos_acc)
    ax1.fill_between(d32cos_steps, d32cos_arr - 3.5, d32cos_arr + 3.5,
                     color=colors["d32cos"], alpha=0.08)

    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("GSM8K Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy vs Training Step", fontsize=13, fontweight="bold")
    ax1.set_xlim(-20, 1080)
    ax1.set_ylim(30, 75)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    # --- Right panel: Accuracy vs Wall Clock ---
    ax2.plot(sg_wall, sg_acc, color=colors["single"], linewidth=2,
             marker="o", markersize=3.5, label="Single GPU (~134 min total)",
             alpha=0.85)

    ax2.plot(d32c_wall, d32c_acc, color=colors["d32c"], linewidth=2,
             marker="s", markersize=5, label="Disagg B×G=32 const lr (~47 min)",
             alpha=0.85)

    ax2.plot(d32cos_wall, d32cos_acc, color=colors["d32cos"], linewidth=2.5,
             marker="^", markersize=6, label="Disagg B×G=32 cosine (~47 min)",
             alpha=0.9)

    ax2.plot(d56_wall, d56_acc, color=colors["d56"], linewidth=2.5,
             marker="D", markersize=7, label="Disagg B×G=56 cosine (~7 min / 100 steps)",
             alpha=0.9, linestyle="--")

    # Speedup arrow between single GPU peak and disagg cosine reaching same accuracy
    sg_peak_wall = sg_wall[np.argmax(sg_acc)]
    d32cos_peak_wall = d32cos_wall[-1]  # final point

    ax2.annotate("", xy=(d32cos_peak_wall, 35),
                 xytext=(sg_peak_wall, 35),
                 arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5))
    speedup = sg_peak_wall / d32cos_peak_wall
    mid_x = (d32cos_peak_wall + sg_peak_wall) / 2
    ax2.text(mid_x, 33, f"{speedup:.1f}× faster", ha="center", fontsize=10,
             fontweight="bold", color="#333333")

    ax2.set_xlabel("Wall Clock Time (minutes)", fontsize=12)
    ax2.set_ylabel("GSM8K Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy vs Wall Clock Time", fontsize=13, fontweight="bold")
    ax2.set_xlim(-2, 145)
    ax2.set_ylim(30, 75)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    # ---------------------------------------------------------------
    # Throughput comparison table at bottom
    # ---------------------------------------------------------------
    table_text = (
        "Config                   B×G   s/step   seqs/sec   Peak Acc   Step 1000   Eval CI\n"
        "Single GPU (const lr)     16    ~8.0     2.0        62.5%      62.5%       ±3.5%\n"
        "Disagg B×G=32 (const)     32    ~2.8     11.4       62.0%      48.0%       ±7.0%\n"
        "Disagg B×G=32 (cosine)    32    ~2.8     11.4       59.0%      59.0%       ±3.5%\n"
        "Disagg B×G=56 (cosine)    56    ~4.0     14.0       50.0%*     —           ±7.0%\n"
        "                                                    * only 100 steps"
    )
    fig.text(0.5, -0.02, table_text, ha="center", fontsize=8, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#F5F5F5", alpha=0.9,
                       edgecolor="#CCCCCC"))

    fig.suptitle("GSM8K Training at 512 Tokens — All Configurations\n"
                 "GRPO on Qwen3-0.6B  |  2× RTX PRO 4500 Blackwell (32 GB)",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    out = "results/gsm8k_all_configs_512tok.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
