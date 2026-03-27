"""Plot GPU memory allocation breakdown: before vs after shared weights.

Creates a stacked bar chart showing exactly where GPU memory goes at each
phase of initialization and training. Compares the baseline (separate weights)
with shared weights mode.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# --- Baseline memory profile (from onyx's server measurements) ---
# Each entry: (phase_label, {component: allocated_GB})
# Components accumulate — each phase adds on top of the previous.

COMPONENTS = [
    "vLLM Engine + KV Cache",
    "Policy Weights",
    "Optimizer States (AdamW m+v)",
    "Ref Model (frozen)",
    "Activations + Gradients (peak)",
]

# Colorblind-safe palette (Wong 2011)
COLORS = {
    "vLLM Engine + KV Cache":          "#0072B2",
    "Policy Weights":                   "#E69F00",
    "Optimizer States (AdamW m+v)":     "#CC79A7",
    "Ref Model (frozen)":               "#D55E00",
    "Activations + Gradients (peak)":   "#009E73",
}

HATCHES = {
    "vLLM Engine + KV Cache":          "//",
    "Policy Weights":                   "xx",
    "Optimizer States (AdamW m+v)":     "--",
    "Ref Model (frozen)":               "\\\\",
    "Activations + Gradients (peak)":   "..",
}

# Memory values in GB (measured on RTX 5090, Qwen3-0.6B bf16)
BASELINE = {
    "vLLM Engine + KV Cache":          9.347,
    "Policy Weights":                   1.192,
    "Optimizer States (AdamW m+v)":     2.384,
    "Ref Model (frozen)":               1.192,
    "Activations + Gradients (peak)":   1.193,  # 15.309 - 14.116
}

SHARED = {
    "vLLM Engine + KV Cache":          9.347,
    "Policy Weights":                   0.0,    # shared with vLLM — no extra alloc
    "Optimizer States (AdamW m+v)":     2.384,
    "Ref Model (frozen)":               1.192,
    "Activations + Gradients (peak)":   1.193,
}


def plot_memory_comparison(output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for ax, (title, data, subtitle) in zip(axes, [
        ("Before: Separate Weights", BASELINE, "Policy + vLLM each hold a full model copy"),
        ("After: Shared Weights", SHARED, "Policy params are views into vLLM's tensors"),
    ]):
        bottoms = np.zeros(1)
        total = 0.0

        for comp in COMPONENTS:
            val = data[comp]
            bar = ax.bar(
                [0], [val], bottom=bottoms,
                color=COLORS[comp], edgecolor="#333333", linewidth=0.8,
                width=0.5, label=comp,
                hatch=HATCHES[comp], alpha=0.85,
            )

            # Label each segment
            if val > 0.3:
                mid_y = bottoms[0] + val / 2
                ax.text(
                    0, mid_y, f"{val:.2f} GB",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.8),
                )
            elif val > 0.05:
                # Small segment — label outside
                mid_y = bottoms[0] + val / 2
                ax.annotate(
                    f"{val:.2f} GB", xy=(0.25, mid_y), fontsize=9,
                    ha="left", va="center",
                )

            bottoms += val
            total += val

        # Total label on top
        ax.text(
            0, total + 0.3, f"Peak: {total:.1f} GB",
            ha="center", va="bottom", fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFF0",
                      edgecolor="black", linewidth=1.5),
        )

        ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold")
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3, linestyle=":")

    axes[0].set_ylabel("GPU Memory Allocated (GB)", fontsize=13)

    # RTX 5090 total memory line
    for ax in axes:
        ax.axhline(y=32, color="red", linestyle="--", linewidth=1.5, alpha=0.6)
        ax.text(0.45, 32.3, "RTX 5090 (32 GB)", ha="right", fontsize=9,
                color="red", alpha=0.8, transform=ax.get_yaxis_transform())
        ax.set_ylim(0, 34)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, fontsize=11,
        framealpha=0.95, edgecolor="black", bbox_to_anchor=(0.5, -0.02),
    )

    # Savings annotation
    baseline_total = sum(BASELINE.values())
    shared_total = sum(SHARED.values())
    savings = baseline_total - shared_total

    fig.text(
        0.5, 0.92,
        f"GPU Memory Profile — RTX 5090, Qwen3-0.6B bf16\n"
        f"Shared weights save {savings:.1f} GB ({savings/baseline_total*100:.0f}% reduction)",
        ha="center", fontsize=15, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.88])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")


def main():
    output_path = Path(__file__).parent.parent / "results" / "memory_profile.png"
    plot_memory_comparison(str(output_path))


if __name__ == "__main__":
    main()
