"""Plot GPU memory allocation across training phases: before vs after shared weights.

Uses real torch.cuda.memory_allocated() measurements from RTX 5090 server.
Two-panel plot: left shows phase-by-phase memory timeline, right shows
component breakdown as stacked bars.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# --- Real measurements from RTX 5090 (torch.cuda.memory_allocated) ---

PHASES = ["After\nvLLM init", "After\npolicy init", "After\nsharing", "After\ngeneration",
          "After\nforward", "After\nbackward", "After\noptimizer", "After\nref model"]

# Allocated GB at each phase (measured)
BASELINE_ALLOC = [9.347, 10.540, 10.540, 10.540, 10.940, 11.781, 12.924, 14.116]
SHARED_ALLOC   = [9.347, 10.540,  9.347,  9.347,  9.747, 10.588, 11.732, 12.924]

# Peak GB at each phase (torch.cuda.max_memory_allocated snapshots)
BASELINE_PEAK = 15.309
SHARED_PEAK   = 15.362  # nearly same due to optimizer moment buffer init spike

# Component breakdown (for stacked bar panel)
COMPONENTS = [
    ("vLLM Engine\n+ KV Cache",          9.347, 9.347, "#0072B2", "//"),
    ("Policy Weights",                    1.192, 0.0,   "#E69F00", "xx"),
    ("Optimizer States\n(AdamW m+v)",     2.384, 2.384, "#CC79A7", "--"),
    ("Ref Model\n(frozen)",               1.192, 1.192, "#D55E00", "\\\\"),
    ("Activations +\nGradients (peak)",   1.193, 1.193, "#009E73", ".."),
]


def plot_memory_profile(output_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8),
                                    gridspec_kw={"width_ratios": [2.2, 1]})

    # --- Left panel: phase-by-phase memory timeline ---
    x = np.arange(len(PHASES))
    width = 0.35

    bars_base = ax1.bar(x - width/2, BASELINE_ALLOC, width,
                        color="#D55E00", alpha=0.75, edgecolor="#333333",
                        linewidth=0.8, hatch="\\\\", label="Separate weights")
    bars_shared = ax1.bar(x + width/2, SHARED_ALLOC, width,
                          color="#0072B2", alpha=0.75, edgecolor="#333333",
                          linewidth=0.8, hatch="//", label="Shared weights")

    # Label the delta on key phases
    for i in range(len(PHASES)):
        delta = SHARED_ALLOC[i] - BASELINE_ALLOC[i]
        if abs(delta) > 0.1:
            y_pos = max(BASELINE_ALLOC[i], SHARED_ALLOC[i]) + 0.3
            ax1.text(x[i], y_pos, f"{delta:+.1f} GB",
                     ha="center", va="bottom", fontsize=9, fontweight="bold",
                     color="#009E73" if delta < 0 else "#D55E00",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                               edgecolor="none", alpha=0.8))

    # Highlight the "after sharing" phase
    ax1.annotate("share_vllm_weights()\nfrees duplicate",
                 xy=(2 + width/2, SHARED_ALLOC[2] + 0.1),
                 xytext=(3.5, 7.5), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFF0",
                           edgecolor="black", alpha=0.9))

    # Peak memory lines
    ax1.axhline(y=BASELINE_PEAK, color="#D55E00", linestyle=":", linewidth=1.5, alpha=0.5)
    ax1.text(len(PHASES) - 0.5, BASELINE_PEAK + 0.2, f"Peak: {BASELINE_PEAK:.1f} GB",
             ha="right", fontsize=9, color="#D55E00", fontweight="bold")
    ax1.axhline(y=SHARED_PEAK, color="#0072B2", linestyle=":", linewidth=1.5, alpha=0.5)
    ax1.text(len(PHASES) - 0.5, SHARED_PEAK - 0.6, f"Peak: {SHARED_PEAK:.1f} GB",
             ha="right", fontsize=9, color="#0072B2", fontweight="bold")

    # RTX 5090 limit
    ax1.axhline(y=32, color="red", linestyle="--", linewidth=1.5, alpha=0.4)
    ax1.text(len(PHASES) - 0.5, 32.3, "RTX 5090 (32 GB)", ha="right",
             fontsize=9, color="red", alpha=0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(PHASES, fontsize=9)
    ax1.set_ylabel("GPU Memory Allocated (GB)", fontsize=12)
    ax1.set_title("Memory Across Training Phases\n(measured with torch.cuda.memory_allocated)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 34)
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(axis="y", alpha=0.3, linestyle=":")

    # --- Right panel: component breakdown (stacked bars) ---
    bar_x = [0, 1]
    bar_labels = ["Separate\nWeights", "Shared\nWeights"]

    bottoms_base = np.zeros(1)
    bottoms_shared = np.zeros(1)

    handles = []
    for name, val_base, val_shared, color, hatch in COMPONENTS:
        # Baseline bar
        ax2.bar([0], [val_base], bottom=bottoms_base, width=0.5,
                color=color, edgecolor="#333333", linewidth=0.8,
                hatch=hatch, alpha=0.85)
        # Shared bar
        ax2.bar([1], [val_shared], bottom=bottoms_shared, width=0.5,
                color=color, edgecolor="#333333", linewidth=0.8,
                hatch=hatch, alpha=0.85)

        # Labels on segments > 0.4 GB
        for bx, val, bot in [(0, val_base, bottoms_base), (1, val_shared, bottoms_shared)]:
            if val > 0.4:
                ax2.text(bx, bot[0] + val/2, f"{val:.1f}",
                         ha="center", va="center", fontsize=10, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                                   edgecolor="none", alpha=0.8))

        handles.append(mpatches.Patch(facecolor=color, edgecolor="#333333",
                                       hatch=hatch, alpha=0.85, label=name))
        bottoms_base += val_base
        bottoms_shared += val_shared

    # Totals
    total_base = sum(v[1] for v in COMPONENTS)
    total_shared = sum(v[2] for v in COMPONENTS)
    savings = total_base - total_shared

    ax2.text(0, total_base + 0.3, f"{total_base:.1f} GB",
             ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax2.text(1, total_shared + 0.3, f"{total_shared:.1f} GB",
             ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Savings arrow
    ax2.annotate(f"-{savings:.1f} GB", xy=(1, total_shared + 0.2),
                 xytext=(0.5, total_base + 1.5), fontsize=12, fontweight="bold",
                 color="#009E73", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#009E73", lw=2))

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(bar_labels, fontsize=11)
    ax2.set_title("Steady-State Breakdown\n(after optimizer init)",
                  fontsize=13, fontweight="bold")
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(0, 34)
    ax2.axhline(y=32, color="red", linestyle="--", linewidth=1.5, alpha=0.4)
    ax2.grid(axis="y", alpha=0.3, linestyle=":")
    ax2.legend(handles=handles, fontsize=9, loc="upper right")

    # Supertitle
    fig.suptitle(
        "GPU Memory Profile — RTX 5090, Qwen3-0.6B bf16\n"
        f"Shared weights save {savings:.1f} GB steady-state "
        f"(peak similar due to optimizer init spike)",
        fontsize=15, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")


def main():
    output_path = Path(__file__).parent.parent / "results" / "memory_profile.png"
    plot_memory_profile(str(output_path))


if __name__ == "__main__":
    main()
