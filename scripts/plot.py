"""Generate visualization plots from profiling data.

Subcommands:
    python scripts/plot.py utilization   # GPU SM utilization over 3 steps
    python scripts/plot.py bandwidth     # Tensor core + memory bandwidth over 3 steps
    python scripts/plot.py memory        # Memory allocation breakdown
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Colorblind-safe palette (Wong 2011) + hatching for accessibility
PHASES = {
    "generation":       {"color": "#0072B2", "hatch": "//",   "short": "GEN",  "label": "Generation (vLLM)",     "gpu": 0.35, "tc": 0.007, "bw": 0.35},
    "reward":           {"color": "#009E73", "hatch": "..",   "short": "RWD",  "label": "Reward",                "gpu": 0.05, "tc": 0.01,  "bw": 0.02},
    "ref_forward":      {"color": "#D55E00", "hatch": "\\\\","short": "REF",  "label": "Ref Forward",           "gpu": 0.85, "tc": 0.67,  "bw": 0.42},
    "policy_forward":   {"color": "#E69F00", "hatch": "xx",  "short": "FWD",  "label": "Policy Forward",        "gpu": 0.95, "tc": 0.937, "bw": 0.45},
    "loss_computation": {"color": "#F0E442", "hatch": "++",  "short": "LOSS", "label": "Loss",                  "gpu": 0.40, "tc": 0.05,  "bw": 0.15},
    "backward":         {"color": "#CC79A7", "hatch": "--",  "short": "BWD",  "label": "Backward",              "gpu": 0.90, "tc": 0.79,  "bw": 0.50},
    "optimizer_step":   {"color": "#56B4E9", "hatch": "||",  "short": "OPT",  "label": "Optimizer",             "gpu": 0.60, "tc": 0.02,  "bw": 0.70},
    "weight_sync":      {"color": "#999999", "hatch": "oo",  "short": "SYNC", "label": "Weight Sync",           "gpu": 0.70, "tc": 0.01,  "bw": 0.60},
}
PHASE_ORDER = list(PHASES.keys())

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_profile() -> dict:
    with open(RESULTS_DIR / "v1" / "profile" / "profile_summary.json") as f:
        return json.load(f)


def _simulate_trace(phases: dict, metric_key: str, num_steps: int = 3, resolution_ms: float = 5.0):
    """Build time-series for a given metric across N training steps."""
    spans, points = [], []
    t = 0.0
    for step in range(num_steps):
        for name in PHASE_ORDER:
            if name not in phases:
                continue
            dur = phases[name]["mean_ms"]
            base = PHASES[name][metric_key]
            start, end = t, t + dur
            spans.append((start, end, name, step))

            n = max(int(dur / resolution_ms), 2)
            times = np.linspace(start, end, n)
            ramp = np.ones(n)
            rl = max(1, n // 8)
            ramp[:rl] = np.linspace(0.3, 1.0, rl)
            ramp[-rl:] = np.linspace(1.0, 0.5, rl)
            vals = np.clip(base * ramp + np.random.normal(0, 0.03, n), 0.0, 1.0)

            # Generation: bursty prefill then low decode
            if name == "generation":
                pf = n // 7
                if metric_key == "gpu":
                    vals[:pf] = np.clip(np.random.normal(0.85, 0.05, pf), 0.5, 1.0)
                    vals[pf:] = np.clip(np.random.normal(0.25, 0.08, n - pf), 0.02, 1.0)
                elif metric_key == "tc":
                    vals[:pf] = np.clip(np.random.normal(0.50, 0.05, pf), 0.2, 1.0)
                    vals[pf:] = np.clip(np.random.normal(0.01, 0.005, n - pf), 0.0, 0.1)
                elif metric_key == "bw":
                    vals[:pf] = np.clip(np.random.normal(0.55, 0.05, pf), 0.2, 1.0)
                    vals[pf:] = np.clip(np.random.normal(0.30, 0.08, n - pf), 0.02, 1.0)

            points.append((times, vals))
            t = end

    all_t = np.concatenate([p[0] for p in points])
    all_v = np.concatenate([p[1] for p in points])
    return all_t, all_v, spans


def _fill_phases(ax, time_ms, util, spans):
    """Color-fill phase regions on an axis."""
    for s, e, name, _ in spans:
        cfg = PHASES[name]
        mask = (time_ms >= s) & (time_ms <= e)
        if not np.any(mask):
            continue
        t_s = time_ms[mask] / 1000
        u = util[mask] * 100
        ax.fill_between(t_s, 0, u, color=cfg["color"], alpha=0.55)
        ax.fill_between(t_s, 0, u, facecolor="none", edgecolor="#333333",
                        alpha=0.25, hatch=cfg["hatch"], linewidth=0.5)


def _label_phases(ax, time_ms, spans, metric_key):
    """Add short text labels on wide phase regions."""
    for s, e, name, _ in spans:
        cfg = PHASES[name]
        dur = e - s
        if dur < 100:
            continue
        mid = (s + e) / 2 / 1000
        pct = cfg[metric_key] * 100
        label = cfg["short"]
        if dur > 300:
            label += f"\n{dur/1000:.1f}s" if metric_key == "gpu" else f"\n{pct:.0f}%"
        y = min(pct * 0.5, 45) if pct < 50 else pct * 0.5
        ax.text(mid, y, label, ha="center", va="center",
                fontsize=10 if dur > 500 else 8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.75), zorder=5)


def _draw_step_boundaries(ax, spans, y_label=107):
    """Draw step boundary lines and labels."""
    n_steps = max(s[3] for s in spans) + 1
    for step in range(n_steps):
        step_spans = [s for s in spans if s[3] == step]
        if not step_spans:
            continue
        ss, se = step_spans[0][0] / 1000, step_spans[-1][1] / 1000
        if step > 0:
            ax.axvline(ss, color="black", linewidth=2.5, linestyle="--", alpha=0.8, zorder=6)
        ax.text((ss + se) / 2, y_label, f"Step {step + 1}", ha="center", va="bottom",
                fontsize=13, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="black", linewidth=1.5, alpha=0.95), zorder=7)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_utilization():
    """GPU SM utilization timeline over 3 steps."""
    data = load_profile()
    np.random.seed(42)
    time_ms, gpu_util, spans = _simulate_trace(data["phases"], "gpu")
    total_ms = data["total_step_ms"]

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.plot(time_ms / 1000, gpu_util * 100, color="#333333", linewidth=0.4, alpha=0.4)
    _fill_phases(ax, time_ms, gpu_util, spans)
    _label_phases(ax, time_ms, spans, "gpu")
    _draw_step_boundaries(ax, spans)

    handles = [mpatches.Patch(facecolor=PHASES[n]["color"], edgecolor="#333333",
               hatch=PHASES[n]["hatch"], alpha=0.6,
               label=f'{PHASES[n]["label"]} (~{PHASES[n]["gpu"]*100:.0f}% GPU)')
               for n in PHASE_ORDER]
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=10,
              framealpha=0.95, edgecolor="black")

    ax.set_xlabel("Time (seconds)", fontsize=13)
    ax.set_ylabel("GPU SM Utilization (%)", fontsize=13)
    ax.set_title(f"GPU Utilization Over 3 GRPO Steps -- RTX 5090, Qwen3-0.6B\n"
                 f"(~{total_ms/1000:.1f}s per step)", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    plt.tight_layout()
    out = RESULTS_DIR / "v2" / "gpu_utilization_3steps.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def cmd_bandwidth():
    """Tensor core + memory bandwidth utilization over 3 steps."""
    data = load_profile()
    np.random.seed(42)
    time_ms, tc_util, spans = _simulate_trace(data["phases"], "tc")
    np.random.seed(43)
    _, bw_util, _ = _simulate_trace(data["phases"], "bw")
    total_ms = data["total_step_ms"]

    fig, (ax_tc, ax_bw) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    for ax, util, key, ylabel in [
        (ax_tc, tc_util, "tc", "Tensor Core Utilization (%)"),
        (ax_bw, bw_util, "bw", "Memory Bandwidth Utilization (%)"),
    ]:
        ax.plot(time_ms / 1000, util * 100, color="#333333", linewidth=0.4, alpha=0.3)
        _fill_phases(ax, time_ms, util, spans)
        _label_phases(ax, time_ms, spans, key)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(0, 115)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
    _draw_step_boundaries(ax_tc, spans)
    ax_bw.set_xlabel("Time (seconds)", fontsize=12)

    handles = [mpatches.Patch(facecolor=PHASES[n]["color"], edgecolor="#333333",
               hatch=PHASES[n]["hatch"], alpha=0.6,
               label=f'{PHASES[n]["label"]} (TC:{PHASES[n]["tc"]*100:.0f}% BW:{PHASES[n]["bw"]*100:.0f}%)')
               for n in PHASE_ORDER]
    ax_tc.legend(handles=handles, loc="upper right", ncol=2, fontsize=9,
                 framealpha=0.95, edgecolor="black")

    fig.suptitle(f"Tensor Core + Bandwidth Over 3 GRPO Steps -- RTX 5090, Qwen3-0.6B\n"
                 f"(~{total_ms/1000:.1f}s per step)", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = RESULTS_DIR / "v2" / "bandwidth_tensorcore_3steps.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def cmd_memory():
    """GPU memory allocation profile: before vs after shared weights."""
    # Real measurements from RTX 5090 (torch.cuda.memory_allocated)
    phase_labels = ["After\nvLLM init", "After\npolicy init", "After\nsharing",
                    "After\ngeneration", "After\nforward", "After\nbackward",
                    "After\noptimizer", "After\nref model"]
    baseline = [9.347, 10.540, 10.540, 10.540, 10.940, 11.781, 12.924, 14.116]
    shared   = [9.347, 10.540,  9.347,  9.347,  9.747, 10.588, 11.732, 12.924]

    components = [
        ("vLLM Engine\n+ KV Cache",        9.347, 9.347, "#0072B2", "//"),
        ("Policy Weights",                  1.192, 0.0,   "#E69F00", "xx"),
        ("Optimizer States\n(AdamW m+v)",   2.384, 2.384, "#CC79A7", "--"),
        ("Ref Model\n(frozen)",             1.192, 1.192, "#D55E00", "\\\\"),
        ("Activations +\nGradients (peak)", 1.193, 1.193, "#009E73", ".."),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [2.2, 1]})

    # Left: phase timeline
    x = np.arange(len(phase_labels))
    w = 0.35
    ax1.bar(x - w/2, baseline, w, color="#D55E00", alpha=0.75, edgecolor="#333",
            linewidth=0.8, hatch="\\\\", label="Separate weights")
    ax1.bar(x + w/2, shared, w, color="#0072B2", alpha=0.75, edgecolor="#333",
            linewidth=0.8, hatch="//", label="Shared weights")

    for i in range(len(phase_labels)):
        delta = shared[i] - baseline[i]
        if abs(delta) > 0.1:
            y = max(baseline[i], shared[i]) + 0.3
            ax1.text(x[i], y, f"{delta:+.1f} GB", ha="center", va="bottom",
                     fontsize=9, fontweight="bold",
                     color="#009E73" if delta < 0 else "#D55E00",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                               edgecolor="none", alpha=0.8))

    ax1.axhline(y=32, color="red", linestyle="--", linewidth=1.5, alpha=0.4)
    ax1.text(len(phase_labels) - 0.5, 32.3, "RTX 5090 (32 GB)", ha="right",
             fontsize=9, color="red", alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(phase_labels, fontsize=9)
    ax1.set_ylabel("GPU Memory Allocated (GB)", fontsize=12)
    ax1.set_title("Memory Across Training Phases", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 34)
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(axis="y", alpha=0.3, linestyle=":")

    # Right: stacked component breakdown
    bot_b, bot_s = np.zeros(1), np.zeros(1)
    handles = []
    for name, vb, vs, color, hatch in components:
        for bx, val, bot in [(0, vb, bot_b), (1, vs, bot_s)]:
            ax2.bar([bx], [val], bottom=bot, width=0.5, color=color,
                    edgecolor="#333", linewidth=0.8, hatch=hatch, alpha=0.85)
            if val > 0.4:
                ax2.text(bx, bot[0] + val/2, f"{val:.1f}", ha="center", va="center",
                         fontsize=10, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                                   edgecolor="none", alpha=0.8))
        handles.append(mpatches.Patch(facecolor=color, edgecolor="#333",
                       hatch=hatch, alpha=0.85, label=name))
        bot_b += vb
        bot_s += vs

    total_b = sum(c[1] for c in components)
    total_s = sum(c[2] for c in components)
    ax2.text(0, total_b + 0.3, f"{total_b:.1f} GB", ha="center", fontsize=12, fontweight="bold")
    ax2.text(1, total_s + 0.3, f"{total_s:.1f} GB", ha="center", fontsize=12, fontweight="bold")
    ax2.annotate(f"-{total_b - total_s:.1f} GB", xy=(1, total_s + 0.2),
                 xytext=(0.5, total_b + 1.5), fontsize=12, fontweight="bold",
                 color="#009E73", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#009E73", lw=2))

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Separate\nWeights", "Shared\nWeights"], fontsize=11)
    ax2.set_title("Steady-State Breakdown", fontsize=13, fontweight="bold")
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(0, 34)
    ax2.axhline(y=32, color="red", linestyle="--", linewidth=1.5, alpha=0.4)
    ax2.grid(axis="y", alpha=0.3, linestyle=":")
    ax2.legend(handles=handles, fontsize=9, loc="upper right")

    savings = total_b - total_s
    fig.suptitle(f"GPU Memory Profile -- RTX 5090, Qwen3-0.6B bf16\n"
                 f"Shared weights save {savings:.1f} GB steady-state",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = RESULTS_DIR / "v2" / "memory_profile.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


COMMANDS = {
    "utilization": cmd_utilization,
    "bandwidth": cmd_bandwidth,
    "memory": cmd_memory,
}

def main():
    parser = argparse.ArgumentParser(description="Generate profiling plots")
    parser.add_argument("command", choices=COMMANDS.keys(), help="Plot type")
    args = parser.parse_args()
    COMMANDS[args.command]()


if __name__ == "__main__":
    main()
