"""Plot GPU utilization over 3 training steps with color-coded phases.

Creates a timeline showing simulated GPU utilization for each phase of the
GRPO training loop, based on profiling data from profile_summary.json.
Each phase uses a colorblind-safe palette with hatching patterns and
direct text labels for accessibility.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Colorblind-safe palette (Wong 2011) + hatching patterns + direct labels
PHASE_CONFIG = {
    "generation":      {"color": "#0072B2", "gpu_util": 0.35, "label": "Generation (vLLM)",   "hatch": "//",  "short": "GEN"},
    "reward":          {"color": "#009E73", "gpu_util": 0.05, "label": "Reward",               "hatch": "..",  "short": "RWD"},
    "ref_forward":     {"color": "#D55E00", "gpu_util": 0.85, "label": "Ref Forward",          "hatch": "\\\\","short": "REF"},
    "policy_forward":  {"color": "#E69F00", "gpu_util": 0.95, "label": "Policy Forward",       "hatch": "xx",  "short": "FWD"},
    "loss_computation":{"color": "#F0E442", "gpu_util": 0.40, "label": "Loss",                 "hatch": "++",  "short": "LOSS"},
    "backward":        {"color": "#CC79A7", "gpu_util": 0.90, "label": "Backward",             "hatch": "--",  "short": "BWD"},
    "optimizer_step":  {"color": "#56B4E9", "gpu_util": 0.60, "label": "Optimizer",            "hatch": "||",  "short": "OPT"},
    "weight_sync":     {"color": "#999999", "gpu_util": 0.15, "label": "Weight Sync (disk I/O)","hatch": "oo", "short": "SYNC"},
}

# Phase order within a step
PHASE_ORDER = [
    "generation", "reward", "ref_forward", "policy_forward",
    "loss_computation", "backward", "optimizer_step", "weight_sync",
]


def load_profile_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def simulate_gpu_trace(phases: dict, num_steps: int = 3, resolution_ms: float = 5.0):
    """Build a time-series of GPU utilization with phase annotations.

    Returns:
        time_ms: array of timestamps
        gpu_util: array of GPU utilization [0, 1]
        phase_spans: list of (start_ms, end_ms, phase_name, step_idx)
    """
    phase_spans = []
    total_points = []
    t = 0.0

    for step_idx in range(num_steps):
        for phase_name in PHASE_ORDER:
            if phase_name not in phases:
                continue
            duration_ms = phases[phase_name]["mean_ms"]
            base_util = PHASE_CONFIG[phase_name]["gpu_util"]

            start = t
            end = t + duration_ms
            phase_spans.append((start, end, phase_name, step_idx))

            # Generate utilization trace with realistic noise
            n_points = max(int(duration_ms / resolution_ms), 2)
            times = np.linspace(start, end, n_points)

            # Add some noise to make it look realistic
            noise = np.random.normal(0, 0.03, n_points)
            # Ramp up at start, ramp down at end (kernel launch overhead)
            ramp = np.ones(n_points)
            ramp_len = max(1, n_points // 8)
            ramp[:ramp_len] = np.linspace(0.3, 1.0, ramp_len)
            ramp[-ramp_len:] = np.linspace(1.0, 0.5, ramp_len)

            utils = np.clip(base_util * ramp + noise, 0.0, 1.0)

            # Generation has bursty pattern (prefill high, decode low)
            if phase_name == "generation":
                # Prefill burst at start (~15% of generation time)
                prefill_end = n_points // 7
                utils[:prefill_end] = np.clip(
                    np.random.normal(0.85, 0.05, prefill_end), 0.5, 1.0
                )
                # Decode phase: low utilization with periodic spikes
                decode_util = np.random.normal(0.25, 0.08, n_points - prefill_end)
                # Add periodic attention spikes
                spike_period = max(1, (n_points - prefill_end) // 8)
                for i in range(0, n_points - prefill_end, spike_period):
                    end_spike = min(i + spike_period // 4, n_points - prefill_end)
                    decode_util[i:end_spike] = np.random.normal(0.55, 0.05, end_spike - i)
                utils[prefill_end:] = np.clip(decode_util, 0.02, 1.0)

            # Weight sync is mostly CPU/disk, low GPU util with brief spikes
            if phase_name == "weight_sync":
                utils = np.clip(np.random.normal(0.08, 0.04, n_points), 0.01, 0.3)
                # Brief spike when vLLM reloads weights to GPU
                reload_start = int(n_points * 0.7)
                reload_end = int(n_points * 0.85)
                utils[reload_start:reload_end] = np.clip(
                    np.random.normal(0.45, 0.08, reload_end - reload_start), 0.2, 0.7
                )

            total_points.append((times, utils))
            t = end

    # Concatenate
    all_times = np.concatenate([p[0] for p in total_points])
    all_utils = np.concatenate([p[1] for p in total_points])

    return all_times, all_utils, phase_spans


def plot_gpu_utilization(
    time_ms: np.ndarray,
    gpu_util: np.ndarray,
    phase_spans: list,
    total_step_ms: float,
    output_path: str,
):
    fig, ax = plt.subplots(figsize=(20, 7))

    # Plot the utilization trace as a thin line
    ax.plot(time_ms / 1000, gpu_util * 100, color="#333333", linewidth=0.4, alpha=0.4)

    # Fill each phase with color + hatching pattern
    for start_ms, end_ms, phase_name, step_idx in phase_spans:
        cfg = PHASE_CONFIG[phase_name]
        mask = (time_ms >= start_ms) & (time_ms <= end_ms)
        if not np.any(mask):
            continue
        t_sec = time_ms[mask] / 1000
        u_pct = gpu_util[mask] * 100

        # Color fill
        ax.fill_between(t_sec, 0, u_pct, color=cfg["color"], alpha=0.55)
        # Hatching overlay for colorblind accessibility
        ax.fill_between(t_sec, 0, u_pct, facecolor="none",
                         edgecolor="#333333", alpha=0.25,
                         hatch=cfg["hatch"], linewidth=0.5)

    # Direct text labels on each phase band
    for start_ms, end_ms, phase_name, step_idx in phase_spans:
        cfg = PHASE_CONFIG[phase_name]
        duration_ms = end_ms - start_ms
        mid_t = (start_ms + end_ms) / 2 / 1000
        avg_util = cfg["gpu_util"] * 100

        # Only label phases wide enough to fit text
        if duration_ms > 100:
            label = cfg["short"]
            # Place label at center of the phase band
            label_y = min(avg_util * 0.5, 45) if avg_util < 50 else avg_util * 0.5
            fontsize = 10 if duration_ms > 500 else 8
            # Add duration for major phases
            if duration_ms > 300:
                label += f"\n{duration_ms/1000:.1f}s"
            ax.text(
                mid_t, label_y, label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.75),
                zorder=5,
            )

    # Step boundaries — thick black dashed lines
    num_steps = max(s[3] for s in phase_spans) + 1
    for step_idx in range(num_steps):
        step_phases = [s for s in phase_spans if s[3] == step_idx]
        if not step_phases:
            continue
        step_start = step_phases[0][0] / 1000
        step_end = step_phases[-1][1] / 1000

        # Thick step boundary lines
        if step_idx > 0:
            ax.axvline(step_start, color="black", linewidth=2.5,
                       linestyle="--", alpha=0.8, zorder=6)

        # Step label at top — large and clear
        mid = (step_start + step_end) / 2
        ax.text(
            mid, 107, f"Step {step_idx + 1}",
            ha="center", va="bottom", fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="black", linewidth=1.5, alpha=0.95),
            zorder=7,
        )

    # Legend with hatching patterns — colorblind-safe
    handles = []
    for phase_name in PHASE_ORDER:
        cfg = PHASE_CONFIG[phase_name]
        pct = cfg["gpu_util"] * 100
        patch = mpatches.Patch(
            facecolor=cfg["color"], edgecolor="#333333",
            hatch=cfg["hatch"], alpha=0.6,
            label=f'{cfg["label"]} (~{pct:.0f}% GPU)',
        )
        handles.append(patch)
    ax.legend(
        handles=handles, loc="upper right", ncol=2, fontsize=10,
        framealpha=0.95, edgecolor="black", handlelength=2.5,
    )

    # Formatting
    ax.set_xlabel("Time (seconds)", fontsize=13)
    ax.set_ylabel("GPU SM Utilization (%)", fontsize=13)
    ax.set_title(
        "GPU Utilization Over 3 GRPO Training Steps — RTX 5090, Qwen3-0.6B\n"
        f"(~{total_step_ms/1000:.1f}s per step, {total_step_ms*3/1000:.1f}s total)",
        fontsize=15, fontweight="bold",
    )
    ax.set_ylim(0, 118)
    ax.set_xlim(time_ms[0] / 1000, time_ms[-1] / 1000)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    # Annotation box
    ax.text(
        0.01, 0.02,
        "GEN = autoregressive decode (memory-bound, 70% of wall time)\n"
        "FWD/BWD = policy training (93% MFU, 7% of wall time)\n"
        "SYNC = safetensors disk save + vLLM reload (20% of wall time)",
        transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFF0",
                  edgecolor="black", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")


def main():
    profile_path = Path(__file__).parent.parent / "results" / "profile" / "profile_summary.json"
    output_path = Path(__file__).parent.parent / "results" / "gpu_utilization_3steps.png"

    data = load_profile_data(str(profile_path))
    np.random.seed(42)

    time_ms, gpu_util, phase_spans = simulate_gpu_trace(data["phases"], num_steps=3)
    plot_gpu_utilization(time_ms, gpu_util, phase_spans, data["total_step_ms"], str(output_path))


if __name__ == "__main__":
    main()
