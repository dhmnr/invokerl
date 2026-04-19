"""Roofline model plot — arithmetic intensity vs achieved throughput."""

from __future__ import annotations

import os

import numpy as np

from invokerl.profiling.constants import GPUSpecs, PHASE_COLORS, RTX_5090
from invokerl.profiling.flops import estimate_flops_per_step


def plot_roofline(
    phase_means: list[float],
    phase_names: list[str],
    output_dir: str,
    gpu: GPUSpecs = RTX_5090,
):
    """Plot roofline model with estimated phase positions.

    Phases below the roof are bottlenecked by memory bandwidth (left of ridge)
    or compute (right of ridge). phase_means is in milliseconds.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flop_est = estimate_flops_per_step()
    fig, ax = plt.subplots(figsize=(14, 9))
    x_range = np.logspace(-1, 4, 500)

    bf16_roof = np.minimum(
        gpu.peak_bf16_tflops * 1000,
        x_range * gpu.mem_bandwidth_gb_s,
    )
    ax.plot(x_range, bf16_roof, "b-", linewidth=2,
            label=f"BF16 peak ({gpu.peak_bf16_tflops:.0f} TFLOPS)")

    fp32_roof = np.minimum(
        gpu.peak_fp32_tflops * 1000,
        x_range * gpu.mem_bandwidth_gb_s,
    )
    ax.plot(x_range, fp32_roof, "r--", linewidth=1.5,
            label=f"FP32 peak ({gpu.peak_fp32_tflops:.0f} TFLOPS)")

    mem_line = x_range * gpu.mem_bandwidth_gb_s
    ax.plot(x_range, mem_line, "k:", linewidth=1, alpha=0.3,
            label=f"Mem BW ({gpu.mem_bandwidth_tb_s:.2f} TB/s)")

    ax.axvline(x=gpu.ridge_point_bf16, color="blue", linestyle=":", alpha=0.3)
    ax.axvline(x=gpu.ridge_point_fp32, color="red", linestyle=":", alpha=0.3)

    for name in phase_names:
        if name not in flop_est or name == "reward":
            continue

        est = flop_est[name]
        ai = est["arithmetic_intensity"]
        if ai <= 0:
            continue

        idx = phase_names.index(name)
        wall_s = phase_means[idx] / 1000.0
        achieved_gflops = est["flops"] / 1e9 / wall_s if wall_s > 0 else 0

        color = PHASE_COLORS.get(name, "#888888")
        ax.scatter([ai], [achieved_gflops],
                   s=150, c=color, edgecolors="black", linewidths=1,
                   zorder=5, label=f"{name} (AI={ai:.1f})")
        ax.annotate(name, (ai, achieved_gflops),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=8, color=color, fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Throughput (GFLOP/s)", fontsize=12)
    ax.set_title(f"Roofline Model — {gpu.name} ({gpu.vram_gb:.0f}GB)", fontsize=14)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0.1, 5000)
    ax.set_ylim(1, gpu.peak_bf16_tflops * 1500)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roofline.png"), dpi=150)
    plt.close()
    print("  Saved roofline.png")
