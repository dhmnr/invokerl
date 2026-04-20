"""Python-timer-based phase breakdown: plots, MFU, bottleneck analysis."""

from __future__ import annotations

import json
import os

import numpy as np

from invokerl.profiling.constants import PHASE_COLORS, PHASE_ORDER, RTX_5090
from invokerl.profiling.flops import estimate_flops_per_step
from invokerl.profiling.roofline import plot_roofline


def timing_report(all_phase_times: list[dict[str, float]], output_dir: str):
    """Generate timing summary from Python-measured phase times.

    Skips the first step as warmup if more than one is provided. Writes
    timing_breakdown.png, timing_pie.png, roofline.png, and profile_summary.json.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    steps = all_phase_times[1:] if len(all_phase_times) > 1 else all_phase_times

    phase_names = [n for n in PHASE_ORDER if any(n in s for s in steps)]
    phase_means = []
    phase_stds = []
    for name in phase_names:
        vals = [s.get(name, 0) * 1000 for s in steps]
        phase_means.append(np.mean(vals))
        phase_stds.append(np.std(vals))

    total_ms = sum(phase_means)
    bar_colors = [PHASE_COLORS.get(n, "#888888") for n in phase_names]

    fig, ax = plt.subplots(figsize=(14, max(6, len(phase_names) * 0.8)))
    bars = ax.barh(
        phase_names, phase_means, xerr=phase_stds, color=bar_colors,
        edgecolor="black", linewidth=0.5, capsize=3,
    )
    ax.set_xlabel("Duration (ms)", fontsize=12)
    ax.set_title(
        f"GRPO Step Timing (Python timer, avg over {len(steps)} steps, "
        f"total: {total_ms / 1000:.1f}s)",
        fontsize=13,
    )
    for bar, mean_ms in zip(bars, phase_means):
        pct = mean_ms / total_ms * 100 if total_ms > 0 else 0
        ax.text(
            bar.get_width() + max(phase_means) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{mean_ms:,.0f} ms ({pct:.1f}%)", va="center", fontsize=9,
        )
    ax.set_xlim(0, max(phase_means) * 1.3 if phase_means else 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timing_breakdown.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pie(
        phase_means, labels=phase_names, autopct="%1.1f%%",
        colors=bar_colors, startangle=90, pctdistance=0.78,
    )
    ax.set_title(f"GRPO Step Time Allocation (total: {total_ms / 1000:.1f}s)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timing_pie.png"), dpi=150)
    plt.close()

    plot_roofline(phase_means, phase_names, output_dir)

    print(f"\n{'=' * 70}")
    print(f"  GRPO Training Step Profiling Summary ({RTX_5090.name})")
    print(f"{'=' * 70}")
    print(f"\n  {'Phase':<20s} {'Mean (ms)':>10s} {'Std':>8s} {'% Step':>8s}")
    print(f"  {'-' * 50}")
    for name, mean, std in zip(phase_names, phase_means, phase_stds):
        pct = mean / total_ms * 100 if total_ms > 0 else 0
        print(f"  {name:<20s} {mean:>10,.0f} {std:>8,.0f} {pct:>7.1f}%")
    print(f"  {'TOTAL':<20s} {total_ms:>10,.0f}")
    print(f"  {'STEPS/HOUR':>28s} {3600 / (total_ms / 1000):.0f}")

    flop_est = estimate_flops_per_step()
    print(f"\n  Estimated FLOPs per step:")
    total_flops = 0
    for name in phase_names:
        if name in flop_est:
            est = flop_est[name]
            flops = est["flops"]
            total_flops += flops
            idx = phase_names.index(name)
            wall_s = phase_means[idx] / 1000.0
            if wall_s > 0 and flops > 0:
                tflops = flops / 1e12 / wall_s
                print(f"    {name:<20s} {flops / 1e12:>8.2f} TFLOP  "
                      f"({tflops:>6.1f} TFLOP/s = "
                      f"{tflops / RTX_5090.peak_bf16_tflops * 100:>4.1f}% of peak BF16)")
    print(f"    {'TOTAL':<20s} {total_flops / 1e12:>8.2f} TFLOP")

    total_wall_s = total_ms / 1000.0
    if total_wall_s > 0 and total_flops > 0:
        mfu = total_flops / 1e12 / total_wall_s / RTX_5090.peak_bf16_tflops * 100
        print(f"\n  Model FLOP Utilization (MFU): {mfu:.1f}%")

    print(f"\n  Bottleneck Analysis:")
    if phase_means:
        top_phase = phase_names[np.argmax(phase_means)]
        top_pct = max(phase_means) / total_ms * 100
        print(f"    Primary bottleneck: {top_phase} ({top_pct:.1f}% of step)")
        if top_phase == "generation":
            print(f"    -> Autoregressive generation is memory-bound and sequential.")
            print(f"    -> Optimize: larger batch, shorter sequences, disaggregate to 2nd GPU.")
        elif top_phase in ("backward", "policy_forward"):
            print(f"    -> Compute-bound phase. Optimize: mixed precision, kernel fusion.")
        elif top_phase == "weight_sync":
            print(f"    -> I/O-bound. Optimize: shared weights, direct GPU copy.")

    summary = {
        "gpu": RTX_5090.name,
        "total_step_ms": total_ms,
        "steps_per_hour": 3600 / (total_ms / 1000) if total_ms > 0 else 0,
        "phases": {
            name: {"mean_ms": mean, "std_ms": std, "pct": mean / total_ms * 100}
            for name, mean, std in zip(phase_names, phase_means, phase_stds)
        },
        "total_tflops": total_flops / 1e12,
        "mfu_pct": (total_flops / 1e12 / (total_ms / 1000) / RTX_5090.peak_bf16_tflops * 100)
        if total_ms > 0 and total_flops > 0 else 0,
    }
    with open(os.path.join(output_dir, "profile_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved all plots + profile_summary.json to {output_dir}/")
