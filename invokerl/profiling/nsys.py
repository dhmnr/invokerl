"""nsys SQLite export → phase breakdown plots + roofline analysis.

The nsys CLI produces `.nsys-rep` files. Export them to SQLite first:
    nsys export --type=sqlite profile.nsys-rep

Then call analyze_profile() or invoke via:
    python -m invokerl.profiling --analyze profile.sqlite
"""

from __future__ import annotations

import json
import os

import numpy as np

from invokerl.profiling.constants import GPUSpecs, PHASE_COLORS, PHASE_ORDER, RTX_5090
from invokerl.profiling.roofline import plot_roofline


def _query_nvtx(conn) -> list[tuple]:
    """Query NVTX events, handling schema variations across nsys versions."""
    for tbl in ["NVTX_EVENTS", "nvtx_events", "NVTX_RANGE"]:
        try:
            rows = conn.execute(f"""
                SELECT s.value AS text, e.start, e.end,
                       (e.end - e.start) AS duration_ns
                FROM {tbl} e
                JOIN StringIds s ON e.textId = s.id
                WHERE e.end IS NOT NULL
                ORDER BY e.start
            """).fetchall()
            if rows:
                return rows
        except Exception:
            pass
        try:
            rows = conn.execute(f"""
                SELECT text, start, end, (end - start) AS duration_ns
                FROM {tbl}
                WHERE text IS NOT NULL AND end IS NOT NULL
                ORDER BY start
            """).fetchall()
            if rows:
                return rows
        except Exception:
            continue
    return []


def _query_kernels_in_window(conn, t_min: int, t_max: int) -> list[tuple]:
    """Query GPU kernels overlapping [t_min, t_max]."""
    for tbl in ["CUPTI_ACTIVITY_KIND_KERNEL", "cupti_activity_kind_kernel"]:
        try:
            rows = conn.execute(f"""
                SELECT s.value, k.start, k.end, (k.end - k.start)
                FROM {tbl} k
                JOIN StringIds s ON k.shortName = s.id
                WHERE k.end >= ? AND k.start <= ?
                ORDER BY k.start
            """, (t_min, t_max)).fetchall()
            if rows:
                return rows
        except Exception:
            pass
        try:
            rows = conn.execute(f"""
                SELECT shortName, start, end, (end - start)
                FROM {tbl}
                WHERE end >= ? AND start <= ?
                ORDER BY start
            """, (t_min, t_max)).fetchall()
            if rows:
                return rows
        except Exception:
            continue
    return []


def _query_memops_in_window(conn, t_min: int, t_max: int) -> list[tuple]:
    """Query GPU memory ops within [t_min, t_max]."""
    results = []
    for tbl in ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"]:
        try:
            rows = conn.execute(f"""
                SELECT 'memop', start, end, (end - start)
                FROM {tbl}
                WHERE end >= ? AND start <= ?
                ORDER BY start
            """, (t_min, t_max)).fetchall()
            results.extend(rows)
        except Exception:
            continue
    return sorted(results, key=lambda x: x[1])


def analyze_profile(sqlite_path: str, output_dir: str, gpu: GPUSpecs = RTX_5090):
    """Parse nsys SQLite export → phase breakdown plots + roofline.

    Generates:
      - phase_durations.png: horizontal bars of each phase's avg wall time
      - phase_pie.png: step time allocation
      - gpu_timeline.png: utilization trace + phase Gantt + idle bubbles
      - phase_gpu_util.png: per-phase kernel time / phase wall time
      - roofline.png: arithmetic intensity vs throughput
      - profile_summary.json
    """
    import sqlite3

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    conn = sqlite3.connect(sqlite_path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"  SQLite tables: {', '.join(tables)}")

    nvtx_events = _query_nvtx(conn)
    if not nvtx_events:
        print("ERROR: No NVTX events found.")
        conn.close()
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"  {len(nvtx_events)} NVTX events")

    training_phases = [
        (name, start, end, dur) for name, start, end, dur in nvtx_events
        if name in PHASE_COLORS and name != "warmup"
    ]
    if not training_phases:
        training_phases = [
            (name, start, end, dur) for name, start, end, dur in nvtx_events
            if name != "warmup" and not name.startswith("step_")
        ]
    if not training_phases:
        print("WARNING: No training phase NVTX events found.")
        training_phases = nvtx_events

    phase_durations: dict[str, list[float]] = {}
    for name, start, end, dur in training_phases:
        phase_durations.setdefault(name, []).append(dur / 1e6)

    phase_names = sorted(
        phase_durations.keys(),
        key=lambda n: PHASE_ORDER.index(n) if n in PHASE_ORDER else 99,
    )
    phase_means = [np.mean(phase_durations[n]) for n in phase_names]
    phase_stds = [np.std(phase_durations[n]) for n in phase_names]
    phase_counts = [len(phase_durations[n]) for n in phase_names]
    bar_colors = [PHASE_COLORS.get(n, "#888888") for n in phase_names]
    total_ms = sum(phase_means)

    # Phase duration bar chart
    fig, ax = plt.subplots(figsize=(14, max(6, len(phase_names) * 0.8)))
    bars = ax.barh(phase_names, phase_means, xerr=phase_stds, color=bar_colors,
                    edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xlabel("Duration (ms)", fontsize=12)
    ax.set_title(
        f"GRPO Training Step: Phase Duration Breakdown "
        f"(avg over {max(phase_counts)} repeats, total: {total_ms / 1000:.1f}s)",
        fontsize=13,
    )
    for bar, mean_ms, count in zip(bars, phase_means, phase_counts):
        pct = mean_ms / total_ms * 100
        label = f"{mean_ms:,.0f} ms ({pct:.1f}%)"
        if count > 1:
            label += f" x{count}"
        ax.text(bar.get_width() + max(phase_means) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9)
    ax.set_xlim(0, max(phase_means) * 1.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_durations.png"), dpi=150)
    plt.close()
    print("  Saved phase_durations.png")

    # Pie chart
    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax.pie(
        phase_means, labels=phase_names, autopct="%1.1f%%",
        colors=bar_colors, startangle=90, pctdistance=0.78,
    )
    for t in autotexts:
        t.set_fontsize(9)
    for t in texts:
        t.set_fontsize(10)
    ax.set_title(f"GRPO Step Time Breakdown (total: {total_ms / 1000:.1f}s)",
                  fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_pie.png"), dpi=150)
    plt.close()
    print("  Saved phase_pie.png")

    # GPU timeline
    t_min = min(s for _, s, _, _ in training_phases)
    t_max = max(e for _, _, e, _ in training_phases)
    total_ns = t_max - t_min
    duration_ms = total_ns / 1e6

    print(f"  Querying GPU kernels in [{t_min}, {t_max}] ({duration_ms / 1000:.1f}s)...")
    kernels = _query_kernels_in_window(conn, t_min, t_max)
    memops = _query_memops_in_window(conn, t_min, t_max)
    conn.close()
    print(f"  {len(kernels)} GPU kernels, {len(memops)} memops")

    n_bins = max(int(duration_ms) + 1, 1)
    gpu_busy = np.zeros(n_bins, dtype=np.float64)

    if kernels:
        for _, k_start, k_end, _ in kernels:
            ks = max(k_start, t_min)
            ke = min(k_end, t_max)
            bin_s = max(0, min(int((ks - t_min) / 1e6), n_bins - 1))
            bin_e = max(0, min(int((ke - t_min) / 1e6), n_bins - 1))
            gpu_busy[bin_s:bin_e + 1] = 1.0

    window = max(1, n_bins // 200)
    gpu_smooth = np.convolve(gpu_busy, np.ones(window) / window, mode="same") if window > 1 else gpu_busy

    fig, axes = plt.subplots(
        3, 1, figsize=(20, 10),
        height_ratios=[1.5, 1, 1.5],
        gridspec_kw={"hspace": 0.3},
    )
    ax_util, ax_phases, ax_bubbles = axes
    time_axis = np.arange(n_bins)
    avg_util = np.mean(gpu_busy) * 100

    ax_util.fill_between(time_axis, gpu_smooth, alpha=0.7, color="#4285F4")
    ax_util.set_ylabel("GPU Util", fontsize=10)
    ax_util.set_ylim(0, 1.15)
    ax_util.set_xlim(0, n_bins)
    ax_util.set_title(
        f"GPU Utilization (avg: {avg_util:.1f}%, "
        f"{len(kernels)} kernels in {duration_ms / 1000:.1f}s)",
        fontsize=12,
    )
    ax_util.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    for name, start, end, dur in training_phases:
        x_start = (start - t_min) / 1e6
        width = dur / 1e6
        color = PHASE_COLORS.get(name, "#888888")
        ax_phases.barh(0, width, left=x_start, height=0.7,
                        color=color, alpha=0.85, edgecolor="black", linewidth=0.3)
    ax_phases.set_xlim(0, n_bins)
    ax_phases.set_yticks([])
    ax_phases.set_ylabel("Phases", fontsize=10)
    legend_patches = [Patch(facecolor=PHASE_COLORS.get(n, "#888"), label=n)
                      for n in phase_names]
    ax_phases.legend(handles=legend_patches, loc="upper right", ncol=4, fontsize=8)

    if kernels:
        gap_starts = []
        gap_durations = []
        prev_end = t_min
        for _, k_start, k_end, _ in kernels:
            gap = k_start - prev_end
            if gap > 100_000:
                gap_starts.append((prev_end - t_min) / 1e6)
                gap_durations.append(gap / 1e6)
            prev_end = max(prev_end, k_end)

        if gap_starts:
            ax_bubbles.bar(gap_starts, gap_durations, width=max(1, n_bins / 500),
                            color="red", alpha=0.6, label="GPU idle gaps")
        total_gap_ms = sum(gap_durations) if gap_durations else 0
        ax_bubbles.set_ylabel("Gap (ms)", fontsize=10)
        ax_bubbles.set_xlabel("Time (ms from start)", fontsize=10)
        ax_bubbles.set_xlim(0, n_bins)
        ax_bubbles.set_title(
            f"GPU Idle Bubbles (>0.1ms gaps, total: {total_gap_ms:.0f}ms = "
            f"{total_gap_ms / duration_ms * 100:.1f}% of step)",
            fontsize=11,
        )
        if gap_starts:
            ax_bubbles.legend(fontsize=9)
    else:
        ax_bubbles.text(0.5, 0.5, "No GPU kernel data (run nsys with --trace=cuda)",
                        transform=ax_bubbles.transAxes, ha="center", fontsize=12)

    plt.savefig(os.path.join(output_dir, "gpu_timeline.png"), dpi=150)
    plt.close()
    print("  Saved gpu_timeline.png")

    # Per-phase GPU utilization
    if kernels:
        k_starts = np.array([k[1] for k in kernels], dtype=np.int64)
        k_ends = np.array([k[2] for k in kernels], dtype=np.int64)

        phase_gpu_util: dict[str, list[float]] = {}
        for pname, p_start, p_end, p_dur in training_phases:
            i_start = np.searchsorted(k_ends, p_start, side="right")
            i_end = np.searchsorted(k_starts, p_end, side="left")
            k_time_ns = 0
            for i in range(i_start, min(i_end, len(kernels))):
                overlap_start = max(k_starts[i], p_start)
                overlap_end = min(k_ends[i], p_end)
                if overlap_end > overlap_start:
                    k_time_ns += overlap_end - overlap_start
            util = k_time_ns / max(p_dur, 1) * 100
            phase_gpu_util.setdefault(pname, []).append(util)

        fig, ax = plt.subplots(figsize=(14, max(6, len(phase_names) * 0.8)))
        util_means = [np.mean(phase_gpu_util.get(n, [0])) for n in phase_names]
        bars = ax.barh(phase_names, util_means, color=bar_colors,
                        edgecolor="black", linewidth=0.5)
        ax.set_xlabel("GPU Utilization within Phase (%)", fontsize=12)
        ax.set_title("Per-Phase GPU Utilization (kernel time / phase wall time)",
                      fontsize=13)
        for bar, util in zip(bars, util_means):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{util:.1f}%", va="center", fontsize=10)
        ax.set_xlim(0, 105)
        ax.axvline(x=100, color="gray", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "phase_gpu_util.png"), dpi=150)
        plt.close()
        print("  Saved phase_gpu_util.png")

    plot_roofline(phase_means, phase_names, output_dir, gpu)

    print(f"\n  {'Phase':<20s} {'Mean (ms)':>10s} {'Std':>8s} "
          f"{'Count':>6s} {'% Step':>8s}")
    print(f"  {'-' * 56}")
    for name, mean, std, cnt in zip(phase_names, phase_means, phase_stds, phase_counts):
        pct = mean / total_ms * 100
        print(f"  {name:<20s} {mean:>10,.0f} {std:>8,.0f} {cnt:>6d} {pct:>7.1f}%")
    print(f"  {'TOTAL':<20s} {total_ms:>10,.0f}")

    if kernels:
        total_kernel_ns = sum(d for _, _, _, d in kernels)
        print(f"\n  GPU utilization: {avg_util:.1f}% "
              f"(kernel time: {total_kernel_ns / 1e9:.2f}s / "
              f"wall time: {total_ns / 1e9:.2f}s)")

        kernel_totals: dict[str, float] = {}
        kernel_counts: dict[str, int] = {}
        for kname, _, _, kdur in kernels:
            kernel_totals[kname] = kernel_totals.get(kname, 0) + kdur / 1e6
            kernel_counts[kname] = kernel_counts.get(kname, 0) + 1
        top_k = sorted(kernel_totals.items(), key=lambda x: -x[1])[:15]

        print(f"\n  Top 15 GPU kernels by total time:")
        print(f"  {'Kernel':<55s} {'Total(ms)':>10s} {'Count':>6s} {'Avg(ms)':>10s}")
        print(f"  {'-' * 85}")
        for kname, ktotal in top_k:
            cnt = kernel_counts[kname]
            kavg = ktotal / cnt
            short = kname[:53] if len(kname) > 53 else kname
            print(f"  {short:<55s} {ktotal:>10,.1f} {cnt:>6d} {kavg:>10.3f}")

    summary = {
        "gpu": gpu.name,
        "total_step_ms": total_ms,
        "phases": {
            name: {"mean_ms": mean, "std_ms": std, "pct": mean / total_ms * 100}
            for name, mean, std in zip(phase_names, phase_means, phase_stds)
        },
        "gpu_utilization_pct": avg_util if kernels else None,
        "num_kernels": len(kernels),
    }
    summary_path = os.path.join(output_dir, "profile_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved profile_summary.json")
