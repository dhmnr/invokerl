"""Profile GRPO training steps with NVTX annotations for Nsight Systems.

Runs a few training steps with fine-grained NVTX markers on each phase:
  - Generation (autoregressive sampling)
  - Reference model forward (log-probs)
  - Reward computation
  - Policy model forward (log-probs + activations)
  - Loss computation
  - Backward pass
  - Optimizer step

Usage:
    # Capture trace with nsys (2 training steps):
    LD_PRELOAD=/usr/local/cuda-13.1/targets/x86_64-linux/lib/libnvrtc.so.13 \
    TILERL_BACKEND=cupy \
    nsys profile --trace=cuda,nvtx --force-overwrite true \
        --output /root/tilerl/results/grpo_profile \
        python3 profile_grpo.py \
            --model_path /workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/ \
            --checkpoint /root/tilerl/grpo_checkpoints/step_200 \
            --num_steps 2

    # Export stats:
    nsys stats /root/tilerl/results/grpo_profile.nsys-rep --report gpukernsum

    # Export to SQLite for custom analysis:
    nsys export --type=sqlite /root/tilerl/results/grpo_profile.nsys-rep

    # Generate plots:
    python3 profile_grpo.py --analyze /root/tilerl/results/grpo_profile.sqlite
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Backend selection
_BACKEND = os.environ.get("TILERL_BACKEND", "").lower()
if _BACKEND == "numpy":
    import numpy as cp
else:
    try:
        import cupy as cp
    except ModuleNotFoundError:
        import numpy as cp

import nvtx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen3 import Qwen3Config, Qwen3Model


# ---------------------------------------------------------------------------
# NVTX-annotated training step
# ---------------------------------------------------------------------------

def free_all_blocks():
    """Release CuPy memory pool blocks back to CUDA."""
    if hasattr(cp, 'get_default_memory_pool'):
        cp.get_default_memory_pool().free_all_blocks()


def profiled_training_step(
    model: Qwen3Model,
    ref_model: Qwen3Model,
    optimizer,
    tokenizer,
    train_data: list[dict],
    cfg,
    rng: np.random.RandomState,
    step: int,
):
    """Run one GRPO training step with NVTX annotations on every phase."""
    from train_grpo_gsm8k import (
        compute_rewards, format_prompt, gather_ref_logprobs,
    )
    from grpo import (
        compute_advantages, clipped_surrogate_fwd, clipped_surrogate_bwd,
        kl_fwd, kl_bwd, gather_logprobs, log_softmax_gather_bwd,
    )
    from optim import clip_grad_norm

    xp = cp

    K = cfg.accumulation_steps
    accumulated_grads = None
    total_loss = total_kl = total_reward = 0.0

    for micro in range(K):
        idx = rng.choice(len(train_data))
        batch_questions = [train_data[idx]]

        # --- Generation phase ---
        with nvtx.annotate("generation", color="blue"):
            input_ids, response_mask, rewards = compute_rewards(
                model, tokenizer, batch_questions,
                group_size=cfg.group_size,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
            )
        if xp is not np and hasattr(cp, 'get_default_memory_pool'):
            cp.get_default_memory_pool().free_all_blocks()

        B, T = input_ids.shape

        # --- Reference model forward ---
        with nvtx.annotate("ref_forward", color="green"):
            ref_lp = gather_ref_logprobs(ref_model, input_ids, response_mask)
        free_all_blocks()

        # --- Advantage computation ---
        with nvtx.annotate("advantages", color="yellow"):
            advantages = compute_advantages(rewards)

        # --- Policy model forward ---
        with nvtx.annotate("policy_forward", color="orange"):
            new_lp, logits_2d, mask = gather_logprobs(
                model, input_ids, response_mask, recompute_attn=True,
            )

        # --- Loss computation ---
        with nvtx.annotate("loss_computation", color="red"):
            # Old log-probs (detached)
            old_lp = new_lp.copy()

            # Clipped surrogate
            surr_loss, surr_per_token = clipped_surrogate_fwd(
                new_lp, old_lp, advantages, mask, clip_eps=cfg.clip_eps,
            )

            # KL penalty
            kl_val, kl_per_token = kl_fwd(new_lp, ref_lp, mask)

            # Total loss
            loss = surr_loss + cfg.beta * kl_val

        total_loss += float(loss)
        total_kl += float(kl_val)
        total_reward += float(rewards.mean())

        # --- Backward pass ---
        with nvtx.annotate("backward", color="purple"):
            # Gradient of loss w.r.t. per-token log-probs
            grad_surr = clipped_surrogate_bwd(
                new_lp, old_lp, advantages, mask, clip_eps=cfg.clip_eps,
            )
            grad_kl = kl_bwd(new_lp, ref_lp, mask)
            grad_lp = grad_surr + cfg.beta * grad_kl

            # Backprop through log_softmax_gather
            V = logits_2d.shape[1]
            shift_targets = input_ids[:, 1:].reshape(B * (T - 1)).astype(xp.int32)
            grad_logits_flat = log_softmax_gather_bwd(
                logits_2d, shift_targets, grad_lp.reshape(-1),
            )

            # Reshape and backprop through model
            full_grad = xp.zeros((B, T, V), dtype=xp.float32)
            full_grad[:, :-1, :] = grad_logits_flat.reshape(B, T - 1, V)

            grads = model.backward(full_grad)

        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            for k in accumulated_grads:
                accumulated_grads[k] += grads[k]
            del grads

        model._cache = {}
        del full_grad, grad_logits_flat, grad_lp, grad_surr, grad_kl
        del new_lp, old_lp, logits_2d, mask, ref_lp, advantages
        del input_ids, response_mask, rewards
        free_all_blocks()

    # --- Optimizer step ---
    with nvtx.annotate("optimizer_step", color="cyan"):
        if K > 1:
            for k in accumulated_grads:
                accumulated_grads[k] /= K
        grad_norm = clip_grad_norm(accumulated_grads, cfg.grad_clip)
        optimizer.step(accumulated_grads)
        del accumulated_grads
        free_all_blocks()

    return {
        "loss": total_loss / K,
        "kl": total_kl / K,
        "reward": total_reward / K,
        "grad_norm": float(grad_norm),
    }


# ---------------------------------------------------------------------------
# Analysis: parse nsys SQLite export and generate plots
# ---------------------------------------------------------------------------

PHASE_COLORS = {
    "generation": "#4285F4",
    "ref_forward": "#34A853",
    "advantages": "#FBBC04",
    "policy_forward": "#FF6D01",
    "loss_computation": "#EA4335",
    "backward": "#9334E6",
    "optimizer_step": "#00BCD4",
    "warmup": "#9E9E9E",
}

# Ordered list of phases for consistent display
PHASE_ORDER = [
    "generation", "ref_forward", "advantages",
    "policy_forward", "loss_computation", "backward", "optimizer_step",
]


def _query_nvtx(conn) -> list[tuple]:
    """Query NVTX events from nsys SQLite, handling schema variations.

    nsys stores NVTX text in two ways:
      1. Directly in the ``text`` column (older exports).
      2. Via ``textId`` foreign key into the ``StringIds`` table (newer exports).
    We try the JOIN approach first, then fall back to direct text.
    """
    for tbl in ["NVTX_EVENTS", "nvtx_events", "NVTX_RANGE"]:
        # Try JOIN with StringIds first (newer nsys exports)
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
        # Fall back to direct text column
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
    """Query GPU kernels that overlap with [t_min, t_max] window.

    Returns (shortName, start, end, duration_ns).
    Resolves shortName via StringIds JOIN if needed.
    """
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
    """Query GPU memory operations within [t_min, t_max]."""
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


def analyze_profile(sqlite_path: str, output_dir: str):
    """Parse nsys SQLite export and generate utilization + bubble plots.

    Generates:
      1. Phase duration bar chart (horizontal bars with ms + %)
      2. Pie chart of step time allocation
      3. GPU timeline: utilization trace + NVTX phase bars + bubble highlights
      4. Top GPU kernels table
      5. Per-phase GPU utilization breakdown
    """
    import sqlite3
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    conn = sqlite3.connect(sqlite_path)

    # List available tables for debugging
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"  SQLite tables: {', '.join(tables)}")

    nvtx_events = _query_nvtx(conn)

    if not nvtx_events:
        print("ERROR: No NVTX events found. Profiling may have failed.")
        print("  Check that the training script has nvtx.annotate() markers")
        print("  and nsys was run with --trace=cuda,nvtx")
        conn.close()
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"  {len(nvtx_events)} NVTX events")

    # Filter out warmup and step-level markers — keep only phase markers
    training_phases = [
        (name, start, end, dur) for name, start, end, dur in nvtx_events
        if name in PHASE_COLORS and name != "warmup"
    ]
    if not training_phases:
        # Maybe the markers include step_ prefixes? Accept anything not warmup
        training_phases = [
            (name, start, end, dur) for name, start, end, dur in nvtx_events
            if name != "warmup" and not name.startswith("step_")
        ]

    if not training_phases:
        print("WARNING: No training phase NVTX events found (only warmup).")
        print("  Available events:", set(name for name, *_ in nvtx_events))
        # Fall through to show whatever we have
        training_phases = nvtx_events

    # Aggregate phases by name
    phase_durations: dict[str, list[float]] = {}
    for name, start, end, dur in training_phases:
        phase_durations.setdefault(name, []).append(dur / 1e6)  # ms

    # Sort phases in canonical order
    phase_names = sorted(
        phase_durations.keys(),
        key=lambda n: PHASE_ORDER.index(n) if n in PHASE_ORDER else 99,
    )
    phase_means = [np.mean(phase_durations[n]) for n in phase_names]
    phase_stds = [np.std(phase_durations[n]) for n in phase_names]
    phase_counts = [len(phase_durations[n]) for n in phase_names]
    bar_colors = [PHASE_COLORS.get(n, "#888888") for n in phase_names]
    total_ms = sum(phase_means)

    # ===== Plot 1: Phase duration horizontal bar chart =====
    fig, ax = plt.subplots(figsize=(14, max(6, len(phase_names) * 0.8)))
    bars = ax.barh(
        phase_names, phase_means, xerr=phase_stds, color=bar_colors,
        edgecolor="black", linewidth=0.5, capsize=3,
    )
    ax.set_xlabel("Duration (ms)", fontsize=12)
    ax.set_title(
        f"GRPO Training Step: Phase Duration Breakdown "
        f"(avg over {max(phase_counts)} repeats, total: {total_ms/1000:.1f}s)",
        fontsize=13,
    )
    for bar, mean_ms, count in zip(bars, phase_means, phase_counts):
        pct = mean_ms / total_ms * 100
        label = f"{mean_ms:,.0f} ms ({pct:.1f}%)"
        if count > 1:
            label += f" ×{count}"
        ax.text(
            bar.get_width() + max(phase_means) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=9,
        )
    ax.set_xlim(0, max(phase_means) * 1.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_durations.png"), dpi=150)
    plt.close()
    print(f"  Saved phase_durations.png")

    # ===== Plot 2: Pie chart =====
    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax.pie(
        phase_means, labels=phase_names, autopct="%1.1f%%",
        colors=bar_colors, startangle=90, pctdistance=0.78,
    )
    for t in autotexts:
        t.set_fontsize(9)
    for t in texts:
        t.set_fontsize(10)
    ax.set_title(
        f"GRPO Step Time Breakdown (total: {total_ms/1000:.1f}s)",
        fontsize=13, pad=20,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_pie.png"), dpi=150)
    plt.close()
    print(f"  Saved phase_pie.png")

    # ===== Plot 3: GPU Timeline with utilization + phases + bubbles =====
    # Find time range from training phases only
    t_min = min(s for _, s, _, _ in training_phases)
    t_max = max(e for _, _, e, _ in training_phases)
    total_ns = t_max - t_min
    duration_ms = total_ns / 1e6

    # Query kernels only within the training window (avoids loading 20M+ rows)
    print(f"  Querying GPU kernels in [{t_min}, {t_max}] window "
          f"({duration_ms/1000:.1f}s)...")
    kernels = _query_kernels_in_window(conn, t_min, t_max)
    memops = _query_memops_in_window(conn, t_min, t_max)
    conn.close()
    print(f"  {len(kernels)} GPU kernels, {len(memops)} memops in window")

    # Build GPU utilization histogram (1ms bins)
    n_bins = max(int(duration_ms) + 1, 1)
    gpu_busy = np.zeros(n_bins, dtype=np.float64)

    if kernels:
        for _, k_start, k_end, _ in kernels:
            ks = max(k_start, t_min)
            ke = min(k_end, t_max)
            bin_s = int((ks - t_min) / 1e6)
            bin_e = int((ke - t_min) / 1e6)
            bin_s = max(0, min(bin_s, n_bins - 1))
            bin_e = max(0, min(bin_e, n_bins - 1))
            gpu_busy[bin_s:bin_e + 1] = 1.0

    # Smooth for readability
    window = max(1, n_bins // 200)
    if window > 1:
        smooth_kern = np.ones(window) / window
        gpu_smooth = np.convolve(gpu_busy, smooth_kern, mode="same")
    else:
        gpu_smooth = gpu_busy

    fig, axes = plt.subplots(
        3, 1, figsize=(20, 10),
        height_ratios=[1.5, 1, 1.5],
        gridspec_kw={"hspace": 0.3},
    )
    ax_util, ax_phases, ax_bubbles = axes

    time_axis = np.arange(n_bins)

    # Row 1: GPU utilization trace
    avg_util = np.mean(gpu_busy) * 100
    ax_util.fill_between(time_axis, gpu_smooth, alpha=0.7, color="#4285F4")
    ax_util.set_ylabel("GPU Util", fontsize=10)
    ax_util.set_ylim(0, 1.15)
    ax_util.set_xlim(0, n_bins)
    ax_util.set_title(
        f"GPU Utilization (avg: {avg_util:.1f}%, "
        f"{len(kernels)} kernels in {duration_ms/1000:.1f}s)",
        fontsize=12,
    )
    ax_util.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    # Row 2: NVTX phase timeline (Gantt chart)
    for name, start, end, dur in training_phases:
        x_start = (start - t_min) / 1e6
        width = dur / 1e6
        color = PHASE_COLORS.get(name, "#888888")
        ax_phases.barh(
            0, width, left=x_start, height=0.7,
            color=color, alpha=0.85, edgecolor="black", linewidth=0.3,
        )
    ax_phases.set_xlim(0, n_bins)
    ax_phases.set_yticks([])
    ax_phases.set_ylabel("Phases", fontsize=10)
    legend_patches = [
        Patch(facecolor=PHASE_COLORS.get(n, "#888"), label=n)
        for n in phase_names
    ]
    ax_phases.legend(
        handles=legend_patches, loc="upper right", ncol=4, fontsize=8,
    )

    # Row 3: Bubble detection — gaps between kernels
    if kernels:
        gap_starts = []
        gap_durations = []
        prev_end = t_min
        for _, k_start, k_end, _ in kernels:  # already sorted by start
            gap = k_start - prev_end
            if gap > 100_000:  # >0.1ms gaps
                gap_starts.append((prev_end - t_min) / 1e6)
                gap_durations.append(gap / 1e6)
            prev_end = max(prev_end, k_end)

        if gap_starts:
            ax_bubbles.bar(
                gap_starts, gap_durations, width=max(1, n_bins / 500),
                color="red", alpha=0.6, label="GPU idle gaps",
            )
        total_gap_ms = sum(gap_durations) if gap_durations else 0
        ax_bubbles.set_ylabel("Gap (ms)", fontsize=10)
        ax_bubbles.set_xlabel("Time (ms from start)", fontsize=10)
        ax_bubbles.set_xlim(0, n_bins)
        ax_bubbles.set_title(
            f"GPU Idle Bubbles (>{0.1}ms gaps, total: {total_gap_ms:.0f}ms = "
            f"{total_gap_ms/duration_ms*100:.1f}% of step)",
            fontsize=11,
        )
        if gap_starts:
            ax_bubbles.legend(fontsize=9)
    else:
        ax_bubbles.text(
            0.5, 0.5, "No GPU kernel data (run nsys with --trace=cuda)",
            transform=ax_bubbles.transAxes, ha="center", fontsize=12,
        )

    plt.savefig(os.path.join(output_dir, "gpu_timeline.png"), dpi=150)
    plt.close()
    print(f"  Saved gpu_timeline.png")

    # ===== Plot 4: Per-phase GPU utilization =====
    if kernels:
        # Build sorted arrays for efficient overlap calculation
        k_starts = np.array([k[1] for k in kernels], dtype=np.int64)
        k_ends = np.array([k[2] for k in kernels], dtype=np.int64)

        phase_gpu_util = {}
        for pname, p_start, p_end, p_dur in training_phases:
            # Binary search for kernels overlapping this phase
            i_start = np.searchsorted(k_ends, p_start, side='right')
            i_end = np.searchsorted(k_starts, p_end, side='left')
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
        bars = ax.barh(
            phase_names, util_means, color=bar_colors,
            edgecolor="black", linewidth=0.5,
        )
        ax.set_xlabel("GPU Utilization within Phase (%)", fontsize=12)
        ax.set_title("Per-Phase GPU Utilization (kernel time / phase wall time)", fontsize=13)
        for bar, util in zip(bars, util_means):
            ax.text(
                bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{util:.1f}%", va="center", fontsize=10,
            )
        ax.set_xlim(0, 105)
        ax.axvline(x=100, color="gray", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "phase_gpu_util.png"), dpi=150)
        plt.close()
        print(f"  Saved phase_gpu_util.png")

    # ===== Print summary =====
    print(f"\n  {'Phase':<20s} {'Mean (ms)':>10s} {'Std':>8s} "
          f"{'Count':>6s} {'% Step':>8s}")
    print(f"  {'-' * 56}")
    for name, mean, std, cnt in zip(phase_names, phase_means, phase_stds, phase_counts):
        pct = mean / total_ms * 100
        print(f"  {name:<20s} {mean:>10,.0f} {std:>8,.0f} {cnt:>6d} {pct:>7.1f}%")
    print(f"  {'TOTAL':<20s} {total_ms:>10,.0f}")

    if kernels:
        # Overall GPU utilization
        total_kernel_ns = sum(d for _, _, _, d in kernels)
        print(f"\n  GPU utilization: {avg_util:.1f}% "
              f"(kernel time: {total_kernel_ns/1e9:.2f}s / "
              f"wall time: {total_ns/1e9:.2f}s)")

        # Top 15 kernels by total time
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile GRPO training steps")
    parser.add_argument("--model_path", type=str,
                        help="Path to Qwen3 weights (safetensors dir)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to load (optional)")
    parser.add_argument("--model_size", type=str, default="0.6b")
    parser.add_argument("--num_steps", type=int, default=2,
                        help="Number of training steps to profile")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)

    # Analysis mode
    parser.add_argument("--analyze", type=str, default=None,
                        help="Path to nsys SQLite export for analysis")
    parser.add_argument("--output_dir", type=str, default="results/profile",
                        help="Output dir for plots")
    args = parser.parse_args()

    # Analysis mode
    if args.analyze:
        print(f"Analyzing profile: {args.analyze}")
        analyze_profile(args.analyze, args.output_dir)
        return

    # Profiling mode — need model_path
    if not args.model_path:
        parser.error("--model_path required for profiling mode")

    from dataclasses import dataclass

    @dataclass
    class ProfileConfig:
        group_size: int = args.group_size
        accumulation_steps: int = args.accumulation_steps
        max_new_tokens: int = args.max_new_tokens
        temperature: float = args.temperature
        lr: float = args.lr
        beta: float = args.beta
        clip_eps: float = 0.2
        grad_clip: float = 1.0
        weight_decay: float = 0.01

    cfg = ProfileConfig()

    print("=" * 60)
    print("GRPO Training Profiler")
    print("=" * 60)

    # Load model
    print("[1/4] Loading model...")
    model_configs = {
        "0.6b": Qwen3Config.qwen3_0_6b,
        "1.7b": Qwen3Config.qwen3_1_7b,
        "4b": Qwen3Config.qwen3_4b,
    }
    model_cfg = model_configs[args.model_size]()
    model = Qwen3Model(model_cfg)
    model.load_weights(args.model_path)

    # Reference model
    print("  Creating reference model...")
    ref_model = Qwen3Model(model_cfg)
    for name, param in model.params.items():
        ref_model.params[name] = param.copy()

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"  Loading checkpoint from {args.checkpoint}...")
        from train_grpo_gsm8k import load_checkpoint
        load_checkpoint(model, args.checkpoint)

    # Optimizer
    print("[2/4] Setting up optimizer...")
    from optim import AdamW, AdamWConfig
    optim_cfg = AdamWConfig(lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer = AdamW(model.params, optim_cfg)

    # Data + tokenizer
    print("[3/4] Loading data...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    from train_grpo_gsm8k import load_gsm8k
    train_data, _ = load_gsm8k()
    print(f"  Train: {len(train_data)} problems")

    rng = np.random.RandomState(42)

    # Warmup step (not profiled by NVTX but captured by nsys)
    print("[4/4] Warmup step...")
    with nvtx.annotate("warmup", color="gray"):
        _ = profiled_training_step(
            model, ref_model, optimizer, tokenizer, train_data, cfg, rng, step=-1,
        )

    # Synchronize before profiling
    if hasattr(cp, 'cuda'):
        cp.cuda.Device().synchronize()

    # Profiled steps
    print(f"\nProfiling {args.num_steps} training steps...")
    for step in range(args.num_steps):
        with nvtx.annotate(f"step_{step}", color="white"):
            metrics = profiled_training_step(
                model, ref_model, optimizer, tokenizer, train_data, cfg, rng, step=step,
            )
        print(f"  [step {step}] loss={metrics['loss']:.4f} "
              f"reward={metrics['reward']:.3f} kl={metrics['kl']:.4f} "
              f"gnorm={metrics['grad_norm']:.2f}")

    # Final sync
    if hasattr(cp, 'cuda'):
        cp.cuda.Device().synchronize()

    print("\nProfiling complete. Use nsys to analyze the trace.")


if __name__ == "__main__":
    main()
