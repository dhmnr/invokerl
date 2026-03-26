"""Profile invokerl GRPO training with NVTX annotations + roofline analysis.

Instruments each training phase with NVTX markers for Nsight Systems:
  - generation       (vLLM autoregressive sampling)
  - reward           (rule-based answer checking)
  - ref_forward      (reference model log-probs)
  - policy_forward   (policy model forward — log-probs with gradients)
  - loss_computation (GRPO clipped surrogate + KL penalty)
  - backward         (autograd backward pass)
  - optimizer_step   (AdamW + grad clipping + LR schedule)
  - weight_sync      (safetensors save + vLLM reload_weights)

Usage:
    # Capture trace with nsys (3 training steps):
    nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
        --force-overwrite true --output results/invokerl_profile \
        python -m invokerl.profile --config invokerl/configs/grpo_gsm8k.yaml \
            --num-steps 3

    # Export to SQLite for custom analysis:
    nsys export --type=sqlite results/invokerl_profile.nsys-rep

    # Generate plots + roofline:
    python -m invokerl.profile --analyze results/invokerl_profile.sqlite \
        --output-dir results/profile_plots

    # Quick Python-only timing (no nsys needed):
    python -m invokerl.profile --config invokerl/configs/grpo_gsm8k.yaml \
        --num-steps 3 --timing-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("invokerl.profile")

# Optional NVTX — gracefully degrade if not installed
try:
    import nvtx

    _has_nvtx = True
except ImportError:
    _has_nvtx = False

    class _FakeAnnotate:
        """No-op context manager when nvtx is not installed."""

        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class nvtx:  # type: ignore[no-redef]
        annotate = _FakeAnnotate


# ---------------------------------------------------------------------------
# Phase colors (consistent with tilerl profiler for visual continuity)
# ---------------------------------------------------------------------------

PHASE_COLORS = {
    "generation": "#4285F4",     # blue
    "reward": "#FBBC04",         # yellow
    "ref_forward": "#34A853",    # green
    "policy_forward": "#FF6D01", # orange
    "loss_computation": "#EA4335",  # red
    "backward": "#9334E6",       # purple
    "optimizer_step": "#00BCD4", # cyan
    "weight_sync": "#795548",    # brown
    "warmup": "#9E9E9E",         # gray
}

PHASE_ORDER = [
    "generation", "reward", "ref_forward", "policy_forward",
    "loss_computation", "backward", "optimizer_step", "weight_sync",
]


# ---------------------------------------------------------------------------
# RTX 5090 hardware specs for roofline model
# ---------------------------------------------------------------------------

@dataclass
class GPUSpecs:
    """Hardware specs for roofline model."""

    name: str = "RTX 5090"
    # Peak compute (TFLOPS)
    peak_bf16_tflops: float = 209.5   # BF16 tensor core
    peak_fp32_tflops: float = 104.8   # FP32 (non-tensor)
    peak_fp16_tflops: float = 209.5   # FP16 tensor core
    # Memory
    mem_bandwidth_tb_s: float = 1.79  # GDDR7 bandwidth
    vram_gb: float = 32.0
    # Derived
    @property
    def mem_bandwidth_gb_s(self) -> float:
        return self.mem_bandwidth_tb_s * 1000

    @property
    def ridge_point_bf16(self) -> float:
        """Arithmetic intensity (FLOP/byte) where compute = memory bound."""
        return self.peak_bf16_tflops * 1000 / self.mem_bandwidth_gb_s

    @property
    def ridge_point_fp32(self) -> float:
        return self.peak_fp32_tflops * 1000 / self.mem_bandwidth_gb_s


RTX_5090 = GPUSpecs()


# ---------------------------------------------------------------------------
# FLOP estimation for transformer operations
# ---------------------------------------------------------------------------

def estimate_flops_per_step(
    hidden_size: int = 1024,
    num_layers: int = 28,
    num_attention_heads: int = 16,
    num_kv_heads: int = 8,
    intermediate_size: int = 3072,
    vocab_size: int = 151936,
    seq_len: int = 512,
    batch_size: int = 4,
    group_size: int = 4,
    accumulation_steps: int = 4,
) -> dict[str, dict]:
    """Estimate FLOPs and memory traffic for each training phase.

    Returns dict mapping phase name → {flops, bytes, arithmetic_intensity, description}.
    All values are per optimizer step (summed over micro-batches).
    """
    B_gen = batch_size * group_size  # sequences during generation
    B_train = B_gen * accumulation_steps  # total sequences per optimizer step
    S = seq_len
    H = hidden_size
    L = num_layers
    V = vocab_size
    I = intermediate_size
    d = H // num_attention_heads  # head_dim

    # FLOPs for a single forward pass through the transformer
    # Per layer: QKV proj + attn + out proj + FFN
    qkv_flops = 2 * S * H * (H + 2 * (num_kv_heads * d))  # Q + K + V
    attn_flops = 2 * S * S * H  # Q @ K^T + attn @ V
    out_proj_flops = 2 * S * H * H
    ffn_flops = 2 * S * H * I * 3  # gate + up + down (SwiGLU)
    per_layer = qkv_flops + attn_flops + out_proj_flops + ffn_flops
    lm_head_flops = 2 * S * H * V  # logits projection

    fwd_flops = (L * per_layer + lm_head_flops)  # single sequence
    bwd_flops = 2 * fwd_flops  # backward ≈ 2x forward

    # Model parameters (approximate)
    params_per_layer = (
        H * (H + 2 * num_kv_heads * d)  # QKV
        + H * H                          # out proj
        + H * I * 3                      # FFN
        + H * 4                          # norms (small)
    )
    total_params = L * params_per_layer + V * H  # + embedding
    param_bytes_bf16 = total_params * 2
    param_bytes_fp32 = total_params * 4

    results = {}

    # Generation: B_gen sequences, S tokens each (autoregressive with KV cache)
    # vLLM batches all B_gen sequences; each decoding step loads model weights once.
    # Memory traffic: S steps * model_weights (batched over B_gen, not per-sequence)
    # Plus KV cache writes: S * B_gen * L * 2 * num_kv_heads * d * 2
    gen_flops = B_gen * fwd_flops  # FLOPs correct: linear ops scale with B
    gen_kv_bytes = S * B_gen * L * 2 * num_kv_heads * d * 2  # KV cache writes
    gen_bytes = S * param_bytes_bf16 + gen_kv_bytes  # S weight loads + KV writes
    results["generation"] = {
        "flops": gen_flops,
        "bytes": gen_bytes,
        "description": f"{B_gen} sequences x {S} tokens (autoregressive, memory-bound)",
    }

    # Reward: CPU-bound string matching, negligible GPU
    results["reward"] = {
        "flops": 0,
        "bytes": 0,
        "description": "CPU-only rule-based matching",
    }

    # Reference forward: K micro-batches of B_gen sequences each (parallel)
    # Each micro-batch loads model weights once; activations scale with batch
    K = accumulation_steps
    ref_fwd_flops = B_train * fwd_flops
    ref_fwd_bytes = K * param_bytes_bf16 + B_train * S * H * 2  # K weight loads + activations
    results["ref_forward"] = {
        "flops": ref_fwd_flops,
        "bytes": ref_fwd_bytes,
        "description": f"{B_train} sequences ({K} micro-batches) forward (bf16, no grad)",
    }

    # Policy forward: same structure but with gradient checkpointing overhead
    pol_fwd_flops = B_train * fwd_flops
    pol_fwd_bytes = K * param_bytes_bf16 + B_train * S * H * 2
    results["policy_forward"] = {
        "flops": pol_fwd_flops,
        "bytes": pol_fwd_bytes,
        "description": f"{B_train} sequences forward with grad (bf16 autocast)",
    }

    # Loss: relatively cheap — ratios, clipping, KL on [B_train, S] tensors
    loss_flops = B_train * S * 20  # ~20 ops per token
    loss_bytes = B_train * S * 4 * 6  # ~6 tensors of float32
    results["loss_computation"] = {
        "flops": loss_flops,
        "bytes": loss_bytes,
        "description": "Clipped surrogate + KL penalty",
    }

    # Backward: ~2x forward FLOPs (with gradient checkpointing: recompute fwd activations)
    # K micro-batches, each loads model weights + gradients + recomputed activations
    bwd_total_flops = B_train * bwd_flops
    bwd_bytes = K * (param_bytes_bf16 + param_bytes_fp32) + B_train * S * H * 4  # weights + grads + activations
    results["backward"] = {
        "flops": bwd_total_flops,
        "bytes": bwd_bytes,
        "description": f"{B_train} sequences backward (gradient checkpointing)",
    }

    # Optimizer: AdamW reads/writes params + m + v (3 tensors per param)
    opt_flops = total_params * 10  # ~10 ops per param (m, v update, decay, step)
    opt_bytes = param_bytes_fp32 * 3  # params + m + v (all fp32 if master weights)
    results["optimizer_step"] = {
        "flops": opt_flops,
        "bytes": opt_bytes,
        "description": "AdamW update + grad clipping",
    }

    # Weight sync: save bf16 weights to disk + vLLM reload
    results["weight_sync"] = {
        "flops": 0,
        "bytes": param_bytes_bf16 * 2,  # write + read
        "description": "safetensors save → disk → vLLM reload_weights",
    }

    # Add arithmetic intensity to all phases
    for phase in results.values():
        if phase["bytes"] > 0:
            phase["arithmetic_intensity"] = phase["flops"] / phase["bytes"]
        else:
            phase["arithmetic_intensity"] = 0.0

    return results


# ---------------------------------------------------------------------------
# Profiled training step
# ---------------------------------------------------------------------------

def profiled_training_step(trainer, step: int) -> tuple[dict, dict[str, float]]:
    """Run one training step with NVTX markers and timing on each phase.

    Returns:
        (metrics, phase_times) where phase_times maps phase name → wall seconds.
    """
    cfg = trainer.config
    phase_times: dict[str, float] = {}

    accumulated_metrics: dict[str, list[float]] = {}
    trainer.optimizer.zero_grad()

    for micro in range(cfg.accumulation_steps):
        # Sample prompts
        prompts = trainer.dataset.sample(cfg.batch_size)

        # --- Generation (vLLM) ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx.annotate("generation", color="blue"):
            # Expand prompts: each prompt gets G completions
            expanded_prompts = [p.prompt for p in prompts for _ in range(cfg.group_size)]
            expanded_truths = [p.ground_truth for p in prompts for _ in range(cfg.group_size)]
            gen_out = trainer.generator.generate(expanded_prompts, trainer.gen_config)
        torch.cuda.synchronize()
        phase_times["generation"] = phase_times.get("generation", 0) + (time.perf_counter() - t0)

        # --- Reward ---
        t0 = time.perf_counter()
        with nvtx.annotate("reward", color="yellow"):
            rewards = trainer.reward_fn.score_batch(
                expanded_prompts, gen_out.texts, ground_truths=expanded_truths,
            )
            rewards = rewards.to(gen_out.token_ids.device)
        phase_times["reward"] = phase_times.get("reward", 0) + (time.perf_counter() - t0)

        # Build batch (need group_ids)
        from invokerl.algorithms.base import RolloutBatch
        B = len(prompts)
        G = cfg.group_size
        group_ids = torch.arange(B, device=gen_out.token_ids.device).repeat_interleave(G)

        # --- Reference forward ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx.annotate("ref_forward", color="green"):
            if trainer.ref_policy is not None:
                ref_log_probs = trainer.ref_policy.forward_no_grad(
                    gen_out.token_ids, gen_out.attention_mask,
                )
            else:
                ref_log_probs = gen_out.log_probs.clone()
        torch.cuda.synchronize()
        phase_times["ref_forward"] = phase_times.get("ref_forward", 0) + (time.perf_counter() - t0)

        batch = RolloutBatch(
            token_ids=gen_out.token_ids,
            prompt_mask=gen_out.prompt_mask,
            response_mask=gen_out.response_mask,
            attention_mask=gen_out.attention_mask,
            rewards=rewards,
            token_rewards=None,
            old_log_probs=gen_out.log_probs,
            ref_log_probs=ref_log_probs,
            group_ids=group_ids,
            group_size=G,
        )
        batch = trainer._batch_to_device(batch)

        # --- Advantages ---
        advantages = trainer.algorithm.compute_advantages(batch)

        # --- Policy forward ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx.annotate("policy_forward", color="orange"):
            new_log_probs = trainer.policy.forward(
                batch.token_ids, batch.attention_mask,
            )
        torch.cuda.synchronize()
        phase_times["policy_forward"] = phase_times.get("policy_forward", 0) + (time.perf_counter() - t0)

        # --- Loss computation ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx.annotate("loss_computation", color="red"):
            loss, metrics = trainer.algorithm.compute_loss(
                new_log_probs, batch, advantages,
            )
            scaled_loss = loss / cfg.accumulation_steps
        torch.cuda.synchronize()
        phase_times["loss_computation"] = phase_times.get("loss_computation", 0) + (time.perf_counter() - t0)

        # --- Backward ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx.annotate("backward", color="purple"):
            scaled_loss.backward()
        torch.cuda.synchronize()
        phase_times["backward"] = phase_times.get("backward", 0) + (time.perf_counter() - t0)

        # Accumulate metrics
        for k, v in metrics.items():
            accumulated_metrics.setdefault(k, []).append(v)

        del batch, loss, scaled_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Optimizer step ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with nvtx.annotate("optimizer_step", color="cyan"):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainer.policy.parameters(), cfg.grad_clip,
        )
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.optimizer.zero_grad()
    torch.cuda.synchronize()
    phase_times["optimizer_step"] = time.perf_counter() - t0

    # --- Weight sync ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with nvtx.annotate("weight_sync", color="brown"):
        trainer.generator.update_weights(trainer.policy.get_state_dict())
    torch.cuda.synchronize()
    phase_times["weight_sync"] = time.perf_counter() - t0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Aggregate metrics
    step_metrics = {k: sum(v) / len(v) for k, v in accumulated_metrics.items()}
    step_metrics["grad_norm"] = float(grad_norm)

    return step_metrics, phase_times


# ---------------------------------------------------------------------------
# Analysis: nsys SQLite → plots + roofline
# ---------------------------------------------------------------------------

def _query_nvtx(conn) -> list[tuple]:
    """Query NVTX events from nsys SQLite, handling schema variations."""
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
    """Query GPU kernels overlapping [t_min, t_max] window."""
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


def analyze_profile(sqlite_path: str, output_dir: str, gpu: GPUSpecs = RTX_5090):
    """Parse nsys SQLite export → phase breakdown plots + roofline model.

    Generates:
      1. Phase duration bar chart (horizontal bars with ms + %)
      2. Pie chart of step time allocation
      3. GPU timeline: utilization trace + phase Gantt + idle bubbles
      4. Per-phase GPU utilization breakdown
      5. Roofline model plot (arithmetic intensity vs throughput)
      6. Top 15 GPU kernels table
      7. JSON summary
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

    # Filter to training phases
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

    # Aggregate by phase
    phase_durations: dict[str, list[float]] = {}
    for name, start, end, dur in training_phases:
        phase_durations.setdefault(name, []).append(dur / 1e6)  # ms

    phase_names = sorted(
        phase_durations.keys(),
        key=lambda n: PHASE_ORDER.index(n) if n in PHASE_ORDER else 99,
    )
    phase_means = [np.mean(phase_durations[n]) for n in phase_names]
    phase_stds = [np.std(phase_durations[n]) for n in phase_names]
    phase_counts = [len(phase_durations[n]) for n in phase_names]
    bar_colors = [PHASE_COLORS.get(n, "#888888") for n in phase_names]
    total_ms = sum(phase_means)

    # ===== Plot 1: Phase duration bar chart =====
    fig, ax = plt.subplots(figsize=(14, max(6, len(phase_names) * 0.8)))
    bars = ax.barh(
        phase_names, phase_means, xerr=phase_stds, color=bar_colors,
        edgecolor="black", linewidth=0.5, capsize=3,
    )
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
        ax.text(
            bar.get_width() + max(phase_means) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=9,
        )
    ax.set_xlim(0, max(phase_means) * 1.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_durations.png"), dpi=150)
    plt.close()
    print("  Saved phase_durations.png")

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
        f"GRPO Step Time Breakdown (total: {total_ms / 1000:.1f}s)",
        fontsize=13, pad=20,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_pie.png"), dpi=150)
    plt.close()
    print("  Saved phase_pie.png")

    # ===== Plot 3: GPU Timeline =====
    t_min = min(s for _, s, _, _ in training_phases)
    t_max = max(e for _, _, e, _ in training_phases)
    total_ns = t_max - t_min
    duration_ms = total_ns / 1e6

    print(f"  Querying GPU kernels in [{t_min}, {t_max}] "
          f"({duration_ms / 1000:.1f}s)...")
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
    if window > 1:
        gpu_smooth = np.convolve(gpu_busy, np.ones(window) / window, mode="same")
    else:
        gpu_smooth = gpu_busy

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
    ax_phases.legend(handles=legend_patches, loc="upper right", ncol=4, fontsize=8)

    if kernels:
        gap_starts = []
        gap_durations = []
        prev_end = t_min
        for _, k_start, k_end, _ in kernels:
            gap = k_start - prev_end
            if gap > 100_000:  # >0.1ms
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
            f"GPU Idle Bubbles (>0.1ms gaps, total: {total_gap_ms:.0f}ms = "
            f"{total_gap_ms / duration_ms * 100:.1f}% of step)",
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
    print("  Saved gpu_timeline.png")

    # ===== Plot 4: Per-phase GPU utilization =====
    if kernels:
        k_starts = np.array([k[1] for k in kernels], dtype=np.int64)
        k_ends = np.array([k[2] for k in kernels], dtype=np.int64)

        phase_gpu_util = {}
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
        bars = ax.barh(
            phase_names, util_means, color=bar_colors,
            edgecolor="black", linewidth=0.5,
        )
        ax.set_xlabel("GPU Utilization within Phase (%)", fontsize=12)
        ax.set_title("Per-Phase GPU Utilization (kernel time / phase wall time)",
                      fontsize=13)
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
        print("  Saved phase_gpu_util.png")

    # ===== Plot 5: Roofline Model =====
    plot_roofline(phase_means, phase_names, output_dir, gpu)

    # ===== Print summary =====
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

    # ===== Save JSON summary =====
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


def plot_roofline(
    phase_means: list[float],
    phase_names: list[str],
    output_dir: str,
    gpu: GPUSpecs = RTX_5090,
):
    """Plot roofline model with estimated phase positions.

    The roofline shows peak attainable GFLOP/s as a function of arithmetic
    intensity (FLOP/byte). Phases below the roof are bottlenecked by either
    memory bandwidth (left of ridge) or compute (right of ridge).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Estimate FLOPs for each phase
    flop_est = estimate_flops_per_step()

    fig, ax = plt.subplots(figsize=(14, 9))

    # Roofline lines
    x_range = np.logspace(-1, 4, 500)  # FLOP/byte

    # BF16 roofline
    bf16_roof = np.minimum(
        gpu.peak_bf16_tflops * 1000,  # GFLOP/s peak
        x_range * gpu.mem_bandwidth_gb_s,  # memory-bound slope
    )
    ax.plot(x_range, bf16_roof, "b-", linewidth=2, label=f"BF16 peak ({gpu.peak_bf16_tflops:.0f} TFLOPS)")

    # FP32 roofline
    fp32_roof = np.minimum(
        gpu.peak_fp32_tflops * 1000,
        x_range * gpu.mem_bandwidth_gb_s,
    )
    ax.plot(x_range, fp32_roof, "r--", linewidth=1.5, label=f"FP32 peak ({gpu.peak_fp32_tflops:.0f} TFLOPS)")

    # Memory bandwidth line
    mem_line = x_range * gpu.mem_bandwidth_gb_s
    ax.plot(x_range, mem_line, "k:", linewidth=1, alpha=0.3, label=f"Mem BW ({gpu.mem_bandwidth_tb_s:.2f} TB/s)")

    # Ridge points
    ax.axvline(x=gpu.ridge_point_bf16, color="blue", linestyle=":", alpha=0.3)
    ax.axvline(x=gpu.ridge_point_fp32, color="red", linestyle=":", alpha=0.3)

    # Plot each phase
    for i, name in enumerate(phase_names):
        if name not in flop_est or name == "reward":
            continue

        est = flop_est[name]
        ai = est["arithmetic_intensity"]
        if ai <= 0:
            continue

        # Estimate achieved throughput from timing
        idx = phase_names.index(name)
        wall_s = phase_means[idx] / 1000.0  # ms → s
        if wall_s > 0:
            achieved_gflops = est["flops"] / 1e9 / wall_s
        else:
            achieved_gflops = 0

        color = PHASE_COLORS.get(name, "#888888")
        ax.scatter(
            [ai], [achieved_gflops],
            s=150, c=color, edgecolors="black", linewidths=1,
            zorder=5, label=f"{name} (AI={ai:.1f})",
        )
        ax.annotate(
            name, (ai, achieved_gflops),
            textcoords="offset points", xytext=(8, 8),
            fontsize=8, color=color, fontweight="bold",
        )

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


# ---------------------------------------------------------------------------
# Python-only timing (no nsys dependency)
# ---------------------------------------------------------------------------

def timing_report(all_phase_times: list[dict[str, float]], output_dir: str):
    """Generate timing summary from Python-measured phase times."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Average over steps (skip first = warmup)
    if len(all_phase_times) > 1:
        steps = all_phase_times[1:]  # skip warmup
    else:
        steps = all_phase_times

    phase_names = list(PHASE_ORDER)
    # Filter to phases that actually appear
    phase_names = [n for n in phase_names if any(n in s for s in steps)]

    phase_means = []
    phase_stds = []
    for name in phase_names:
        vals = [s.get(name, 0) * 1000 for s in steps]  # s → ms
        phase_means.append(np.mean(vals))
        phase_stds.append(np.std(vals))

    total_ms = sum(phase_means)
    bar_colors = [PHASE_COLORS.get(n, "#888888") for n in phase_names]

    # Bar chart
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

    # Pie chart
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pie(
        phase_means, labels=phase_names, autopct="%1.1f%%",
        colors=bar_colors, startangle=90, pctdistance=0.78,
    )
    ax.set_title(f"GRPO Step Time Allocation (total: {total_ms / 1000:.1f}s)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timing_pie.png"), dpi=150)
    plt.close()

    # Roofline (estimated)
    plot_roofline(phase_means, phase_names, output_dir)

    # Text summary
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

    # FLOP estimates
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

    # MFU estimate
    total_wall_s = total_ms / 1000.0
    if total_wall_s > 0 and total_flops > 0:
        mfu = total_flops / 1e12 / total_wall_s / RTX_5090.peak_bf16_tflops * 100
        print(f"\n  Model FLOP Utilization (MFU): {mfu:.1f}%")
        print(f"  (total FLOPs / wall time / peak BF16 TFLOPS)")

    # Bottleneck analysis
    print(f"\n  Bottleneck Analysis:")
    if phase_means:
        top_phase = phase_names[np.argmax(phase_means)]
        top_pct = max(phase_means) / total_ms * 100
        print(f"    Primary bottleneck: {top_phase} ({top_pct:.1f}% of step)")
        if top_phase == "generation":
            print(f"    -> Autoregressive generation is inherently sequential and memory-bound.")
            print(f"    -> Optimize: larger batch, shorter sequences, or disaggregate to 2nd GPU.")
        elif top_phase in ("backward", "policy_forward"):
            print(f"    -> Compute-bound phase. Optimize: mixed precision, kernel fusion.")
        elif top_phase == "weight_sync":
            print(f"    -> I/O-bound. Optimize: direct tensor copy, skip disk, async sync.")

    # Save JSON
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile invokerl GRPO training step",
    )
    parser.add_argument(
        "--config", type=str,
        help="YAML config file (for profiling mode)",
    )
    parser.add_argument(
        "--num-steps", type=int, default=3,
        help="Number of training steps to profile",
    )
    parser.add_argument(
        "--timing-only", action="store_true",
        help="Python-only timing (no nsys needed)",
    )
    parser.add_argument(
        "--no-ref", action="store_true",
        help="Skip reference model (saves memory)",
    )
    parser.add_argument(
        "--analyze", type=str, default=None,
        help="Path to nsys SQLite export for analysis",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/profile",
        help="Output directory for plots and summaries",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Analysis mode ---
    if args.analyze:
        print(f"Analyzing profile: {args.analyze}")
        analyze_profile(args.analyze, args.output_dir)
        return

    # --- Profiling mode ---
    if not args.config:
        parser.error("--config required for profiling mode (or use --analyze)")

    from invokerl.train import (
        build_algorithm,
        build_dataset,
        build_generator,
        build_policy,
        build_ref_policy,
        build_reward,
        build_trainer_config,
        load_config,
    )
    from invokerl.engine.trainer import Trainer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    cfg = load_config(args.config)
    trainer_config = build_trainer_config(cfg)

    print("=" * 70)
    print("  invokerl GRPO Training Profiler")
    print("=" * 70)
    print(f"  Model: {trainer_config.model_name_or_path}")
    print(f"  Steps: {args.num_steps} (+ 1 warmup)")
    print(f"  Accum: {trainer_config.accumulation_steps}, "
          f"Group: {trainer_config.group_size}, "
          f"Batch: {trainer_config.batch_size}")
    print(f"  Mode: {'timing-only (Python)' if args.timing_only else 'NVTX (use with nsys)'}")
    print()

    # Build components
    print("[1/6] Loading algorithm...")
    algorithm = build_algorithm(cfg)

    print("[2/6] Loading dataset...")
    train_dataset = build_dataset(cfg, split="train")
    print(f"  {len(train_dataset)} training items")

    print("[3/6] Loading reward function...")
    reward_fn = build_reward(cfg)

    print("[4/6] Initializing vLLM generator...")
    generator = build_generator(cfg, trainer_config)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("[5/6] Loading policy model...")
    policy = build_policy(cfg)

    print("[6/6] Loading reference model...")
    if args.no_ref:
        ref_policy = None
        print("  Skipped (--no-ref)")
    else:
        ref_policy = build_ref_policy(cfg)
        if ref_policy is None:
            print("  Not needed for this algorithm")

    # Build trainer (no eval dataset needed for profiling)
    trainer = Trainer(
        config=trainer_config,
        algorithm=algorithm,
        generator=generator,
        policy=policy,
        ref_policy=ref_policy,
        reward_fn=reward_fn,
        dataset=train_dataset,
        eval_dataset=None,
    )

    # Warmup step
    print("\n[warmup] Running warmup step (not profiled)...")
    with nvtx.annotate("warmup", color="gray"):
        _, warmup_times = profiled_training_step(trainer, step=-1)
    warmup_total = sum(warmup_times.values())
    print(f"  Warmup: {warmup_total:.1f}s")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Profiled steps
    all_phase_times: list[dict[str, float]] = []
    print(f"\nProfiling {args.num_steps} training steps...")

    for step in range(args.num_steps):
        with nvtx.annotate(f"step_{step}", color="white"):
            metrics, phase_times = profiled_training_step(trainer, step)

        all_phase_times.append(phase_times)
        step_total = sum(phase_times.values())
        print(
            f"  [step {step}] loss={metrics.get('loss', 0):.4f} "
            f"reward={metrics.get('reward', 0):.3f} "
            f"kl={metrics.get('kl', 0):.4f} "
            f"gnorm={metrics.get('grad_norm', 0):.2f} "
            f"total={step_total:.1f}s"
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Generate timing report
    if args.timing_only:
        timing_report(all_phase_times, args.output_dir)
    else:
        # Still produce Python timing as a complement to nsys
        timing_report(all_phase_times, args.output_dir)
        print("\n  NVTX markers captured. Analyze with:")
        print(f"    nsys export --type=sqlite <your_profile>.nsys-rep")
        print(f"    python -m invokerl.profile --analyze <your_profile>.sqlite")


if __name__ == "__main__":
    main()
