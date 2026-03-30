"""Profile invokerl training with NVTX annotations and phase timing.

Instruments each training phase with NVTX markers for Nsight Systems and
produces a JSON timing summary with FLOP estimates and MFU.

Usage:
    # Quick Python-only timing (no nsys needed):
    python scripts/profile.py --config invokerl/configs/grpo_gsm8k.yaml --num-steps 3

    # With nsys trace capture:
    nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
        --force-overwrite true --output results/profile \
        python scripts/profile.py --config invokerl/configs/grpo_gsm8k.yaml --num-steps 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger("invokerl.profile")

# Optional NVTX -- gracefully degrade if not installed
try:
    import nvtx
    _has_nvtx = True
except ImportError:
    _has_nvtx = False

    class _FakeAnnotate:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class nvtx:  # type: ignore[no-redef]
        annotate = _FakeAnnotate


PHASE_ORDER = [
    "generation", "reward", "ref_forward", "policy_forward",
    "loss_computation", "backward", "optimizer_step", "weight_sync",
]


# ---------------------------------------------------------------------------
# GPU specs for roofline / MFU estimation
# ---------------------------------------------------------------------------

@dataclass
class GPUSpecs:
    name: str = "RTX 5090"
    peak_bf16_tflops: float = 209.5
    peak_fp32_tflops: float = 104.8
    mem_bandwidth_tb_s: float = 1.79
    vram_gb: float = 32.0

    @property
    def mem_bandwidth_gb_s(self) -> float:
        return self.mem_bandwidth_tb_s * 1000

RTX_5090 = GPUSpecs()


# ---------------------------------------------------------------------------
# FLOP estimation
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

    Returns dict mapping phase name -> {flops, bytes, arithmetic_intensity}.
    """
    B_gen = batch_size * group_size
    B_train = B_gen * accumulation_steps
    S, H, L, V, I = seq_len, hidden_size, num_layers, vocab_size, intermediate_size
    d = H // num_attention_heads
    K = accumulation_steps

    # Per-layer FLOPs (single sequence, single layer)
    qkv_flops = 2 * S * H * (H + 2 * num_kv_heads * d)
    attn_flops = 2 * S * S * H
    out_proj_flops = 2 * S * H * H
    ffn_flops = 2 * S * H * I * 3  # SwiGLU: gate + up + down
    per_layer = qkv_flops + attn_flops + out_proj_flops + ffn_flops
    lm_head_flops = 2 * S * H * V

    fwd_flops = L * per_layer + lm_head_flops
    bwd_flops = 2 * fwd_flops

    # Model size
    params_per_layer = H * (H + 2 * num_kv_heads * d) + H * H + H * I * 3 + H * 4
    total_params = L * params_per_layer + V * H
    param_bytes_bf16 = total_params * 2
    param_bytes_fp32 = total_params * 4

    results = {
        "generation": {
            "flops": B_gen * fwd_flops,
            "bytes": S * param_bytes_bf16 + S * B_gen * L * 2 * num_kv_heads * d * 2,
        },
        "reward": {"flops": 0, "bytes": 0},
        "ref_forward": {
            "flops": B_train * fwd_flops,
            "bytes": K * param_bytes_bf16 + B_train * S * H * 2,
        },
        "policy_forward": {
            "flops": B_train * fwd_flops,
            "bytes": K * param_bytes_bf16 + B_train * S * H * 2,
        },
        "loss_computation": {
            "flops": B_train * S * 20,
            "bytes": B_train * S * 4 * 6,
        },
        "backward": {
            "flops": B_train * bwd_flops,
            "bytes": K * (param_bytes_bf16 + param_bytes_fp32) + B_train * S * H * 4,
        },
        "optimizer_step": {
            "flops": total_params * 10,
            "bytes": param_bytes_fp32 * 3,
        },
        "weight_sync": {
            "flops": 0,
            "bytes": param_bytes_bf16 * 2,
        },
    }

    for phase in results.values():
        phase["arithmetic_intensity"] = (
            phase["flops"] / phase["bytes"] if phase["bytes"] > 0 else 0.0
        )

    return results


# ---------------------------------------------------------------------------
# Profiled training step
# ---------------------------------------------------------------------------

def profiled_training_step(trainer, step: int) -> tuple[dict, dict[str, float]]:
    """Run one training step with NVTX markers and per-phase timing.

    Returns (metrics, phase_times) where phase_times maps phase -> wall seconds.
    """
    cfg = trainer.config
    phase_times: dict[str, float] = {}
    accumulated_metrics: dict[str, list[float]] = {}
    trainer.optimizer.zero_grad()

    for micro in range(cfg.accumulation_steps):
        prompts = trainer.dataset.sample(cfg.batch_size)

        # Generation
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx.annotate("generation", color="blue"):
            expanded_prompts = [p.prompt for p in prompts for _ in range(cfg.group_size)]
            expanded_truths = [p.ground_truth for p in prompts for _ in range(cfg.group_size)]
            gen_out = trainer.generator.generate(expanded_prompts, trainer.gen_config)
        torch.cuda.synchronize()
        phase_times["generation"] = phase_times.get("generation", 0) + (time.perf_counter() - t0)

        # Reward
        t0 = time.perf_counter()
        with nvtx.annotate("reward", color="yellow"):
            rewards = trainer.reward_fn.score_batch(
                expanded_prompts, gen_out.texts, ground_truths=expanded_truths,
            ).to(gen_out.token_ids.device)
        phase_times["reward"] = phase_times.get("reward", 0) + (time.perf_counter() - t0)

        # Build batch
        from invokerl.algorithms.base import RolloutBatch
        B, G = len(prompts), cfg.group_size
        group_ids = torch.arange(B, device=gen_out.token_ids.device).repeat_interleave(G)

        # Ref forward
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
            token_ids=gen_out.token_ids, prompt_mask=gen_out.prompt_mask,
            response_mask=gen_out.response_mask, attention_mask=gen_out.attention_mask,
            rewards=rewards, token_rewards=None,
            old_log_probs=gen_out.log_probs, ref_log_probs=ref_log_probs,
            group_ids=group_ids, group_size=G,
        )
        batch = trainer._batch_to_device(batch)
        advantages = trainer.algorithm.compute_advantages(batch)

        # Policy forward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx.annotate("policy_forward", color="orange"):
            new_log_probs = trainer.policy.forward(batch.token_ids, batch.attention_mask)
        torch.cuda.synchronize()
        phase_times["policy_forward"] = phase_times.get("policy_forward", 0) + (time.perf_counter() - t0)

        # Loss
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx.annotate("loss_computation", color="red"):
            loss, metrics = trainer.algorithm.compute_loss(new_log_probs, batch, advantages)
            scaled_loss = loss / cfg.accumulation_steps
        torch.cuda.synchronize()
        phase_times["loss_computation"] = phase_times.get("loss_computation", 0) + (time.perf_counter() - t0)

        # Backward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx.annotate("backward", color="purple"):
            scaled_loss.backward()
        torch.cuda.synchronize()
        phase_times["backward"] = phase_times.get("backward", 0) + (time.perf_counter() - t0)

        for k, v in metrics.items():
            accumulated_metrics.setdefault(k, []).append(v)
        del batch, loss, scaled_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Optimizer step
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with nvtx.annotate("optimizer_step", color="cyan"):
        grad_norm = torch.nn.utils.clip_grad_norm_(trainer.policy.parameters(), cfg.grad_clip)
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.optimizer.zero_grad()
    torch.cuda.synchronize()
    phase_times["optimizer_step"] = time.perf_counter() - t0

    # Weight sync
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with nvtx.annotate("weight_sync", color="brown"):
        trainer.generator.update_weights(trainer.policy.get_state_dict())
    torch.cuda.synchronize()
    phase_times["weight_sync"] = time.perf_counter() - t0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    step_metrics = {k: sum(v) / len(v) for k, v in accumulated_metrics.items()}
    step_metrics["grad_norm"] = float(grad_norm)
    return step_metrics, phase_times


# ---------------------------------------------------------------------------
# Timing report
# ---------------------------------------------------------------------------

def timing_report(
    all_phase_times: list[dict[str, float]],
    output_dir: str,
    gpu: GPUSpecs = RTX_5090,
):
    """Generate timing summary from per-phase measurements."""
    os.makedirs(output_dir, exist_ok=True)

    # Skip warmup step
    steps = all_phase_times[1:] if len(all_phase_times) > 1 else all_phase_times
    phase_names = [n for n in PHASE_ORDER if any(n in s for s in steps)]

    phase_means = [np.mean([s.get(n, 0) * 1000 for s in steps]) for n in phase_names]
    phase_stds = [np.std([s.get(n, 0) * 1000 for s in steps]) for n in phase_names]
    total_ms = sum(phase_means)

    # Text summary
    print(f"\n{'=' * 70}")
    print(f"  GRPO Training Step Profile ({gpu.name})")
    print(f"{'=' * 70}")
    print(f"\n  {'Phase':<20s} {'Mean (ms)':>10s} {'Std':>8s} {'% Step':>8s}")
    print(f"  {'-' * 50}")
    for name, mean, std in zip(phase_names, phase_means, phase_stds):
        pct = mean / total_ms * 100 if total_ms > 0 else 0
        print(f"  {name:<20s} {mean:>10,.0f} {std:>8,.0f} {pct:>7.1f}%")
    print(f"  {'TOTAL':<20s} {total_ms:>10,.0f}")
    print(f"  {'STEPS/HOUR':>28s} {3600 / (total_ms / 1000):.0f}" if total_ms > 0 else "")

    # FLOP estimates + MFU
    flop_est = estimate_flops_per_step()
    total_flops = 0
    print(f"\n  FLOPs per step:")
    for name in phase_names:
        if name in flop_est and flop_est[name]["flops"] > 0:
            flops = flop_est[name]["flops"]
            total_flops += flops
            idx = phase_names.index(name)
            wall_s = phase_means[idx] / 1000.0
            if wall_s > 0:
                tflops = flops / 1e12 / wall_s
                print(f"    {name:<20s} {flops / 1e12:>8.2f} TFLOP  "
                      f"({tflops:>6.1f} TFLOP/s = "
                      f"{tflops / gpu.peak_bf16_tflops * 100:>4.1f}% peak)")
    print(f"    {'TOTAL':<20s} {total_flops / 1e12:>8.2f} TFLOP")

    total_wall_s = total_ms / 1000.0
    if total_wall_s > 0 and total_flops > 0:
        mfu = total_flops / 1e12 / total_wall_s / gpu.peak_bf16_tflops * 100
        print(f"\n  MFU: {mfu:.1f}%")

    # Save JSON
    summary = {
        "gpu": gpu.name,
        "total_step_ms": total_ms,
        "steps_per_hour": 3600 / (total_ms / 1000) if total_ms > 0 else 0,
        "phases": {
            name: {"mean_ms": mean, "std_ms": std, "pct": mean / total_ms * 100}
            for name, mean, std in zip(phase_names, phase_means, phase_stds)
        },
        "total_tflops": total_flops / 1e12,
        "mfu_pct": (total_flops / 1e12 / total_wall_s / gpu.peak_bf16_tflops * 100)
        if total_wall_s > 0 and total_flops > 0 else 0,
    }
    with open(os.path.join(output_dir, "profile_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved profile_summary.json to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile invokerl training")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--num-steps", type=int, default=3, help="Steps to profile")
    parser.add_argument("--no-ref", action="store_true", help="Skip reference model")
    parser.add_argument("--output-dir", type=str, default="results/profile", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from invokerl.train import (
        build_algorithm, build_dataset, build_generator,
        build_policy, build_ref_policy, build_reward,
        build_trainer_config, load_config,
    )
    from invokerl.engine.trainer import Trainer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    trainer_config = build_trainer_config(cfg)

    print(f"{'=' * 60}")
    print(f"  invokerl Profiler")
    print(f"{'=' * 60}")
    print(f"  Model: {trainer_config.model_name_or_path}")
    print(f"  Steps: {args.num_steps} (+ 1 warmup)")
    print(f"  Accum: {trainer_config.accumulation_steps}, "
          f"Group: {trainer_config.group_size}, "
          f"Batch: {trainer_config.batch_size}")
    print()

    algorithm = build_algorithm(cfg)
    train_dataset = build_dataset(cfg, split="train")
    reward_fn = build_reward(cfg)
    generator = build_generator(cfg, trainer_config)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    policy = build_policy(cfg)
    ref_policy = None if args.no_ref else build_ref_policy(cfg)

    trainer = Trainer(
        config=trainer_config, algorithm=algorithm, generator=generator,
        policy=policy, ref_policy=ref_policy, reward_fn=reward_fn,
        dataset=train_dataset, eval_dataset=None,
    )

    # Warmup
    print("[warmup] Running warmup step...")
    with nvtx.annotate("warmup", color="gray"):
        _, warmup_times = profiled_training_step(trainer, step=-1)
    print(f"  Warmup: {sum(warmup_times.values()):.1f}s")
    torch.cuda.synchronize()

    # Profile
    all_phase_times: list[dict[str, float]] = []
    print(f"\nProfiling {args.num_steps} steps...")
    for step in range(args.num_steps):
        with nvtx.annotate(f"step_{step}", color="white"):
            metrics, phase_times = profiled_training_step(trainer, step)
        all_phase_times.append(phase_times)
        total = sum(phase_times.values())
        print(f"  [step {step}] loss={metrics.get('loss', 0):.4f} "
              f"reward={metrics.get('reward', 0):.3f} "
              f"kl={metrics.get('kl', 0):.4f} total={total:.1f}s")

    torch.cuda.synchronize()
    timing_report(all_phase_times, args.output_dir)


if __name__ == "__main__":
    main()
