"""CLI entry point for the profiler.

Three modes:

    # Python-timer-based phase breakdown + roofline (no nsys needed):
    python -m invokerl.profiling --config <config.yaml> --num-steps 3

    # Run under nsys to capture NVTX markers + kernel-level trace:
    nsys profile --trace=cuda,nvtx --output results/prof.nsys-rep \\
        python -m invokerl.profiling --config <config.yaml> --num-steps 3

    # Analyze an nsys SQLite export (after `nsys export --type=sqlite ...`):
    python -m invokerl.profiling --analyze results/prof.sqlite

Library use (opt-in Perfetto trace):

    from invokerl.profiling import profile
    with profile() as p:
        trainer.step()
    p.summary()
    p.export_perfetto("trace.json")
"""

from __future__ import annotations

import argparse
import logging
import random

import torch

from invokerl.profiling._nvtx import nvtx
from invokerl.profiling.nsys import analyze_profile
from invokerl.profiling.step import profiled_training_step
from invokerl.profiling.timing import timing_report
from invokerl.profiling.trace import profile as perfetto_profile


def _build_trainer(config_path: str, no_ref: bool, seed: int):
    """Build a Trainer from a YAML config using upstream's helpers."""
    from invokerl.engine.trainer import Trainer
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

    cfg = load_config(config_path)
    trainer_config = build_trainer_config(cfg)
    # Silence eval/save during profiling.
    trainer_config.eval_every = 0
    trainer_config.save_every = 0

    algorithm = build_algorithm(cfg)
    train_dataset = build_dataset(cfg, split="train")
    reward_fn = build_reward(cfg)
    generator = build_generator(cfg, trainer_config)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    policy = build_policy(cfg)
    ref_policy = None if no_ref else build_ref_policy(cfg)

    return Trainer(
        config=trainer_config,
        algorithm=algorithm,
        generator=generator,
        policy=policy,
        ref_policy=ref_policy,
        reward_fn=reward_fn,
        dataset=train_dataset,
        eval_dataset=None,
    )


def run_profiling(
    config_path: str,
    num_steps: int = 3,
    output_dir: str = "results/profile",
    perfetto: bool = False,
    no_ref: bool = False,
    seed: int = 42,
) -> None:
    """Run N profiled training steps after one warmup step.

    Writes timing plots, roofline, and summary JSON to `output_dir`.
    If `perfetto=True`, also exports trace.json for ui.perfetto.dev.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    trainer = _build_trainer(config_path, no_ref=no_ref, seed=seed)

    print("=" * 70)
    print("  invokerl Training Profiler")
    print("=" * 70)
    print(f"  Steps: {num_steps} (+ 1 warmup)")
    print(f"  Output: {output_dir}")
    if perfetto:
        print(f"  Perfetto: trace.json will be exported")
    print()

    print("\n[warmup] Running warmup step (not profiled)...")
    with nvtx.annotate("warmup", color="gray"):
        _, warmup_times = profiled_training_step(trainer, step=-1)
    print(f"  Warmup: {sum(warmup_times.values()):.1f}s")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    all_phase_times: list[dict[str, float]] = []
    print(f"\nProfiling {num_steps} training steps...")

    if perfetto:
        # Capture ONE Perfetto trace around the first measured step; the rest
        # run normally so timing averages are meaningful.
        with perfetto_profile() as ptrace:
            with nvtx.annotate("step_0", color="white"):
                metrics, phase_times = profiled_training_step(trainer, step=0)
        all_phase_times.append(phase_times)
        print(f"  [step 0] loss={metrics.get('loss', 0):.4f} "
              f"reward={metrics.get('reward', 0):.3f} "
              f"kl={metrics.get('kl', 0):.4f} "
              f"total={sum(phase_times.values()):.1f}s (traced)")

        import os
        trace_path = os.path.join(output_dir, "trace.json")
        os.makedirs(output_dir, exist_ok=True)
        ptrace.export_perfetto(trace_path)
        ptrace.summary()
        start = 1
    else:
        start = 0

    for step in range(start, num_steps):
        with nvtx.annotate(f"step_{step}", color="white"):
            metrics, phase_times = profiled_training_step(trainer, step)

        all_phase_times.append(phase_times)
        print(f"  [step {step}] loss={metrics.get('loss', 0):.4f} "
              f"reward={metrics.get('reward', 0):.3f} "
              f"kl={metrics.get('kl', 0):.4f} "
              f"gnorm={metrics.get('grad_norm', 0):.2f} "
              f"total={sum(phase_times.values()):.1f}s")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timing_report(all_phase_times, output_dir)
    print("\n  If you ran under nsys, analyze the trace with:")
    print(f"    nsys export --type=sqlite <profile>.nsys-rep")
    print(f"    python -m invokerl.profiling --analyze <profile>.sqlite")


def main():
    parser = argparse.ArgumentParser(
        description="Profile invokerl training step",
    )
    parser.add_argument("--config", type=str,
                        help="YAML config file (required unless --analyze)")
    parser.add_argument("--num-steps", type=int, default=3,
                        help="Number of training steps to profile")
    parser.add_argument("--no-ref", action="store_true",
                        help="Skip reference model (saves memory)")
    parser.add_argument("--perfetto", action="store_true",
                        help="Also export a Chrome/Perfetto trace for one step")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Path to nsys SQLite export for analysis")
    parser.add_argument("--output-dir", type=str, default="results/profile")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.analyze:
        print(f"Analyzing profile: {args.analyze}")
        analyze_profile(args.analyze, args.output_dir)
        return

    if not args.config:
        parser.error("--config required unless --analyze is given")

    run_profiling(
        config_path=args.config,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        perfetto=args.perfetto,
        no_ref=args.no_ref,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
