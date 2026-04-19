"""NVTX-instrumented training step.

Reimplements trainer.train_step() with NVTX markers and cuda.synchronize()
around each phase so wall-clock time can be attributed per-phase. Used by
the CLI profiler for both timing-only reports and nsys captures.
"""

from __future__ import annotations

import time

import torch

from invokerl.algorithms.base import RolloutBatch
from invokerl.profiling._nvtx import nvtx


def profiled_training_step(trainer, step: int) -> tuple[dict, dict[str, float]]:
    """Run one training step with NVTX markers and per-phase timing.

    Returns (metrics, phase_times) where phase_times maps phase name → seconds.
    """
    cfg = trainer.config
    phase_times: dict[str, float] = {}

    accumulated_metrics: dict[str, list[float]] = {}
    trainer.optimizer.zero_grad()

    for _ in range(cfg.accumulation_steps):
        prompts = trainer.dataset.sample(cfg.batch_size)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx.annotate("generation", color="blue"):
            expanded_prompts = [p.prompt for p in prompts for _ in range(cfg.group_size)]
            expanded_truths = [p.ground_truth for p in prompts for _ in range(cfg.group_size)]
            gen_out = trainer.generator.generate(expanded_prompts, trainer.gen_config)
        torch.cuda.synchronize()
        phase_times["generation"] = phase_times.get("generation", 0) + (time.perf_counter() - t0)

        t0 = time.perf_counter()
        with nvtx.annotate("reward", color="yellow"):
            rewards = trainer.reward_fn.score_batch(
                expanded_prompts, gen_out.texts, ground_truths=expanded_truths,
            )
            rewards = rewards.to(gen_out.token_ids.device)
        phase_times["reward"] = phase_times.get("reward", 0) + (time.perf_counter() - t0)

        B = len(prompts)
        G = cfg.group_size
        group_ids = torch.arange(B, device=gen_out.token_ids.device).repeat_interleave(G)

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

        advantages = trainer.algorithm.compute_advantages(batch)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx.annotate("policy_forward", color="orange"):
            new_log_probs = trainer.policy.forward(
                batch.token_ids, batch.attention_mask,
            )
        torch.cuda.synchronize()
        phase_times["policy_forward"] = phase_times.get("policy_forward", 0) + (time.perf_counter() - t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with nvtx.annotate("loss_computation", color="red"):
            loss, metrics = trainer.algorithm.compute_loss(
                new_log_probs, batch, advantages,
            )
            scaled_loss = loss / cfg.accumulation_steps
        torch.cuda.synchronize()
        phase_times["loss_computation"] = phase_times.get("loss_computation", 0) + (time.perf_counter() - t0)

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

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with nvtx.annotate("weight_sync", color="brown"):
        if not getattr(trainer, "_weights_shared", False):
            trainer.generator.update_weights(trainer.policy.get_state_dict())
    torch.cuda.synchronize()
    phase_times["weight_sync"] = time.perf_counter() - t0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    step_metrics = {k: sum(v) / len(v) for k, v in accumulated_metrics.items()}
    step_metrics["grad_norm"] = float(grad_norm)

    return step_metrics, phase_times
