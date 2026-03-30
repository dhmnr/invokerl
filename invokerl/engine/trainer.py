"""Training loop -- orchestrates generation, reward, loss, and optimization.

Step: sample prompts -> generate completions (vLLM) -> score rewards ->
compute ref log-probs -> advantages -> forward -> loss -> backward -> step -> sync.
"""

from __future__ import annotations

import glob as glob_mod
import json
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import AdamW

from invokerl.algorithms.base import BaseAlgorithm, RolloutBatch
from invokerl.data.base import BaseDataset, PromptItem
from invokerl.engine.generator import BaseGenerator, GenerationConfig
from invokerl.engine.policy import PolicyModel
from invokerl.rewards.base import BaseReward

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Training hyperparameters."""

    model_name_or_path: str = ""
    algorithm: str = "grpo"

    # Optimization
    total_steps: int = 200
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 50
    lr_schedule: str = "constant"       # "constant" or "cosine"
    lr_end: float = 0.0
    accumulation_steps: int = 4

    # Generation
    batch_size: int = 4
    group_size: int = 4
    max_new_tokens: int = 384
    temperature: float = 0.9
    top_k: int = 50

    # Logging & checkpointing
    log_every: int = 10
    eval_every: int = 50
    save_every: int = 100
    eval_samples: int = 50
    max_checkpoints: int = 2
    output_dir: str = "./checkpoints"


class Trainer:
    """Orchestrates: generate -> reward -> forward -> loss -> backward -> step -> sync."""

    def __init__(
        self,
        config: TrainerConfig,
        algorithm: BaseAlgorithm,
        generator: BaseGenerator,
        policy: PolicyModel,
        ref_policy: PolicyModel | None,
        reward_fn: BaseReward,
        dataset: BaseDataset,
        eval_dataset: BaseDataset | None = None,
    ):
        self.config = config
        self.algorithm = algorithm
        self.generator = generator
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_fn = reward_fn
        self.dataset = dataset
        self.eval_dataset = eval_dataset

        # Shared weights: policy params point to vLLM's GPU memory.
        # Must happen BEFORE optimizer creation so moment buffers
        # are initialized on the shared storage.
        self._weights_shared = False
        if hasattr(generator, "get_model_params"):
            try:
                vllm_params = generator.get_model_params()
                shared = policy.share_vllm_weights(vllm_params)
                if shared > 0:
                    self._weights_shared = True
                    self._vllm_params = vllm_params  # prevent GC
                    logger.info("Shared weights enabled: update_weights() is a no-op")
            except Exception as e:
                logger.warning("Shared weights setup failed: %s. Using copy sync.", e)

        self.optimizer = AdamW(
            self.policy.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self._make_lr_lambda(),
        )
        self.gen_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
        )
        self.history: list[dict] = []
        self.step = 0
        self._weight_version = 0  # incremented each optimizer step
        self._disagg_mode = False  # set True by train_disagg(); skips auto weight sync
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _make_lr_lambda(self):
        """Build LR schedule: linear warmup then constant or cosine decay."""
        warmup = self.config.warmup_steps
        total = self.config.total_steps
        schedule = self.config.lr_schedule
        min_ratio = self.config.lr_end / self.config.lr if self.config.lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            if schedule == "cosine":
                progress = min((step - warmup) / max(1, total - warmup), 1.0)
                return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        return lr_lambda

    # -- Rollout (no gradients) ------------------------------------------------

    @torch.no_grad()
    def rollout(self, prompts: list[PromptItem]) -> RolloutBatch:
        """Generate completions, score rewards, compute ref log-probs."""
        B, G = len(prompts), self.config.group_size

        expanded_prompts = [p.prompt for p in prompts for _ in range(G)]
        expanded_truths = [p.ground_truth for p in prompts for _ in range(G)]

        gen_out = self.generator.generate(expanded_prompts, self.gen_config)

        rewards = self.reward_fn.score_batch(
            expanded_prompts, gen_out.texts, ground_truths=expanded_truths,
        ).to(gen_out.token_ids.device)

        if self.ref_policy is not None:
            ref_log_probs = self.ref_policy.forward_no_grad(
                gen_out.token_ids, gen_out.attention_mask,
            )
        else:
            ref_log_probs = gen_out.log_probs.clone()

        group_ids = torch.arange(B, device=gen_out.token_ids.device).repeat_interleave(G)

        return RolloutBatch(
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
            weight_version=self._weight_version,
        )

    # -- Training step ---------------------------------------------------------

    def train_step(self, batch: RolloutBatch) -> tuple[Tensor, dict[str, float]]:
        """One micro-batch: advantages -> forward -> loss -> backward."""
        batch = self._batch_to_device(batch)
        advantages = self.algorithm.compute_advantages(batch)
        new_log_probs = self.policy.forward(batch.token_ids, batch.attention_mask)
        loss, metrics = self.algorithm.compute_loss(new_log_probs, batch, advantages)

        (loss / self.config.accumulation_steps).backward()
        return loss, metrics

    def _batch_to_device(self, batch: RolloutBatch) -> RolloutBatch:
        """Move batch tensors to the policy model's device."""
        dev = self.policy.device
        return RolloutBatch(
            token_ids=batch.token_ids.to(dev),
            prompt_mask=batch.prompt_mask.to(dev),
            response_mask=batch.response_mask.to(dev),
            attention_mask=batch.attention_mask.to(dev),
            rewards=batch.rewards.to(dev),
            token_rewards=batch.token_rewards.to(dev) if batch.token_rewards is not None else None,
            old_log_probs=batch.old_log_probs.to(dev),
            ref_log_probs=batch.ref_log_probs.to(dev),
            group_ids=batch.group_ids.to(dev) if batch.group_ids is not None else None,
            group_size=batch.group_size,
            weight_version=batch.weight_version,
            extras=batch.extras,
        )

    # -- Optimizer step --------------------------------------------------------

    def optimizer_step(self) -> float:
        """Clip gradients, step optimizer, sync weights. Returns grad norm."""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.grad_clip,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # In disagg mode, the pipeline handles weight sync on its own schedule.
        if not self._disagg_mode:
            if self._weights_shared:
                torch.cuda.synchronize()
                # Verify sharing is intact after first step
                if self.step == 0 and hasattr(self, "_vllm_params"):
                    if not self.policy.verify_shared_weights(self._vllm_params):
                        logger.warning("Shared weights broken -- falling back to copy sync")
                        self._weights_shared = False
            else:
                self.generator.update_weights(self.policy.get_state_dict())

        self._weight_version += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return float(grad_norm)

    # -- Evaluation ------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, max_samples: int | None = None) -> dict[str, float]:
        """Evaluate current policy (greedy decoding)."""
        if self.eval_dataset is None:
            return {}
        n = max_samples or self.config.eval_samples
        if n <= 0:
            return {}

        items = self.eval_dataset.items[:n]
        eval_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens, temperature=0.0, top_k=1,
        )
        gen_out = self.generator.generate([i.prompt for i in items], eval_config)
        rewards = self.reward_fn.score_batch(
            [i.prompt for i in items], gen_out.texts,
            ground_truths=[i.ground_truth for i in items],
        )
        return {
            "eval_accuracy": float(rewards.mean()),
            "eval_correct": int(rewards.sum()),
            "eval_total": n,
        }

    # -- Checkpointing ---------------------------------------------------------

    def save_checkpoint(self, step: int) -> str:
        ckpt_dir = os.path.join(self.config.output_dir, f"step_{step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        self.policy.model.save_pretrained(ckpt_dir)
        self.policy.tokenizer.save_pretrained(ckpt_dir)

        torch.save({
            "step": step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "history": self.history,
            "weight_version": self._weight_version,
        }, os.path.join(ckpt_dir, "training_state.pt"))

        logger.info("Checkpoint saved: %s", ckpt_dir)
        self._cleanup_old_checkpoints()
        return ckpt_dir

    def _cleanup_old_checkpoints(self) -> None:
        if self.config.max_checkpoints <= 0:
            return
        pattern = os.path.join(self.config.output_dir, "step_*")
        ckpt_dirs = sorted(
            glob_mod.glob(pattern),
            key=lambda d: int(os.path.basename(d).split("_")[1]),
        )
        while len(ckpt_dirs) > self.config.max_checkpoints:
            shutil.rmtree(ckpt_dirs.pop(0), ignore_errors=True)

    def load_checkpoint(self, ckpt_dir: str) -> int:
        """Load checkpoint. Returns step number to resume from."""
        self._weights_shared = False

        from transformers import AutoModelForCausalLM
        self.policy.model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, dtype=self.policy.dtype,
            device_map=self.policy.device, attn_implementation="sdpa",
        )
        self.policy.model.train()

        # Sync checkpoint weights to vLLM before re-sharing
        self.generator.update_weights(self.policy.get_state_dict())

        if hasattr(self, "_vllm_params"):
            try:
                vllm_params = self.generator.get_model_params()
                if self.policy.share_vllm_weights(vllm_params) > 0:
                    self._weights_shared = True
                    self._vllm_params = vllm_params
                    logger.info("Re-established shared weights after checkpoint load")
            except Exception as e:
                logger.warning("Could not re-share weights after resume: %s", e)

        self.optimizer = AdamW(
            self.policy.parameters(), lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        state_path = os.path.join(ckpt_dir, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.policy.device)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.history = state.get("history", [])
            self._weight_version = state.get("weight_version", 0)
            return state["step"]
        return 0

    # -- Main loop -------------------------------------------------------------

    def train(self, resume_from: str | None = None) -> list[dict]:
        """Run the full training loop."""
        cfg = self.config
        start_step = 0

        if resume_from:
            start_step = self.load_checkpoint(resume_from)
            logger.info("Resumed from step %d", start_step)
            self.generator.update_weights(self.policy.get_state_dict())

        # Baseline eval
        eval_metrics = self.evaluate()
        if eval_metrics:
            logger.info("Baseline: %d/%d = %.1f%%",
                        eval_metrics["eval_correct"], eval_metrics["eval_total"],
                        eval_metrics["eval_accuracy"] * 100)

        logger.info("Starting training: %d steps, accum=%d, group_size=%d",
                    cfg.total_steps, cfg.accumulation_steps, cfg.group_size)

        for step in range(start_step, cfg.total_steps):
            t0 = time.time()
            self.step = step
            self.algorithm.on_step_start(step)

            accumulated_metrics: dict[str, list[float]] = {}
            self.optimizer.zero_grad()

            staleness_values: list[int] = []
            for _ in range(cfg.accumulation_steps):
                prompts = self.dataset.sample(cfg.batch_size)
                batch = self.rollout(prompts)
                staleness_values.append(self._weight_version - batch.weight_version)
                loss, metrics = self.train_step(batch)

                del batch, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                for k, v in metrics.items():
                    accumulated_metrics.setdefault(k, []).append(v)

            grad_norm = self.optimizer_step()
            dt = time.time() - t0

            step_metrics = {k: sum(v) / len(v) for k, v in accumulated_metrics.items()}
            avg_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0.0
            max_staleness = max(staleness_values) if staleness_values else 0
            step_metrics.update({
                "step": step, "grad_norm": grad_norm,
                "lr": self.scheduler.get_last_lr()[0], "time": dt,
                "staleness": avg_staleness,
                "staleness_max": max_staleness,
                "weight_version": self._weight_version,
            })

            self.algorithm.on_step_end(step, step_metrics)
            self.history.append(step_metrics)

            if step % cfg.log_every == 0 or step == start_step:
                logger.info(
                    "[step %4d] loss=%.4f reward=%.3f kl=%.4f gnorm=%.2f lr=%.2e time=%.1fs",
                    step, step_metrics.get("loss", 0), step_metrics.get("reward", 0),
                    step_metrics.get("kl", 0), grad_norm, step_metrics["lr"], dt,
                )

            if cfg.eval_every > 0 and (step + 1) % cfg.eval_every == 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    logger.info("--- Eval step %d: %d/%d = %.1f%%",
                                step + 1, eval_metrics["eval_correct"],
                                eval_metrics["eval_total"], eval_metrics["eval_accuracy"] * 100)
                    step_metrics.update(eval_metrics)

            if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                self.save_checkpoint(step + 1)

        # Final eval + checkpoint
        final_eval = self.evaluate()
        if final_eval:
            logger.info("Final: %d/%d = %.1f%%",
                        final_eval["eval_correct"], final_eval["eval_total"],
                        final_eval["eval_accuracy"] * 100)

        self.save_checkpoint(cfg.total_steps)

        with open(os.path.join(cfg.output_dir, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    # -- Disaggregated training loop -------------------------------------------

    def train_disagg(
        self,
        pipeline,  # DisaggPipeline — imported lazily to avoid circular deps
        resume_from: str | None = None,
    ) -> list[dict]:
        """Train using the async disaggregated pipeline.

        Generation runs in a background thread on a separate GPU.
        This method consumes pre-generated batches from the pipeline queue,
        trains, and periodically syncs weights back to the generation engine.
        """
        cfg = self.config
        start_step = 0

        # Tell optimizer_step() to skip auto weight sync — pipeline handles it.
        self._disagg_mode = True

        if resume_from:
            start_step = self.load_checkpoint(resume_from)
            logger.info("Resumed from step %d (disagg mode)", start_step)

        # Baseline eval BEFORE starting gen thread — vLLM is not thread-safe.
        eval_metrics = self.evaluate()
        if eval_metrics:
            logger.info("Baseline: %d/%d = %.1f%%",
                        eval_metrics["eval_correct"], eval_metrics["eval_total"],
                        eval_metrics["eval_accuracy"] * 100)

        # Start the async generation thread.
        pipeline.start(initial_state_dict=self.policy.get_state_dict())

        logger.info("Starting disagg training: %d steps, accum=%d, group_size=%d, "
                    "sync_every=%d, buffer=%d",
                    cfg.total_steps, cfg.accumulation_steps, cfg.group_size,
                    pipeline.config.sync_every, pipeline.config.buffer_size)

        try:
            for step in range(start_step, cfg.total_steps):
                t0 = time.time()
                self.step = step
                self.algorithm.on_step_start(step)

                accumulated_metrics: dict[str, list[float]] = {}
                self.optimizer.zero_grad()

                staleness_values: list[int] = []
                t_wait, t_train = 0.0, 0.0
                for _ in range(cfg.accumulation_steps):
                    tw0 = time.time()
                    batch = pipeline.get_batch()
                    t_wait += time.time() - tw0
                    if batch is None:
                        pipeline.check_health()
                        logger.error("Pipeline returned None at step %d", step)
                        break

                    staleness = pipeline.weight_version - batch.weight_version
                    staleness_values.append(staleness)
                    tt0 = time.time()
                    loss, metrics = self.train_step(batch)
                    t_train += time.time() - tt0

                    del batch, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    for k, v in metrics.items():
                        accumulated_metrics.setdefault(k, []).append(v)

                to0 = time.time()
                grad_norm = self.optimizer_step()
                t_optim = time.time() - to0
                pipeline.step_version()

                # Periodic weight sync (non-blocking — gen thread applies it).
                sync_ms = 0.0
                if pipeline.should_sync():
                    pipeline.sync_weights(self.policy.get_state_dict())
                    sync_ms = pipeline.last_sync_ms  # from previous sync

                dt = time.time() - t0

                step_metrics = {k: sum(v) / len(v) for k, v in accumulated_metrics.items()}
                avg_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0.0
                max_staleness = max(staleness_values) if staleness_values else 0
                # Use pipeline.weight_version as single source of truth.
                wv = pipeline.weight_version
                step_metrics.update({
                    "step": step, "grad_norm": grad_norm,
                    "lr": self.scheduler.get_last_lr()[0], "time": dt,
                    "staleness": avg_staleness,
                    "staleness_max": max_staleness,
                    "weight_version": wv,
                    "sync_ms": sync_ms,
                    "queue_size": pipeline._queue.qsize(),
                    "t_wait": t_wait,
                    "t_train": t_train,
                    "t_optim": t_optim,
                })

                self.algorithm.on_step_end(step, step_metrics)
                self.history.append(step_metrics)

                if step % cfg.log_every == 0 or step == start_step:
                    logger.info(
                        "[step %4d] loss=%.4f reward=%.3f kl=%.4f gnorm=%.2f "
                        "lr=%.2e time=%.1fs [wait=%.1fs train=%.1fs optim=%.1fs] "
                        "stale=%.1f sync=%.0fms q=%d",
                        step, step_metrics.get("loss", 0), step_metrics.get("reward", 0),
                        step_metrics.get("kl", 0), grad_norm, step_metrics["lr"],
                        dt, t_wait, t_train, t_optim,
                        avg_staleness, sync_ms, pipeline._queue.qsize(),
                    )

                if cfg.eval_every > 0 and (step + 1) % cfg.eval_every == 0:
                    # Hold vLLM lock during eval to prevent concurrent vLLM access
                    # (vLLM's LLM class is not thread-safe).
                    with pipeline._vllm_lock:
                        eval_metrics = self.evaluate()
                    if eval_metrics:
                        logger.info("--- Eval step %d: %d/%d = %.1f%%",
                                    step + 1, eval_metrics["eval_correct"],
                                    eval_metrics["eval_total"], eval_metrics["eval_accuracy"] * 100)
                        step_metrics.update(eval_metrics)

                if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                    self.save_checkpoint(step + 1)

        finally:
            pipeline_stats = pipeline.stop()
            logger.info("Pipeline stats: %s", pipeline_stats)

        # Final eval + checkpoint.
        final_eval = self.evaluate()
        if final_eval:
            logger.info("Final: %d/%d = %.1f%%",
                        final_eval["eval_correct"], final_eval["eval_total"],
                        final_eval["eval_accuracy"] * 100)

        self.save_checkpoint(cfg.total_steps)

        with open(os.path.join(cfg.output_dir, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
