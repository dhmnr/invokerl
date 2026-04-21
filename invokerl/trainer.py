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

from invokerl._logging import log_step, training_progress
from invokerl.algorithms.base import BaseAlgorithm, RolloutBatch
from invokerl.datasets.base import BaseDataset, PromptItem
from invokerl.generator import BaseGenerator, GenerationConfig
from invokerl.policy import PolicyModel
from invokerl.profiling import annotate
from invokerl.rewards.base import BaseReward

logger = logging.getLogger(__name__)


# ============================================================================
# RolloutSource — abstracts where training batches come from
# ============================================================================
# Three variants, same interface:
#   - SyncRollout: blocking rollout() on the main thread (single-GPU)
#   - AsyncRollout: pulls pre-generated batches from a DisaggPipeline queue
#   - DistributedRollout: same as AsyncRollout but rank 0 produces and the
#     rest receive via broadcast (FSDP multi-GPU)
#
# The unified Trainer.train() calls source.next() and doesn't care which.


class _SyncRollout:
    """Synchronous rollout on the training thread. Used when pipeline is None."""

    def __init__(self, trainer):
        self._trainer = trainer

    def start(self) -> None:
        pass

    def next(self) -> RolloutBatch | None:
        prompts = self._trainer.dataset.sample(self._trainer.config.batch_size)
        return self._trainer.rollout(prompts)

    def stop(self) -> dict:
        return {}

    @property
    def weight_version(self) -> int:
        return self._trainer._weight_version

    def step_version(self) -> None:
        # Sync mode: optimizer_step() already incremented _weight_version.
        pass

    def sync_weights_if_due(self, state_dict) -> float:
        # Sync mode: optimizer_step() already copied weights to vLLM.
        return 0.0

    @property
    def vllm_lock(self):
        return None  # no concurrent vLLM access in sync mode


class _AsyncRollout:
    """Async rollout via DisaggPipeline. Pipeline owns the background gen thread."""

    def __init__(self, pipeline):
        self._pipeline = pipeline

    def start(self, initial_state_dict=None) -> None:
        self._pipeline.start(initial_state_dict=initial_state_dict)

    def next(self) -> RolloutBatch | None:
        batch = self._pipeline.get_batch()
        if batch is None:
            self._pipeline.check_health()
        return batch

    def stop(self) -> dict:
        return self._pipeline.stop()

    @property
    def weight_version(self) -> int:
        return self._pipeline.weight_version

    def step_version(self) -> None:
        self._pipeline.step_version()

    def sync_weights_if_due(self, state_dict) -> float:
        if self._pipeline.should_sync():
            self._pipeline.sync_weights(state_dict)
            return self._pipeline.last_sync_ms
        return 0.0

    @property
    def vllm_lock(self):
        return self._pipeline._vllm_lock

    @property
    def queue_size(self) -> int:
        return self._pipeline._queue.qsize()


class _DistributedRollout:
    """FSDP + async: rank 0 produces, others receive via broadcast."""

    def __init__(self, pipeline, rank: int, device):
        self._pipeline = pipeline  # only non-None on rank 0
        self._rank = rank
        self._device = device

    def start(self, initial_state_dict=None) -> None:
        if self._rank == 0:
            self._pipeline.start(initial_state_dict=initial_state_dict)

    def next(self) -> RolloutBatch | None:
        from invokerl.distributed import broadcast_batch

        if self._rank == 0:
            batch = self._pipeline.get_batch()
            if batch is None:
                self._pipeline.check_health()
        else:
            batch = None
        return broadcast_batch(batch, src=0, device=self._device)

    def stop(self) -> dict:
        if self._rank == 0:
            return self._pipeline.stop()
        return {}

    @property
    def weight_version(self) -> int:
        return self._pipeline.weight_version if self._rank == 0 else 0

    def step_version(self) -> None:
        if self._rank == 0:
            self._pipeline.step_version()

    def sync_weights_if_due(self, state_dict) -> float:
        # All ranks must participate in the FSDP allgather (caller of
        # this function is expected to have already produced state_dict
        # collectively). Rank 0 decides whether to actually sync, then
        # broadcasts the decision so every rank's control flow matches.
        if self._rank == 0:
            should = self._pipeline.should_sync()
            flag = torch.tensor([1 if should else 0], device=self._device)
        else:
            flag = torch.tensor([0], device=self._device)
        torch.distributed.broadcast(flag, src=0)
        if flag.item() and self._rank == 0:
            self._pipeline.sync_weights(state_dict)
            return self._pipeline.last_sync_ms
        return 0.0

    @property
    def vllm_lock(self):
        return self._pipeline._vllm_lock if self._rank == 0 else None

    @property
    def queue_size(self) -> int:
        return self._pipeline._queue.qsize() if self._rank == 0 else 0


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
    lr_schedule: str = "constant"  # "constant" or "cosine"
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
            self.policy.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self._make_lr_lambda(),
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

        with annotate("generation", color="blue"):
            gen_out = self.generator.generate(expanded_prompts, self.gen_config)

        with annotate("reward", color="yellow"):
            rewards = self.reward_fn.score_batch(
                expanded_prompts,
                gen_out.texts,
                ground_truths=expanded_truths,
            ).to(gen_out.token_ids.device)

        with annotate("ref_forward", color="green"):
            if self.ref_policy is not None:
                ref_log_probs = self.ref_policy.forward_no_grad(
                    gen_out.token_ids,
                    gen_out.attention_mask,
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
        with annotate("policy_forward", color="orange"):
            new_log_probs = self.policy.forward(batch.token_ids, batch.attention_mask)
        with annotate("loss_computation", color="red"):
            loss, metrics = self.algorithm.compute_loss(new_log_probs, batch, advantages)

        with annotate("backward", color="purple"):
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
        with annotate("optimizer_step", color="cyan"):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.grad_clip,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # In disagg mode, the pipeline handles weight sync on its own schedule.
        if not self._disagg_mode:
            with annotate("weight_sync", color="brown"):
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
            max_new_tokens=self.config.max_new_tokens,
            temperature=0.0,
            top_k=1,
        )
        gen_out = self.generator.generate([i.prompt for i in items], eval_config)
        rewards = self.reward_fn.score_batch(
            [i.prompt for i in items],
            gen_out.texts,
            ground_truths=[i.ground_truth for i in items],
        )
        return {
            "eval_accuracy": float(rewards.mean()),
            "eval_correct": int(rewards.sum()),
            "eval_total": n,
        }

    # -- Checkpointing ---------------------------------------------------------

    def save_checkpoint(self, step: int) -> str:
        """Save model + optimizer checkpoint.

        All CUDA tensors are copied to CPU before serialization to avoid
        conflicts with vLLM's CUDA graph memory pool. Without this, torch.save
        can trigger cudaErrorLaunchFailure when CUDA graphs are active.

        IMPORTANT: With FSDP, ALL ranks must call this method because
        get_state_dict() does an allgather collective. Only rank 0 writes
        files to disk (non-rank-0 gets an empty state_dict from
        get_full_state_dict with rank0_only=True).
        """
        # Synchronize the training device to flush pending CUDA ops before
        # reading GPU tensors. Only sync the train device — never the gen
        # device from this thread (disrupts vLLM's CUDA graph pool).
        train_device = next(self.policy.model.parameters()).device
        torch.cuda.synchronize(train_device)

        # get_state_dict() is an FSDP collective — all ranks must participate.
        # rank 0 gets the full state dict; other ranks get empty dict.
        state_dict = self.policy.get_state_dict()

        # Only rank 0 (or non-distributed) writes to disk.
        from invokerl.distributed import is_main_rank

        if not is_main_rank():
            return ""

        ckpt_dir = os.path.join(self.config.output_dir, f"step_{step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        self.policy.model.save_pretrained(ckpt_dir, state_dict=cpu_state_dict)
        del cpu_state_dict
        self.policy.tokenizer.save_pretrained(ckpt_dir)

        # Move optimizer state to CPU before torch.save.
        cpu_optim_state = self._optimizer_state_to_cpu(self.optimizer.state_dict())
        torch.save(
            {
                "step": step,
                "optimizer": cpu_optim_state,
                "scheduler": self.scheduler.state_dict(),
                "history": self.history,
                "weight_version": self._weight_version,
            },
            os.path.join(ckpt_dir, "training_state.pt"),
        )
        del cpu_optim_state

        logger.info("Checkpoint saved: %s", ckpt_dir)
        self._cleanup_old_checkpoints()
        return ckpt_dir

    @staticmethod
    def _optimizer_state_to_cpu(state_dict: dict) -> dict:
        """Recursively move all tensors in an optimizer state_dict to CPU."""
        cpu_state = {"state": {}, "param_groups": state_dict["param_groups"]}
        for param_id, param_state in state_dict["state"].items():
            cpu_state["state"][param_id] = {
                k: v.cpu() if isinstance(v, Tensor) else v for k, v in param_state.items()
            }
        return cpu_state

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
            ckpt_dir,
            dtype=self.policy.dtype,
            device_map=self.policy.device,
            attn_implementation="sdpa",
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
            self.policy.parameters(),
            lr=self.config.lr,
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

    # -- Main training loop (unified across all modes) -------------------------

    def train(
        self,
        resume_from: str | None = None,
        pipeline=None,
    ) -> list[dict]:
        """Run the full training loop. Seamless across single-GPU / disagg / FSDP.

            trainer.train()                       # single GPU
            trainer.train(pipeline=disagg_pipe)   # async disagg, or disagg+FSDP

        Mode is inferred from what you pass: no pipeline → sync; pipeline +
        FSDP-wrapped policy → distributed disagg; pipeline only → disagg.
        """
        cfg = self.config

        # -- Mode detection ----------------------------------------------------
        is_fsdp = getattr(self.policy, "_fsdp_wrapped", False)
        if pipeline is None:
            source = _SyncRollout(self)
            rank, world_size, is_main = 0, 1, True
        elif is_fsdp:
            from invokerl.distributed import get_rank, get_world_size, is_main_rank

            rank, world_size = get_rank(), get_world_size()
            is_main = is_main_rank()
            # Only rank 0 owns the pipeline; other ranks pass through broadcast.
            source = _DistributedRollout(
                pipeline if is_main else None,
                rank,
                self.policy.device,
            )
        else:
            source = _AsyncRollout(pipeline)
            rank, world_size, is_main = 0, 1, True

        # Tell optimizer_step() to skip auto weight sync when a pipeline
        # is providing its own (possibly async) sync.
        self._disagg_mode = pipeline is not None

        # -- Resume ------------------------------------------------------------
        start_step = 0
        if resume_from and is_main:
            start_step = self.load_checkpoint(resume_from)
            logger.info("Resumed from step %d", start_step)
            if pipeline is None:
                # In sync mode, resync loaded weights to vLLM.
                self.generator.update_weights(self.policy.get_state_dict())

        if is_fsdp:
            start_step_t = torch.tensor([start_step], device=self.policy.device)
            torch.distributed.broadcast(start_step_t, src=0)
            start_step = int(start_step_t.item())

        # -- Baseline eval (rank 0 only) --------------------------------------
        if is_main:
            eval_metrics = self.evaluate()
            if eval_metrics:
                logger.info(
                    "Baseline: %d/%d = %.1f%%",
                    eval_metrics["eval_correct"],
                    eval_metrics["eval_total"],
                    eval_metrics["eval_accuracy"] * 100,
                )

        if is_fsdp:
            from invokerl.distributed import barrier as _barrier

            _barrier()

        # -- Start async pipeline (if any) ------------------------------------
        if pipeline is not None:
            # FSDP: all ranks must participate in get_state_dict() (allgather);
            # only rank 0 uses the result. Non-FSDP: just rank 0.
            initial_state = self.policy.get_state_dict() if (is_fsdp or is_main) else None
            if is_main:
                source.start(initial_state_dict=initial_state)
            else:
                source.start()  # no-op for non-rank-0 in _DistributedRollout
            del initial_state
            if is_fsdp:
                _barrier()

        mode_label = (
            "single GPU"
            if pipeline is None
            else f"disagg+FSDP (world={world_size})"
            if is_fsdp
            else "disagg"
        )
        if is_main:
            logger.info(
                "Starting training [%s]: %d steps, accum=%d, group_size=%d",
                mode_label,
                cfg.total_steps,
                cfg.accumulation_steps,
                cfg.group_size,
            )

        # -- Main loop ---------------------------------------------------------
        # Progress bar lives across the whole loop. Non-main ranks get a
        # no-op so we don't fight with FSDP output and bar ordering.
        from contextlib import ExitStack

        def _noop_advance(*_, **__):
            pass

        _stack = ExitStack()
        if is_main:
            advance_progress = _stack.enter_context(training_progress(cfg.total_steps, start_step))
        else:
            advance_progress = _noop_advance

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
                    batch = source.next()
                    t_wait += time.time() - tw0
                    if batch is None:
                        logger.error("Rollout returned None at step %d", step)
                        break

                    staleness_values.append(source.weight_version - batch.weight_version)

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

                source.step_version()

                # Post-step weight sync (async modes). For FSDP the state_dict
                # call is itself an allgather — ALL ranks must participate.
                sync_ms = 0.0
                if pipeline is not None:
                    state_dict = self.policy.get_state_dict()
                    sync_ms = source.sync_weights_if_due(state_dict)
                    del state_dict

                dt = time.time() - t0

                # -- Logging / eval / checkpoint (rank 0 only) -----------------
                if is_main:
                    step_metrics = {k: sum(v) / len(v) for k, v in accumulated_metrics.items()}
                    avg_stale = (
                        sum(staleness_values) / len(staleness_values) if staleness_values else 0.0
                    )
                    max_stale = max(staleness_values) if staleness_values else 0
                    step_metrics.update(
                        {
                            "step": step,
                            "grad_norm": float(grad_norm),
                            "lr": self.scheduler.get_last_lr()[0],
                            "time": dt,
                            "staleness": avg_stale,
                            "staleness_max": max_stale,
                            "weight_version": source.weight_version,
                        }
                    )
                    if pipeline is not None:
                        step_metrics.update(
                            {
                                "t_wait": t_wait,
                                "t_train": t_train,
                                "t_optim": t_optim,
                                "sync_ms": sync_ms,
                                "queue_size": source.queue_size,
                            }
                        )
                    if is_fsdp:
                        step_metrics["world_size"] = world_size

                    self.algorithm.on_step_end(step, step_metrics)
                    self.history.append(step_metrics)

                    if step % cfg.log_every == 0 or step == start_step:
                        log_step(
                            step=step,
                            dt=dt,
                            metrics=step_metrics,
                            is_disagg=pipeline is not None,
                            is_fsdp=is_fsdp,
                        )

                    if cfg.eval_every > 0 and (step + 1) % cfg.eval_every == 0:
                        eval_metrics = self._locked_evaluate(source.vllm_lock)
                        if eval_metrics:
                            logger.info(
                                "--- Eval step %d: %d/%d = %.1f%%",
                                step + 1,
                                eval_metrics["eval_correct"],
                                eval_metrics["eval_total"],
                                eval_metrics["eval_accuracy"] * 100,
                            )
                            step_metrics.update(eval_metrics)

                # Checkpoint (FSDP: all ranks must participate in get_state_dict).
                if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                    self._locked_save_checkpoint(step + 1, source.vllm_lock)

                if is_fsdp:
                    _barrier()

                advance_progress()

        finally:
            _stack.close()  # tears down the progress bar cleanly
            stats = source.stop()
            if stats and is_main:
                logger.info("Pipeline stats: %s", stats)

        # -- Final eval + checkpoint ------------------------------------------
        if is_main:
            final_eval = self.evaluate()
            if final_eval:
                logger.info(
                    "Final: %d/%d = %.1f%%",
                    final_eval["eval_correct"],
                    final_eval["eval_total"],
                    final_eval["eval_accuracy"] * 100,
                )

        self.save_checkpoint(cfg.total_steps)

        if is_main:
            with open(os.path.join(cfg.output_dir, "training_history.json"), "w") as f:
                json.dump(self.history, f, indent=2)

        if is_fsdp:
            _barrier()
        return self.history

    # -- Helpers used by the unified loop --------------------------------------

    def _locked_evaluate(self, lock) -> dict[str, float]:
        """Run evaluate() under the vLLM lock if one is provided (disagg mode)."""
        if lock is None:
            return self.evaluate()
        with lock:
            return self.evaluate()

    def _locked_save_checkpoint(self, step: int, lock) -> None:
        """Save checkpoint under the vLLM lock if provided.

        In disagg mode, holding the lock prevents the gen thread from doing
        concurrent CUDA copies (weight sync) while we copy tensors to CPU.
        """
        if lock is None:
            self.save_checkpoint(step)
        else:
            with lock:
                self.save_checkpoint(step)
