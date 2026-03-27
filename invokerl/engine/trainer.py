"""Main training loop — orchestrates generation, reward, loss, and optimization.

This is the glue between the algorithm (hackable) and the infrastructure
(generator, policy model, optimizer). Researchers don't modify this file.

Training step:
    1. Sample prompts from dataset
    2. Generate completions via vLLM (→ token_ids, old_log_probs)
    3. Score completions with reward function (→ rewards)
    4. Compute reference log-probs (→ ref_log_probs)
    5. Build RolloutBatch
    6. Call algorithm.compute_advantages(batch) (→ advantages)
    7. Forward pass through policy model (→ new_log_probs, with gradients)
    8. Call algorithm.compute_loss(new_log_probs, batch, advantages) (→ loss)
    9. loss.backward() + optimizer.step()
    10. Sync weights to generation engine
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Training hyperparameters."""

    # Model
    model_name_or_path: str = ""

    # Algorithm (set by config file)
    algorithm: str = "grpo"

    # Training
    total_steps: int = 200
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 50
    lr_schedule: str = "constant"   # "constant" or "cosine" (after warmup)
    lr_end: float = 0.0             # final LR for cosine schedule (default: 0)
    accumulation_steps: int = 4     # micro-batches per optimizer step

    # Generation
    batch_size: int = 4             # prompts per micro-batch
    group_size: int = 4             # completions per prompt
    max_new_tokens: int = 384
    temperature: float = 0.9
    top_k: int = 50

    # Logging & checkpointing
    log_every: int = 10
    eval_every: int = 50
    save_every: int = 100
    eval_samples: int = 50
    max_checkpoints: int = 2          # keep only the N most recent checkpoints
    output_dir: str = "./checkpoints"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Main training loop.

    Orchestrates: generate → reward → forward → loss → backward → step → sync.
    The algorithm controls only compute_advantages() and compute_loss().
    """

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

        # Shared weights: make policy params point to vLLM's GPU memory.
        # Must happen BEFORE optimizer creation so moment buffers are
        # initialized on the shared storage.
        self._weights_shared = False
        if hasattr(generator, "get_model_params"):
            try:
                vllm_params = generator.get_model_params()
                shared = policy.share_vllm_weights(vllm_params)
                if shared > 0:
                    self._weights_shared = True
                    self._vllm_params = vllm_params  # prevent GC
                    logger.info(
                        "Shared weights enabled: update_weights() is a no-op"
                    )
            except Exception as e:
                logger.warning("Shared weights setup failed: %s. Using copy sync.", e)

        # Optimizer
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # LR scheduler: linear warmup, then constant or cosine decay
        import math
        _warmup = config.warmup_steps
        _total = config.total_steps
        _schedule = config.lr_schedule
        _lr_min_ratio = config.lr_end / config.lr if config.lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if step < _warmup:
                return (step + 1) / _warmup
            if _schedule == "cosine":
                progress = (step - _warmup) / max(1, _total - _warmup)
                progress = min(progress, 1.0)
                return _lr_min_ratio + (1.0 - _lr_min_ratio) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
            return 1.0  # constant

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

        # Generation config
        self.gen_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
        )

        # History
        self.history: list[dict] = []
        self.step = 0

        # Output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # ----- Rollout phase (no gradients) ------------------------------------

    @torch.no_grad()
    def rollout(self, prompts: list[PromptItem]) -> RolloutBatch:
        """Generate completions, score rewards, compute ref log-probs.

        Args:
            prompts: Sampled prompt items (one per prompt in the batch).

        Returns:
            RolloutBatch ready for the algorithm.
        """
        cfg = self.config
        B = len(prompts)
        G = cfg.group_size

        # Expand prompts: each prompt gets G completions
        expanded_prompts = [p.prompt for p in prompts for _ in range(G)]
        expanded_truths = [p.ground_truth for p in prompts for _ in range(G)]

        # Generate completions + old_log_probs
        gen_out = self.generator.generate(expanded_prompts, self.gen_config)

        # Score rewards
        rewards = self.reward_fn.score_batch(
            expanded_prompts,
            gen_out.texts,
            ground_truths=expanded_truths,
        )
        rewards = rewards.to(gen_out.token_ids.device)

        # Reference log-probs
        if self.ref_policy is not None:
            ref_log_probs = self.ref_policy.forward_no_grad(
                gen_out.token_ids, gen_out.attention_mask,
            )
        else:
            # If no ref model, use old_log_probs (degrades to no KL penalty)
            ref_log_probs = gen_out.log_probs.clone()

        # Group IDs: prompt i → completions [i*G, ..., (i+1)*G - 1]
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
        )

    # ----- Training step ---------------------------------------------------

    def _batch_to_device(self, batch: RolloutBatch) -> RolloutBatch:
        """Move all batch tensors to the policy model's device."""
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
            extras=batch.extras,
        )

    def train_step(self, batch: RolloutBatch) -> tuple[Tensor, dict[str, float]]:
        """Run one micro-batch: advantages → forward → loss.

        Args:
            batch: RolloutBatch from rollout().

        Returns:
            (loss, metrics) — loss has gradients, metrics are detached floats.
        """
        # Move batch to training device (generator returns CPU tensors)
        batch = self._batch_to_device(batch)

        # Compute advantages (algorithm-defined credit assignment)
        advantages = self.algorithm.compute_advantages(batch)

        # Forward pass through current policy (with gradients)
        new_log_probs = self.policy.forward(
            batch.token_ids, batch.attention_mask,
        )

        # Compute loss (algorithm-defined objective)
        loss, metrics = self.algorithm.compute_loss(
            new_log_probs, batch, advantages,
        )

        # Scale loss by accumulation steps before backward
        scaled_loss = loss / self.config.accumulation_steps
        scaled_loss.backward()

        return loss, metrics

    # ----- Optimizer step --------------------------------------------------

    def optimizer_step(self) -> float:
        """Clip gradients, step optimizer, update LR, sync weights.

        Returns:
            grad_norm: Global gradient norm before clipping.
        """
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.grad_clip,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        if self._weights_shared:
            # Shared weights mode: optimizer updated params in-place,
            # vLLM already sees the new values. Just ensure writes are visible.
            torch.cuda.synchronize()

            # Verify sharing is intact after first step (catches silent copies)
            if self.step == 0 and hasattr(self, "_vllm_params"):
                if not self.policy.verify_shared_weights(self._vllm_params):
                    logger.warning("Shared weights broken — falling back to copy sync")
                    self._weights_shared = False
        else:
            # Copy-based sync (fallback)
            self.generator.update_weights(self.policy.get_state_dict())

        # Free training tensors (activations, gradients) before generation
        # reclaims GPU memory so vLLM doesn't OOM on the next rollout
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return float(grad_norm)

    # ----- Evaluation ------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, max_samples: int | None = None) -> dict[str, float]:
        """Evaluate current policy on the eval dataset.

        Generates one completion per problem (greedy), checks answer.

        Returns:
            Dict with "eval_accuracy", "eval_n", etc.
        """
        if self.eval_dataset is None:
            return {}

        n = max_samples or self.config.eval_samples
        if n <= 0:
            return {}

        items = self.eval_dataset.items[:n]
        prompts = [item.prompt for item in items]
        truths = [item.ground_truth for item in items]

        # Greedy generation for eval
        eval_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=0.0,  # greedy
            top_k=1,
        )
        gen_out = self.generator.generate(prompts, eval_config)

        # Score
        rewards = self.reward_fn.score_batch(
            prompts, gen_out.texts, ground_truths=truths,
        )
        accuracy = float(rewards.mean())
        correct = int(rewards.sum())

        return {
            "eval_accuracy": accuracy,
            "eval_correct": correct,
            "eval_total": n,
        }

    # ----- Checkpointing ---------------------------------------------------

    def save_checkpoint(self, step: int) -> str:
        """Save model checkpoint and training state.

        Returns:
            Path to the saved checkpoint directory.
        """
        ckpt_dir = os.path.join(self.config.output_dir, f"step_{step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save model weights
        self.policy.model.save_pretrained(ckpt_dir)
        self.policy.tokenizer.save_pretrained(ckpt_dir)

        # Save training state
        state = {
            "step": step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "history": self.history,
        }
        torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))

        logger.info(f"Checkpoint saved to {ckpt_dir}")

        # Remove old checkpoints if we exceed max_checkpoints
        self._cleanup_old_checkpoints()

        return ckpt_dir

    def _cleanup_old_checkpoints(self) -> None:
        """Remove oldest checkpoints if we exceed max_checkpoints."""
        import glob as glob_mod
        max_keep = self.config.max_checkpoints
        if max_keep <= 0:
            return

        pattern = os.path.join(self.config.output_dir, "step_*")
        ckpt_dirs = sorted(
            glob_mod.glob(pattern),
            key=lambda d: int(os.path.basename(d).split("_")[1]),
        )

        while len(ckpt_dirs) > max_keep:
            oldest = ckpt_dirs.pop(0)
            import shutil
            shutil.rmtree(oldest, ignore_errors=True)
            logger.info(f"Removed old checkpoint: {oldest}")

    def load_checkpoint(self, ckpt_dir: str) -> int:
        """Load model and training state from checkpoint.

        Returns:
            The step number to resume from.
        """
        # Load model weights
        from transformers import AutoModelForCausalLM
        self.policy.model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            dtype=self.policy.dtype,
            device_map=self.policy.device,
            attn_implementation="sdpa",
        )
        self.policy.model.train()

        # Reinitialize optimizer with new parameters
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Load training state
        state_path = os.path.join(ckpt_dir, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.policy.device)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.history = state.get("history", [])
            return state["step"]

        return 0

    # ----- Main loop -------------------------------------------------------

    def train(self, resume_from: str | None = None) -> list[dict]:
        """Run the full training loop.

        Args:
            resume_from: Optional checkpoint directory to resume from.

        Returns:
            Training history (list of per-step metric dicts).
        """
        cfg = self.config
        start_step = 0

        if resume_from:
            start_step = self.load_checkpoint(resume_from)
            logger.info(f"Resumed from step {start_step}")
            # Sync loaded weights to generator
            self.generator.update_weights(self.policy.get_state_dict())

        # Baseline eval
        logger.info("Running baseline evaluation...")
        eval_metrics = self.evaluate()
        if eval_metrics:
            logger.info(
                f"Baseline: {eval_metrics['eval_correct']}/{eval_metrics['eval_total']} "
                f"= {eval_metrics['eval_accuracy']:.1%}"
            )

        # Training
        logger.info(
            f"Starting training: {cfg.total_steps} steps, "
            f"accum={cfg.accumulation_steps}, group_size={cfg.group_size}"
        )

        end_step = cfg.total_steps  # total_steps is the absolute target
        for step in range(start_step, end_step):
            t0 = time.time()
            self.step = step
            self.algorithm.on_step_start(step)

            # Gradient accumulation over K micro-batches
            accumulated_metrics: dict[str, list[float]] = {}
            self.optimizer.zero_grad()

            for micro in range(cfg.accumulation_steps):
                # Sample prompts and generate rollouts
                prompts = self.dataset.sample(cfg.batch_size)
                batch = self.rollout(prompts)

                # Forward + loss + backward (gradients accumulate)
                loss, metrics = self.train_step(batch)

                # Free intermediate tensors between micro-batches
                # (activations from backward, batch tensors)
                del batch, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Accumulate metrics
                for k, v in metrics.items():
                    accumulated_metrics.setdefault(k, []).append(v)

            # Optimizer step (clip + step + sync weights)
            grad_norm = self.optimizer_step()

            dt = time.time() - t0

            # Aggregate metrics
            step_metrics = {
                k: sum(v) / len(v) for k, v in accumulated_metrics.items()
            }
            step_metrics.update({
                "step": step,
                "grad_norm": grad_norm,
                "lr": self.scheduler.get_last_lr()[0],
                "time": dt,
            })

            self.algorithm.on_step_end(step, step_metrics)
            self.history.append(step_metrics)

            # Logging
            if step % cfg.log_every == 0 or step == start_step:
                reward_str = f"reward={step_metrics.get('reward', 0):.3f}"
                kl_str = f"kl={step_metrics.get('kl', 0):.4f}"
                logger.info(
                    f"[step {step:>4d}] loss={step_metrics.get('loss', 0):.4f} "
                    f"{reward_str} {kl_str} "
                    f"gnorm={grad_norm:.2f} lr={step_metrics['lr']:.2e} "
                    f"time={dt:.1f}s"
                )

            # Evaluation
            if cfg.eval_every > 0 and (step + 1) % cfg.eval_every == 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    logger.info(
                        f"--- Eval at step {step + 1}: "
                        f"{eval_metrics['eval_correct']}/{eval_metrics['eval_total']} "
                        f"= {eval_metrics['eval_accuracy']:.1%}"
                    )
                    step_metrics.update(eval_metrics)

            # Checkpoint
            if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                self.save_checkpoint(step + 1)

        # Final eval
        logger.info("Running final evaluation...")
        final_eval = self.evaluate()
        if final_eval:
            logger.info(
                f"Final: {final_eval['eval_correct']}/{final_eval['eval_total']} "
                f"= {final_eval['eval_accuracy']:.1%}"
            )

        # Save final checkpoint
        self.save_checkpoint(end_step)

        # Save history
        history_path = os.path.join(cfg.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
