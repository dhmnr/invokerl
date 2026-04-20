"""Disaggregated async pipeline — generation and training on separate GPUs.

Architecture:
    Gen thread (GPU 0) → [batch queue] → Train thread (main, GPU 1)

    The generation thread continuously produces RolloutBatches and pushes them
    into a bounded queue (double buffer). The main thread pulls batches and
    trains. Weights are synced periodically from the policy model to vLLM.

    Each batch carries a `weight_version` stamp from generation time so the
    trainer can track how many optimizer steps have elapsed since the batch
    was generated (staleness).

Usage:
    invokerl --config ... --disagg --gen-device cuda:0 --train-device cuda:1

For single-GPU mode, the existing synchronous Trainer.train() is used instead.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import torch
from torch import Tensor

from invokerl.algorithms.base import RolloutBatch
from invokerl.data.base import BaseDataset
from invokerl.generator import BaseGenerator, GenerationConfig
from invokerl.policy import PolicyModel
from invokerl.rewards.base import BaseReward

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Disaggregated pipeline settings."""

    gen_device: str = "cuda:0"       # GPU for vLLM generation
    train_device: str = "cuda:1"     # GPU for policy training
    sync_every: int = 1              # sync weights every N optimizer steps
    buffer_size: int = 2             # max batches in queue (double buffer)
    max_staleness: int = 0           # drop batches staler than this (0 = no limit)


class DisaggPipeline:
    """Async double-buffered generation/training pipeline.

    Start the pipeline, then call get_batch() from the training loop.
    After each optimizer step, call step_version() and optionally sync_weights().

    The generation thread blocks when the queue is full (back-pressure).
    The training thread blocks when the queue is empty (waiting for gen).
    This naturally balances the two when T_gen ≈ T_train.
    """

    def __init__(
        self,
        config: PipelineConfig,
        generator: BaseGenerator,
        ref_policy: PolicyModel | None,
        reward_fn: BaseReward,
        dataset: BaseDataset,
        gen_config: GenerationConfig,
        batch_size: int = 4,
        group_size: int = 4,
        use_vllm_ref: bool = False,
    ):
        self.config = config
        self.generator = generator
        self.ref_policy = ref_policy
        self.reward_fn = reward_fn
        self.dataset = dataset
        self.gen_config = gen_config
        self.batch_size = batch_size
        self.group_size = group_size
        # Use vLLM's compute_log_probs() for ref log-probs instead of a
        # separate PolicyModel. Saves ~1.2 GB and benefits from TP scaling.
        self.use_vllm_ref = use_vllm_ref

        # Bounded queue: gen thread produces, train thread consumes.
        self._queue: queue.Queue[RolloutBatch | None] = queue.Queue(
            maxsize=config.buffer_size,
        )

        # Weight versioning.
        # _weight_version: incremented by training (via step_version()).
        # _gen_version: version of weights currently in vLLM.
        self._weight_version = 0
        self._gen_version = 0
        self._version_lock = threading.Lock()

        # vLLM lock: serializes all vLLM access (generate + evaluate).
        # The gen thread holds this during generate(); the training thread
        # acquires it for mid-training evaluate() calls.
        self._vllm_lock = threading.Lock()

        # Non-blocking weight sync: training thread posts state_dict here,
        # gen thread applies it between generation calls.
        self._pending_sync: dict[str, Tensor] | None = None
        self._pending_sync_lock = threading.Lock()
        self._last_sync_ms: float = 0.0

        # Lifecycle control.
        self._stop = threading.Event()
        self._gen_thread: threading.Thread | None = None
        self._gen_error: Exception | None = None

        # Counters.
        self.batches_generated = 0
        self.batches_consumed = 0
        self.batches_dropped = 0
        self.syncs_done = 0
        self._total_staleness = 0
        self._max_staleness = 0

        # Phase timing (gen thread, cumulative).
        self._t_generate = 0.0
        self._t_reward = 0.0
        self._t_ref = 0.0

    # -- Generation thread -----------------------------------------------------

    def _gen_loop(self) -> None:
        """Background loop: generate rollouts until stopped.

        Between generation calls, applies any pending weight sync posted by
        the training thread. This avoids lock contention — the gen thread is
        the only one touching vLLM, so no concurrent access issues.
        """
        try:
            while not self._stop.is_set():
                # Apply pending weight sync between generations (~50ms when pending).
                self._apply_pending_sync()

                with self._version_lock:
                    version = self._gen_version

                batch = self._generate_one(version)

                # Enqueue (blocks if full — back-pressure from training).
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=0.5)
                        break
                    except queue.Full:
                        # Check for pending sync while waiting on full queue too.
                        self._apply_pending_sync()
                        continue

                self.batches_generated += 1
        except Exception as e:
            logger.error("Generation thread failed: %s", e, exc_info=True)
            self._gen_error = e
            self._queue.put(None)  # unblock training thread

    def _apply_pending_sync(self) -> None:
        """Apply any pending weight sync. Called by gen thread between generations."""
        with self._pending_sync_lock:
            state_dict = self._pending_sync
            self._pending_sync = None

        if state_dict is None:
            return

        t0 = time.perf_counter()
        # Hold vllm_lock to prevent concurrent evaluate() during the brief copy.
        with self._vllm_lock:
            self.generator.update_weights(state_dict)
        with self._version_lock:
            self._gen_version = self._weight_version
        dt_ms = (time.perf_counter() - t0) * 1000
        self.syncs_done += 1
        self._last_sync_ms = dt_ms
        logger.debug(
            "Weight sync #%d: %.1fms (version %d → gen)",
            self.syncs_done, dt_ms, self._gen_version,
        )

    @torch.no_grad()
    def _generate_one(self, weight_version: int) -> RolloutBatch:
        """Produce one rollout batch on the generation GPU."""
        B, G = self.batch_size, self.group_size
        prompts = self.dataset.sample(B)

        expanded_prompts = [p.prompt for p in prompts for _ in range(G)]
        expanded_truths = [p.ground_truth for p in prompts for _ in range(G)]

        # Hold vllm_lock during generate() to serialize with evaluate().
        t0 = time.perf_counter()
        with self._vllm_lock:
            gen_out = self.generator.generate(expanded_prompts, self.gen_config)
        self._t_generate += time.perf_counter() - t0

        t0 = time.perf_counter()
        rewards = self.reward_fn.score_batch(
            expanded_prompts, gen_out.texts, ground_truths=expanded_truths,
        ).to(gen_out.token_ids.device)
        self._t_reward += time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.ref_policy is not None:
            ref_log_probs = self.ref_policy.forward_no_grad(
                gen_out.token_ids, gen_out.attention_mask,
            )
        elif self.use_vllm_ref:
            # Use vLLM to compute ref log-probs instead of a separate model.
            # This saves ~1.2 GB (no frozen PolicyModel needed) and gets TP
            # benefits for free. The ref is the initial (pre-training) weights,
            # frozen at pipeline start — but here we approximate with the
            # generation-time weights (staleness ≤ sync_every steps).
            with self._vllm_lock:
                ref_log_probs = self.generator.compute_log_probs(
                    gen_out.token_ids, gen_out.attention_mask,
                )
        else:
            ref_log_probs = gen_out.log_probs.clone()
        self._t_ref += time.perf_counter() - t0

        group_ids = torch.arange(
            B, device=gen_out.token_ids.device,
        ).repeat_interleave(G)

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
            weight_version=weight_version,
        )

    # -- Batch consumption -----------------------------------------------------

    def get_batch(self, timeout: float = 60.0) -> RolloutBatch | None:
        """Pull the next rollout batch from the queue.

        Returns None if the generation thread errored or if no batch
        is available within `timeout` seconds.
        """
        self.check_health()

        try:
            batch = self._queue.get(timeout=timeout)
        except queue.Empty:
            logger.warning("No batch available after %.0fs — gen may be slow", timeout)
            return None

        if batch is None:
            return None  # gen thread signaled an error

        staleness = self._weight_version - batch.weight_version
        self._total_staleness += staleness
        self._max_staleness = max(self._max_staleness, staleness)
        self.batches_consumed += 1

        # Optionally drop stale batches.
        if self.config.max_staleness > 0 and staleness > self.config.max_staleness:
            self.batches_dropped += 1
            logger.debug(
                "Dropped batch (staleness %d > max %d)", staleness, self.config.max_staleness,
            )
            return self.get_batch(timeout)

        return batch

    # -- Weight sync -----------------------------------------------------------

    def sync_weights(self, state_dict: dict[str, Tensor]) -> float:
        """Request async weight sync. Non-blocking — returns immediately.

        The gen thread applies the sync between generation calls (~50ms).
        Returns 0.0 since the actual sync happens asynchronously.
        Use `last_sync_ms` to check the actual time of the most recent sync.

        The state_dict is cloned to prevent races with the next optimizer step.
        """
        with self._pending_sync_lock:
            self._pending_sync = {k: v.detach().clone() for k, v in state_dict.items()}
        return 0.0

    def sync_weights_blocking(self, state_dict: dict[str, Tensor]) -> float:
        """Synchronous weight sync (used before pipeline starts). Returns ms."""
        t0 = time.perf_counter()
        self.generator.update_weights(state_dict)
        with self._version_lock:
            self._gen_version = self._weight_version
        dt_ms = (time.perf_counter() - t0) * 1000
        self.syncs_done += 1
        self._last_sync_ms = dt_ms
        logger.debug(
            "Weight sync #%d: %.1fms (version %d → gen)",
            self.syncs_done, dt_ms, self._weight_version,
        )
        return dt_ms

    @property
    def last_sync_ms(self) -> float:
        """Actual time of the most recent sync (set by gen thread)."""
        return self._last_sync_ms

    def step_version(self) -> int:
        """Increment weight version after an optimizer step. Returns new version."""
        self._weight_version += 1
        return self._weight_version

    def should_sync(self) -> bool:
        """True if weights should be synced based on sync_every config."""
        return (self._weight_version - self._gen_version) >= self.config.sync_every

    @property
    def weight_version(self) -> int:
        return self._weight_version

    # -- Lifecycle -------------------------------------------------------------

    def start(self, initial_state_dict: dict[str, Tensor] | None = None) -> None:
        """Start the generation thread.

        Args:
            initial_state_dict: If provided, sync these weights to vLLM before
                                starting generation. Required if the policy has
                                been loaded from a checkpoint.
        """
        if self._gen_thread is not None:
            raise RuntimeError("Pipeline already running")

        if initial_state_dict is not None:
            self.sync_weights_blocking(initial_state_dict)

        self._stop.clear()
        self._gen_thread = threading.Thread(
            target=self._gen_loop, name="invokerl-gen", daemon=True,
        )
        self._gen_thread.start()
        logger.info(
            "Disagg pipeline started: gen=%s train=%s buffer=%d sync_every=%d",
            self.config.gen_device,
            self.config.train_device,
            self.config.buffer_size,
            self.config.sync_every,
        )

    def stop(self) -> dict[str, float]:
        """Stop the pipeline and return summary stats."""
        self._stop.set()
        if self._gen_thread is not None:
            self._gen_thread.join(timeout=10)
            if self._gen_thread.is_alive():
                logger.warning("Generation thread did not exit cleanly")
            self._gen_thread = None

        # Drain queue.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        avg_staleness = self._total_staleness / max(1, self.batches_consumed)
        n = max(1, self.batches_generated)
        stats = {
            "batches_generated": self.batches_generated,
            "batches_consumed": self.batches_consumed,
            "batches_dropped": self.batches_dropped,
            "weight_syncs": self.syncs_done,
            "avg_staleness": avg_staleness,
            "max_staleness": self._max_staleness,
            "gen_total_s": round(self._t_generate, 1),
            "gen_avg_ms": round(self._t_generate / n * 1000, 1),
            "reward_total_s": round(self._t_reward, 1),
            "ref_total_s": round(self._t_ref, 1),
        }

        logger.info(
            "Pipeline stopped: gen=%d train=%d dropped=%d syncs=%d "
            "avg_stale=%.2f max_stale=%d",
            self.batches_generated,
            self.batches_consumed,
            self.batches_dropped,
            self.syncs_done,
            avg_staleness,
            self._max_staleness,
        )
        return stats

    def check_health(self) -> None:
        """Raise if the generation thread has failed."""
        if self._gen_error is not None:
            raise RuntimeError(
                f"Generation thread failed: {self._gen_error}"
            ) from self._gen_error
