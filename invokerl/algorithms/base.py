"""Base algorithm interface for invokerl.

Every RL post-training algorithm implements this interface. The only methods
a researcher needs to write are compute_advantages() and compute_loss().
Everything else — generation, model forward/backward, optimization, weight
sync — is handled by the trainer.

Credit assignment is a first-class concept: override compute_advantages()
to experiment with different ways of turning rewards into per-token learning
signals (group normalization, GAE, token-level reward shaping, PRM, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RolloutBatch:
    """A batch of generated rollouts with rewards. This is the input to every
    algorithm's compute_loss().

    Produced by the trainer's rollout phase (generation + reward scoring).
    Algorithms should treat this as read-only.
    """

    # Token IDs: prompt + completion, padded to max length
    token_ids: Tensor  # [B, T] int64

    # Masks
    prompt_mask: Tensor  # [B, T] bool — True for prompt tokens
    response_mask: Tensor  # [B, T] bool — True for response tokens
    attention_mask: Tensor  # [B, T] bool — True for non-padding tokens

    # Rewards — can be sequence-level or token-level
    rewards: Tensor  # [B] scalar per-sequence reward
    token_rewards: Tensor | None  # [B, T] per-token rewards (optional, for credit assignment)

    # Log-probabilities from generation rollout and reference model
    old_log_probs: Tensor  # [B, T] from policy at generation time
    ref_log_probs: Tensor  # [B, T] from frozen reference model

    # Group structure (for GRPO-style algorithms)
    group_ids: Tensor | None  # [B] int — which prompt each completion belongs to
    group_size: int = 1  # number of completions per prompt

    # Off-policy tracking: which weight version generated this batch.
    # staleness = current_weight_version - weight_version.
    # In synchronous mode this is always 0. In async pipelines it tracks
    # how many optimizer steps have happened since generation.
    weight_version: int = 0

    # Optional extras (value predictions, etc.)
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base algorithm
# ---------------------------------------------------------------------------


class BaseAlgorithm(ABC):
    """Base class for all RL post-training algorithms.

    To implement a new algorithm:
        1. Override compute_advantages() — how rewards become learning signals
        2. Override compute_loss() — the policy optimization objective

    That's it. The trainer handles everything else.
    """

    def __init__(self, **kwargs):
        """Store hyperparameters. Subclasses should accept explicit kwargs."""
        self.config = kwargs

    # -- Credit assignment (override to experiment) ---------------------------

    def compute_advantages(
        self,
        batch: RolloutBatch,
    ) -> Tensor:
        """Turn rewards into per-token advantages.

        This is the primary hook for credit assignment experiments. The default
        implementation broadcasts sequence-level rewards to all response tokens.

        Override to implement:
        - Group normalization (GRPO)
        - GAE with value function (PPO)
        - Token-level reward shaping
        - Process reward model scores
        - Custom baselines

        Args:
            batch: The rollout batch with rewards and masks.

        Returns:
            advantages: [B, T] per-token advantage values.
        """
        # Default: sequence-level reward broadcast to all response tokens
        adv = batch.rewards.unsqueeze(1).expand_as(batch.response_mask)
        return adv * batch.response_mask.float()

    # -- Policy objective (override to implement) -----------------------------

    @abstractmethod
    def compute_loss(
        self,
        new_log_probs: Tensor,
        batch: RolloutBatch,
        advantages: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the policy loss given current log-probs and advantages.

        This is called by the trainer after:
        1. Generating completions (→ batch.old_log_probs)
        2. Scoring rewards (→ batch.rewards)
        3. Computing reference log-probs (→ batch.ref_log_probs)
        4. Running policy forward (→ new_log_probs)
        5. Computing advantages (→ advantages)

        Args:
            new_log_probs: [B, T] log-probs from current policy (has gradients).
            batch: The rollout batch (rewards, old_log_probs, ref_log_probs, masks).
            advantages: [B, T] from compute_advantages().

        Returns:
            loss: Scalar loss tensor (with gradients for backward).
            metrics: Dict of loggable metrics (reward, kl, policy_loss, etc.).
                     Values should be Python floats (detached).
        """
        ...

    # -- Optional hooks -------------------------------------------------------

    def on_step_start(self, step: int) -> None:
        """Called at the start of each training step. For scheduling, etc."""
        pass

    def on_step_end(self, step: int, metrics: dict) -> None:
        """Called at the end of each training step."""
        pass
