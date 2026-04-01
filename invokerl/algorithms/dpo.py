"""DPO — Direct Preference Optimization.

Paper: Rafailov et al., "Direct Preference Optimization: Your Language Model
is Secretly a Reward Model" (2023). https://arxiv.org/abs/2305.18290

Core idea: bypass reward modeling by directly optimizing the policy on
preference pairs. The implicit reward is the log-ratio between the policy
and reference model. Loss is a sigmoid cross-entropy on the reward margin
between chosen and rejected completions.

L = -log σ(β (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

Requires paired data: batch must have group_size=2 where odd indices are
chosen and even indices are rejected (or vice versa, configured via
chosen_first).
"""

from __future__ import annotations

import torch
from torch import Tensor

from invokerl.algorithms.base import BaseAlgorithm, RolloutBatch


class DPO(BaseAlgorithm):
    """Direct Preference Optimization.

    Hyperparameters:
        beta: Temperature parameter controlling deviation from reference.
        label_smoothing: Smoothing for the preference labels (0 = hard labels).
        chosen_first: If True, even indices are chosen, odd are rejected.
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        chosen_first: bool = True,
        **kwargs,
    ):
        super().__init__(beta=beta, label_smoothing=label_smoothing, **kwargs)
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.chosen_first = chosen_first

    def compute_advantages(self, batch: RolloutBatch) -> Tensor:
        """DPO doesn't use advantages — return zeros."""
        return torch.zeros_like(batch.response_mask, dtype=torch.float32)

    def compute_loss(
        self,
        new_log_probs: Tensor,
        batch: RolloutBatch,
        advantages: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """DPO loss on preference pairs.

        Expects batch to contain paired sequences: [chosen_0, rejected_0,
        chosen_1, rejected_1, ...] with group_size=2.
        """
        mask = batch.response_mask.float()
        ref_lp = batch.ref_log_probs

        # Per-sequence sum of log-probs (only response tokens).
        policy_lp_sum = (new_log_probs * mask).sum(dim=1)  # [B]
        ref_lp_sum = (ref_lp * mask).sum(dim=1)            # [B]

        # Log-ratio: log π(y|x) - log π_ref(y|x)
        log_ratio = policy_lp_sum - ref_lp_sum  # [B]

        # Split into chosen / rejected pairs.
        if self.chosen_first:
            chosen_lr = log_ratio[0::2]   # even indices
            rejected_lr = log_ratio[1::2]  # odd indices
        else:
            chosen_lr = log_ratio[1::2]
            rejected_lr = log_ratio[0::2]

        # DPO loss: -log σ(β * (chosen_lr - rejected_lr))
        margin = self.beta * (chosen_lr - rejected_lr)

        if self.label_smoothing > 0:
            # Smoothed: (1 - ε) * -log σ(margin) + ε * -log σ(-margin)
            loss = (
                (1 - self.label_smoothing) * -torch.nn.functional.logsigmoid(margin)
                + self.label_smoothing * -torch.nn.functional.logsigmoid(-margin)
            ).mean()
        else:
            loss = -torch.nn.functional.logsigmoid(margin).mean()

        with torch.no_grad():
            chosen_rewards = self.beta * chosen_lr
            rejected_rewards = self.beta * rejected_lr
            reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

            metrics = {
                "loss": loss.item(),
                "reward": batch.rewards.mean().item(),
                "chosen_reward": chosen_rewards.mean().item(),
                "rejected_reward": rejected_rewards.mean().item(),
                "reward_accuracy": reward_accuracy.item(),
                "margin": margin.mean().item(),
            }

        return loss, metrics
