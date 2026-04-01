"""SimPO — Simple Preference Optimization.

Paper: Meng et al., "SimPO: Simple Preference Optimization with a
Reference-Free Reward" (2024). https://arxiv.org/abs/2405.14734

Core idea: use length-normalized average log-probability as the implicit
reward, eliminating the need for a reference model. The reward for a
sequence is simply avg_log_prob = (1/|y|) Σ log π(y_t|y_<t, x).

L = -log σ(β (r_w - r_l) - γ)

where r = avg_log_prob and γ is a target margin.
"""

from __future__ import annotations

import torch
from torch import Tensor

from invokerl.algorithms.base import BaseAlgorithm, RolloutBatch


class SimPO(BaseAlgorithm):
    """Simple Preference Optimization (reference-free).

    Hyperparameters:
        beta: Temperature parameter.
        gamma: Target reward margin between chosen and rejected.
        chosen_first: If True, even indices are chosen, odd are rejected.
    """

    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 0.5,
        chosen_first: bool = True,
        **kwargs,
    ):
        super().__init__(beta=beta, gamma=gamma, **kwargs)
        self.beta = beta
        self.gamma = gamma
        self.chosen_first = chosen_first

    def compute_advantages(self, batch: RolloutBatch) -> Tensor:
        """SimPO doesn't use advantages — return zeros."""
        return torch.zeros_like(batch.response_mask, dtype=torch.float32)

    def compute_loss(
        self,
        new_log_probs: Tensor,
        batch: RolloutBatch,
        advantages: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """SimPO loss on preference pairs.

        No reference model needed — uses length-normalized avg log-prob
        as the implicit reward.
        """
        mask = batch.response_mask.float()

        # Length-normalized average log-prob per sequence (response only).
        response_lengths = mask.sum(dim=1).clamp(min=1.0)  # [B]
        avg_log_prob = (new_log_probs * mask).sum(dim=1) / response_lengths  # [B]

        # Split into chosen / rejected pairs.
        if self.chosen_first:
            chosen_reward = avg_log_prob[0::2]
            rejected_reward = avg_log_prob[1::2]
        else:
            chosen_reward = avg_log_prob[1::2]
            rejected_reward = avg_log_prob[0::2]

        # SimPO loss: -log σ(β * (r_w - r_l) - γ)
        margin = self.beta * (chosen_reward - rejected_reward) - self.gamma
        loss = -torch.nn.functional.logsigmoid(margin).mean()

        with torch.no_grad():
            reward_accuracy = (chosen_reward > rejected_reward).float().mean()

            metrics = {
                "loss": loss.item(),
                "reward": batch.rewards.mean().item(),
                "chosen_reward": chosen_reward.mean().item(),
                "rejected_reward": rejected_reward.mean().item(),
                "reward_accuracy": reward_accuracy.item(),
                "margin": margin.mean().item(),
                "avg_log_prob": avg_log_prob.mean().item(),
            }

        return loss, metrics
