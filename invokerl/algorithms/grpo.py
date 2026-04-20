"""GRPO — Group Relative Policy Optimization.

Paper: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning
in Open Language Models" (2024). https://arxiv.org/abs/2402.03300

Core idea: generate multiple completions per prompt, compute advantages by
normalizing rewards within each group, optimize a clipped surrogate objective
with a KL penalty against the reference policy.

Credit assignment hook: override compute_advantages() to experiment with
different ways of turning per-sequence rewards into per-token learning signals
(group normalization, token-level shaping, PRM scores, etc.).
"""

from __future__ import annotations

import torch
from torch import Tensor

from invokerl.algorithms.base import BaseAlgorithm, RolloutBatch


class GRPO(BaseAlgorithm):
    """Group Relative Policy Optimization.

    Hyperparameters:
        clip_eps: PPO-style clipping epsilon for the surrogate objective.
        beta: KL penalty coefficient against the reference policy.
        kl_type: KL divergence estimator — "schulman" (default) or "simple".
    """

    def __init__(
        self,
        clip_eps: float = 0.2,
        beta: float = 0.04,
        kl_type: str = "schulman",
        **kwargs,
    ):
        super().__init__(clip_eps=clip_eps, beta=beta, kl_type=kl_type, **kwargs)
        self.clip_eps = clip_eps
        self.beta = beta
        self.kl_type = kl_type

    # -- Credit assignment -----------------------------------------------------

    def compute_advantages(self, batch: RolloutBatch) -> Tensor:
        """Group-normalized advantages.

        For each prompt group, normalize rewards to zero-mean unit-variance.
        This makes the algorithm invariant to reward scale and encourages
        relative ranking within each group.

        If token_rewards are provided (credit assignment experiment), uses those
        instead of per-sequence rewards.

        Returns:
            advantages: [B, T] per-token advantages.
        """
        G = batch.group_size
        mask = batch.response_mask.float()

        if batch.token_rewards is not None:
            # Token-level credit assignment: normalize per-token rewards
            # within each group, per position
            tr = batch.token_rewards  # [B, T]
            tr_grouped = tr.view(-1, G, tr.shape[1])  # [num_groups, G, T]
            mean = tr_grouped.mean(dim=1, keepdim=True)
            std = tr_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
            adv = ((tr_grouped - mean) / std).view(-1, tr.shape[1])
            return adv * mask

        # Default: sequence-level group normalization
        rewards = batch.rewards  # [B]
        grouped = rewards.view(-1, G)  # [num_groups, G]
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        adv = ((grouped - mean) / std).view(-1)  # [B]

        # Broadcast to per-token
        return adv.unsqueeze(1).expand_as(mask) * mask

    # -- Policy loss -----------------------------------------------------------

    def compute_loss(
        self,
        new_log_probs: Tensor,
        batch: RolloutBatch,
        advantages: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Clipped surrogate objective with KL penalty.

        L = -E_t[ min(ratio * A, clip(ratio) * A) ] + beta * KL(ref || policy)

        Args:
            new_log_probs: [B, T] log-probs from current policy (has grad).
            batch: Rollout data (old_log_probs, ref_log_probs, masks).
            advantages: [B, T] from compute_advantages().

        Returns:
            loss: Scalar loss (minimized).
            metrics: Loggable metrics dict.
        """
        mask = batch.response_mask.float()
        old_lp = batch.old_log_probs
        ref_lp = batch.ref_log_probs

        # -- Clipped surrogate --
        log_ratio = new_log_probs - old_lp
        ratio = log_ratio.exp()

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2)

        # -- KL penalty (Schulman estimator: r - log(r) - 1 where r = ref/policy) --
        log_r = ref_lp - new_log_probs  # log(ref/policy)
        if self.kl_type == "schulman":
            # Schulman (2020): KL ≈ exp(log_r) - log_r - 1
            # Unbiased, non-negative estimator of KL(ref || policy)
            kl_per_token = log_r.exp() - log_r - 1.0
        else:
            # Simple: KL ≈ -log_r  (= new_lp - ref_lp, biased but stable)
            kl_per_token = -log_r

        # -- Total loss --
        token_loss = policy_loss + self.beta * kl_per_token
        num_tokens = mask.sum().clamp(min=1.0)
        loss = (token_loss * mask).sum() / num_tokens

        # -- Metrics --
        with torch.no_grad():
            metrics = {
                "loss": loss.item(),
                "reward": batch.rewards.mean().item(),
                "kl": (kl_per_token * mask).sum().item() / num_tokens.item(),
                "policy_loss": (policy_loss * mask).sum().item() / num_tokens.item(),
                "clip_frac": (((ratio - 1.0).abs() > self.clip_eps).float() * mask).sum().item()
                / num_tokens.item(),
                "approx_kl": ((0.5 * log_ratio.pow(2)) * mask).sum().item() / num_tokens.item(),
                "advantages_mean": advantages[mask.bool()].mean().item() if mask.any() else 0.0,
            }

        return loss, metrics
