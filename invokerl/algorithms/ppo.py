"""PPO — Proximal Policy Optimization for RLHF.

Paper: Schulman et al., "Proximal Policy Optimization Algorithms" (2017).
https://arxiv.org/abs/1707.06347

Applied to RLHF: Ouyang et al., "Training language models to follow
instructions with human feedback" (2022). https://arxiv.org/abs/2203.02176

Core idea: clipped surrogate objective with a value function baseline
and entropy bonus. GAE (Generalized Advantage Estimation) computes
per-token advantages from value predictions and rewards.

L = L_policy + c1 * L_value + c2 * L_entropy
"""

from __future__ import annotations

import torch
from torch import Tensor

from invokerl.algorithms.base import BaseAlgorithm, RolloutBatch


class PPO(BaseAlgorithm):
    """Proximal Policy Optimization with GAE.

    Hyperparameters:
        clip_eps: Clipping epsilon for the policy surrogate.
        vf_clip_eps: Clipping epsilon for the value function.
        vf_coef: Value function loss coefficient.
        entropy_coef: Entropy bonus coefficient.
        gae_gamma: GAE discount factor.
        gae_lambda: GAE lambda for bias-variance tradeoff.
        beta: KL penalty against reference (0 = no KL penalty).
    """

    def __init__(
        self,
        clip_eps: float = 0.2,
        vf_clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gae_gamma: float = 1.0,
        gae_lambda: float = 0.95,
        beta: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            clip_eps=clip_eps, vf_coef=vf_coef, entropy_coef=entropy_coef,
            gae_gamma=gae_gamma, gae_lambda=gae_lambda, beta=beta, **kwargs,
        )
        self.clip_eps = clip_eps
        self.vf_clip_eps = vf_clip_eps
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.beta = beta

    def compute_advantages(self, batch: RolloutBatch) -> Tensor:
        """GAE advantages from value predictions and rewards.

        Requires batch.extras["values"] — [B, T] value predictions from
        the value head (computed during rollout). If not provided, falls
        back to sequence-level reward broadcast.
        """
        mask = batch.response_mask.float()
        values = batch.extras.get("values")

        if values is None:
            # Fallback: broadcast sequence reward (no value function).
            adv = batch.rewards.unsqueeze(1).expand_as(mask)
            return adv * mask

        # GAE reverse scan.
        B, T = mask.shape
        advantages = torch.zeros_like(mask)
        last_gae = torch.zeros(B, device=mask.device)

        # Token rewards if available, else assign sequence reward to last token.
        if batch.token_rewards is not None:
            rewards = batch.token_rewards
        else:
            rewards = torch.zeros_like(mask)
            # Place sequence reward at the last response token.
            for i in range(B):
                resp_idx = batch.response_mask[i].nonzero(as_tuple=True)[0]
                if len(resp_idx) > 0:
                    rewards[i, resp_idx[-1]] = batch.rewards[i]

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros(B, device=mask.device)
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + self.gae_gamma * next_value - values[:, t]
            last_gae = delta + self.gae_gamma * self.gae_lambda * last_gae * mask[:, t]
            advantages[:, t] = last_gae

        return advantages * mask

    def compute_loss(
        self,
        new_log_probs: Tensor,
        batch: RolloutBatch,
        advantages: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """PPO clipped surrogate + value loss + entropy bonus.

        Requires batch.extras["values"] for the value loss component.
        If not present, only computes the policy loss.
        """
        mask = batch.response_mask.float()
        old_lp = batch.old_log_probs
        num_tokens = mask.sum().clamp(min=1.0)

        # Normalize advantages.
        adv_masked = advantages[mask.bool()]
        if adv_masked.numel() > 1:
            advantages = (advantages - adv_masked.mean()) / (adv_masked.std() + 1e-8)
            advantages = advantages * mask

        # -- Clipped policy loss --
        log_ratio = new_log_probs - old_lp
        ratio = log_ratio.exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2)
        policy_loss_mean = (policy_loss * mask).sum() / num_tokens

        # -- Entropy bonus --
        # Approximate entropy from log-probs: H ≈ -mean(log_prob)
        entropy = -(new_log_probs * mask).sum() / num_tokens
        entropy_loss = -self.entropy_coef * entropy

        # -- Value loss (if value predictions available) --
        values = batch.extras.get("values")
        old_values = batch.extras.get("old_values")
        vf_loss_mean = torch.tensor(0.0, device=mask.device)

        if values is not None and old_values is not None:
            returns = advantages + old_values
            # Clipped value loss.
            vf_clipped = old_values + (values - old_values).clamp(
                -self.vf_clip_eps, self.vf_clip_eps,
            )
            vf_loss1 = (values - returns).pow(2)
            vf_loss2 = (vf_clipped - returns).pow(2)
            vf_loss_mean = 0.5 * (torch.max(vf_loss1, vf_loss2) * mask).sum() / num_tokens

        # -- KL penalty (optional) --
        kl_loss = torch.tensor(0.0, device=mask.device)
        if self.beta > 0:
            ref_lp = batch.ref_log_probs
            log_r = ref_lp - new_log_probs
            kl_per_token = log_r.exp() - log_r - 1.0  # Schulman estimator
            kl_loss = self.beta * (kl_per_token * mask).sum() / num_tokens

        # -- Total loss --
        loss = policy_loss_mean + self.vf_coef * vf_loss_mean + entropy_loss + kl_loss

        with torch.no_grad():
            metrics = {
                "loss": loss.item(),
                "reward": batch.rewards.mean().item(),
                "policy_loss": policy_loss_mean.item(),
                "vf_loss": vf_loss_mean.item(),
                "entropy": entropy.item(),
                "clip_frac": (
                    ((ratio - 1.0).abs() > self.clip_eps).float() * mask
                ).sum().item() / num_tokens.item(),
                "approx_kl": (
                    (0.5 * log_ratio.pow(2)) * mask
                ).sum().item() / num_tokens.item(),
                "advantages_mean": advantages[mask.bool()].mean().item()
                if mask.any()
                else 0.0,
            }
            if self.beta > 0:
                metrics["kl"] = (kl_per_token * mask).sum().item() / num_tokens.item()

        return loss, metrics
