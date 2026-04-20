"""DAPO — Dynamic Actor-Policy Optimization.

Paper: Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System with
Dynamic Sampling" (2025). https://arxiv.org/abs/2503.14476

Core idea: GRPO variant with four key modifications:
1. Clip-higher: asymmetric clipping (wider upper bound) for positive advantages
2. Dynamic sampling: filter out sequences where all completions are correct
   or all are incorrect (no learning signal from zero-variance groups)
3. Token-level loss: normalize by total response tokens, not by sequence count
4. Overlong penalty: penalize sequences that hit max_new_tokens (truncated)
"""

from __future__ import annotations

import torch
from torch import Tensor

from invokerl.algorithms.base import BaseAlgorithm, RolloutBatch


class DAPO(BaseAlgorithm):
    """Dynamic Actor-Policy Optimization.

    Hyperparameters:
        clip_eps_low: Lower clipping bound for policy ratio.
        clip_eps_high: Upper clipping bound (asymmetric, > clip_eps_low).
        beta: KL penalty coefficient.
        overlong_penalty: Penalty applied to sequences that were truncated.
        filter_zero_var: Drop groups with zero reward variance (dynamic sampling).
    """

    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.28,
        beta: float = 0.04,
        overlong_penalty: float = -1.0,
        filter_zero_var: bool = True,
        **kwargs,
    ):
        super().__init__(
            clip_eps_low=clip_eps_low,
            clip_eps_high=clip_eps_high,
            beta=beta,
            overlong_penalty=overlong_penalty,
            **kwargs,
        )
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.beta = beta
        self.overlong_penalty = overlong_penalty
        self.filter_zero_var = filter_zero_var

    def compute_advantages(self, batch: RolloutBatch) -> Tensor:
        """Group-normalized advantages with dynamic sampling.

        Like GRPO, but filters out groups where all rewards are equal
        (zero variance = no learning signal).
        """
        G = batch.group_size
        mask = batch.response_mask.float()
        rewards = batch.rewards  # [B]
        grouped = rewards.view(-1, G)  # [num_groups, G]

        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True)

        if self.filter_zero_var:
            # Dynamic sampling: zero out advantages for zero-variance groups.
            # These groups have all-correct or all-incorrect completions.
            valid = (std > 1e-8).float()
            std = std.clamp(min=1e-8)
            adv = ((grouped - mean) / std) * valid
        else:
            std = std.clamp(min=1e-8)
            adv = (grouped - mean) / std

        adv = adv.view(-1)  # [B]

        # Apply overlong penalty to truncated sequences.
        if self.overlong_penalty != 0.0 and batch.extras.get("truncated") is not None:
            truncated = batch.extras["truncated"]  # [B] bool
            adv = torch.where(
                truncated, torch.tensor(self.overlong_penalty, device=adv.device), adv
            )

        return adv.unsqueeze(1).expand_as(mask) * mask

    def compute_loss(
        self,
        new_log_probs: Tensor,
        batch: RolloutBatch,
        advantages: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """DAPO loss: asymmetric clipping + token-level normalization + KL.

        Key differences from GRPO:
        - Clip-higher: upper clip bound > lower clip bound for positive advantages
        - Token-level loss: sum over tokens / total_tokens (not per-sequence mean)
        """
        mask = batch.response_mask.float()
        old_lp = batch.old_log_probs
        ref_lp = batch.ref_log_probs
        num_tokens = mask.sum().clamp(min=1.0)

        # -- Asymmetric clipped surrogate (clip-higher) --
        log_ratio = new_log_probs - old_lp
        ratio = log_ratio.exp()

        surr1 = ratio * advantages
        # Clip-higher: tighter lower bound, wider upper bound.
        surr2 = (
            ratio.clamp(
                1.0 - self.clip_eps_low,
                1.0 + self.clip_eps_high,
            )
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2)

        # -- KL penalty (Schulman estimator) --
        log_r = ref_lp - new_log_probs
        kl_per_token = log_r.exp() - log_r - 1.0

        # -- Token-level total loss --
        token_loss = policy_loss + self.beta * kl_per_token
        loss = (token_loss * mask).sum() / num_tokens

        # -- Metrics --
        with torch.no_grad():
            G = batch.group_size
            grouped_rewards = batch.rewards.view(-1, G)
            num_groups = grouped_rewards.shape[0]
            zero_var_groups = (grouped_rewards.std(dim=1) < 1e-8).sum().item()

            metrics = {
                "loss": loss.item(),
                "reward": batch.rewards.mean().item(),
                "kl": (kl_per_token * mask).sum().item() / num_tokens.item(),
                "policy_loss": (policy_loss * mask).sum().item() / num_tokens.item(),
                "clip_frac": (((ratio - 1.0).abs() > self.clip_eps_low).float() * mask).sum().item()
                / num_tokens.item(),
                "approx_kl": ((0.5 * log_ratio.pow(2)) * mask).sum().item() / num_tokens.item(),
                "zero_var_groups": zero_var_groups,
                "filtered_frac": zero_var_groups / max(1, num_groups),
                "advantages_mean": advantages[mask.bool()].mean().item() if mask.any() else 0.0,
            }

        return loss, metrics
