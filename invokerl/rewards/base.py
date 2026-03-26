"""Reward function interface.

Reward functions score completions. They're pluggable — swap between
rule-based (answer matching), model-based (reward model), or custom
scoring without changing any algorithm code.

For credit assignment experiments, reward functions can return per-token
rewards (not just per-sequence scalars).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseReward(ABC):
    """Base reward function interface."""

    @abstractmethod
    def score(
        self,
        prompt: str,
        completion: str,
        tokens: list[int] | None = None,
        ground_truth: str | None = None,
    ) -> float | Tensor:
        """Score a single completion.

        Args:
            prompt: The original prompt.
            completion: The generated completion text.
            tokens: Optional token IDs for token-level reward.
            ground_truth: Expected answer (for rule-based rewards).

        Returns:
            float: Scalar reward (sequence-level).
            Tensor: [T] per-token rewards (for credit assignment experiments).
        """
        ...

    def score_batch(
        self,
        prompts: list[str],
        completions: list[str],
        token_ids: Tensor | None = None,
        ground_truths: list[str] | None = None,
    ) -> Tensor:
        """Score a batch of completions.

        Default implementation calls score() in a loop. Override for
        batched scoring (e.g., reward model inference).

        Args:
            prompts: List of prompts (may contain duplicates for groups).
            completions: List of completion texts, same length as prompts.
            token_ids: Optional token IDs for token-level rewards.
            ground_truths: Expected answers (for rule-based rewards).

        Returns:
            rewards: [B] tensor of scalar rewards.
        """
        rewards = []
        for i, (p, c) in enumerate(zip(prompts, completions)):
            toks = token_ids[i].tolist() if token_ids is not None else None
            gt = ground_truths[i] if ground_truths is not None else None
            r = self.score(p, c, toks, ground_truth=gt)
            rewards.append(r if isinstance(r, (int, float)) else r.item())
        return torch.tensor(rewards, dtype=torch.float32)
