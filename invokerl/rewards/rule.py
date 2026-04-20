"""Rule-based reward functions.

Score completions by checking if the extracted answer matches ground truth.
Used for math (GSM8K, MATH), code (execution result), and other tasks
where correctness is objectively verifiable.
"""

from __future__ import annotations

import re

import torch
from torch import Tensor

from invokerl.rewards.base import BaseReward


def _normalize_number(s: str) -> str | None:
    """Try to parse a string as a number and return a canonical form.

    Handles: integers, decimals, commas, percentages, negative numbers,
    trailing periods/dots.
    """
    s = s.strip().rstrip(".").replace(",", "").replace("%", "")
    s = s.replace("$", "").replace("€", "").replace("£", "")
    try:
        val = float(s)
        # Return as int if possible (to match "42" with "42.0")
        if val == int(val):
            return str(int(val))
        return f"{val:.6f}".rstrip("0").rstrip(".")
    except (ValueError, OverflowError):
        return None


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from a completion.

    Handles Qwen3-style thinking tags: if <think>...</think> is present,
    searches for the answer in the text AFTER the last </think> tag first,
    falling back to the full text if nothing is found there.

    Tries multiple patterns in priority order:
    1. #### <number> (GSM8K format)
    2. \\boxed{<number>} (MATH format)
    3. Last number in the text (fallback)
    """
    # If the model uses <think> tags, prefer content after </think>
    think_end = text.rfind("</think>")
    if think_end != -1:
        after_think = text[think_end + len("</think>") :]
        # Try structured patterns in post-thinking content first
        result = _extract_from_text(after_think)
        if result is not None:
            return result

    # Fall back to searching the full text
    return _extract_from_text(text)


def _extract_from_text(text: str) -> str | None:
    """Extract answer from a text segment using structured patterns."""
    # Pattern 1: GSM8K "#### <answer>" — capture numbers/words, stop at tags or newlines
    match = re.search(r"####\s*([^<\n]+)", text)
    if match:
        return match.group(1).strip()

    # Pattern 2: LaTeX \boxed{answer}
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()

    # Pattern 3: Last number in the text
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def check_answer(prediction: str | None, ground_truth: str) -> bool:
    """Check if prediction matches ground truth numerically."""
    if prediction is None:
        return False

    pred_norm = _normalize_number(prediction)
    gold_norm = _normalize_number(ground_truth)

    if pred_norm is not None and gold_norm is not None:
        return pred_norm == gold_norm

    # Fallback: exact string match (stripped)
    return prediction.strip() == ground_truth.strip()


class ExactMatch(BaseReward):
    """Binary reward: 1.0 if extracted answer matches ground truth, 0.0 otherwise.

    Works with any dataset that provides ground_truth in PromptItem.
    The ground truth is passed via the prompt metadata or directly.
    """

    def __init__(self, ground_truths: dict[str, str] | None = None):
        """
        Args:
            ground_truths: Optional mapping from prompt → expected answer.
                           If None, ground truth must be passed at score time.
        """
        self._ground_truths = ground_truths or {}

    def score(
        self,
        prompt: str,
        completion: str,
        tokens: list[int] | None = None,
        ground_truth: str | None = None,
    ) -> float:
        """Score a completion by exact answer matching.

        Args:
            prompt: The prompt string.
            completion: The generated completion.
            tokens: Unused (for interface compatibility).
            ground_truth: Expected answer. Falls back to self._ground_truths.

        Returns:
            1.0 if correct, 0.0 if incorrect.
        """
        if ground_truth is None:
            ground_truth = self._ground_truths.get(prompt)
        if ground_truth is None:
            raise ValueError(f"No ground truth for prompt (first 80 chars): {prompt[:80]!r}")

        predicted = extract_answer(completion)
        return 1.0 if check_answer(predicted, ground_truth) else 0.0

    def score_batch(
        self,
        prompts: list[str],
        completions: list[str],
        token_ids: Tensor | None = None,
        ground_truths: list[str] | None = None,
    ) -> Tensor:
        """Score a batch of completions.

        Args:
            prompts: List of prompt strings.
            completions: List of completion strings.
            token_ids: Unused.
            ground_truths: List of expected answers (one per prompt).

        Returns:
            rewards: [B] tensor of 0.0/1.0.
        """
        rewards = []
        for i, (p, c) in enumerate(zip(prompts, completions)):
            gt = ground_truths[i] if ground_truths is not None else None
            rewards.append(self.score(p, c, ground_truth=gt))
        return torch.tensor(rewards, dtype=torch.float32)
