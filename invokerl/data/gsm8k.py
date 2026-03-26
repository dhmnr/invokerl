"""GSM8K dataset — 8.5K grade-school math word problems.

Loads from HuggingFace datasets. Each problem has a question and a numeric
answer. Used with rule-based reward (exact numeric match).
"""

from __future__ import annotations

import random
import re

from datasets import load_dataset

from invokerl.data.base import BaseDataset, PromptItem


# GSM8K answer format: "#### <number>" at the end of the solution
_ANSWER_RE = re.compile(r"####\s*(.+)")


def _extract_gold_answer(solution: str) -> str:
    """Extract the numeric answer from GSM8K solution string."""
    match = _ANSWER_RE.search(solution)
    if match:
        return match.group(1).strip().replace(",", "")
    return solution.strip()


def _format_prompt(question: str) -> str:
    """Format a GSM8K question as a Qwen3 ChatML prompt.

    Qwen3 expects <|im_start|>/<|im_end|> chat markers. Without them,
    the model doesn't know it's in assistant mode and produces garbage.
    """
    return (
        "<|im_start|>system\n"
        "You are a helpful math assistant. Solve the problem step by step. "
        "End your response with #### followed by the numeric answer.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class GSM8KDataset(BaseDataset):
    """GSM8K training and test dataset.

    Args:
        split: "train" or "test".
        max_samples: Limit dataset size (0 = all).
        seed: Random seed for sampling.
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: int = 0,
        seed: int = 42,
    ):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        self.items: list[PromptItem] = []

        for row in ds:
            question = row["question"]
            answer = _extract_gold_answer(row["answer"])
            self.items.append(
                PromptItem(
                    prompt=_format_prompt(question),
                    ground_truth=answer,
                    metadata={"question": question, "raw_answer": row["answer"]},
                )
            )

        if max_samples > 0:
            self.items = self.items[:max_samples]

        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> PromptItem:
        return self.items[idx]

    def sample(self, n: int) -> list[PromptItem]:
        """Sample n random prompts."""
        return self._rng.choices(self.items, k=n)
