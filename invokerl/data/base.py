"""Dataset interface for training data.

Datasets provide prompts for generation and optional ground truth for
reward computation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PromptItem:
    """A single training prompt with optional ground truth."""

    prompt: str                 # formatted prompt string
    ground_truth: str | None    # expected answer (for rule-based rewards)
    metadata: dict | None       # any extra info (source, difficulty, etc.)


class BaseDataset(ABC):
    """Abstract dataset interface."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> PromptItem:
        ...

    @abstractmethod
    def sample(self, n: int) -> list[PromptItem]:
        """Sample n random prompts for a training batch."""
        ...
