"""Generic probabilistic matcher contract; no model is trained in Phase 0."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Matcher(ABC):
    """Fit a permitted model later and score pairwise feature payloads."""

    @abstractmethod
    def fit(self, features: object, labels: object) -> None:
        """Fit the matcher using leakage-safe, future training data."""

    @abstractmethod
    def score(self, features: object) -> Sequence[float]:
        """Return one model score per supplied candidate pair."""
