"""Generic training contracts; Phase 0 does not construct labeled pairs."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Trainer(ABC):
    """Train a matcher from candidate-respecting, entity-safe inputs later."""

    @abstractmethod
    def train(self, candidate_pairs: object, features: object, labels: object) -> object:
        """Return a trained artifact after Phase 1+ data contracts exist."""


class HardNegativeMiner(ABC):
    """Find difficult false positives without contaminating validation splits."""

    @abstractmethod
    def mine(self, scored_pairs: object, labels: object) -> object:
        """Return future hard-negative records from a split-safe candidate set."""
