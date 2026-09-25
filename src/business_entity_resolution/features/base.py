"""Generic pairwise feature contract with no assumed record fields."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from business_entity_resolution.contracts import CandidatePair


class PairFeatureGenerator(ABC):
    """Generate deterministic pair-level features from inspected record fields."""

    @abstractmethod
    def generate(self, pairs: Iterable[CandidatePair], records: object) -> object:
        """Return a feature payload for candidate pairs without leaking labels."""
