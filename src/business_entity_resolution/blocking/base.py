"""Generic blocking contract with no field-specific strategy implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from business_entity_resolution.contracts import CandidatePair


class Blocker(ABC):
    """Generate a bounded candidate set instead of an all-pairs comparison."""

    @abstractmethod
    def generate(self, anchors: object, candidates: object) -> Iterable[CandidatePair]:
        """Return candidate pairs using a strategy defined after schema inspection."""
