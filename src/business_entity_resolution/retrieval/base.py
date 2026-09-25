"""Generic candidate retrieval contract for future strategy unions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from business_entity_resolution.contracts import CandidatePair


class CandidateRetriever(ABC):
    """Coordinate candidate retrieval and deduplication across strategies."""

    @abstractmethod
    def retrieve(self, anchors: object, candidates: object) -> Iterable[CandidatePair]:
        """Return the deduplicated candidate set consumed by later components."""
