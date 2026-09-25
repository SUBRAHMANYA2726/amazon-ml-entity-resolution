"""Generic normalization contract; field rules await dataset inspection."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Normalizer(ABC):
    """Create comparison-ready representations without deciding a match."""

    @abstractmethod
    def normalize(self, records: object) -> object:
        """Return normalized representations while preserving the raw payload."""
