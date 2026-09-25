"""Generic inference orchestration contract for the future full pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod


class InferencePipeline(ABC):
    """Run the future schema-aware inference sequence on supplied test data."""

    @abstractmethod
    def run(self, inputs: object) -> object:
        """Return an inference artifact after all data contracts are implemented."""
