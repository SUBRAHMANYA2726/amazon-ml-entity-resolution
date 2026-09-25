"""Generic submission-generation contract with no assumed output schema."""

from __future__ import annotations

from abc import ABC, abstractmethod


class SubmissionGenerator(ABC):
    """Generate the official format only after its exact contract is inspected."""

    @abstractmethod
    def generate(self, predictions: object) -> object:
        """Return a submission artifact using future official requirements."""
