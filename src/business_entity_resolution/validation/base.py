"""Contract for invoking the supplied official validator in a later phase."""

from __future__ import annotations

from abc import ABC, abstractmethod

from business_entity_resolution.contracts import ValidationResult


class SubmissionValidator(ABC):
    """Validate generated artifacts against the official supplied validator."""

    @abstractmethod
    def validate(self, submission: object, candidate_artifact: object) -> ValidationResult:
        """Return official-validator findings after that script is supplied."""
