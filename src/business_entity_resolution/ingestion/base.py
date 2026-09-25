"""Schema-agnostic ingestion contracts for a future dataset audit."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path

from business_entity_resolution.contracts import ValidationResult


class DataIngestor(ABC):
    """Load a supplied dataset after its actual file contract is inspected."""

    @abstractmethod
    def load(self, location: Path) -> object:
        """Return an opaque dataset payload from a supplied location."""


class DatasetProfiler(ABC):
    """Produce a data-quality report without assuming a schema in Phase 0."""

    @abstractmethod
    def profile(self, dataset: object) -> Mapping[str, object]:
        """Return machine-readable profile metadata for an inspected dataset."""


class SchemaValidator(ABC):
    """Validate a schema contract established only after the dataset audit."""

    @abstractmethod
    def validate(self, dataset: object) -> ValidationResult:
        """Validate a dataset against the future, inspected schema contract."""
