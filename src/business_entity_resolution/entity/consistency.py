"""Generic entity-safety contracts with no unsupported cardinality constraint."""

from __future__ import annotations

from abc import ABC, abstractmethod

from business_entity_resolution.contracts import ConsistencyReport


class EntityConsistencyChecker(ABC):
    """Audit future predictions without imposing unproven entity constraints."""

    @abstractmethod
    def check(self, predictions: object, candidates: object) -> ConsistencyReport:
        """Return duplicate, validity, and consistency findings after schema review."""


class SingletonAuditor(ABC):
    """Audit no-match decisions separately from match coverage in a later phase."""

    @abstractmethod
    def audit(self, predictions: object, ground_truth: object | None = None) -> ConsistencyReport:
        """Return singleton-related audit findings when the required inputs exist."""
