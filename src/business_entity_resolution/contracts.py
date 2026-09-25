"""Schema-agnostic data contracts shared by future pipeline components.

These types describe internal pipeline boundaries only. They do not prescribe
challenge filenames, source labels, column names, or submission fields.
"""

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class CandidatePair:
    """A candidate relationship identified by opaque internal record references."""

    anchor_ref: str
    candidate_ref: str
    provenance: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """A generic validation outcome for schemas, submissions, or artifacts."""

    is_valid: bool
    messages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """A container for future metric values without calculating any metric."""

    values: Mapping[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    """A future validation-derived threshold decision."""

    threshold: float
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    """A generic entity-level consistency audit result."""

    is_consistent: bool
    messages: tuple[str, ...] = ()
