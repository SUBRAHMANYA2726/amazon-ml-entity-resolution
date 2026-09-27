"""Official-submission validation interface."""

from business_entity_resolution.validation.base import SubmissionValidator
from business_entity_resolution.validation.entity_level import (
    DEFAULT_THRESHOLD_GRID,
    EntityPredictionRecord,
    ThresholdRecord,
    build_entity_predictions,
    evaluate_threshold_grid,
    perform_ambiguity_analysis,
    perform_conflict_analysis,
    perform_error_analysis,
    select_validation_threshold,
)
from business_entity_resolution.validation.submission_validation import (
    ValidationResult,
    validate_matching_results,
    validate_candidate_pairs,
    validate_subset,
    validate_all,
    check_deterministic_ordering,
)

__all__ = [
    "SubmissionValidator",
    "DEFAULT_THRESHOLD_GRID",
    "EntityPredictionRecord",
    "ThresholdRecord",
    "build_entity_predictions",
    "evaluate_threshold_grid",
    "perform_ambiguity_analysis",
    "perform_conflict_analysis",
    "perform_error_analysis",
    "select_validation_threshold",
    "ValidationResult",
    "validate_matching_results",
    "validate_candidate_pairs",
    "validate_subset",
    "validate_all",
    "check_deterministic_ordering",
]

