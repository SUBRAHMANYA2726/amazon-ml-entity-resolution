"""Evaluation and threshold-selection interfaces."""

from business_entity_resolution.evaluation.base import Evaluator, ThresholdOptimizer
from business_entity_resolution.evaluation.metrics import (
    ThresholdEvaluationMetrics,
    build_validation_predictions_artifact,
    compute_distribution_summary,
    evaluate_predictions,
    extract_prediction_diagnostics,
)

__all__ = [
    "Evaluator",
    "ThresholdEvaluationMetrics",
    "ThresholdOptimizer",
    "build_validation_predictions_artifact",
    "compute_distribution_summary",
    "evaluate_predictions",
    "extract_prediction_diagnostics",
]
