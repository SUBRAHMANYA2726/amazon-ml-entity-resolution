"""Generic evaluation contracts without computing Phase 0 metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from business_entity_resolution.contracts import EvaluationReport, ThresholdSelection


class Evaluator(ABC):
    """Evaluate predictions at the validated entity level in a later phase."""

    @abstractmethod
    def evaluate(self, predictions: object, ground_truth: object) -> EvaluationReport:
        """Return a metric report using the official, inspected evaluation contract."""


class ThresholdOptimizer(ABC):
    """Select a threshold from validation evidence rather than a fixed default."""

    @abstractmethod
    def select(self, scored_predictions: object, ground_truth: object) -> ThresholdSelection:
        """Return a reproducible threshold selection from future validation data."""
