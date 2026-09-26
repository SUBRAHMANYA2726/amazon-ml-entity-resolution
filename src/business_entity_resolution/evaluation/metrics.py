"""Evaluation metrics, diagnostic threshold sweeps, and error analysis for Phase 6.

Calculates:
1. Validation ranking & threshold metrics: PR-AUC, ROC-AUC, Precision, Recall, F0.5, FP, FN.
2. Probability distribution diagnostics across all, positive, and negative pairs.
3. High-confidence false positives and low-confidence true positives.
4. Construction of deterministic validation prediction artifact sorted by:
   source1_entity_id ASC, match_probability DESC, candidate_entity_id ASC.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from business_entity_resolution.matching.evaluation import compute_f_beta
from business_entity_resolution.models.pairwise import validate_probabilities
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)

DEFAULT_DIAGNOSTIC_THRESHOLDS: tuple[float, ...] = (
    0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
)


@dataclass(frozen=True, slots=True)
class ThresholdEvaluationMetrics:
    """Pair-level performance metrics at a single diagnostic threshold."""

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f0_5: float
    f1: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_distribution_summary(values: np.ndarray | pd.Series | Sequence[float]) -> dict[str, float]:
    """Compute distribution summary statistics (min, max, mean, std, percentiles)."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def evaluate_predictions(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    diagnostic_thresholds: Sequence[float] = DEFAULT_DIAGNOSTIC_THRESHOLDS,
) -> dict[str, Any]:
    """Calculate pair-level validation metrics across diagnostic thresholds.

    IMPORTANT: Phase 6 does NOT declare a final production threshold.
    Threshold metrics are purely diagnostic.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32).ravel()
    y_prob_arr = validate_probabilities(np.asarray(y_prob, dtype=np.float64))

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError(
            f"y_true length ({len(y_true_arr)}) does not match y_prob length ({len(y_prob_arr)})"
        )

    n_pos = int((y_true_arr == 1).sum())
    n_neg = int((y_true_arr == 0).sum())

    # Calculate global ranking metrics
    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr)) if n_pos > 0 else 0.0
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr)) if n_pos > 0 and n_neg > 0 else 0.0

    sweep_results: list[dict[str, Any]] = []
    for thresh in diagnostic_thresholds:
        preds = (y_prob_arr >= thresh).astype(np.int32)
        tp = int(((preds == 1) & (y_true_arr == 1)).sum())
        fp = int(((preds == 1) & (y_true_arr == 0)).sum())
        tn = int(((preds == 0) & (y_true_arr == 0)).sum())
        fn = int(((preds == 0) & (y_true_arr == 1)).sum())

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f0_5 = compute_f_beta(precision, recall, beta=0.5)
        f1 = compute_f_beta(precision, recall, beta=1.0)

        m = ThresholdEvaluationMetrics(
            threshold=float(thresh),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f0_5=f0_5,
            f1=f1,
        )
        sweep_results.append(m.to_dict())

    # Overall, positive, and negative probability distributions
    pos_probs = y_prob_arr[y_true_arr == 1]
    neg_probs = y_prob_arr[y_true_arr == 0]

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "total_pairs": len(y_true_arr),
        "total_positives": n_pos,
        "total_negatives": n_neg,
        "probability_distributions": {
            "overall": compute_distribution_summary(y_prob_arr),
            "positive_pairs": compute_distribution_summary(pos_probs),
            "negative_pairs": compute_distribution_summary(neg_probs),
        },
        "diagnostic_threshold_sweep": sweep_results,
    }


def build_validation_predictions_artifact(
    val_df: pd.DataFrame,
    match_probabilities: np.ndarray | Sequence[float],
) -> pd.DataFrame:
    """Build the deterministic validation prediction artifact.

    Mandatory Columns:
    - source1_entity_id
    - candidate_entity_id
    - candidate_source
    - match_probability
    - ground_truth_label
    - baseline_score
    - candidate provenance (blocking_strategy_count, blocking_multi_strategy)
    - rank within source1 entity

    Sort deterministically by:
    1. source1_entity_id ascending
    2. match_probability descending
    3. candidate_entity_id ascending
    """
    probs = validate_probabilities(np.asarray(match_probabilities, dtype=np.float64))
    if len(probs) != len(val_df):
        raise ValueError(
            f"Probabilities length ({len(probs)}) does not match validation DataFrame rows ({len(val_df)})"
        )

    res_df = pd.DataFrame()
    res_df["source1_entity_id"] = val_df["source1_entity_id"].astype(str)
    res_df["candidate_entity_id"] = val_df["candidate_entity_id"].astype(str)
    res_df["candidate_source"] = val_df["candidate_source"].astype(str)
    res_df["match_probability"] = probs
    res_df["ground_truth_label"] = val_df["ground_truth_label"].astype(int)

    # Carry forward baseline_score if present in Phase 5
    if "baseline_score" in val_df.columns:
        res_df["baseline_score"] = val_df["baseline_score"].astype(float)
    else:
        res_df["baseline_score"] = np.nan

    # Carry forward provenance columns present in Phase 5
    provenance_cols = ["blocking_strategy_count", "blocking_multi_strategy"]
    for col in provenance_cols:
        if col in val_df.columns:
            res_df[col] = val_df[col]

    # Deterministic sort:
    # 1. source1_entity_id ASC
    # 2. match_probability DESC
    # 3. candidate_entity_id ASC
    res_df = res_df.sort_values(
        by=["source1_entity_id", "match_probability", "candidate_entity_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    # Compute rank within source1 entity (1-indexed)
    res_df["rank_within_source1"] = res_df.groupby("source1_entity_id").cumcount() + 1

    return res_df


def extract_prediction_diagnostics(
    pred_df: pd.DataFrame,
    top_n: int = 25,
) -> dict[str, Any]:
    """Extract high-confidence false positives and low-confidence true positives for error analysis."""
    fp_candidates = pred_df[pred_df["ground_truth_label"] == 0].sort_values(
        by=["match_probability", "candidate_entity_id"], ascending=[False, True]
    ).head(top_n)

    tp_candidates = pred_df[pred_df["ground_truth_label"] == 1].sort_values(
        by=["match_probability", "candidate_entity_id"], ascending=[True, True]
    ).head(top_n)

    return {
        "high_confidence_false_positives": fp_candidates.to_dict(orient="records"),
        "low_confidence_true_positives": tp_candidates.to_dict(orient="records"),
    }
