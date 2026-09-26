"""Evaluation metrics, diagnostic threshold sweeps, and error analysis for Phase 4.

Computes:
1. Top-K recall (Top-1, Top-3, Top-5, Top-10) overall, for S2, and for S3.
2. Diagnostic threshold sweep: Precision, Recall, F0.5 at thresholds [0.50 .. 0.95].
3. Baseline matching error analysis:
   - False negatives (true matches with low baseline score).
   - False positives (non-matches with high baseline score).
   - Root cause categorization.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F-beta score. Default beta=0.5 weights precision more heavily than recall."""
    if precision + recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    return float((1.0 + beta_sq) * (precision * recall) / (beta_sq * precision + recall))


@dataclass(frozen=True, slots=True)
class TopKRecallMetrics:
    """Recall metrics at different Top-K rank cutoffs."""

    top_1_recall: float
    top_3_recall: float
    top_5_recall: float
    top_10_recall: float
    total_true_pairs: int
    true_pairs_in_candidates: int
    true_at_1: int
    true_at_3: int
    true_at_5: int
    true_at_10: int


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """Pair-level precision, recall, and F0.5 at a single diagnostic threshold."""

    threshold: float
    predicted_pairs: int
    true_positives: int
    precision: float
    recall: float
    f0_5: float


@dataclass(frozen=True, slots=True)
class BaselineEvaluationReport:
    """Complete Phase 4 Baseline Matching Evaluation Report."""

    total_candidate_pairs: int
    total_true_pairs_gt: int
    true_pairs_in_candidate_pool: int
    phase3_candidate_recall: float

    # Top-K ranking performance
    overall_top_k: TopKRecallMetrics
    s2_top_k: TopKRecallMetrics
    s3_top_k: TopKRecallMetrics

    # Diagnostic threshold sweep
    diagnostic_thresholds: tuple[ThresholdMetrics, ...]

    # Error analysis summaries
    false_negative_count: int
    false_positive_count: int
    top_false_negatives: tuple[dict[str, Any], ...]
    top_false_positives: tuple[dict[str, Any], ...]
    error_patterns: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "total_candidate_pairs": self.total_candidate_pairs,
            "total_true_pairs_gt": self.total_true_pairs_gt,
            "true_pairs_in_candidate_pool": self.true_pairs_in_candidate_pool,
            "phase3_candidate_recall": round(self.phase3_candidate_recall, 4),
            "overall_top_k": {
                "top_1_recall": round(self.overall_top_k.top_1_recall, 4),
                "top_3_recall": round(self.overall_top_k.top_3_recall, 4),
                "top_5_recall": round(self.overall_top_k.top_5_recall, 4),
                "top_10_recall": round(self.overall_top_k.top_10_recall, 4),
                "true_at_1": self.overall_top_k.true_at_1,
                "true_at_3": self.overall_top_k.true_at_3,
                "true_at_5": self.overall_top_k.true_at_5,
                "true_at_10": self.overall_top_k.true_at_10,
            },
            "s2_top_k": {
                "top_1_recall": round(self.s2_top_k.top_1_recall, 4),
                "top_3_recall": round(self.s2_top_k.top_3_recall, 4),
                "top_5_recall": round(self.s2_top_k.top_5_recall, 4),
                "top_10_recall": round(self.s2_top_k.top_10_recall, 4),
            },
            "s3_top_k": {
                "top_1_recall": round(self.s3_top_k.top_1_recall, 4),
                "top_3_recall": round(self.s3_top_k.top_3_recall, 4),
                "top_5_recall": round(self.s3_top_k.top_5_recall, 4),
                "top_10_recall": round(self.s3_top_k.top_10_recall, 4),
            },
            "diagnostic_thresholds": [
                {
                    "threshold": tm.threshold,
                    "predicted_pairs": tm.predicted_pairs,
                    "true_positives": tm.true_positives,
                    "precision": round(tm.precision, 4),
                    "recall": round(tm.recall, 4),
                    "f0_5": round(tm.f0_5, 4),
                }
                for tm in self.diagnostic_thresholds
            ],
            "error_patterns": self.error_patterns,
            "false_negative_sample_size": len(self.top_false_negatives),
            "false_positive_sample_size": len(self.top_false_positives),
        }


class BaselineEvaluator:
    """Evaluates deterministic baseline matcher ranking and score calibration."""

    def __init__(self, ground_truth: Mapping[str, set[str]]) -> None:
        self.ground_truth = {str(k): set(v) for k, v in ground_truth.items()}

    def evaluate(
        self,
        ranked_df: pd.DataFrame,
        s1_lookup: Mapping[str, Mapping[str, Any]] | None = None,
        cand_lookup: Mapping[str, Mapping[str, Any]] | None = None,
        diagnostic_threshold_grid: Sequence[float] = (
            0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
        ),
    ) -> BaselineEvaluationReport:
        """Run complete Phase 4 baseline evaluation against ground truth."""
        LOGGER.info("Evaluating baseline matcher performance across %d pairs...", len(ranked_df))

        # Check / create ground truth indicator
        if "is_true_match" not in ranked_df.columns:
            is_true = [
                1 if cid in self.ground_truth.get(aid, set()) else 0
                for aid, cid in zip(ranked_df["source1_entity_id"], ranked_df["candidate_entity_id"])
            ]
            df = ranked_df.copy()
            df["is_true_match"] = is_true
        else:
            df = ranked_df

        # Anchors evaluated
        evaluated_anchors = set(df["source1_entity_id"])
        total_gt_pairs = sum(len(self.ground_truth.get(aid, set())) for aid in evaluated_anchors)
        total_gt_s2 = sum(
            len([cid for cid in self.ground_truth.get(aid, set()) if cid.startswith("S2-")])
            for aid in evaluated_anchors
        )
        total_gt_s3 = sum(
            len([cid for cid in self.ground_truth.get(aid, set()) if cid.startswith("S3-")])
            for aid in evaluated_anchors
        )

        true_pairs_in_pool = int(df["is_true_match"].sum())
        p3_cand_recall = true_pairs_in_pool / total_gt_pairs if total_gt_pairs > 0 else 0.0

        # 1. Top-K Recall Metrics
        overall_top_k = self._calc_top_k(df, total_gt_pairs)
        s2_df = df[df["candidate_source"] == "S2"]
        s2_top_k = self._calc_top_k(s2_df, total_gt_s2)
        s3_df = df[df["candidate_source"] == "S3"]
        s3_top_k = self._calc_top_k(s3_df, total_gt_s3)

        # 2. Diagnostic Threshold Sweep
        threshold_metrics_list = []
        for thresh in diagnostic_threshold_grid:
            sub = df[df["baseline_score"] >= thresh]
            n_pred = len(sub)
            n_tp = int(sub["is_true_match"].sum())
            prec = n_tp / n_pred if n_pred > 0 else 0.0
            rec = n_tp / total_gt_pairs if total_gt_pairs > 0 else 0.0
            f05 = compute_f_beta(prec, rec, beta=0.5)
            threshold_metrics_list.append(
                ThresholdMetrics(
                    threshold=thresh,
                    predicted_pairs=n_pred,
                    true_positives=n_tp,
                    precision=prec,
                    recall=rec,
                    f0_5=f05,
                )
            )

        # 3. Error Analysis
        top_fn, top_fp, error_patterns = self._analyze_errors(
            df, s1_lookup or {}, cand_lookup or {}
        )

        return BaselineEvaluationReport(
            total_candidate_pairs=len(df),
            total_true_pairs_gt=total_gt_pairs,
            true_pairs_in_candidate_pool=true_pairs_in_pool,
            phase3_candidate_recall=p3_cand_recall,
            overall_top_k=overall_top_k,
            s2_top_k=s2_top_k,
            s3_top_k=s3_top_k,
            diagnostic_thresholds=tuple(threshold_metrics_list),
            false_negative_count=len(top_fn),
            false_positive_count=len(top_fp),
            top_false_negatives=tuple(top_fn[:30]),
            top_false_positives=tuple(top_fp[:30]),
            error_patterns=error_patterns,
        )

    def _calc_top_k(self, df: pd.DataFrame, total_true_pairs: int) -> TopKRecallMetrics:
        """Compute recall at Top-1, Top-3, Top-5, Top-10 ranks."""
        true_in_cands = int(df["is_true_match"].sum())
        at_1 = int(df[(df["baseline_rank"] <= 1) & (df["is_true_match"] == 1)].shape[0])
        at_3 = int(df[(df["baseline_rank"] <= 3) & (df["is_true_match"] == 1)].shape[0])
        at_5 = int(df[(df["baseline_rank"] <= 5) & (df["is_true_match"] == 1)].shape[0])
        at_10 = int(df[(df["baseline_rank"] <= 10) & (df["is_true_match"] == 1)].shape[0])

        denom = total_true_pairs if total_true_pairs > 0 else 1
        return TopKRecallMetrics(
            top_1_recall=at_1 / denom,
            top_3_recall=at_3 / denom,
            top_5_recall=at_5 / denom,
            top_10_recall=at_10 / denom,
            total_true_pairs=total_true_pairs,
            true_pairs_in_candidates=true_in_cands,
            true_at_1=at_1,
            true_at_3=at_3,
            true_at_5=at_5,
            true_at_10=at_10,
        )

    def _analyze_errors(
        self,
        df: pd.DataFrame,
        s1_lookup: Mapping[str, Mapping[str, Any]],
        cand_lookup: Mapping[str, Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
        """Identify and categorize false negatives and false positives."""
        pattern_counts: Counter[str] = Counter()

        # False Negatives: True matches with low baseline score (e.g. < 0.60)
        fn_df = df[(df["is_true_match"] == 1) & (df["baseline_score"] < 0.60)].sort_values(
            by="baseline_score", ascending=True
        )

        false_negatives: list[dict[str, Any]] = []
        for _, row in fn_df.iterrows():
            aid = str(row["source1_entity_id"])
            cid = str(row["candidate_entity_id"])
            s1_rec = s1_lookup.get(aid, {})
            cand_rec = cand_lookup.get(cid, {})

            # Diagnose pattern
            s1_name = str(s1_rec.get("business_name", ""))
            c_name = str(cand_rec.get("business_name", ""))
            s1_addr = str(s1_rec.get("business_address", ""))
            c_addr = str(cand_rec.get("business_address", ""))

            cause = "other"
            if not c_addr or c_addr.strip().lower() in ("none", "nan", ""):
                cause = "missing_candidate_address"
            elif len(s1_name.split()) > 0 and len(c_name.split()) > 0 and sorted(s1_name.lower().split()) == sorted(c_name.lower().split()):
                cause = "word_order_variation"
            elif any(len(tok) <= 2 for tok in s1_name.split()) or any(len(tok) <= 2 for tok in c_name.split()):
                cause = "abbreviation_or_acronym"
            elif len(s1_addr) > 0 and len(c_addr) > 0:
                s1_num = "".join(filter(str.isdigit, s1_addr[:10]))
                c_num = "".join(filter(str.isdigit, c_addr[:10]))
                if s1_num and c_num and s1_num != c_num:
                    cause = "address_number_discrepancy"
                else:
                    cause = "address_formatting_variation"
            else:
                cause = "spelling_or_typo"

            pattern_counts[f"FN_{cause}"] += 1
            false_negatives.append({
                "source1_entity_id": aid,
                "candidate_entity_id": cid,
                "candidate_source": row["candidate_source"],
                "baseline_score": round(float(row["baseline_score"]), 4),
                "baseline_rank": int(row["baseline_rank"]),
                "name_sim": round(float(row["name_similarity_score"]), 4),
                "addr_sim": round(float(row["address_similarity_score"]), 4),
                "s1_name": s1_name,
                "cand_name": c_name,
                "s1_address": s1_addr,
                "cand_address": c_addr,
                "diagnosed_cause": cause,
            })

        # False Positives: Non-matches with high baseline score (e.g. >= 0.70)
        fp_df = df[(df["is_true_match"] == 0) & (df["baseline_score"] >= 0.70)].sort_values(
            by="baseline_score", ascending=False
        )

        false_positives: list[dict[str, Any]] = []
        for _, row in fp_df.iterrows():
            aid = str(row["source1_entity_id"])
            cid = str(row["candidate_entity_id"])
            s1_rec = s1_lookup.get(aid, {})
            cand_rec = cand_lookup.get(cid, {})

            s1_name = str(s1_rec.get("business_name", ""))
            c_name = str(cand_rec.get("business_name", ""))
            s1_addr = str(s1_rec.get("business_address", ""))
            c_addr = str(cand_rec.get("business_address", ""))

            cause = "other"
            if s1_name.strip().lower() == c_name.strip().lower():
                cause = "identical_name_distinct_location"
            elif s1_addr.strip().lower() == c_addr.strip().lower():
                cause = "co_located_different_business"
            else:
                cause = "high_token_overlap_collision"

            pattern_counts[f"FP_{cause}"] += 1
            false_positives.append({
                "source1_entity_id": aid,
                "candidate_entity_id": cid,
                "candidate_source": row["candidate_source"],
                "baseline_score": round(float(row["baseline_score"]), 4),
                "baseline_rank": int(row["baseline_rank"]),
                "name_sim": round(float(row["name_similarity_score"]), 4),
                "addr_sim": round(float(row["address_similarity_score"]), 4),
                "s1_name": s1_name,
                "cand_name": c_name,
                "s1_address": s1_addr,
                "cand_address": c_addr,
                "diagnosed_cause": cause,
            })

        return false_negatives, false_positives, dict(pattern_counts)
