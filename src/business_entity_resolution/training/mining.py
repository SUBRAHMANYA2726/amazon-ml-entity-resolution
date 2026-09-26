"""Hard-negative mining implementation for Phase 7.

Identifies difficult negative candidate pairs from the training partition that
exhibit high predicted match probability or strong lexical/address similarity
to reference entities, while guaranteeing zero validation leakage.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from business_entity_resolution.training.base import HardNegativeMiner
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)

REQUIRED_HARD_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "model_probability",
    "baseline_score",
    "ground_truth_label",
    "hard_negative_reason",
)


@dataclass(frozen=True, slots=True)
class HardNegativeMiningConfig:
    """Configuration for hard negative selection rules."""

    negative_probability_percentile: float = 99.5
    min_model_probability: float | None = None
    min_baseline_score: float = 0.65
    min_name_similarity: float = 0.85
    enable_probability_threshold: bool = True
    enable_baseline_threshold: bool = True
    enable_name_collision: bool = True


@dataclass(frozen=True, slots=True)
class HardNegativeMiningResult:
    """Artifact containing mined hard negatives and detailed diagnostic statistics."""

    hard_negatives_df: pd.DataFrame
    statistics: dict[str, Any]
    selection_criteria_description: str
    hard_features_df: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "statistics": self.statistics,
            "selection_criteria_description": self.selection_criteria_description,
        }


class ProductionHardNegativeMiner(HardNegativeMiner):
    """Mines high-difficulty false positives strictly from the training partition."""

    def __init__(self, config: HardNegativeMiningConfig | None = None) -> None:
        self.config = config or HardNegativeMiningConfig()

    def mine(
        self,
        scored_pairs: object,
        labels: object | None = None,
    ) -> HardNegativeMiningResult:
        """Mine hard negatives from scored candidate pairs.

        Parameters
        ----------
        scored_pairs : pd.DataFrame
            Scored training pairs containing features and model probabilities.
        labels : pd.Series or np.ndarray or None
            Binary ground truth labels (if not in scored_pairs).

        Returns
        -------
        HardNegativeMiningResult
            Mined hard negatives dataframe and diagnostic statistics.
        """
        if not isinstance(scored_pairs, pd.DataFrame):
            raise TypeError(
                f"Expected pd.DataFrame for scored_pairs, got {type(scored_pairs).__name__}"
            )

        df = scored_pairs.copy()

        # 0. Check required identifier columns
        required_cols = {"source1_entity_id", "candidate_entity_id", "candidate_source"}
        missing_ids = required_cols - set(df.columns)
        if missing_ids:
            raise ValueError(
                f"Missing required columns: {sorted(missing_ids)}"
            )

        # 1. Resolve labels
        if labels is not None:
            df["ground_truth_label"] = np.asarray(labels, dtype=np.int32).ravel()
        elif "ground_truth_label" not in df.columns:
            raise ValueError(
                "Missing 'ground_truth_label'. Hard negatives must be verified against true labels."
            )

        # 2. Verify model probability exists
        if "model_probability" not in df.columns:
            raise ValueError(
                "Missing 'model_probability' column. Run model scoring before hard negative mining."
            )

        total_pairs = len(df)
        total_positives = int((df["ground_truth_label"] == 1).sum())
        total_negatives = int((df["ground_truth_label"] == 0).sum())

        LOGGER.info(
            "Analyzing %d training pairs (%d positive, %d negative) for hard negatives...",
            total_pairs,
            total_positives,
            total_negatives,
        )

        # 3. Filter strictly to negative pairs (y == 0)
        neg_df = df[df["ground_truth_label"] == 0].copy()

        # 4. Compute empirical threshold from actual negative training-score distribution
        if self.config.min_model_probability is not None:
            derived_prob_threshold = float(self.config.min_model_probability)
            percentile_used = None
            derivation_method = (
                f"Explicit threshold override: min_model_probability = {derived_prob_threshold:.6f}"
            )
        else:
            pct = float(self.config.negative_probability_percentile)
            if len(neg_df) == 0:
                derived_prob_threshold = 0.50
                percentile_used = pct
                derivation_method = "Fallback: no negative training pairs available"
            else:
                neg_probs = neg_df["model_probability"].to_numpy(dtype=np.float64)
                derived_prob_threshold = float(np.percentile(neg_probs, pct))
                percentile_used = pct
                derivation_method = (
                    f"Calculated from negative training score distribution at the {pct:.1f}th percentile: "
                    f"np.percentile(negative_scores, {pct}) = {derived_prob_threshold:.6f}"
                )

        LOGGER.info(
            "Hard negative probability threshold: %.6f (derivation: %s)",
            derived_prob_threshold,
            derivation_method,
        )

        # 5. Apply multi-criteria hard negative selection
        cond_prob = pd.Series(False, index=neg_df.index)
        if self.config.enable_probability_threshold:
            cond_prob = neg_df["model_probability"] >= derived_prob_threshold

        cond_base = pd.Series(False, index=neg_df.index)
        if self.config.enable_baseline_threshold and "baseline_score" in neg_df.columns:
            cond_base = neg_df["baseline_score"] >= self.config.min_baseline_score

        cond_name = pd.Series(False, index=neg_df.index)
        if self.config.enable_name_collision:
            core_match = (
                (neg_df["name_core_exact_match"] == 1.0)
                if "name_core_exact_match" in neg_df.columns
                else pd.Series(False, index=neg_df.index)
            )
            high_name_sim = (
                (neg_df["name_token_jaccard"] >= self.config.min_name_similarity)
                if "name_token_jaccard" in neg_df.columns
                else pd.Series(False, index=neg_df.index)
            )
            cond_name = core_match | high_name_sim

        # Combined hard negative mask
        hard_mask = cond_prob | cond_base | cond_name
        hard_negs = neg_df[hard_mask].copy()

        # 6. Programmatic Validation: Check that every single selected hard negative has label == 0
        non_zero_labels = (hard_negs["ground_truth_label"] != 0).sum()
        if non_zero_labels > 0:
            raise ValueError(
                f"FATAL INTEGRITY VIOLATION: {non_zero_labels} hard negatives have ground_truth_label != 0! "
                "Hard negatives MUST strictly have label == 0."
            )

        # 7. Assign documented categorical reasons
        reasons: list[str] = []
        primary_categories: list[str] = []

        for _, row in hard_negs.iterrows():
            r = []
            p = float(row.get("model_probability", 0.0))
            b = float(row.get("baseline_score", 0.0))

            if p >= 0.50:
                r.append("high_model_probability_ge_0.50")
            elif p >= derived_prob_threshold:
                r.append(f"empirical_model_probability_ge_{derived_prob_threshold:.4f}")

            if b >= self.config.min_baseline_score:
                r.append("high_baseline_similarity")

            if row.get("name_core_exact_match") == 1.0:
                r.append("exact_core_name_collision")
            elif row.get("name_token_jaccard", 0.0) >= self.config.min_name_similarity:
                r.append("high_name_similarity")

            if row.get("postal_exact_match") == 1.0:
                r.append("same_postal_code")
            if row.get("address_exact_match") == 1.0:
                r.append("same_address")

            if not r:
                r.append("lexical_near_neighbor")

            reason_str = ";".join(r)
            reasons.append(reason_str)

            # Assign single primary category for aggregation
            if p >= 0.50:
                primary = "high_confidence_model_false_positive"
            elif "exact_core_name_collision" in r:
                primary = "exact_core_name_collision"
            elif "high_baseline_similarity" in r:
                primary = "high_baseline_similarity"
            elif "high_name_similarity" in r:
                primary = "high_name_similarity"
            else:
                primary = "moderate_probability_confusion"
            primary_categories.append(primary)

        hard_negs["hard_negative_reason"] = reasons
        hard_negs["primary_category"] = primary_categories

        # 8. Format clean output dataset
        out_df = pd.DataFrame()
        for col in REQUIRED_HARD_NEGATIVE_COLUMNS:
            if col in hard_negs.columns:
                out_df[col] = hard_negs[col]
            else:
                out_df[col] = np.nan

        # Deterministic sort
        out_df = out_df.sort_values(
            by=["model_probability", "baseline_score", "source1_entity_id", "candidate_entity_id"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)

        # 9. Compute machine-readable statistics
        n_hard = len(out_df)
        pct_hard = (n_hard / total_negatives * 100.0) if total_negatives > 0 else 0.0

        s2_count = int((out_df["candidate_source"] == "S2").sum())
        s3_count = int((out_df["candidate_source"] == "S3").sum())
        if s2_count == 0 and s3_count == 0:
            # Fallback to checking candidate_entity_id prefix
            s2_count = int(out_df["candidate_entity_id"].str.startswith("S2-").sum())
            s3_count = int(out_df["candidate_entity_id"].str.startswith("S3-").sum())

        prob_buckets = {
            "p >= 0.50": int((out_df["model_probability"] >= 0.50).sum()),
            "0.30 <= p < 0.50": int(((out_df["model_probability"] >= 0.30) & (out_df["model_probability"] < 0.50)).sum()),
            "0.20 <= p < 0.30": int(((out_df["model_probability"] >= 0.20) & (out_df["model_probability"] < 0.30)).sum()),
            "0.10 <= p < 0.20": int(((out_df["model_probability"] >= 0.10) & (out_df["model_probability"] < 0.20)).sum()),
            "0.05 <= p < 0.10": int(((out_df["model_probability"] >= 0.05) & (out_df["model_probability"] < 0.10)).sum()),
            "p < 0.05": int((out_df["model_probability"] < 0.05).sum()),
        }

        sim_buckets = {
            "baseline >= 0.80": int((out_df["baseline_score"] >= 0.80).sum()),
            "0.70 <= baseline < 0.80": int(((out_df["baseline_score"] >= 0.70) & (out_df["baseline_score"] < 0.80)).sum()),
            "0.65 <= baseline < 0.70": int(((out_df["baseline_score"] >= 0.65) & (out_df["baseline_score"] < 0.70)).sum()),
            "baseline < 0.65": int((out_df["baseline_score"] < 0.65).sum()),
        }

        reason_counts = dict(Counter(reasons).most_common(10))
        category_counts = dict(Counter(primary_categories).most_common())

        criteria_desc = (
            f"Mined negative candidate pairs (ground_truth_label == 0) meeting at least one of: "
            f"(1) model_probability >= {derived_prob_threshold:.6f} "
            f"({derivation_method}), "
            f"(2) baseline_score >= {self.config.min_baseline_score:.2f}, or "
            f"(3) name_core_exact_match == 1.0 or name_token_jaccard >= {self.config.min_name_similarity:.2f}."
        )

        n_training_negs_ge_thresh = int((neg_df["model_probability"] >= derived_prob_threshold).sum())
        pct_training_negs_ge_thresh = (
            round(n_training_negs_ge_thresh / len(neg_df) * 100.0, 4) if len(neg_df) > 0 else 0.0
        )

        stats = {
            "total_training_pairs": total_pairs,
            "total_positive_candidates": total_positives,
            "total_negative_candidates": total_negatives,
            "hard_negative_count": n_hard,
            "hard_negative_percentage": round(pct_hard, 4),
            "s2_hard_negative_count": s2_count,
            "s3_hard_negative_count": s3_count,
            "probability_buckets": prob_buckets,
            "similarity_buckets": sim_buckets,
            "major_hard_negative_categories": category_counts,
            "top_detailed_reasons": reason_counts,
            "empirical_threshold_info": {
                "percentile_used": percentile_used,
                "derived_probability_threshold": round(derived_prob_threshold, 6),
                "derivation_method": derivation_method,
                "training_negatives_evaluated": len(neg_df),
                "training_negatives_ge_threshold": n_training_negs_ge_thresh,
                "percentage_training_negatives_ge_threshold": pct_training_negs_ge_thresh,
            },
            "selection_criteria": {
                "min_model_probability": round(derived_prob_threshold, 6),
                "derived_model_probability_threshold": round(derived_prob_threshold, 6),
                "empirical_percentile": percentile_used,
                "derivation_method": derivation_method,
                "min_baseline_score": self.config.min_baseline_score,
                "min_name_similarity": self.config.min_name_similarity,
            },
        }

        LOGGER.info(
            "Hard negative mining complete: identified %d hard negatives (%.2f%% of negatives). "
            "S2: %d, S3: %d.",
            n_hard,
            pct_hard,
            s2_count,
            s3_count,
        )

        return HardNegativeMiningResult(
            hard_negatives_df=out_df,
            statistics=stats,
            selection_criteria_description=criteria_desc,
            hard_features_df=hard_negs.copy().reset_index(drop=True),
        )
