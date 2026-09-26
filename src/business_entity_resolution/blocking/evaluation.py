"""Phase 3 Ground Truth Candidate Recall Evaluation and Diagnostics.

Computes:
1. Pair-Level Candidate Recall (overall, S2, S3)
2. S1 Entity-Level Coverage (percentage of anchors with 100% of true matches retrieved)
3. Candidate Explosion Statistics (mean, median, p90, p95, p99, max per anchor)
4. Candidate Quality Distribution (0, 1, 2-10, 11-50, 51-100, >100)
5. Strategy-by-Strategy Recall Table
6. Cumulative Ablation Analysis
7. Missed True Pair Analysis with root cause categorization
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from business_entity_resolution.blocking.union import AnchorCandidates
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CandidateRecallReport:
    """Complete candidate generation and recall evaluation report."""

    total_anchors_evaluated: int
    total_true_anchors_with_matches: int
    total_singleton_anchors: int

    # Pair-level recall
    total_true_pairs: int
    true_pairs_retrieved: int
    pair_recall: float

    # S2 and S3 specific recall
    total_true_s2_pairs: int
    true_s2_retrieved: int
    s2_recall: float

    total_true_s3_pairs: int
    true_s3_retrieved: int
    s3_recall: float

    # Entity-level coverage
    anchors_with_full_coverage: int
    entity_coverage: float

    # Candidate volume & distribution
    total_candidate_pairs: int
    avg_candidates_per_anchor: float
    median_candidates_per_anchor: float
    p90_candidates_per_anchor: float
    p95_candidates_per_anchor: float
    p99_candidates_per_anchor: float
    max_candidates_per_anchor: int

    # Candidate quality buckets (percentages)
    pct_zero_candidates: float
    pct_one_candidate: float
    pct_two_to_ten: float
    pct_eleven_to_fifty: float
    pct_fifty_one_to_hundred: float
    pct_over_hundred: float

    # Strategy-by-strategy breakdown
    strategy_metrics: tuple[dict[str, Any], ...]
    ablation_metrics: tuple[dict[str, Any], ...]
    missed_pairs_sample: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "total_anchors_evaluated": self.total_anchors_evaluated,
            "total_true_anchors_with_matches": self.total_true_anchors_with_matches,
            "total_singleton_anchors": self.total_singleton_anchors,
            "total_true_pairs": self.total_true_pairs,
            "true_pairs_retrieved": self.true_pairs_retrieved,
            "pair_recall": round(self.pair_recall, 4),
            "total_true_s2_pairs": self.total_true_s2_pairs,
            "true_s2_retrieved": self.true_s2_retrieved,
            "s2_recall": round(self.s2_recall, 4),
            "total_true_s3_pairs": self.total_true_s3_pairs,
            "true_s3_retrieved": self.true_s3_retrieved,
            "s3_recall": round(self.s3_recall, 4),
            "anchors_with_full_coverage": self.anchors_with_full_coverage,
            "entity_coverage": round(self.entity_coverage, 4),
            "total_candidate_pairs": self.total_candidate_pairs,
            "avg_candidates_per_anchor": round(self.avg_candidates_per_anchor, 2),
            "median_candidates_per_anchor": round(self.median_candidates_per_anchor, 1),
            "p90_candidates_per_anchor": round(self.p90_candidates_per_anchor, 1),
            "p95_candidates_per_anchor": round(self.p95_candidates_per_anchor, 1),
            "p99_candidates_per_anchor": round(self.p99_candidates_per_anchor, 1),
            "max_candidates_per_anchor": self.max_candidates_per_anchor,
            "pct_zero_candidates": round(self.pct_zero_candidates, 2),
            "pct_one_candidate": round(self.pct_one_candidate, 2),
            "pct_two_to_ten": round(self.pct_two_to_ten, 2),
            "pct_eleven_to_fifty": round(self.pct_eleven_to_fifty, 2),
            "pct_fifty_one_to_hundred": round(self.pct_fifty_one_to_hundred, 2),
            "pct_over_hundred": round(self.pct_over_hundred, 2),
            "strategy_metrics": list(self.strategy_metrics),
            "ablation_metrics": list(self.ablation_metrics),
            "missed_pairs_sample": list(self.missed_pairs_sample),
        }


def parse_ground_truth(gt_series_or_df: pd.DataFrame | pd.Series | Mapping[str, str]) -> dict[str, set[str]]:
    """Parse ground truth mapping of anchor_id -> set of true matched IDs."""
    parsed: dict[str, set[str]] = {}

    if isinstance(gt_series_or_df, pd.DataFrame):
        aids = gt_series_or_df["source1_entity_id"].astype(str).tolist()
        matches = gt_series_or_df["matched_entity_ids"].fillna("").astype(str).tolist()
        for aid, raw_val in zip(aids, matches):
            cleaned = raw_val.strip()
            if not cleaned:
                parsed[aid] = set()
            else:
                parsed[aid] = set(cleaned.split(","))
    elif isinstance(gt_series_or_df, Mapping):
        for aid, val in gt_series_or_df.items():
            if val is None or pd.isna(val) or not str(val).strip():
                parsed[str(aid)] = set()
            else:
                parsed[str(aid)] = set(str(val).strip().split(","))

    return parsed


class CandidateRecallEvaluator:
    """Evaluates candidate generation performance against training ground truth."""

    def __init__(self, ground_truth: Mapping[str, set[str]] | pd.DataFrame) -> None:
        if isinstance(ground_truth, pd.DataFrame):
            self.ground_truth = parse_ground_truth(ground_truth)
        else:
            self.ground_truth = {str(k): set(v) for k, v in ground_truth.items()}

    def evaluate(
        self,
        anchor_candidates: Mapping[str, AnchorCandidates],
        strategy_candidate_maps: Sequence[tuple[str, Mapping[str, Mapping[str, Iterable[str]]]]] | None = None,
        anchor_records_df: pd.DataFrame | None = None,
        candidate_records_df: pd.DataFrame | None = None,
    ) -> CandidateRecallReport:
        """Run complete candidate recall evaluation over the candidate pool."""
        anchor_ids = list(anchor_candidates.keys())
        total_anchors = len(anchor_ids)

        total_true_pairs = 0
        true_pairs_retrieved = 0
        total_true_s2 = 0
        true_s2_retrieved = 0
        total_true_s3 = 0
        true_s3_retrieved = 0

        anchors_with_matches = 0
        singleton_anchors = 0
        full_coverage_anchors = 0

        candidate_counts: list[int] = []
        missed_pairs: list[dict[str, Any]] = []

        # Candidate record lookup for missed-pair diagnostics
        cand_lookup = {}
        if candidate_records_df is not None and not candidate_records_df.empty:
            cand_id_col = "entity_id" if "entity_id" in candidate_records_df.columns else candidate_records_df.columns[0]
            for _, r in candidate_records_df.iterrows():
                cand_lookup[str(r[cand_id_col])] = r.to_dict()

        anchor_lookup = {}
        if anchor_records_df is not None and not anchor_records_df.empty:
            anchor_id_col = "entity_id" if "entity_id" in anchor_records_df.columns else anchor_records_df.columns[0]
            for _, r in anchor_records_df.iterrows():
                anchor_lookup[str(r[anchor_id_col])] = r.to_dict()

        for aid in anchor_ids:
            acands = anchor_candidates[aid]
            retrieved_ids = set(acands.candidate_ids())
            cand_count = len(retrieved_ids)
            candidate_counts.append(cand_count)

            true_matches = self.ground_truth.get(aid, set())
            if not true_matches:
                singleton_anchors += 1
                continue

            anchors_with_matches += 1
            true_count = len(true_matches)
            total_true_pairs += true_count

            # Check retrieved matches
            matched_subset = true_matches.intersection(retrieved_ids)
            match_found_count = len(matched_subset)
            true_pairs_retrieved += match_found_count

            if match_found_count == true_count:
                full_coverage_anchors += 1

            # Count by source (S2 vs S3)
            for t_id in true_matches:
                src = t_id.split("-")[0] if "-" in t_id else "unknown"
                is_retrieved = t_id in retrieved_ids

                if src == "S2":
                    total_true_s2 += 1
                    if is_retrieved:
                        true_s2_retrieved += 1
                elif src == "S3":
                    total_true_s3 += 1
                    if is_retrieved:
                        true_s3_retrieved += 1

                if not is_retrieved and len(missed_pairs) < 100:
                    missed_info: dict[str, Any] = {
                        "source1_entity_id": aid,
                        "true_candidate_entity_id": t_id,
                        "target_source": src,
                        "anchor_data": anchor_lookup.get(aid, {}),
                        "candidate_data": cand_lookup.get(t_id, {}),
                        "retrieved_count": cand_count,
                    }
                    missed_pairs.append(missed_info)

        # Pair recall calculations
        pair_recall = true_pairs_retrieved / total_true_pairs if total_true_pairs > 0 else 1.0
        s2_recall = true_s2_retrieved / total_true_s2 if total_true_s2 > 0 else 1.0
        s3_recall = true_s3_retrieved / total_true_s3 if total_true_s3 > 0 else 1.0
        entity_coverage = full_coverage_anchors / anchors_with_matches if anchors_with_matches > 0 else 1.0

        # Candidate counts distribution
        counts_arr = np.array(candidate_counts, dtype=float) if candidate_counts else np.array([0.0])
        total_candidate_pairs = int(counts_arr.sum())
        avg_candidates = float(np.mean(counts_arr))
        median_candidates = float(np.median(counts_arr))
        p90_candidates = float(np.percentile(counts_arr, 90))
        p95_candidates = float(np.percentile(counts_arr, 95))
        p99_candidates = float(np.percentile(counts_arr, 99))
        max_candidates = int(np.max(counts_arr))

        # Quality buckets
        pct_zero = float(np.mean(counts_arr == 0) * 100)
        pct_one = float(np.mean(counts_arr == 1) * 100)
        pct_2_10 = float(np.mean((counts_arr >= 2) & (counts_arr <= 10)) * 100)
        pct_11_50 = float(np.mean((counts_arr >= 11) & (counts_arr <= 50)) * 100)
        pct_51_100 = float(np.mean((counts_arr >= 51) & (counts_arr <= 100)) * 100)
        pct_over_100 = float(np.mean(counts_arr > 100) * 100)

        # Strategy-level and Ablation metrics
        strategy_metrics = []
        ablation_metrics = []
        if strategy_candidate_maps:
            strategy_metrics = self._evaluate_strategies_individually(
                strategy_candidate_maps, anchor_ids, total_true_pairs
            )
            ablation_metrics = self._evaluate_ablation(
                strategy_candidate_maps, anchor_ids, total_true_pairs
            )

        return CandidateRecallReport(
            total_anchors_evaluated=total_anchors,
            total_true_anchors_with_matches=anchors_with_matches,
            total_singleton_anchors=singleton_anchors,
            total_true_pairs=total_true_pairs,
            true_pairs_retrieved=true_pairs_retrieved,
            pair_recall=pair_recall,
            total_true_s2_pairs=total_true_s2,
            true_s2_retrieved=true_s2_retrieved,
            s2_recall=s2_recall,
            total_true_s3_pairs=total_true_s3,
            true_s3_retrieved=true_s3_retrieved,
            s3_recall=s3_recall,
            anchors_with_full_coverage=full_coverage_anchors,
            entity_coverage=entity_coverage,
            total_candidate_pairs=total_candidate_pairs,
            avg_candidates_per_anchor=avg_candidates,
            median_candidates_per_anchor=median_candidates,
            p90_candidates_per_anchor=p90_candidates,
            p95_candidates_per_anchor=p95_candidates,
            p99_candidates_per_anchor=p99_candidates,
            max_candidates_per_anchor=max_candidates,
            pct_zero_candidates=pct_zero,
            pct_one_candidate=pct_one,
            pct_two_to_ten=pct_2_10,
            pct_eleven_to_fifty=pct_11_50,
            pct_fifty_one_to_hundred=pct_51_100,
            pct_over_hundred=pct_over_100,
            strategy_metrics=tuple(strategy_metrics),
            ablation_metrics=tuple(ablation_metrics),
            missed_pairs_sample=tuple(missed_pairs[:20]),
        )

    def _evaluate_strategies_individually(
        self,
        strategy_maps: Sequence[tuple[str, Mapping[str, Mapping[str, Iterable[str]]]]],
        anchor_ids: Sequence[str],
        total_true_pairs: int,
    ) -> list[dict[str, Any]]:
        """Compute candidate count and recall for each strategy alone."""
        results = []
        for strat_name, cand_map in strategy_maps:
            cand_pairs_count = sum(len(cands) for cands in cand_map.values())
            true_found = 0
            for aid in anchor_ids:
                retrieved = set(cand_map.get(aid, {}).keys())
                true_matches = self.ground_truth.get(aid, set())
                true_found += len(retrieved.intersection(true_matches))

            rec = true_found / total_true_pairs if total_true_pairs > 0 else 0.0
            avg_cands = cand_pairs_count / len(anchor_ids) if anchor_ids else 0.0
            results.append({
                "strategy": strat_name,
                "candidate_pairs": cand_pairs_count,
                "true_pairs_found": true_found,
                "recall": round(rec, 4),
                "avg_candidates_per_anchor": round(avg_cands, 2),
            })
        return results

    def _evaluate_ablation(
        self,
        strategy_maps: Sequence[tuple[str, Mapping[str, Mapping[str, Iterable[str]]]]],
        anchor_ids: Sequence[str],
        total_true_pairs: int,
    ) -> list[dict[str, Any]]:
        """Compute cumulative ablation recall and candidate count."""
        cumulative_cands: dict[str, set[str]] = defaultdict(set)
        results = []

        for strat_name, cand_map in strategy_maps:
            for aid in anchor_ids:
                cumulative_cands[aid].update(cand_map.get(aid, {}).keys())

            total_cands = sum(len(c) for c in cumulative_cands.values())
            true_found = 0
            for aid in anchor_ids:
                true_matches = self.ground_truth.get(aid, set())
                true_found += len(cumulative_cands[aid].intersection(true_matches))

            rec = true_found / total_true_pairs if total_true_pairs > 0 else 0.0
            avg_cands = total_cands / len(anchor_ids) if anchor_ids else 0.0

            results.append({
                "stage": f"+ {strat_name}",
                "total_candidate_pairs": total_cands,
                "true_pairs_found": true_found,
                "recall": round(rec, 4),
                "avg_candidates_per_anchor": round(avg_cands, 2),
            })

        return results
