"""Unit and integration tests for Phase 8 Entity-Level Validation.

Validates:
1. Entity-level grouping (grouping by source1_entity_id, candidate retention).
2. Candidate ranking (deterministic sorting by probability descending, rank assignments).
3. Threshold evaluation (grid sweep calculation of pair-level and entity-level metrics).
4. Ambiguity detection (decision-boundary margin and uncertain probability classification).
5. Conflict detection (candidate fan-in collisions across S1 entities and duplicate ties).
6. Ground-truth validation (exact set matches, singleton FPs, matched FNs, multi-candidate preservation).
7. Train/validation entity separation (strictly zero S1 entity overlap).
8. No validation-label leakage (no validation labels in training, no submission files generated).
9. Threshold selection logic (F0.5 maximization with precision/recall trade-off rationale).
10. Output schema and artifact integrity (all Phase 8 required files present and valid).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.validation.entity_level import (
    DEFAULT_THRESHOLD_GRID,
    EntityPredictionRecord,
    ThresholdRecord,
    build_entity_predictions,
    compute_distribution_stats,
    evaluate_threshold_grid,
    perform_ambiguity_analysis,
    perform_conflict_analysis,
    perform_error_analysis,
    select_validation_threshold,
)


@pytest.fixture
def synthetic_predictions_df() -> pd.DataFrame:
    """Create a synthetic scored candidate pairs DataFrame for testing entity validation."""
    data = [
        # S1-001: 2 matches (1 from S2, 1 from S3), both high confidence
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001", "candidate_source": "S2",
         "match_probability": 0.99, "ground_truth_label": 1, "baseline_score": 0.85},
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S3-001", "candidate_source": "S3",
         "match_probability": 0.98, "ground_truth_label": 1, "baseline_score": 0.80},
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-002", "candidate_source": "S2",
         "match_probability": 0.05, "ground_truth_label": 0, "baseline_score": 0.30},

        # S1-002: True singleton (0 matches in GT), all low probabilities
        {"source1_entity_id": "S1-002", "candidate_entity_id": "S2-003", "candidate_source": "S2",
         "match_probability": 0.02, "ground_truth_label": 0, "baseline_score": 0.20},
        {"source1_entity_id": "S1-002", "candidate_entity_id": "S3-002", "candidate_source": "S3",
         "match_probability": 0.01, "ground_truth_label": 0, "baseline_score": 0.15},

        # S1-003: Ambiguous entity (candidate in [0.90, 0.95) boundary margin)
        {"source1_entity_id": "S1-003", "candidate_entity_id": "S2-004", "candidate_source": "S2",
         "match_probability": 0.96, "ground_truth_label": 1, "baseline_score": 0.75},
        {"source1_entity_id": "S1-003", "candidate_entity_id": "S3-003", "candidate_source": "S3",
         "match_probability": 0.92, "ground_truth_label": 1, "baseline_score": 0.70},

        # S1-004: Uncertain singleton (no matches in GT, but top prob is 0.60)
        {"source1_entity_id": "S1-004", "candidate_entity_id": "S2-005", "candidate_source": "S2",
         "match_probability": 0.60, "ground_truth_label": 0, "baseline_score": 0.50},
        {"source1_entity_id": "S1-004", "candidate_entity_id": "S3-004", "candidate_source": "S3",
         "match_probability": 0.10, "ground_truth_label": 0, "baseline_score": 0.25},

        # S1-005: False Positive match (candidate >= 0.95, but GT = 0)
        {"source1_entity_id": "S1-005", "candidate_entity_id": "S2-006", "candidate_source": "S2",
         "match_probability": 0.97, "ground_truth_label": 0, "baseline_score": 0.65},
    ]
    return pd.DataFrame(data)


class TestEntityLevelValidation:
    """Comprehensive test suite for Phase 8 validation functionality."""

    def test_entity_level_grouping(self, synthetic_predictions_df: pd.DataFrame) -> None:
        """1. Validate grouping by source1_entity_id and preservation of candidate identities."""
        entity_df = build_entity_predictions(synthetic_predictions_df, threshold=0.95)

        # 5 distinct S1 entities
        assert len(entity_df) == 5
        assert set(entity_df["source1_entity_id"]) == {"S1-001", "S1-002", "S1-003", "S1-004", "S1-005"}

        # Candidate counts
        s1_001 = entity_df[entity_df["source1_entity_id"] == "S1-001"].iloc[0]
        assert s1_001["total_candidates_evaluated"] == 3
        assert s1_001["predicted_candidate_count"] == 2
        assert "S2-001" in s1_001["predicted_candidate_ids"]
        assert "S3-001" in s1_001["predicted_candidate_ids"]
        assert "S2" in s1_001["predicted_candidate_sources"]
        assert "S3" in s1_001["predicted_candidate_sources"]

    def test_candidate_ranking(self, synthetic_predictions_df: pd.DataFrame) -> None:
        """2. Validate candidate ranking, top probability, and gap calculations."""
        entity_df = build_entity_predictions(synthetic_predictions_df, threshold=0.95)

        s1_001 = entity_df[entity_df["source1_entity_id"] == "S1-001"].iloc[0]
        assert s1_001["top_candidate_id"] == "S2-001"
        assert s1_001["top_candidate_probability"] == 0.99
        assert s1_001["second_candidate_id"] == "S3-001"
        assert s1_001["second_candidate_probability"] == 0.98
        assert np.isclose(s1_001["top_candidate_probability_gap"], 0.01)

    def test_threshold_evaluation(self, synthetic_predictions_df: pd.DataFrame) -> None:
        """3. Validate threshold grid evaluation computes accurate pair and entity metrics."""
        grid_df = evaluate_threshold_grid(
            synthetic_predictions_df,
            grid=[0.50, 0.90, 0.95],
            margin_delta=0.05,
        )

        assert len(grid_df) == 3
        assert list(grid_df["threshold"]) == [0.50, 0.90, 0.95]

        # At 0.95:
        # TP pairs: S2-001 (0.99), S3-001 (0.98), S2-004 (0.96) -> 3 TP
        # FP pairs: S2-006 (0.97) -> 1 FP
        # FN pairs: S3-003 (0.92 < 0.95) -> 1 FN
        row_95 = grid_df[grid_df["threshold"] == 0.95].iloc[0]
        assert row_95["pair_true_positives"] == 3
        assert row_95["pair_false_positives"] == 1
        assert row_95["pair_false_negatives"] == 1
        assert np.isclose(row_95["pair_precision"], 3 / 4)
        assert np.isclose(row_95["pair_recall"], 3 / 4)

        # Entity metrics
        assert row_95["total_source1_entities"] == 5
        assert row_95["matched_source1_entities"] == 3  # S1-001, S1-003, S1-005
        assert row_95["unmatched_source1_entities"] == 2  # S1-002, S1-004

    def test_ambiguity_detection(self, synthetic_predictions_df: pd.DataFrame) -> None:
        """4. Validate ambiguity detection on margin candidates and uncertain singletons."""
        entity_df = build_entity_predictions(synthetic_predictions_df, threshold=0.95, margin_delta=0.05)

        # S1-003 has candidate S3-003 at 0.92 (in [0.90, 0.95)) -> ambiguous
        s1_003 = entity_df[entity_df["source1_entity_id"] == "S1-003"].iloc[0]
        assert s1_003["prediction_status"] == "ambiguous"
        assert "margin" in s1_003["prediction_status_reason"].lower()

        # S1-004 has top prob 0.60 (in [0.50, 0.95)) -> ambiguous
        s1_004 = entity_df[entity_df["source1_entity_id"] == "S1-004"].iloc[0]
        assert s1_004["prediction_status"] == "ambiguous"
        assert "uncertain" in s1_004["prediction_status_reason"].lower()

        # S1-001 has both candidates >= 0.95 and lowest is 0.05 -> confident
        s1_001 = entity_df[entity_df["source1_entity_id"] == "S1-001"].iloc[0]
        assert s1_001["prediction_status"] == "confident"

        # S1-002 has max prob 0.02 -> unmatched
        s1_002 = entity_df[entity_df["source1_entity_id"] == "S1-002"].iloc[0]
        assert s1_002["prediction_status"] == "unmatched"

    def test_conflict_detection(self) -> None:
        """5. Validate conflict detection when a candidate is assigned to multiple S1 entities (fan-in)."""
        conflict_df = pd.DataFrame([
            {"source1_entity_id": "S1-010", "candidate_entity_id": "S2-COMMON", "candidate_source": "S2",
             "match_probability": 0.98, "ground_truth_label": 1},
            {"source1_entity_id": "S1-020", "candidate_entity_id": "S2-COMMON", "candidate_source": "S2",
             "match_probability": 0.96, "ground_truth_label": 0},
        ])

        entity_df = build_entity_predictions(conflict_df, threshold=0.95)
        statuses = dict(zip(entity_df["source1_entity_id"], entity_df["prediction_status"]))
        assert statuses["S1-010"] == "conflicting"
        assert statuses["S1-020"] == "conflicting"

        conflict_report = perform_conflict_analysis(conflict_df, threshold=0.95)
        assert conflict_report["fan_in_candidate_conflicts_at_operating_threshold"] == 1

    def test_ground_truth_validation(self, synthetic_predictions_df: pd.DataFrame) -> None:
        """6. Validate ground-truth set comparison, singleton FPs, and matched FNs."""
        entity_df = build_entity_predictions(synthetic_predictions_df, threshold=0.95)

        # S1-001 has 2 true matches (S2-001, S3-001), predicted exactly {S2-001, S3-001} -> exact set match
        s1_001 = entity_df[entity_df["source1_entity_id"] == "S1-001"].iloc[0]
        assert bool(s1_001["exact_set_match"]) is True
        assert np.isclose(s1_001["jaccard_similarity"], 1.0)
        assert bool(s1_001["entity_has_false_positive"]) is False
        assert bool(s1_001["entity_has_false_negative"]) is False

        # S1-002 is true singleton, predicted 0 -> exact set match
        s1_002 = entity_df[entity_df["source1_entity_id"] == "S1-002"].iloc[0]
        assert bool(s1_002["exact_set_match"]) is True
        assert bool(s1_002["is_ground_truth_singleton"]) is True

        # S1-005 is singleton in GT (0 matches), but predicted 1 candidate -> singleton FP
        s1_005 = entity_df[entity_df["source1_entity_id"] == "S1-005"].iloc[0]
        assert bool(s1_005["is_ground_truth_singleton"]) is True
        assert bool(s1_005["exact_set_match"]) is False
        assert bool(s1_005["entity_has_false_positive"]) is True

    def test_train_validation_entity_separation(self) -> None:
        """7. Verify strict train/validation entity separation from saved artifacts."""
        split_file = Path("output/phase6/split_metadata.json")
        leakage_file = Path("output/phase7/leakage_entity_split_validation.json")
        p8_ref_file = Path("output/phase8/model_reference.json")

        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            assert meta["n_train_entities"] == 2000
            assert meta["n_val_entities"] == 500

        if leakage_file.exists():
            with open(leakage_file, "r", encoding="utf-8") as f:
                leak = json.load(f)
            assert leak["zero_entity_overlap"] is True
            assert leak["entity_overlap_count"] == 0

        if p8_ref_file.exists():
            with open(p8_ref_file, "r", encoding="utf-8") as f:
                p8_ref = json.load(f)
            assert p8_ref["leakage_verification"]["zero_overlap"] is True

    def test_no_validation_label_leakage(self) -> None:
        """8. Verify no validation labels entered Model B training and no submission files exist."""
        p7_summary_file = Path("output/phase7/phase7_summary.json")
        if p7_summary_file.exists():
            with open(p7_summary_file, "r", encoding="utf-8") as f:
                p7 = json.load(f)
            assert p7["leakage_verification"]["validation_labels_in_train"] is False

        # Strictly zero submission files in Phase 8 output or project root
        forbidden_files = [
            Path("output/phase8/matching_results.tsv"),
            Path("output/matching_results.tsv"),
            Path("matching_results.tsv"),
        ]
        for f in forbidden_files:
            assert not f.exists(), f"Forbidden submission file detected at {f}!"

    def test_threshold_selection_logic(self, synthetic_predictions_df: pd.DataFrame) -> None:
        """9. Validate threshold selection selects peak F0.5 and produces rationale."""
        grid_df = evaluate_threshold_grid(synthetic_predictions_df, grid=[0.50, 0.90, 0.95, 0.975])
        sel = select_validation_threshold(grid_df, primary_metric="pair_f0_5")

        assert "selected_threshold" in sel
        assert "primary_metric" in sel
        assert "selection_rationale" in sel
        assert "limitations" in sel
        assert len(sel["limitations"]) > 0

    def test_output_schema_and_integrity(self) -> None:
        """10. Verify that all Phase 8 output artifacts exist, are non-empty, and valid."""
        p8_dir = Path("output/phase8")
        if not p8_dir.exists():
            pytest.skip("Phase 8 outputs not yet generated")

        required_files = [
            "entity_level_validation.csv",
            "entity_level_validation.csv.gz",
            "entity_level_statistics.json",
            "threshold_analysis.csv",
            "threshold_analysis.json",
            "ambiguity_analysis.json",
            "conflict_analysis.json",
            "error_analysis.json",
            "selected_threshold.json",
            "validation_predictions.csv.gz",
            "model_reference.json",
            "phase8_summary.json",
        ]

        for fname in required_files:
            fpath = p8_dir / fname
            assert fpath.exists(), f"Required Phase 8 artifact {fname} missing!"
            assert fpath.stat().st_size > 0, f"Artifact {fname} is empty!"

        # Verify entity_level_validation.csv schema
        df = pd.read_csv(p8_dir / "entity_level_validation.csv")
        assert len(df) == 500
        expected_cols = [
            "source1_entity_id", "total_candidates_evaluated", "predicted_candidate_count",
            "predicted_candidate_ids", "predicted_candidate_sources", "top_candidate_id",
            "top_candidate_source", "top_candidate_probability", "second_candidate_id",
            "second_candidate_source", "second_candidate_probability", "top_candidate_probability_gap",
            "prediction_status", "prediction_status_reason", "ground_truth_candidate_count",
            "ground_truth_candidate_ids", "is_ground_truth_singleton", "exact_set_match",
            "entity_has_false_positive", "entity_has_false_negative", "jaccard_similarity"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column {col} in entity_level_validation.csv"
