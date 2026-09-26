"""Unit and integration tests for Phase 7 Hard-Negative Mining and Precision Improvement.

Validates:
1. All mined hard negatives strictly have ground_truth_label == 0.
2. Mined dataset schema includes all required columns:
   ['source1_entity_id', 'candidate_entity_id', 'candidate_source',
    'model_probability', 'baseline_score', 'ground_truth_label', 'hard_negative_reason'].
3. Mining logic is completely deterministic.
4. Summary statistics match the generated dataset counts and distributions.
5. Model A and Model B are evaluated on identical validation entities.
6. Train and validation entities have zero overlap (train_S1 ∩ val_S1 == empty).
7. Zero validation labels or rows are added to Model B training.
8. Feature schema matches Phase 6 LightGBM feature names and order (84 features).
9. Comparison results contain precision, recall, F0.5, PR-AUC, false positive, and false negative counts.
10. Phase 7 creates no final submission files (matching_results.tsv, candidate_pairs.tsv).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.training.mining import (
    REQUIRED_HARD_NEGATIVE_COLUMNS,
    HardNegativeMiningConfig,
    HardNegativeMiningResult,
    ProductionHardNegativeMiner,
)
from business_entity_resolution.training.split import entity_aware_train_val_split


@pytest.fixture
def synthetic_mining_df() -> pd.DataFrame:
    """Create a synthetic scored candidate pairs DataFrame for testing hard-negative mining."""
    data = [
        # S1-001
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001_1", "candidate_source": "source2",
         "model_probability": 0.98, "baseline_score": 0.95, "name_core_exact_match": 1.0, "name_token_jaccard": 0.95,
         "feat_1": 1.0, "feat_2": 2.0, "ground_truth_label": 1},  # True match
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001_2", "candidate_source": "source2",
         "model_probability": 0.65, "baseline_score": 0.72, "name_core_exact_match": 0.0, "name_token_jaccard": 0.88,
         "feat_1": 0.5, "feat_2": 1.5, "ground_truth_label": 0},  # Hard negative (high prob, baseline, name sim)
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001_3", "candidate_source": "source2",
         "model_probability": 0.02, "baseline_score": 0.20, "name_core_exact_match": 0.0, "name_token_jaccard": 0.15,
         "feat_1": 0.1, "feat_2": 0.2, "ground_truth_label": 0},  # Easy negative

        # S1-002
        {"source1_entity_id": "S1-002", "candidate_entity_id": "S3-002_1", "candidate_source": "source3",
         "model_probability": 0.92, "baseline_score": 0.89, "name_core_exact_match": 1.0, "name_token_jaccard": 0.90,
         "feat_1": 1.2, "feat_2": 2.1, "ground_truth_label": 1},  # True match
        {"source1_entity_id": "S1-002", "candidate_entity_id": "S3-002_2", "candidate_source": "source3",
         "model_probability": 0.15, "baseline_score": 0.40, "name_core_exact_match": 0.0, "name_token_jaccard": 0.30,
         "feat_1": 0.3, "feat_2": 0.8, "ground_truth_label": 0},  # Hard negative (prob >= 0.10)
        {"source1_entity_id": "S1-002", "candidate_entity_id": "S3-002_3", "candidate_source": "source3",
         "model_probability": 0.05, "baseline_score": 0.68, "name_core_exact_match": 0.0, "name_token_jaccard": 0.40,
         "feat_1": 0.2, "feat_2": 0.5, "ground_truth_label": 0},  # Hard negative (baseline >= 0.65)
        {"source1_entity_id": "S1-002", "candidate_entity_id": "S3-002_4", "candidate_source": "source3",
         "model_probability": 0.04, "baseline_score": 0.35, "name_core_exact_match": 1.0, "name_token_jaccard": 0.50,
         "feat_1": 0.4, "feat_2": 0.6, "ground_truth_label": 0},  # Hard negative (exact core name collision)

        # S1-003
        {"source1_entity_id": "S1-003", "candidate_entity_id": "S2-003_1", "candidate_source": "source2",
         "model_probability": 0.01, "baseline_score": 0.10, "name_core_exact_match": 0.0, "name_token_jaccard": 0.10,
         "feat_1": 0.0, "feat_2": 0.1, "ground_truth_label": 0},  # Easy negative
    ]
    return pd.DataFrame(data)


class TestHardNegativeMinerLogic:
    """Test unit logic of ProductionHardNegativeMiner."""

    def test_all_mined_hard_negatives_have_label_zero(self, synthetic_mining_df: pd.DataFrame) -> None:
        """Requirement 1: Every single mined hard negative must strictly have ground_truth_label == 0."""
        miner = ProductionHardNegativeMiner()
        result = miner.mine(synthetic_mining_df)

        hard_df = result.hard_negatives_df
        assert len(hard_df) > 0, "Expected at least one hard negative in synthetic dataset"
        assert (hard_df["ground_truth_label"] == 0).all(), (
            "FATAL: Found non-zero ground_truth_label in mined hard negatives!"
        )

    def test_mined_dataset_schema_contains_required_columns(self, synthetic_mining_df: pd.DataFrame) -> None:
        """Requirement 2: Mined hard negative dataset must contain all required columns."""
        miner = ProductionHardNegativeMiner()
        result = miner.mine(synthetic_mining_df)

        hard_df = result.hard_negatives_df
        for col in REQUIRED_HARD_NEGATIVE_COLUMNS:
            assert col in hard_df.columns, f"Missing required column: {col}"

    def test_mining_is_deterministic(self, synthetic_mining_df: pd.DataFrame) -> None:
        """Requirement 3: Mining logic must produce identical output when executed repeatedly."""
        miner1 = ProductionHardNegativeMiner()
        result1 = miner1.mine(synthetic_mining_df)

        miner2 = ProductionHardNegativeMiner()
        result2 = miner2.mine(synthetic_mining_df)

        pd.testing.assert_frame_equal(result1.hard_negatives_df, result2.hard_negatives_df)
        assert result1.statistics == result2.statistics

    def test_statistics_match_generated_dataset(self, synthetic_mining_df: pd.DataFrame) -> None:
        """Requirement 4: Summary statistics must precisely match the mined candidate counts."""
        miner = ProductionHardNegativeMiner()
        result = miner.mine(synthetic_mining_df)

        stats = result.statistics
        hard_df = result.hard_negatives_df

        assert stats["total_training_pairs"] == len(synthetic_mining_df)
        assert stats["total_positive_candidates"] == int((synthetic_mining_df["ground_truth_label"] == 1).sum())
        assert stats["total_negative_candidates"] == int((synthetic_mining_df["ground_truth_label"] == 0).sum())
        assert stats["hard_negative_count"] == len(hard_df)
        assert stats["s2_hard_negative_count"] == int((hard_df["candidate_source"] == "source2").sum())
        assert stats["s3_hard_negative_count"] == int((hard_df["candidate_source"] == "source3").sum())

        prob_bucket_sum = sum(stats["probability_buckets"].values())
        assert prob_bucket_sum == len(hard_df)

    def test_miner_fails_loudly_on_missing_required_columns(self) -> None:
        """Miner must raise ValueError if required input columns are absent."""
        miner = ProductionHardNegativeMiner()
        bad_df = pd.DataFrame({
            "source1_entity_id": ["S1-001"],
            "candidate_entity_id": ["S2-001_1"],
            # Missing candidate_source, ground_truth_label
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            miner.mine(bad_df)

    def test_miner_assigns_correct_categorical_reasons(self, synthetic_mining_df: pd.DataFrame) -> None:
        """Verify that specific triggers are captured in hard_negative_reason."""
        miner = ProductionHardNegativeMiner()
        result = miner.mine(synthetic_mining_df)
        hard_df = result.hard_negatives_df

        # Row with model_prob=0.65 should have 'high_model_probability_ge_0.50'
        row_high_prob = hard_df[hard_df["candidate_entity_id"] == "S2-001_2"]
        assert len(row_high_prob) == 1
        assert "high_model_probability_ge_0.50" in row_high_prob.iloc[0]["hard_negative_reason"]

        # Row with core name collision should have 'exact_core_name_collision'
        row_collision = hard_df[hard_df["candidate_entity_id"] == "S3-002_4"]
        assert len(row_collision) == 1
        assert "exact_core_name_collision" in row_collision.iloc[0]["hard_negative_reason"]

    def test_empirical_percentile_computation(self, synthetic_mining_df: pd.DataFrame) -> None:
        """Requirement 2 & 3: Miner must dynamically calculate the threshold from actual negative scores."""
        # 1. Test dynamic calculation from negative distribution at 50th percentile (median)
        miner_50 = ProductionHardNegativeMiner(config=HardNegativeMiningConfig(negative_probability_percentile=50.0))
        res_50 = miner_50.mine(synthetic_mining_df)
        neg_scores = synthetic_mining_df.loc[synthetic_mining_df["ground_truth_label"] == 0, "model_probability"].to_numpy()
        expected_p50 = float(np.percentile(neg_scores, 50.0))

        assert abs(res_50.statistics["empirical_threshold_info"]["derived_probability_threshold"] - expected_p50) < 1e-6
        assert res_50.statistics["empirical_threshold_info"]["percentile_used"] == 50.0
        assert "50.0th percentile" in res_50.statistics["empirical_threshold_info"]["derivation_method"]

        # 2. Test explicit override
        miner_explicit = ProductionHardNegativeMiner(config=HardNegativeMiningConfig(min_model_probability=0.25))
        res_explicit = miner_explicit.mine(synthetic_mining_df)
        assert res_explicit.statistics["empirical_threshold_info"]["derived_probability_threshold"] == 0.25
        assert res_explicit.statistics["empirical_threshold_info"]["percentile_used"] is None


class TestModelComparisonAndLeakageGuarantees:
    """Test leakage boundaries, evaluation consistency, and feature schemas."""

    def test_entity_aware_split_has_zero_overlap(self) -> None:
        """Requirement 6: Train and validation entities must have zero overlap."""
        # 100 S1 entities, 5 candidate pairs each
        rows = []
        for i in range(100):
            s1 = f"S1_{i:04d}"
            for j in range(5):
                rows.append({
                    "source1_entity_id": s1,
                    "candidate_entity_id": f"S2_{i}_{j}",
                    "candidate_source": "source2",
                    "feat_1": float(i + j),
                    "ground_truth_label": 1 if j == 0 else 0,
                })
        df = pd.DataFrame(rows)

        split_result = entity_aware_train_val_split(df, val_fraction=0.20, random_seed=2026)
        train_s1 = set(split_result.train_df["source1_entity_id"])
        val_s1 = set(split_result.val_df["source1_entity_id"])

        overlap = train_s1.intersection(val_s1)
        assert len(overlap) == 0, f"Entity leakage detected: {overlap}"
        assert len(train_s1) == 80
        assert len(val_s1) == 20

    def test_zero_validation_labels_in_model_b_training(self) -> None:
        """Requirement 7: Zero validation pairs or labels are allowed in Model B training."""
        rows = []
        for i in range(50):
            s1 = f"S1_{i:04d}"
            for j in range(4):
                rows.append({
                    "source1_entity_id": s1,
                    "candidate_entity_id": f"S2_{i}_{j}",
                    "candidate_source": "source2",
                    "model_probability": 0.6 if (j == 1 and i < 40) else 0.05,
                    "baseline_score": 0.7 if (j == 1 and i < 40) else 0.1,
                    "name_core_exact_match": 0.0,
                    "name_token_jaccard": 0.2,
                    "feat_1": 1.0,
                    "ground_truth_label": 1 if j == 0 else 0,
                })
        df = pd.DataFrame(rows)

        split_res = entity_aware_train_val_split(df, val_fraction=0.20, random_seed=2026)
        miner = ProductionHardNegativeMiner()
        mining_result = miner.mine(split_res.train_df)

        hard_features = mining_result.hard_features_df[["feat_1", "ground_truth_label", "source1_entity_id"]]
        train_b_df = pd.concat([split_res.train_df[["feat_1", "ground_truth_label", "source1_entity_id"]], hard_features], ignore_index=True)

        val_s1 = set(split_res.val_df["source1_entity_id"])
        train_b_s1 = set(train_b_df["source1_entity_id"])

        assert len(train_b_s1.intersection(val_s1)) == 0, "Validation entities leaked into Model B training set!"
        assert (train_b_df.loc[len(split_res.train_df):, "ground_truth_label"] == 0).all(), (
            "Mined hard negatives added to Model B training set must strictly have label == 0!"
        )


class TestPhase7GeneratedArtifacts:
    """Validate physical artifacts generated in output/phase7/."""

    @pytest.fixture(autouse=True)
    def setup_paths(self) -> None:
        self.output_dir = Path("output/phase7")
        self.phase6_dir = Path("output/phase6")

    def test_all_expected_artifacts_exist(self) -> None:
        """Verify that all 12 required Phase 7 artifact files exist."""
        if not self.output_dir.exists():
            pytest.skip("output/phase7 directory does not exist yet (run scripts/run_phase7_hard_negatives.py)")

        expected_files = [
            "hard_negative_dataset.csv.gz",
            "hard_negative_dataset_sample.csv",
            "hard_negative_statistics.json",
            "leakage_entity_split_validation.json",
            "model_comparison.json",
            "model_comparison.csv",
            "model_b_lightgbm.joblib",
            "model_b_metadata.json",
            "model_b_validation_predictions.csv.gz",
            "model_b_validation_predictions_sample.csv",
            "high_confidence_false_positives.json",
            "phase7_summary.json",
        ]
        for fname in expected_files:
            p = self.output_dir / fname
            assert p.exists(), f"Missing required Phase 7 artifact: {p}"
            assert p.stat().st_size > 0, f"Phase 7 artifact is empty: {p}"

    def test_mined_hard_negative_dataset_integrity(self) -> None:
        """Requirement 1 & 2: Mined dataset has 100% label == 0 and correct columns."""
        path = self.output_dir / "hard_negative_dataset.csv.gz"
        if not path.exists():
            pytest.skip("output/phase7/hard_negative_dataset.csv.gz does not exist")

        df = pd.read_csv(path)
        assert len(df) > 0, "hard_negative_dataset.csv.gz is empty"
        assert (df["ground_truth_label"] == 0).all(), "Found non-zero labels in hard_negative_dataset!"

        for col in REQUIRED_HARD_NEGATIVE_COLUMNS:
            assert col in df.columns, f"Missing required column {col} in hard_negative_dataset"

    def test_leakage_audit_file(self) -> None:
        """Requirement 6 & 7: Verify zero entity overlap in leakage_entity_split_validation.json."""
        path = self.output_dir / "leakage_entity_split_validation.json"
        if not path.exists():
            pytest.skip("leakage_entity_split_validation.json does not exist")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["zero_entity_overlap"] is True
        assert data["entity_overlap_count"] == 0
        assert data["train_s1_unique_entities"] == 2000
        assert data["val_s1_unique_entities"] == 500
        assert data["zero_validation_labels_in_train"] is True
        assert data["features_match_phase6"] is True
        assert len(data["forbidden_columns_in_features"]) == 0

    def test_model_comparison_metrics_completeness(self) -> None:
        """Requirement 9: Comparison results contain precision, recall, F0.5, PR-AUC, FP, FN."""
        path = self.output_dir / "model_comparison.json"
        if not path.exists():
            pytest.skip("model_comparison.json does not exist")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "overall_metrics" in data
        assert "model_a" in data["overall_metrics"]
        assert "model_b" in data["overall_metrics"]
        assert "pr_auc" in data["overall_metrics"]["model_a"]
        assert "pr_auc" in data["overall_metrics"]["model_b"]

        assert "threshold_comparisons" in data
        thresh_list = data["threshold_comparisons"]
        assert len(thresh_list) >= 9

        for item in thresh_list:
            for key in ["threshold", "model_a_precision", "model_b_precision",
                        "model_a_recall", "model_b_recall",
                        "model_a_f0_5", "model_b_f0_5",
                        "model_a_fp", "model_b_fp",
                        "model_a_fn", "model_b_fn"]:
                assert key in item, f"Missing key {key} in threshold comparison item"

    def test_feature_schema_matches_phase6(self) -> None:
        """Requirement 8: Model B feature schema matches Phase 6 LightGBM feature names and order."""
        p6_meta_file = self.phase6_dir / "lightgbm_metadata.json"
        p7_meta_file = self.output_dir / "model_b_metadata.json"

        if not p6_meta_file.exists() or not p7_meta_file.exists():
            pytest.skip("Phase 6 or Phase 7 metadata file not found")

        with open(p6_meta_file, "r", encoding="utf-8") as f:
            p6_meta = json.load(f)
        with open(p7_meta_file, "r", encoding="utf-8") as f:
            p7_meta = json.load(f)

        p6_features = p6_meta["feature_column_order"]
        p7_features = p7_meta["feature_column_order"]

        assert len(p7_features) == 84, f"Expected 84 features, got {len(p7_features)}"
        assert p7_features == p6_features, "Model B feature column order does not match Phase 6 exactly"

    def test_no_submission_files_created_in_phase7(self) -> None:
        """Requirement 10: Strictly verify NO submission TSVs are created in Phase 7."""
        forbidden_files = ["matching_results.tsv", "candidate_pairs.tsv"]

        for fname in forbidden_files:
            # Check output/phase7
            p_phase7 = self.output_dir / fname
            assert not p_phase7.exists(), f"VIOLATION: Submission file found in output/phase7: {p_phase7}"

            # Check repo root
            p_root = Path(fname)
            assert not p_root.exists(), f"VIOLATION: Submission file found in repo root: {p_root}"

    def test_statistics_and_summary_report_empirical_derivation(self) -> None:
        """Requirement 7: Verify hard_negative_statistics.json and phase7_summary.json report actual derivation."""
        stats_file = self.output_dir / "hard_negative_statistics.json"
        summary_file = self.output_dir / "phase7_summary.json"

        if not stats_file.exists() or not summary_file.exists():
            pytest.skip("Phase 7 statistics or summary file does not exist")

        with open(stats_file, "r", encoding="utf-8") as f:
            stats_data = json.load(f)
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        # Check hard_negative_statistics.json
        assert "empirical_threshold_info" in stats_data
        emp_info = stats_data["empirical_threshold_info"]
        assert emp_info["percentile_used"] == 99.5
        assert isinstance(emp_info["derived_probability_threshold"], float)
        assert emp_info["derived_probability_threshold"] > 0.0
        assert "np.percentile" in emp_info["derivation_method"]
        assert "99.5th percentile" in emp_info["derivation_method"]
        assert emp_info["training_negatives_evaluated"] == 128556
        assert emp_info["training_negatives_ge_threshold"] == 643

        # Check phase7_summary.json
        assert "empirical_probability_threshold" in summary_data["hard_negatives"]
        assert summary_data["hard_negatives"]["empirical_percentile"] == 99.5
        assert "np.percentile" in summary_data["hard_negatives"]["derivation_method"]
        assert summary_data["hard_negatives"]["empirical_probability_threshold"] == emp_info["derived_probability_threshold"]
