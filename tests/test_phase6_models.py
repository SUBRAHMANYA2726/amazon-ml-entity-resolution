"""Unit tests for Phase 6 Supervised Pairwise ML Matching.

Tests:
1. Feature leakage prevention.
2. Entity-aware train/validation split (grouping by source1_entity_id).
3. Zero overlap between train and validation source1 IDs.
4. Deterministic split behavior.
5. Fail-loud behavior on invalid splits or corrupted inputs.
6. Model probability validation (0 <= p <= 1, no NaN, no inf).
7. Correct feature column ordering and rejection of leaked columns.
8. Model output shape and predictions format.
9. Serialization and reloadability.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.training.split import (
    EntitySplitResult,
    EntitySplitStatistics,
    entity_aware_train_val_split,
)


@pytest.fixture
def synthetic_pairwise_df() -> pd.DataFrame:
    """Create a small deterministic synthetic pairwise dataset across 10 S1 entities."""
    data = []
    # 10 S1 anchors: S1-001 through S1-010
    # Each anchor has 5 candidate pairs: 1 positive and 4 negatives (1:4 ratio)
    for i in range(1, 11):
        s1_id = f"S1-{i:03d}"
        for j in range(1, 6):
            cand_id = f"S2-{i:03d}_{j:02d}"
            label = 1 if j == 1 else 0
            # 4 mock feature columns
            row = {
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "candidate_source": "source2",
                "name_exact_match": 1.0 if label == 1 else 0.0,
                "name_token_jaccard": 0.9 if label == 1 else 0.2,
                "address_token_jaccard": 0.85 if label == 1 else 0.15,
                "baseline_score": 0.88 if label == 1 else 0.25,
                "ground_truth_label": label,
            }
            data.append(row)
    return pd.DataFrame(data)


class TestEntityAwareSplit:
    """Tests for entity-aware train/validation split logic."""

    def test_split_preserves_all_rows_and_entities(self, synthetic_pairwise_df: pd.DataFrame) -> None:
        result = entity_aware_train_val_split(
            synthetic_pairwise_df,
            val_fraction=0.20,
            random_seed=2026,
        )

        assert len(result.train_df) + len(result.val_df) == len(synthetic_pairwise_df)
        assert result.statistics.n_total_entities == 10
        assert result.statistics.n_val_entities == 2
        assert result.statistics.n_train_entities == 8
        assert result.statistics.n_train_pairs == 40
        assert result.statistics.n_val_pairs == 10

    def test_zero_overlap_between_train_and_val_s1_ids(self, synthetic_pairwise_df: pd.DataFrame) -> None:
        result = entity_aware_train_val_split(
            synthetic_pairwise_df,
            val_fraction=0.30,
            random_seed=2026,
        )

        train_s1 = set(result.train_df["source1_entity_id"])
        val_s1 = set(result.val_df["source1_entity_id"])

        # Strict zero overlap
        overlap = train_s1.intersection(val_s1)
        assert overlap == set(), f"Overlap detected between train and val S1 IDs: {overlap}"
        assert result.train_entity_ids == frozenset(train_s1)
        assert result.val_entity_ids == frozenset(val_s1)

    def test_split_is_deterministic(self, synthetic_pairwise_df: pd.DataFrame) -> None:
        result1 = entity_aware_train_val_split(
            synthetic_pairwise_df,
            val_fraction=0.20,
            random_seed=2026,
        )
        result2 = entity_aware_train_val_split(
            synthetic_pairwise_df,
            val_fraction=0.20,
            random_seed=2026,
        )

        assert sorted(result1.train_entity_ids) == sorted(result2.train_entity_ids)
        assert sorted(result1.val_entity_ids) == sorted(result2.val_entity_ids)
        pd.testing.assert_frame_equal(result1.train_df, result2.train_df)
        pd.testing.assert_frame_equal(result1.val_df, result2.val_df)

    def test_split_fails_on_missing_columns(self, synthetic_pairwise_df: pd.DataFrame) -> None:
        bad_df = synthetic_pairwise_df.drop(columns=["source1_entity_id"])
        with pytest.raises(ValueError, match="Entity column 'source1_entity_id' not found"):
            entity_aware_train_val_split(bad_df)

        bad_label_df = synthetic_pairwise_df.drop(columns=["ground_truth_label"])
        with pytest.raises(ValueError, match="Label column 'ground_truth_label' not found"):
            entity_aware_train_val_split(bad_label_df)

    def test_split_statistics_accuracy(self, synthetic_pairwise_df: pd.DataFrame) -> None:
        result = entity_aware_train_val_split(
            synthetic_pairwise_df,
            val_fraction=0.20,
            random_seed=2026,
        )

        stats = result.statistics
        assert stats.random_seed == 2026
        assert stats.train_positive_pairs == 8
        assert stats.train_negative_pairs == 32
        assert pytest.approx(stats.train_positive_rate) == 8 / 40
        assert stats.val_positive_pairs == 2
        assert stats.val_negative_pairs == 8
        assert pytest.approx(stats.val_positive_rate) == 2 / 10
        # Imbalance ratio: 32 / 8 = 4.0
        assert pytest.approx(stats.train_scale_pos_weight) == 4.0


class TestLeakagePrevention:
    """Tests ensuring forbidden identifier and target columns are rejected."""

    def test_forbidden_columns_raise_error(self, synthetic_pairwise_df: pd.DataFrame) -> None:
        from business_entity_resolution.models.pairwise import (
            FORBIDDEN_FEATURE_COLUMNS,
            validate_and_extract_features,
        )

        for col in ["source1_entity_id", "candidate_entity_id", "ground_truth_label"]:
            assert col in FORBIDDEN_FEATURE_COLUMNS
            sub_df = synthetic_pairwise_df[[col, "name_exact_match"]]
            with pytest.raises(ValueError, match="FATAL DATA LEAKAGE: Forbidden columns detected"):
                validate_and_extract_features(sub_df)

    def test_feature_ordering_and_missingness(self, synthetic_pairwise_df: pd.DataFrame) -> None:
        from business_entity_resolution.models.pairwise import validate_and_extract_features

        feature_cols = ["name_exact_match", "name_token_jaccard", "address_token_jaccard", "baseline_score"]
        clean_df = synthetic_pairwise_df[feature_cols]

        # Pass in reverse order
        scrambled_df = clean_df[feature_cols[::-1]]
        arr, names = validate_and_extract_features(scrambled_df, expected_feature_names=feature_cols)
        assert names == feature_cols
        # Check values match expected column order
        np.testing.assert_array_equal(arr[:, 0], clean_df["name_exact_match"].to_numpy())

        # Missing expected column
        incomplete_df = clean_df.drop(columns=["baseline_score"])
        with pytest.raises(ValueError, match="Feature matrix is missing 1 expected features"):
            validate_and_extract_features(incomplete_df, expected_feature_names=feature_cols)


class TestProbabilityValidation:
    """Tests ensuring probability outputs strictly satisfy 0 <= p <= 1 and contain no NaN/inf."""

    def test_valid_probabilities(self) -> None:
        from business_entity_resolution.models.pairwise import validate_probabilities

        valid = np.array([0.0, 0.05, 0.5, 0.99, 1.0])
        res = validate_probabilities(valid)
        np.testing.assert_array_equal(res, valid)

    def test_rejects_nan_and_inf(self) -> None:
        from business_entity_resolution.models.pairwise import validate_probabilities

        with pytest.raises(ValueError, match="Probabilities contain NaN values"):
            validate_probabilities(np.array([0.5, np.nan, 0.8]))

        with pytest.raises(ValueError, match="Probabilities contain Infinite values"):
            validate_probabilities(np.array([0.5, np.inf, 0.8]))

    def test_rejects_out_of_bounds(self) -> None:
        from business_entity_resolution.models.pairwise import validate_probabilities

        with pytest.raises(ValueError, match="Probabilities outside valid range"):
            validate_probabilities(np.array([-0.01, 0.5, 0.8]))

        with pytest.raises(ValueError, match="Probabilities outside valid range"):
            validate_probabilities(np.array([0.5, 1.01, 0.8]))


class TestLightGBMMatcher:
    """Tests for LightGBM pairwise matcher fitting, scoring, importance, and serialization."""

    def test_lightgbm_fit_score_and_reload(self, synthetic_pairwise_df: pd.DataFrame) -> None:
        from business_entity_resolution.models.pairwise import LightGBMMatcher

        feature_cols = ["name_exact_match", "name_token_jaccard", "address_token_jaccard", "baseline_score"]
        X = synthetic_pairwise_df[feature_cols]
        y = synthetic_pairwise_df["ground_truth_label"]

        matcher = LightGBMMatcher(
            scale_pos_weight=4.0,
            random_seed=2026,
            feature_names=feature_cols,
            n_estimators=15,
            num_leaves=7,
        )
        matcher.fit(X, y)
        assert matcher.is_fitted

        scores = matcher.score(X)
        assert len(scores) == len(synthetic_pairwise_df)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

        # Feature importance
        imp_df = matcher.get_feature_importances()
        assert len(imp_df) == len(feature_cols)
        assert "split_importance" in imp_df.columns
        assert "gain_importance" in imp_df.columns
        assert (imp_df["importance_label"] == "MODEL FEATURE IMPORTANCE").all()

        # Serialization & Reload
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            meta_path = Path(tmpdir) / "metadata.json"
            matcher.save(model_path, meta_path)

            assert model_path.exists()
            assert meta_path.exists()

            reloaded = LightGBMMatcher.load(model_path)
            assert reloaded.is_fitted
            assert reloaded.feature_names == tuple(feature_cols)
            reloaded_scores = reloaded.score(X)
            np.testing.assert_allclose(scores, reloaded_scores, rtol=1e-5)


class TestCatBoostMatcher:
    """Tests for CatBoost pairwise matcher fitting, scoring, importance, and serialization."""

    def test_catboost_fit_score_and_reload(self, synthetic_pairwise_df: pd.DataFrame) -> None:
        from business_entity_resolution.models.pairwise import CatBoostMatcher

        feature_cols = ["name_exact_match", "name_token_jaccard", "address_token_jaccard", "baseline_score"]
        X = synthetic_pairwise_df[feature_cols]
        y = synthetic_pairwise_df["ground_truth_label"]

        matcher = CatBoostMatcher(
            scale_pos_weight=4.0,
            random_seed=2026,
            feature_names=feature_cols,
            iterations=15,
            depth=3,
        )
        matcher.fit(X, y)
        assert matcher.is_fitted

        scores = matcher.score(X)
        assert len(scores) == len(synthetic_pairwise_df)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

        # Feature importance
        imp_df = matcher.get_feature_importances()
        assert len(imp_df) == len(feature_cols)
        assert "importance" in imp_df.columns
        assert (imp_df["importance_label"] == "MODEL FEATURE IMPORTANCE").all()

        # Serialization & Reload
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            meta_path = Path(tmpdir) / "metadata.json"
            matcher.save(model_path, meta_path)

            assert model_path.exists()
            assert meta_path.exists()

            reloaded = CatBoostMatcher.load(model_path)
            assert reloaded.is_fitted
            assert reloaded.feature_names == tuple(feature_cols)
            reloaded_scores = reloaded.score(X)
            np.testing.assert_allclose(scores, reloaded_scores, rtol=1e-5)


class TestPhase6Evaluation:
    """Tests for Phase 6 validation evaluation and prediction artifact generation."""

    def test_evaluate_predictions_metrics(self) -> None:
        from business_entity_resolution.evaluation.metrics import evaluate_predictions

        y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])

        results = evaluate_predictions(y_true, y_prob, diagnostic_thresholds=[0.5])
        assert "pr_auc" in results
        assert "roc_auc" in results
        assert 0.0 <= results["pr_auc"] <= 1.0
        assert 0.0 <= results["roc_auc"] <= 1.0
        assert results["total_pairs"] == 10
        assert results["total_positives"] == 3
        assert results["total_negatives"] == 7

        sweep = results["diagnostic_threshold_sweep"]
        assert len(sweep) == 1
        m = sweep[0]
        assert m["threshold"] == 0.5
        # Predicted >= 0.5 are indices 0 (prob 0.9, y=1), 1 (prob 0.8, y=0), 2 (prob 0.7, y=1), 3 (prob 0.6, y=0)
        # TP = 2, FP = 2, FN = 1, TN = 5
        assert m["true_positives"] == 2
        assert m["false_positives"] == 2
        assert m["false_negatives"] == 1
        assert m["true_negatives"] == 5
        assert pytest.approx(m["precision"]) == 2 / 4
        assert pytest.approx(m["recall"]) == 2 / 3
        assert 0.0 <= m["f0_5"] <= 1.0

        # Distributions
        dists = results["probability_distributions"]
        assert "overall" in dists
        assert "positive_pairs" in dists
        assert "negative_pairs" in dists
        assert dists["overall"]["count"] == 10
        assert dists["positive_pairs"]["count"] == 3
        assert dists["negative_pairs"]["count"] == 7

    def test_validation_predictions_artifact_structure_and_sorting(self) -> None:
        from business_entity_resolution.evaluation.metrics import (
            build_validation_predictions_artifact,
            extract_prediction_diagnostics,
        )

        val_df = pd.DataFrame([
            {"source1_entity_id": "S1-B", "candidate_entity_id": "C-2", "candidate_source": "source2", "ground_truth_label": 0, "baseline_score": 0.4},
            {"source1_entity_id": "S1-A", "candidate_entity_id": "C-1", "candidate_source": "source2", "ground_truth_label": 1, "baseline_score": 0.9},
            {"source1_entity_id": "S1-B", "candidate_entity_id": "C-3", "candidate_source": "source3", "ground_truth_label": 1, "baseline_score": 0.7},
            {"source1_entity_id": "S1-A", "candidate_entity_id": "C-4", "candidate_source": "source2", "ground_truth_label": 0, "baseline_score": 0.2},
        ])
        probs = np.array([0.35, 0.85, 0.75, 0.85])

        artifact = build_validation_predictions_artifact(val_df, probs)

        # Mandatory columns
        expected_cols = [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "match_probability",
            "ground_truth_label",
            "baseline_score",
            "rank_within_source1",
        ]
        for col in expected_cols:
            assert col in artifact.columns

        # Sorting verification:
        # S1-A has C-1 (prob 0.85) and C-4 (prob 0.85). Ties broken by candidate_entity_id ASC: C-1 then C-4
        assert artifact.iloc[0]["source1_entity_id"] == "S1-A"
        assert artifact.iloc[0]["candidate_entity_id"] == "C-1"
        assert artifact.iloc[0]["rank_within_source1"] == 1

        assert artifact.iloc[1]["source1_entity_id"] == "S1-A"
        assert artifact.iloc[1]["candidate_entity_id"] == "C-4"
        assert artifact.iloc[1]["rank_within_source1"] == 2

        # S1-B has C-3 (prob 0.75) and C-2 (prob 0.35)
        assert artifact.iloc[2]["source1_entity_id"] == "S1-B"
        assert artifact.iloc[2]["candidate_entity_id"] == "C-3"
        assert artifact.iloc[2]["rank_within_source1"] == 1

        assert artifact.iloc[3]["source1_entity_id"] == "S1-B"
        assert artifact.iloc[3]["candidate_entity_id"] == "C-2"
        assert artifact.iloc[3]["rank_within_source1"] == 2

        # Error cases
        error_cases = extract_prediction_diagnostics(artifact, top_n=5)
        assert "high_confidence_false_positives" in error_cases
        assert "low_confidence_true_positives" in error_cases
        # S1-A / C-4 is FP with prob 0.85
        assert error_cases["high_confidence_false_positives"][0]["candidate_entity_id"] == "C-4"


class TestPhase6ArtifactsReload:
    """Tests ensuring saved Phase 6 model artifacts and metadata reload cleanly."""

    def test_reload_saved_models_if_exist(self) -> None:
        import json
        from business_entity_resolution.models.pairwise import CatBoostMatcher, LightGBMMatcher

        lgb_path = Path("output/phase6/lightgbm_model.joblib")
        lgb_meta = Path("output/phase6/lightgbm_metadata.json")
        if lgb_path.exists() and lgb_meta.exists():
            with open(lgb_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            assert meta["model_type"] == "lightgbm"
            assert meta["feature_count"] == 84
            assert len(meta["feature_column_order"]) == 84
            assert meta["random_seed"] == 2026

            model = LightGBMMatcher.load(lgb_path)
            assert model.is_fitted
            assert len(model.feature_names) == 84

        cb_path = Path("output/phase6/catboost_model.joblib")
        cb_meta = Path("output/phase6/catboost_metadata.json")
        if cb_path.exists() and cb_meta.exists():
            with open(cb_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            assert meta["model_type"] == "catboost"
            assert meta["feature_count"] == 84
            assert len(meta["feature_column_order"]) == 84
            assert meta["random_seed"] == 2026

            model = CatBoostMatcher.load(cb_path)
            assert model.is_fitted
            assert len(model.feature_names) == 84
