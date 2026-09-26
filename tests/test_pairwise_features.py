"""Unit tests for Phase 5 Pairwise Feature Engineering and Diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.contracts import CandidatePair
from business_entity_resolution.features.diagnostics import (
    compute_redundancy_diagnostics,
    compute_separation_diagnostics,
    summarize_class_distribution,
    validate_feature_matrix,
)
from business_entity_resolution.features.metadata import (
    FEATURE_NAMES,
    FEATURE_REGISTRY,
    export_feature_metadata,
)
from business_entity_resolution.features.pairwise import PairwiseFeatureGenerator


@pytest.fixture
def sample_records() -> tuple[dict[str, dict], dict[str, dict]]:
    """Synthetic S1 anchors and S2/S3 candidate records."""
    s1_records = {
        "S1-100": {
            "entity_id": "S1-100",
            "business_name": "Novent Owl PLLC",
            "business_name__cleaned": "novent owl pllc",
            "business_name__name_core": "novent owl",
            "business_name__name_sorted": "novent owl pllc",
            "business_address": "100 Main St, Des Plaines, IL 60016",
            "business_address__cleaned": "100 main st des plaines il 60016",
            "business_address__address_sorted": "100 60016 des il main plaines st",
            "business_address__address_postal": "60016",
            "business_address__address_house_number": "100",
            "country": "US",
            "country__cleaned": "us",
        },
        "S1-200": {
            "entity_id": "S1-200",
            "business_name": "Shakti Agro Limited",
            "business_name__cleaned": "shakti agro limited",
            "business_name__name_core": "shakti agro",
            "business_name__name_sorted": "agro limited shakti",
            "business_address": "Beside Zudio, Sambalpur, 768001 Orissa",
            "business_address__cleaned": "beside zudio sambalpur 768001 orissa",
            "business_address__address_postal": "768001",
            "country": "India",
            "country__cleaned": "india",
        },
    }
    cand_records = {
        "S2-101": {
            "entity_id": "S2-101",
            "business_name": "Novent Owl",
            "business_name__cleaned": "novent owl",
            "business_name__name_core": "novent owl",
            "business_name__name_sorted": "novent owl",
            "business_address": "100 Main Street, Des Plaines, IL 60016",
            "business_address__cleaned": "100 main street des plaines il 60016",
            "business_address__address_sorted": "100 60016 des il main plaines street",
            "business_address__address_postal": "60016",
            "business_address__address_house_number": "100",
            "country": "US",
            "country__cleaned": "us",
        },
        "S3-102": {
            "entity_id": "S3-102",
            "business_name": "Completely Unrelated Corp",
            "business_name__cleaned": "completely unrelated corp",
            "business_name__name_core": "completely unrelated",
            "business_address": "999 Other Rd, Chicago, IL 60601",
            "business_address__cleaned": "999 other rd chicago il 60601",
            "business_address__address_postal": "60601",
            "business_address__address_house_number": "999",
            "country": "US",
            "country__cleaned": "us",
        },
        "S2-201": {
            "entity_id": "S2-201",
            "business_name": "Shakthi Agro Private Limited",
            "business_name__cleaned": "shakthi agro private limited",
            "business_name__name_core": "shakthi agro",
            "business_address": "Beside Zudio, Sambalpur, 768001 Orissa",
            "business_address__cleaned": "beside zudio sambalpur 768001 orissa",
            "business_address__address_postal": "768001",
            "country": "India",
            "country__cleaned": "india",
        },
    }
    return s1_records, cand_records


class TestPairwiseFeatureGenerator:
    """Test feature generation, feature validity, missingness, and provenance."""

    def test_feature_generation_columns_and_counts(
        self, sample_records: tuple[dict, dict]
    ) -> None:
        s1_lookup, cand_lookup = sample_records
        pairs_df = pd.DataFrame([
            {
                "source1_entity_id": "S1-100",
                "candidate_entity_id": "S2-101",
                "candidate_source": "S2",
                "strategies": "exact_name;name_token;tfidf_name",
                "strategy_count": 3,
            },
            {
                "source1_entity_id": "S1-100",
                "candidate_entity_id": "S3-102",
                "candidate_source": "S3",
                "strategies": "char_ngram",
                "strategy_count": 1,
            },
            {
                "source1_entity_id": "S1-200",
                "candidate_entity_id": "S2-201",
                "candidate_source": "S2",
                "strategies": "name_token;char_ngram",
                "strategy_count": 2,
            },
        ])
        gt = {"S1-100": {"S2-101"}, "S1-200": {"S2-201"}}

        generator = PairwiseFeatureGenerator()
        feat_df, feat_names = generator.generate_features(
            candidate_pairs_df=pairs_df,
            s1_records=s1_lookup,
            cand_records=cand_lookup,
            ground_truth=gt,
        )

        assert len(feat_df) == 3
        assert feat_names == FEATURE_NAMES
        assert set(feat_names).issubset(set(feat_df.columns))

        # ID columns and label column presence
        assert "source1_entity_id" in feat_df.columns
        assert "candidate_entity_id" in feat_df.columns
        assert "candidate_source" in feat_df.columns
        assert "ground_truth_label" in feat_df.columns
        # Label must NOT be in feature columns list!
        assert "ground_truth_label" not in feat_names

        # Label verification
        assert feat_df.loc[0, "ground_truth_label"] == 1
        assert feat_df.loc[1, "ground_truth_label"] == 0
        assert feat_df.loc[2, "ground_truth_label"] == 1

    def test_feature_ranges_and_missing_handling(
        self, sample_records: tuple[dict, dict]
    ) -> None:
        s1_lookup, cand_lookup = sample_records
        # Add a record with missing address
        s1_lookup["S1-300"] = {
            "entity_id": "S1-300",
            "business_name": "Orphan Records Inc",
            "business_address": None,
            "country": "France",
        }
        cand_lookup["S2-301"] = {
            "entity_id": "S2-301",
            "business_name": "Orphan Records",
            "business_address": None,
            "country": "France",
        }
        pairs_df = pd.DataFrame([
            {"source1_entity_id": "S1-300", "candidate_entity_id": "S2-301", "candidate_source": "S2"},
        ])

        generator = PairwiseFeatureGenerator()
        feat_df, _ = generator.generate_features(pairs_df, s1_lookup, cand_lookup)

        # Ensure no NaNs or Infs anywhere in feature matrix
        for fname in FEATURE_NAMES:
            val = feat_df.loc[0, fname]
            assert not np.isnan(val), f"Feature {fname} produced NaN"
            assert not np.isinf(val), f"Feature {fname} produced Inf"

        # Check missing indicators
        assert feat_df.loc[0, "missing_s1_address"] == 1.0
        assert feat_df.loc[0, "missing_cand_address"] == 1.0
        assert feat_df.loc[0, "missing_both_address"] == 1.0
        assert feat_df.loc[0, "address_exact_match"] == 0.0

        # Generic open-set country features (France)
        assert feat_df.loc[0, "country_exact_match"] == 1.0
        assert feat_df.loc[0, "country_conflict"] == 0.0

    def test_provenance_and_source_features(
        self, sample_records: tuple[dict, dict]
    ) -> None:
        s1_lookup, cand_lookup = sample_records
        pairs_df = pd.DataFrame([
            {
                "source1_entity_id": "S1-100",
                "candidate_entity_id": "S2-101",
                "candidate_source": "S2",
                "strategies": "exact_name;name_token",
                "strategy_count": 2,
            },
            {
                "source1_entity_id": "S1-100",
                "candidate_entity_id": "S3-102",
                "candidate_source": "S3",
                "strategies": "tfidf_addr",
                "strategy_count": 1,
            },
        ])
        generator = PairwiseFeatureGenerator()
        feat_df, _ = generator.generate_features(pairs_df, s1_lookup, cand_lookup)

        # Row 0: S2, exact_name, name_token, strategy_count=2
        assert feat_df.loc[0, "source_cand_is_s2"] == 1.0
        assert feat_df.loc[0, "source_cand_is_s3"] == 0.0
        assert feat_df.loc[0, "blocking_exact_name"] == 1.0
        assert feat_df.loc[0, "blocking_name_token"] == 1.0
        assert feat_df.loc[0, "blocking_strategy_count"] == 2.0
        assert feat_df.loc[0, "blocking_multi_strategy"] == 1.0

        # Row 1: S3, tfidf_addr, strategy_count=1
        assert feat_df.loc[1, "source_cand_is_s2"] == 0.0
        assert feat_df.loc[1, "source_cand_is_s3"] == 1.0
        assert feat_df.loc[1, "blocking_tfidf_addr"] == 1.0
        assert feat_df.loc[1, "blocking_exact_name"] == 0.0
        assert feat_df.loc[1, "blocking_strategy_count"] == 1.0
        assert feat_df.loc[1, "blocking_multi_strategy"] == 0.0

    def test_deterministic_output(self, sample_records: tuple[dict, dict]) -> None:
        s1_lookup, cand_lookup = sample_records
        pairs_df = pd.DataFrame([
            {"source1_entity_id": "S1-100", "candidate_entity_id": "S2-101", "candidate_source": "S2"},
        ])
        generator = PairwiseFeatureGenerator()
        df1, _ = generator.generate_features(pairs_df, s1_lookup, cand_lookup)
        df2, _ = generator.generate_features(pairs_df, s1_lookup, cand_lookup)
        pd.testing.assert_frame_equal(df1, df2)


class TestFeatureDiagnostics:
    """Test feature validation, class distribution, and separation diagnostics."""

    def test_feature_quality_validation(self) -> None:
        df = pd.DataFrame({
            "name_exact_match": [1.0, 0.0, 1.0],
            "name_length_s1": [10.0, 20.0, 15.0],
            "source_s1_is_s1": [1.0, 1.0, 1.0],  # Constant feature
        })
        cols = ["name_exact_match", "name_length_s1", "source_s1_is_s1"]
        report = validate_feature_matrix(df, cols)

        assert report.total_rows == 3
        assert report.total_features == 3
        assert "source_s1_is_s1" in report.constant_features
        assert len(report.nan_features) == 0
        assert len(report.inf_features) == 0

    def test_separation_diagnostics(self) -> None:
        df = pd.DataFrame({
            "name_token_jaccard": [1.0, 0.9, 0.1, 0.0],
            "address_token_jaccard": [1.0, 0.8, 0.0, 0.1],
            "ground_truth_label": [1, 1, 0, 0],
        })
        cols = ["name_token_jaccard", "address_token_jaccard"]
        diags = compute_separation_diagnostics(df, cols)

        assert len(diags) == 2
        # Cohen's d must be strongly positive for highly discriminative features
        assert diags[0]["cohens_d_separation"] > 2.0
        assert diags[0]["pos_mean"] > diags[0]["neg_mean"]

    def test_summarize_class_distribution(self) -> None:
        df = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-2"],
            "candidate_source": ["S2", "S3", "S2", "S3"],
            "ground_truth_label": [1, 0, 0, 1],
        })
        dist = summarize_class_distribution(df)
        assert dist["total_candidate_pairs"] == 4
        assert dist["positive_pairs"] == 2
        assert dist["negative_pairs"] == 2
        assert dist["positive_percentage"] == 50.0
        assert dist["s2_distribution"]["positives"] == 1
        assert dist["s3_distribution"]["positives"] == 1
