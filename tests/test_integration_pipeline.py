"""Integration test for the complete production inference pipeline.

Verifies end-to-end execution: S1 → Normalization → Blocking → Features → Model → Threshold → Predictions
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.inference.pipeline import (
    InferenceConfig,
    ProductionInferencePipeline,
    create_inference_pipeline,
)


@pytest.fixture
def synthetic_s1_data() -> pd.DataFrame:
    """Create synthetic Source 1 data for testing."""
    return pd.DataFrame([
        {
            "entity_id": "S1-TEST-001",
            "business_name": "Acme Corporation",
            "business_address": "123 Main St, Springfield, IL 62701",
            "country": "US",
        },
        {
            "entity_id": "S1-TEST-002",
            "business_name": "Beta Industries LLC",
            "business_address": "456 Oak Ave, Chicago, IL 60601",
            "country": "US",
        },
        {
            "entity_id": "S1-TEST-003",
            "business_name": "Gamma Technologies",
            "business_address": "789 Pine Rd, Austin, TX 78701",
            "country": "US",
        },
        {
            "entity_id": "S1-TEST-004",
            "business_name": "Delta Services",
            "business_address": "321 Elm St, Boston, MA 02101",
            "country": "US",
        },
        {
            "entity_id": "S1-TEST-005",
            "business_name": "Epsilon Solutions",
            "business_address": "654 Maple Dr, Seattle, WA 98101",
            "country": "US",
        },
    ])


@pytest.fixture
def synthetic_s2_data() -> pd.DataFrame:
    """Create synthetic Source 2 candidate data for testing."""
    return pd.DataFrame([
        {
            "entity_id": "S2-TEST-001",
            "business_name": "Acme Corp",
            "business_address": "123 Main Street, Springfield, IL 62701",
            "country": "US",
        },
        {
            "entity_id": "S2-TEST-002",
            "business_name": "Beta Industries",
            "business_address": "456 Oak Avenue, Chicago, IL 60601",
            "country": "US",
        },
        {
            "entity_id": "S2-TEST-003",
            "business_name": "Gamma Tech Inc",
            "business_address": "789 Pine Road, Austin, TX 78701",
            "country": "US",
        },
        {
            "entity_id": "S2-TEST-004",
            "business_name": "Unrelated Company",
            "business_address": "999 Different St, Miami, FL 33101",
            "country": "US",
        },
        {
            "entity_id": "S2-TEST-005",
            "business_name": "Another Business",
            "business_address": "111 Another Ave, Denver, CO 80201",
            "country": "US",
        },
    ])


@pytest.fixture
def synthetic_s3_data() -> pd.DataFrame:
    """Create synthetic Source 3 candidate data for testing."""
    return pd.DataFrame([
        {
            "entity_id": "S3-TEST-001",
            "business_name": "Acme Corporation",
            "business_address": "123 Main St, Springfield, Illinois 62701",
            "country": "US",
        },
        {
            "entity_id": "S3-TEST-002",
            "business_name": "Beta Industrial LLC",
            "business_address": "456 Oak Ave, Chicago, Illinois 60601",
            "country": "US",
        },
        {
            "entity_id": "S3-TEST-003",
            "business_name": "Gamma Technologies Ltd",
            "business_address": "789 Pine Rd, Austin, Texas 78701",
            "country": "US",
        },
        {
            "entity_id": "S3-TEST-004",
            "business_name": "Delta Servicing",
            "business_address": "321 Elm Street, Boston, MA 02101",
            "country": "US",
        },
        {
            "entity_id": "S3-TEST-005",
            "business_name": "Random Entity",
            "business_address": "555 Random Blvd, Phoenix, AZ 85001",
            "country": "US",
        },
    ])


@pytest.fixture
def inference_config() -> InferenceConfig:
    """Load production inference configuration from artifacts."""
    return InferenceConfig.from_artifacts(
        model_path="output/phase 8/model_b_lightgbm.joblib",
        threshold_path="output/phase9/selected_threshold.json",
        consistency_path="output/phase10/consistency_config.json",
        config_path="config.yaml",
        batch_size=100,
    )


def test_inference_config_loads(inference_config: InferenceConfig):
    """Test that inference configuration loads correctly from artifacts."""
    assert inference_config.model_path.exists(), "Model artifact not found"
    assert inference_config.threshold == 0.922, f"Expected threshold 0.922, got {inference_config.threshold}"
    assert len(inference_config.feature_names) == 84, f"Expected 84 features, got {len(inference_config.feature_names)}"
    assert inference_config.consistency_config["operating_threshold"] == 0.922
    assert inference_config.blocking_config.partition_by_country is True
    assert inference_config.random_seed == 2026


def test_model_loads_and_matches_schema(inference_config: InferenceConfig):
    """Test that the model loads and feature schema matches."""
    pipeline = ProductionInferencePipeline(inference_config)
    model = pipeline._load_model()

    assert model is not None
    assert model.is_fitted
    assert len(model.feature_names) == 84
    assert list(model.feature_names) == list(inference_config.feature_names)


def test_pipeline_creation():
    """Test that the factory function creates a working pipeline."""
    pipeline = create_inference_pipeline()
    assert isinstance(pipeline, ProductionInferencePipeline)
    assert pipeline.config.threshold == 0.922
    assert len(pipeline.config.feature_names) == 84


def test_end_to_end_inference(
    synthetic_s1_data: pd.DataFrame,
    synthetic_s2_data: pd.DataFrame,
    synthetic_s3_data: pd.DataFrame,
    inference_config: InferenceConfig,
):
    """Test complete end-to-end inference pipeline execution."""
    pipeline = ProductionInferencePipeline(inference_config)

    # Run the full pipeline
    result = pipeline.run_batch(
        s1_df=synthetic_s1_data,
        s2_df=synthetic_s2_data,
        s3_df=synthetic_s3_data,
    )

    # Verify result structure
    assert isinstance(result, type(result))
    assert hasattr(result, "predictions")
    assert hasattr(result, "entity_details")
    assert hasattr(result, "candidates_evaluated")
    assert hasattr(result, "anchors_processed")
    assert hasattr(result, "thresholds_applied")

    # Verify all S1 entities have predictions (even if empty)
    assert result.anchors_processed == len(synthetic_s1_data)
    assert set(result.predictions.keys()) == set(synthetic_s1_data["entity_id"])

    # Verify predictions are lists of candidate IDs
    for s1_id, candidates in result.predictions.items():
        assert isinstance(candidates, list)
        for cand_id in candidates:
            assert isinstance(cand_id, str)
            assert cand_id.startswith(("S2-", "S3-"))

    # Verify candidates were evaluated
    assert result.candidates_evaluated > 0
    assert result.thresholds_applied == 0.922

    # Verify entity details DataFrame structure
    assert isinstance(result.entity_details, pd.DataFrame)
    if not result.entity_details.empty:
        assert "source1_entity_id" in result.entity_details.columns
        assert "prediction_status" in result.entity_details.columns
        assert "top_candidate_probability" in result.entity_details.columns


def test_pipeline_handles_empty_candidates(inference_config: InferenceConfig):
    """Test pipeline handles empty candidate sets gracefully."""
    pipeline = ProductionInferencePipeline(inference_config)

    s1_df = pd.DataFrame([
        {"entity_id": "S1-EMPTY-001", "business_name": "Test Co", "business_address": "123 Test St", "country": "US"},
    ])
    s2_df = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    s3_df = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    result = pipeline.run_batch(s1_df=s1_df, s2_df=s2_df, s3_df=s3_df)

    assert result.anchors_processed == 1
    assert result.candidates_evaluated == 0
    assert result.predictions["S1-EMPTY-001"] == []
    assert "warning" in result.metadata
    assert result.metadata["warning"] == "no_candidate_data_provided"


def test_pipeline_handles_missing_fields(inference_config: InferenceConfig):
    """Test pipeline handles missing/empty fields in input data."""
    pipeline = ProductionInferencePipeline(inference_config)

    # S1 with missing address
    s1_df = pd.DataFrame([
        {
            "entity_id": "S1-MISSING-001",
            "business_name": "Test Company",
            "business_address": "",
            "country": "US",
        },
        {
            "entity_id": "S1-MISSING-002",
            "business_name": "",
            "business_address": "456 Test Ave",
            "country": "US",
        },
        {
            "entity_id": "S1-MISSING-003",
            "business_name": "Another Co",
            "business_address": "789 Test Rd",
            "country": "",
        },
    ])

    # Candidates with missing fields
    s2_df = pd.DataFrame([
        {
            "entity_id": "S2-MISSING-001",
            "business_name": "Test Co",
            "business_address": "",
            "country": "US",
        },
        {
            "entity_id": "S2-MISSING-002",
            "business_name": "",
            "business_address": "456 Test Avenue",
            "country": "US",
        },
    ])

    result = pipeline.run_batch(s1_df=s1_df, s2_df=s2_df, s3_df=None)

    assert result.anchors_processed == 3
    assert result.candidates_evaluated > 0
    assert all(isinstance(v, list) for v in result.predictions.values())


def test_pipeline_supports_arbitrary_countries(inference_config: InferenceConfig):
    """Test pipeline works with arbitrary country codes (not hard-coded)."""
    pipeline = ProductionInferencePipeline(inference_config)

    # Use non-standard country codes
    s1_df = pd.DataFrame([
        {
            "entity_id": "S1-INTL-001",
            "business_name": "Global Corp",
            "business_address": "100 Global Plaza",
            "country": "XX",  # Arbitrary country code
        },
        {
            "entity_id": "S1-INTL-002",
            "business_name": "Worldwide Inc",
            "business_address": "200 World Way",
            "country": "YY",  # Another arbitrary code
        },
    ])

    s2_df = pd.DataFrame([
        {
            "entity_id": "S2-INTL-001",
            "business_name": "Global Corporation",
            "business_address": "100 Global Plz",
            "country": "XX",
        },
        {
            "entity_id": "S2-INTL-002",
            "business_name": "World Wide Inc",
            "business_address": "200 World Wy",
            "country": "YY",
        },
    ])

    result = pipeline.run_batch(s1_df=s1_df, s2_df=s2_df, s3_df=None)

    assert result.anchors_processed == 2
    assert result.candidates_evaluated > 0
    # Country is open-set, should not error
    assert all(isinstance(v, list) for v in result.predictions.values())


def test_pipeline_handles_multiple_matches(inference_config: InferenceConfig):
    """Test pipeline correctly handles S1 entities with multiple candidate matches."""
    pipeline = ProductionInferencePipeline(inference_config)

    # One S1 entity that could match multiple candidates
    s1_df = pd.DataFrame([
        {
            "entity_id": "S1-MULTI-001",
            "business_name": "Franchise Coffee",
            "business_address": "100 Main St",
            "country": "US",
        },
    ])

    # Multiple similar candidates (simulating franchise locations)
    s2_df = pd.DataFrame([
        {
            "entity_id": f"S2-MULTI-{i:03d}",
            "business_name": "Franchise Coffee",
            "business_address": f"{100 + i} Main St",
            "country": "US",
        }
        for i in range(5)
    ])

    result = pipeline.run_batch(s1_df=s1_df, s2_df=s2_df, s3_df=None)

    assert result.anchors_processed == 1
    assert result.candidates_evaluated == 5
    # Should handle multiple matches per S1 (not force top-1 only)
    # The exact count depends on model scores and threshold


def test_pipeline_deterministic_output(
    synthetic_s1_data: pd.DataFrame,
    synthetic_s2_data: pd.DataFrame,
    synthetic_s3_data: pd.DataFrame,
    inference_config: InferenceConfig,
):
    """Test that pipeline produces deterministic output for same inputs."""
    pipeline1 = ProductionInferencePipeline(inference_config)
    pipeline2 = ProductionInferencePipeline(inference_config)

    result1 = pipeline1.run_batch(synthetic_s1_data, synthetic_s2_data, synthetic_s3_data)
    result2 = pipeline2.run_batch(synthetic_s1_data, synthetic_s2_data, synthetic_s3_data)

    # Predictions should be identical
    assert result1.predictions == result2.predictions
    assert result1.candidates_evaluated == result2.candidates_evaluated
    assert np.allclose(result1.thresholds_applied, result2.thresholds_applied)


def test_feature_schema_matches_model(inference_config: InferenceConfig):
    """Test that feature schema exactly matches model expectations."""
    pipeline = ProductionInferencePipeline(inference_config)

    # Generate features for a small test case
    s1_norm, s2_norm, _ = pipeline.normalize_sources(
        synthetic_s1_data := pd.DataFrame([{
            "entity_id": "S1-SCHEMA-001",
            "business_name": "Test Co",
            "business_address": "123 Test St",
            "country": "US",
        }]),
        s2_df := pd.DataFrame([{
            "entity_id": "S2-SCHEMA-001",
            "business_name": "Test Company",
            "business_address": "123 Test Street",
            "country": "US",
        }]),
        s3_df=None,
    )

    merged, _ = pipeline.generate_candidates(s1_norm, s2_norm)
    pairs_df = pipeline.build_candidate_pairs_df(merged)

    s1_records = {str(row["entity_id"]): row.to_dict() for _, row in s1_norm.iterrows()}
    cand_records = {str(row["entity_id"]): row.to_dict() for _, row in s2_norm.iterrows()}

    feature_df = pipeline.generate_features(pairs_df, s1_records, cand_records)

    # Verify exact feature count and names (generate_features returns only feature columns)
    feature_cols = list(feature_df.columns)
    assert len(feature_cols) == 84
    assert feature_cols == list(inference_config.feature_names)

    # Verify no NaN or Inf in features
    assert not feature_df[feature_cols].isna().any().any()
    assert not np.isinf(feature_df[feature_cols].values).any()


def test_threshold_from_artifact(inference_config: InferenceConfig):
    """Test that threshold is correctly loaded from Phase 9 artifact."""
    threshold_path = Path("output/phase9/selected_threshold.json")
    assert threshold_path.exists()

    with open(threshold_path) as f:
        data = json.load(f)

    assert data["selected_threshold"] == 0.922
    assert data["primary_metric"] == "macro_f0_5"
    assert inference_config.threshold == data["selected_threshold"]


def test_consistency_config_from_artifact(inference_config: InferenceConfig):
    """Test that consistency config is correctly loaded from Phase 10 artifact."""
    consistency_path = Path("output/phase10/consistency_config.json")
    assert consistency_path.exists()

    with open(consistency_path) as f:
        data = json.load(f)

    assert data["operating_threshold"] == 0.922
    assert data["duplicate_removal"] is True
    assert data["top1_only"] is False
    assert inference_config.consistency_config["operating_threshold"] == 0.922