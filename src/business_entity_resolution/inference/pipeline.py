"""Production Inference Pipeline for Business Entity Resolution.

End-to-end inference engine integrating all phases:
Normalization → Blocking → Candidates → Features → Scoring → Threshold → Consistency → Predictions

Supports arbitrary countries, batching, streaming, and production-safe error handling.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from business_entity_resolution.blocking.pipeline import MultiStrategyBlockingPipeline, BlockingConfig
from business_entity_resolution.config.settings import load_settings, Settings
from business_entity_resolution.features.metadata import FEATURE_NAMES
from business_entity_resolution.features.pairwise import PairwiseFeatureGenerator
from business_entity_resolution.matching.baseline import DeterministicBaselineMatcher
from business_entity_resolution.models.pairwise import LightGBMMatcher
from business_entity_resolution.normalization.pipeline import NormalizationPipeline
from business_entity_resolution.utils.logging import get_logger
from business_entity_resolution.validation.entity_level import build_entity_predictions

LOGGER = get_logger(__name__)

DEFAULT_THRESHOLD = 0.922
DEFAULT_CONSISTENCY_CONFIG = {
    "operating_threshold": 0.922,
    "duplicate_removal": True,
    "deterministic_ordering": "match_probability DESC, candidate_entity_id ASC",
    "fan_in_resolution": "keep_all_above_threshold_with_safety_check",
    "top1_only": False,
    "country_conflict_veto": False,
    "margin_veto": False,
    "singleton_extra_veto": False,
    "per_source_quota": None,
}


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Production inference configuration loaded from artifacts."""

    model_path: Path
    threshold: float
    consistency_config: Mapping[str, Any]
    feature_names: tuple[str, ...]
    blocking_config: BlockingConfig
    normalization_settings: Any
    random_seed: int = 2026
    batch_size: int = 1000

    @classmethod
    def from_artifacts(
        cls,
        model_path: str | Path,
        threshold_path: str | Path | None = None,
        consistency_path: str | Path | None = None,
        config_path: str | Path = "config.yaml",
        batch_size: int = 1000,
    ) -> InferenceConfig:
        """Load configuration from production artifacts."""
        settings = load_settings(config_path)

        # Load threshold from Phase 9 artifact
        threshold = DEFAULT_THRESHOLD
        if threshold_path:
            with open(threshold_path, "r") as f:
                thresh_data = json.load(f)
            threshold = float(thresh_data.get("selected_threshold", DEFAULT_THRESHOLD))

        # Load consistency config from Phase 10 artifact
        consistency = DEFAULT_CONSISTENCY_CONFIG
        if consistency_path:
            with open(consistency_path, "r") as f:
                consistency = json.load(f)

        # Build blocking config from settings
        blocking_opts = settings.blocking.options
        blocking_config = BlockingConfig(
            enable_exact=blocking_opts.get("exact", {}).get("enabled", True),
            enable_name_token=blocking_opts.get("name_token", {}).get("enabled", True),
            enable_address_token=blocking_opts.get("address_token", {}).get("enabled", True),
            enable_char_ngram=blocking_opts.get("char_ngram", {}).get("enabled", True),
            enable_tfidf=blocking_opts.get("tfidf", {}).get("enabled", True),
            partition_by_country=blocking_opts.get("candidate_limits", {}).get("partition_by_country", True),
            exact_max_block_size=blocking_opts.get("exact", {}).get("max_block_size", 5000),
            min_token_length=blocking_opts.get("name_token", {}).get("min_token_length", 3),
            max_token_frequency=blocking_opts.get("name_token", {}).get("max_token_frequency", 10000),
            name_token_max_cands=blocking_opts.get("name_token", {}).get("max_candidates_per_query", 50),
            address_max_block_size=blocking_opts.get("address_token", {}).get("max_block_size", 5000),
            address_max_cands=blocking_opts.get("address_token", {}).get("max_candidates_per_query", 50),
            char_ngram_size=blocking_opts.get("char_ngram", {}).get("ngram_size", 3),
            char_ngram_min_length=blocking_opts.get("char_ngram", {}).get("min_length", 3),
            char_ngram_top_k=blocking_opts.get("char_ngram", {}).get("top_k", 20),
            char_ngram_min_overlap=blocking_opts.get("char_ngram", {}).get("min_overlap", 0.35),
            tfidf_top_k_name=blocking_opts.get("tfidf", {}).get("top_k_name", 10),
            tfidf_top_k_address=blocking_opts.get("tfidf", {}).get("top_k_address", 5),
            tfidf_min_similarity=blocking_opts.get("tfidf", {}).get("min_similarity", 0.15),
            tfidf_batch_size=blocking_opts.get("tfidf", {}).get("batch_size", 2000),
            max_candidates_per_anchor=blocking_opts.get("candidate_limits", {}).get("max_candidates_per_anchor", 100),
        )

        return cls(
            model_path=Path(model_path),
            threshold=threshold,
            consistency_config=consistency,
            feature_names=tuple(FEATURE_NAMES),
            blocking_config=blocking_config,
            normalization_settings=settings.normalization,
            random_seed=settings.random_seed,
            batch_size=batch_size,
        )


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Production inference output for a batch of Source 1 entities."""

    predictions: dict[str, list[str]]  # s1_entity_id -> list of matched candidate_ids
    entity_details: pd.DataFrame  # Per-entity prediction details
    candidates_evaluated: int
    anchors_processed: int
    thresholds_applied: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ProductionInferencePipeline:
    """Production-ready inference pipeline for entity resolution.

    Orchestrates the complete pipeline:
    1. Normalize Source 1 and candidate records
    2. Block candidates per country partition
    3. Generate pairwise features
    4. Score with LightGBM model
    5. Apply F0.5-optimized threshold
    6. Enforce consistency rules
    7. Return predictions
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self._model: LightGBMMatcher | None = None
        self._blocking_pipeline: MultiStrategyBlockingPipeline | None = None
        self._normalization_pipeline: NormalizationPipeline | None = None
        self._feature_generator: PairwiseFeatureGenerator | None = None
        self._baseline_matcher: DeterministicBaselineMatcher | None = None

    def _load_model(self) -> LightGBMMatcher:
        """Load the LightGBM model with feature schema validation."""
        if self._model is None:
            LOGGER.info("Loading LightGBM model from %s", self.config.model_path)
            self._model = LightGBMMatcher.load(self.config.model_path)

            # Verify feature count matches schema
            model_features = len(self._model.feature_names)
            schema_features = len(self.config.feature_names)
            if model_features != schema_features:
                raise ValueError(
                    f"Feature count mismatch: model has {model_features} features, "
                    f"schema expects {schema_features}"
                )

            # Verify feature order matches
            if list(self._model.feature_names) != list(self.config.feature_names):
                raise ValueError(
                    f"Feature order mismatch!\n"
                    f"Model:  {list(self._model.feature_names)[:5]}...\n"
                    f"Schema: {list(self.config.feature_names)[:5]}..."
                )

            LOGGER.info("Model loaded successfully: %d features verified", model_features)
        return self._model

    def _get_blocking_pipeline(self) -> MultiStrategyBlockingPipeline:
        """Get or create the blocking pipeline."""
        if self._blocking_pipeline is None:
            self._blocking_pipeline = MultiStrategyBlockingPipeline(self.config.blocking_config)
        return self._blocking_pipeline

    def _get_normalization_pipeline(self) -> NormalizationPipeline:
        """Get or create the normalization pipeline."""
        if self._normalization_pipeline is None:
            if self.config.normalization_settings is None:
                raise ValueError("Normalization settings not configured")
            self._normalization_pipeline = NormalizationPipeline(self.config.normalization_settings)
        return self._normalization_pipeline

    def _get_feature_generator(self) -> PairwiseFeatureGenerator:
        """Get or create the feature generator."""
        if self._feature_generator is None:
            self._feature_generator = PairwiseFeatureGenerator(self._get_baseline_matcher())
        return self._feature_generator

    def _get_baseline_matcher(self) -> DeterministicBaselineMatcher:
        """Get or create the baseline matcher."""
        if self._baseline_matcher is None:
            self._baseline_matcher = DeterministicBaselineMatcher()
        return self._baseline_matcher

    def normalize_sources(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame | None = None,
        s3_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        """Normalize all source dataframes, preserving raw columns.

        Args:
            s1_df: Source 1 (anchor) entities
            s2_df: Source 2 candidate entities (optional)
            s3_df: Source 3 candidate entities (optional)

        Returns:
            Tuple of normalized DataFrames
        """
        norm_pipe = self._get_normalization_pipeline()
        s1_norm = norm_pipe.normalize_frame(s1_df)
        s2_norm = norm_pipe.normalize_frame(s2_df) if s2_df is not None else None
        s3_norm = norm_pipe.normalize_frame(s3_df) if s3_df is not None else None
        return s1_norm, s2_norm, s3_norm

    def generate_candidates(
        self,
        s1_norm: pd.DataFrame,
        candidates_norm: pd.DataFrame,
    ) -> tuple[dict[str, Any], list[tuple[str, dict[str, dict[str, set[str]]]]]]:
        """Run multi-strategy blocking to generate candidate pairs.

        Args:
            s1_norm: Normalized Source 1 entities
            candidates_norm: Combined normalized candidates (S2 + S3)

        Returns:
            Tuple of (merged_anchor_candidates, strategy_candidate_maps)
        """
        blocking = self._get_blocking_pipeline()
        blocking.fit(candidates_norm)
        merged, strategy_list = blocking.block(s1_norm)
        return merged, strategy_list

    def build_candidate_pairs_df(
        self,
        merged_candidates: dict[str, Any],
    ) -> pd.DataFrame:
        """Convert merged candidates to pairwise DataFrame for feature generation.

        Args:
            merged_candidates: Output from blocking pipeline (AnchorCandidates objects)

        Returns:
            DataFrame with columns: source1_entity_id, candidate_entity_id, candidate_source, strategies, strategy_count
        """
        rows = []
        for anchor_id, anchor_cands in merged_candidates.items():
            for prov_cand in anchor_cands.candidates:
                rows.append({
                    "source1_entity_id": anchor_id,
                    "candidate_entity_id": prov_cand.candidate_id,
                    "candidate_source": prov_cand.source,
                    "strategies": ";".join(sorted(prov_cand.strategies)),
                    "strategy_count": len(prov_cand.strategies),
                })
        return pd.DataFrame(rows)

    def generate_features(
        self,
        candidate_pairs_df: pd.DataFrame,
        s1_records: Mapping[str, Mapping[str, Any]],
        cand_records: Mapping[str, Mapping[str, Any]],
        baseline_scored_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate pairwise features for all candidate pairs.

        Args:
            candidate_pairs_df: Pairs from blocking
            s1_records: Normalized S1 records keyed by entity_id
            cand_records: Normalized candidate records keyed by entity_id
            baseline_scored_df: Pre-scored baseline (optional)

        Returns:
            Feature matrix DataFrame with feature columns in exact schema order
        """
        generator = self._get_feature_generator()
        feat_df, feature_names = generator.generate_features(
            candidate_pairs_df=candidate_pairs_df,
            s1_records=s1_records,
            cand_records=cand_records,
            baseline_scored_df=baseline_scored_df,
            ground_truth=None,  # Never leak labels in inference
        )

        # Ensure exact feature order matches schema
        missing = set(self.config.feature_names) - set(feat_df.columns)
        if missing:
            raise ValueError(f"Missing features in generated matrix: {missing}")
        return feat_df[list(self.config.feature_names)]

    def score_candidates(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Score candidate pairs with the LightGBM model.

        Args:
            feature_df: Feature matrix with columns in schema order

        Returns:
            Array of match probabilities in [0, 1]
        """
        model = self._load_model()
        probs = model.score(feature_df[list(self.config.feature_names)])
        return probs

    def apply_threshold_and_consistency(
        self,
        candidate_pairs_df: pd.DataFrame,
        probabilities: np.ndarray,
        all_s1_ids: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """Apply threshold and consistency rules to produce final predictions.

        Args:
            candidate_pairs_df: DataFrame with source1_entity_id, candidate_entity_id, candidate_source
            probabilities: Model match probabilities
            all_s1_ids: Complete list of S1 entity IDs (including those with zero candidates)

        Returns:
            Dict mapping S1 entity_id to list of matched candidate entity_ids
        """
        threshold = self.config.threshold
        consistency = self.config.consistency_config

        # Build prediction DataFrame
        pred_df = candidate_pairs_df[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
        pred_df["match_probability"] = probabilities

        # Sort deterministically: probability DESC, candidate_id ASC
        pred_df = pred_df.sort_values(
            by=["source1_entity_id", "match_probability", "candidate_entity_id"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

        # Apply threshold
        above_thresh = pred_df[pred_df["match_probability"] >= threshold]

        # Remove duplicate candidate assignments per S1 (deterministic ordering already applied)
        if consistency.get("duplicate_removal", True):
            above_thresh = above_thresh.drop_duplicates(
                subset=["source1_entity_id", "candidate_entity_id"],
                keep="first",
            )

        # Fan-in resolution: candidates assigned to multiple S1 entities
        if consistency.get("fan_in_resolution") == "keep_all_above_threshold_with_safety_check":
            # Keep all assignments above threshold (multi-match allowed)
            pass
        elif consistency.get("top1_only", False):
            # Keep only top-1 per S1
            above_thresh = above_thresh.groupby("source1_entity_id").head(1).reset_index(drop=True)

        # Build predictions dict
        predictions: dict[str, list[str]] = {}
        for _, row in above_thresh.iterrows():
            s1_id = row["source1_entity_id"]
            cand_id = row["candidate_entity_id"]
            if s1_id not in predictions:
                predictions[s1_id] = []
            predictions[s1_id].append(cand_id)

        # Ensure all S1 entities appear in output (even singletons with empty list)
        if all_s1_ids is not None:
            s1_universe = all_s1_ids
        else:
            s1_universe = candidate_pairs_df["source1_entity_id"].unique()
        for s1_id in s1_universe:
            if s1_id not in predictions:
                predictions[s1_id] = []

        return predictions

    def run_batch(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame | None = None,
        s3_df: pd.DataFrame | None = None,
    ) -> InferenceResult:
        """Run complete inference pipeline on a batch of data.

        Args:
            s1_df: Source 1 anchor entities (required)
            s2_df: Source 2 candidates (optional)
            s3_df: Source 3 candidates (optional)

        Returns:
            InferenceResult with predictions and metadata
        """
        if s1_df.empty:
            raise ValueError("Source 1 DataFrame cannot be empty")

        if s2_df is None and s3_df is None:
            raise ValueError("At least one candidate source (S2 or S3) must be provided")

        LOGGER.info("Starting production inference on %d S1 anchors", len(s1_df))

        # Combine candidates
        cand_dfs = []
        if s2_df is not None and not s2_df.empty:
            cand_dfs.append(s2_df)
        if s3_df is not None and not s3_df.empty:
            cand_dfs.append(s3_df)
        candidates_df = pd.concat(cand_dfs, ignore_index=True) if cand_dfs else pd.DataFrame()

        if candidates_df.empty:
            LOGGER.warning("No candidate data provided; returning empty predictions for all S1 entities")
            predictions = {str(row["entity_id"]): [] for _, row in s1_df.iterrows()}
            return InferenceResult(
                predictions=predictions,
                entity_details=pd.DataFrame(),
                candidates_evaluated=0,
                anchors_processed=len(s1_df),
                thresholds_applied=self.config.threshold,
                metadata={"warning": "no_candidate_data_provided"},
            )

        # 1. Normalize all sources
        LOGGER.info("Normalizing %d S1 + %d candidates", len(s1_df), len(candidates_df))
        s1_norm, s2_norm, s3_norm = self.normalize_sources(s1_df, s2_df, s3_df)

        # Combine normalized candidates
        norm_cand_dfs = []
        if s2_norm is not None:
            norm_cand_dfs.append(s2_norm)
        if s3_norm is not None:
            norm_cand_dfs.append(s3_norm)
        candidates_norm = pd.concat(norm_cand_dfs, ignore_index=True) if norm_cand_dfs else pd.DataFrame()

        # 2. Generate candidates via blocking
        LOGGER.info("Running multi-strategy blocking...")
        merged, strategy_list = self.generate_candidates(s1_norm, candidates_norm)

        # 3. Build candidate pairs DataFrame
        candidate_pairs_df = self.build_candidate_pairs_df(merged)
        n_pairs = len(candidate_pairs_df)
        LOGGER.info("Generated %d candidate pairs", n_pairs)

        if n_pairs == 0:
            LOGGER.warning("No candidate pairs generated; returning empty predictions for all S1 entities")
            predictions = {str(row["entity_id"]): [] for _, row in s1_df.iterrows()}
            return InferenceResult(
                predictions=predictions,
                entity_details=pd.DataFrame(),
                candidates_evaluated=0,
                anchors_processed=len(s1_df),
                thresholds_applied=self.config.threshold,
                metadata={"warning": "zero_candidates_generated"},
            )

        # 4. Build record lookups
        s1_records = {str(row["entity_id"]): row.to_dict() for _, row in s1_norm.iterrows()}
        cand_records = {str(row["entity_id"]): row.to_dict() for _, row in candidates_norm.iterrows()}

        # 5. Generate features
        LOGGER.info("Generating pairwise features...")
        feature_df = self.generate_features(
            candidate_pairs_df=candidate_pairs_df,
            s1_records=s1_records,
            cand_records=cand_records,
            baseline_scored_df=None,  # Will be computed inside
        )

        # 6. Score with model
        LOGGER.info("Scoring %d candidate pairs with LightGBM...", len(feature_df))
        probabilities = self.score_candidates(feature_df)

        # 7. Apply threshold and consistency
        LOGGER.info("Applying threshold %.3f and consistency rules...", self.config.threshold)
        all_s1_ids = s1_df["entity_id"].astype(str).tolist()
        predictions = self.apply_threshold_and_consistency(candidate_pairs_df, probabilities, all_s1_ids)

        # Build entity details for inspection
        pred_df = candidate_pairs_df[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
        pred_df["match_probability"] = probabilities
        # Add dummy ground_truth_label for inference mode (not available)
        pred_df["ground_truth_label"] = 0
        pred_df = pred_df.sort_values(
            by=["source1_entity_id", "match_probability", "candidate_entity_id"],
            ascending=[True, False, True],
        )

        entity_details = build_entity_predictions(
            val_preds_df=pred_df,
            threshold=self.config.threshold,
        )

        return InferenceResult(
            predictions=predictions,
            entity_details=entity_details,
            candidates_evaluated=n_pairs,
            anchors_processed=len(s1_df),
            thresholds_applied=self.config.threshold,
            metadata={
                "model_path": str(self.config.model_path),
                "feature_count": len(self.config.feature_names),
                "consistency_config": dict(self.config.consistency_config),
            },
        )

    def run_streaming(
        self,
        s1_iterator,
        s2_iterator,
        s3_iterator,
        chunk_size: int | None = None,
    ) -> list[InferenceResult]:
        """Run inference in streaming mode for large datasets.

        Args:
            s1_iterator: Iterator yielding S1 DataFrame chunks
            s2_iterator: Iterator yielding S2 DataFrame chunks
            s3_iterator: Iterator yielding S3 DataFrame chunks
            chunk_size: Override default batch size

        Returns:
            List of InferenceResult per chunk
        """
        batch_size = chunk_size or self.config.batch_size
        results = []

        for s1_chunk in s1_iterator:
            # For streaming, we need candidates available
            # This is a simplified version; production would coordinate candidate loading
            s2_chunk = next(s2_iterator, pd.DataFrame())
            s3_chunk = next(s3_iterator, pd.DataFrame())

            if s1_chunk.empty:
                continue

            result = self.run_batch(s1_chunk, s2_chunk, s3_chunk)
            results.append(result)

        return results


def create_inference_pipeline(
    model_path: str | Path = "output/phase 8/model_b_lightgbm.joblib",
    threshold_path: str | Path = "output/phase9/selected_threshold.json",
    consistency_path: str | Path = "output/phase10/consistency_config.json",
    config_path: str | Path = "config.yaml",
    batch_size: int = 1000,
) -> ProductionInferencePipeline:
    """Factory function to create a configured production inference pipeline.

    Args:
        model_path: Path to LightGBM model artifact
        threshold_path: Path to Phase 9 threshold artifact
        consistency_path: Path to Phase 10 consistency config
        config_path: Path to main configuration file
        batch_size: Processing batch size

    Returns:
        Configured ProductionInferencePipeline instance
    """
    config = InferenceConfig.from_artifacts(
        model_path=model_path,
        threshold_path=threshold_path,
        consistency_path=consistency_path,
        config_path=config_path,
        batch_size=batch_size,
    )
    return ProductionInferencePipeline(config)


class InferencePipeline(ABC):
    """Abstract base class for inference pipelines (Phase 0 contract)."""

    @abstractmethod
    def run(self, inputs: object) -> object:
        """Return an inference artifact after all data contracts are implemented."""