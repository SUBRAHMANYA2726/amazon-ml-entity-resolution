"""Phase 5 Pairwise feature-generation module."""

from business_entity_resolution.features.base import PairFeatureGenerator
from business_entity_resolution.features.diagnostics import (
    FeatureColumnStat,
    FeatureQualityReport,
    compute_redundancy_diagnostics,
    compute_separation_diagnostics,
    summarize_class_distribution,
    validate_feature_matrix,
)
from business_entity_resolution.features.metadata import (
    FEATURE_NAMES,
    FEATURE_REGISTRY,
    FEATURE_SPECS,
    FeatureSpec,
    export_feature_metadata,
)
from business_entity_resolution.features.pairwise import PairwiseFeatureGenerator

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_REGISTRY",
    "FEATURE_SPECS",
    "FeatureColumnStat",
    "FeatureQualityReport",
    "FeatureSpec",
    "PairFeatureGenerator",
    "PairwiseFeatureGenerator",
    "compute_redundancy_diagnostics",
    "compute_separation_diagnostics",
    "export_feature_metadata",
    "summarize_class_distribution",
    "validate_feature_matrix",
]
