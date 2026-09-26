"""Phase 4 Baseline Matching and Retrieval module."""

from business_entity_resolution.matching.baseline import (
    BaselinePairScore,
    BaselineScoreWeights,
    DeterministicBaselineMatcher,
)
from business_entity_resolution.matching.evaluation import (
    BaselineEvaluationReport,
    BaselineEvaluator,
    ThresholdMetrics,
    TopKRecallMetrics,
    compute_f_beta,
)
from business_entity_resolution.matching.similarity import (
    FieldComparisonResult,
    char_ngram_jaccard,
    common_prefix_ratio,
    common_suffix_ratio,
    compare_structured_field,
    exact_equality,
    is_blank,
    jaro_winkler_similarity,
    levenshtein_similarity,
    token_jaccard,
    token_overlap,
)

__all__ = [
    "BaselineEvaluationReport",
    "BaselineEvaluator",
    "BaselinePairScore",
    "BaselineScoreWeights",
    "DeterministicBaselineMatcher",
    "FieldComparisonResult",
    "ThresholdMetrics",
    "TopKRecallMetrics",
    "char_ngram_jaccard",
    "common_prefix_ratio",
    "common_suffix_ratio",
    "compare_structured_field",
    "compute_f_beta",
    "exact_equality",
    "is_blank",
    "jaro_winkler_similarity",
    "levenshtein_similarity",
    "token_jaccard",
    "token_overlap",
]
