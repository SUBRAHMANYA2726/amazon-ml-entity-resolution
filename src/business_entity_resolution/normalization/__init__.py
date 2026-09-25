"""Raw-preserving normalization components for the challenge pipeline."""

from business_entity_resolution.normalization.base import Normalizer
from business_entity_resolution.normalization.diagnostics import (
    FieldDiagnostics,
    diagnose_column,
    summarize,
)
from business_entity_resolution.normalization.fields import (
    STAGE_NAMES,
    FieldNormalizer,
    FieldResult,
    NormalizerRegistry,
    classify_column,
)
from business_entity_resolution.normalization.pipeline import (
    NormalizationPipeline,
    NormalizationPlan,
    normalize_records,
    representations_for,
)
from business_entity_resolution.normalization.text import (
    TextRepresentation,
    apply_text_pipeline,
    collapse_whitespace,
    fold_punctuation,
    is_missing,
    normalize_punctuation,
    strip_control_characters,
    to_unicode,
)
from business_entity_resolution.normalization.tokens import (
    alphanumeric_tokens,
    char_ngrams,
    compact,
    join_tokens,
    normalized_tokens,
    whitespace_tokens,
)

__all__ = [
    "STAGE_NAMES",
    "FieldDiagnostics",
    "FieldNormalizer",
    "FieldResult",
    "Normalizer",
    "NormalizerRegistry",
    "NormalizationPipeline",
    "NormalizationPlan",
    "TextRepresentation",
    "alphanumeric_tokens",
    "apply_text_pipeline",
    "char_ngrams",
    "classify_column",
    "collapse_whitespace",
    "compact",
    "diagnose_column",
    "fold_punctuation",
    "is_missing",
    "join_tokens",
    "normalize_punctuation",
    "normalize_records",
    "normalized_tokens",
    "representations_for",
    "strip_control_characters",
    "summarize",
    "to_unicode",
    "whitespace_tokens",
]
