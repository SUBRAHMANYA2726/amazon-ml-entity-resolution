"""
Amazon ML Entity Resolution Package
"""

from .config import config, Config
from .dataset import DatasetLoader, parse_ground_truth_matches, expand_ground_truth, get_source_from_id
from .normalization import (
    TextNormalizer,
    BusinessNameNormalizer,
    AddressNormalizer,
    StructuredFieldNormalizer,
    Tokenizer,
    EntityNormalizer,
    NormalizedField,
    normalize_dataset,
    normalization_diagnostics,
)

__version__ = "0.1.0"

__all__ = [
    "config",
    "Config",
    "DatasetLoader",
    "parse_ground_truth_matches",
    "expand_ground_truth",
    "get_source_from_id",
    "TextNormalizer",
    "BusinessNameNormalizer",
    "AddressNormalizer",
    "StructuredFieldNormalizer",
    "Tokenizer",
    "EntityNormalizer",
    "NormalizedField",
    "normalize_dataset",
    "normalization_diagnostics",
]