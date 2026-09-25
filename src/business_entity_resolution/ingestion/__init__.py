"""Dataset ingestion, discovery, and profiling components."""

from business_entity_resolution.ingestion.base import (
    DataIngestor,
    DatasetProfiler,
    SchemaValidator,
)
from business_entity_resolution.ingestion.dataset import (
    LoadedFile,
    iter_chunks,
    read_filtered,
    read_header,
    read_sample,
)
from business_entity_resolution.ingestion.discovery import (
    DiscoveredDataset,
    DiscoveredFile,
    DatasetDiscoveryError,
    discover_dataset,
)
from business_entity_resolution.ingestion.ground_truth import (
    analyse_ground_truth,
    coverage_against_sources,
    validate_ground_truth_ids,
)
from business_entity_resolution.ingestion.profiling import (
    FileProfile,
    SampleProfile,
    build_pair_sample,
    collect_country_distribution,
    exact_pattern_counts,
    profile_file,
    profile_sample,
)
from business_entity_resolution.ingestion.variation import (
    VariationExample,
    VariationReport,
    compare_pairs,
)

__all__ = [
    "DataIngestor",
    "DatasetDiscoveryError",
    "DatasetProfiler",
    "DiscoveredDataset",
    "DiscoveredFile",
    "FileProfile",
    "LoadedFile",
    "SampleProfile",
    "SchemaValidator",
    "VariationExample",
    "VariationReport",
    "analyse_ground_truth",
    "build_pair_sample",
    "collect_country_distribution",
    "compare_pairs",
    "coverage_against_sources",
    "discover_dataset",
    "exact_pattern_counts",
    "iter_chunks",
    "profile_file",
    "profile_sample",
    "read_filtered",
    "read_header",
    "read_sample",
    "validate_ground_truth_ids",
]
