"""Machine-readable feature schema and metadata definitions for Phase 5.

Every pairwise feature has an explicit entry detailing:
- Feature name and group
- Data type
- Expected numeric range
- Missing value policy (None/missingness imputed deterministically)
- Source input fields
- Human-readable definition
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Specification metadata for a single pairwise feature."""

    name: str
    group: str
    dtype: str
    expected_range: tuple[float | None, float | None]
    missing_allowed: bool
    source_fields: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Convert specification to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "group": self.group,
            "dtype": self.dtype,
            "expected_range": list(self.expected_range),
            "missing_allowed": self.missing_allowed,
            "source_fields": list(self.source_fields),
            "description": self.description,
        }


# Complete registry of all Phase 5 feature definitions
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # Group A: Name Features
    FeatureSpec("name_exact_match", "name", "float32", (0.0, 1.0), False, ("business_name",), "Normalized business name exact string match indicator"),
    FeatureSpec("name_core_exact_match", "name", "float32", (0.0, 1.0), False, ("business_name",), "Legal-suffix-free core business name exact match indicator"),
    FeatureSpec("name_sorted_exact_match", "name", "float32", (0.0, 1.0), False, ("business_name",), "Token-sorted business name exact equality (word-order invariant)"),
    FeatureSpec("name_length_s1", "name", "float32", (0.0, None), False, ("business_name",), "Character length of normalized S1 business name"),
    FeatureSpec("name_length_cand", "name", "float32", (0.0, None), False, ("business_name",), "Character length of normalized candidate business name"),
    FeatureSpec("name_length_diff", "name", "float32", (0.0, None), False, ("business_name",), "Absolute difference between S1 and candidate name character lengths"),
    FeatureSpec("name_length_ratio", "name", "float32", (0.0, 1.0), False, ("business_name",), "Ratio of min to max character length between S1 and candidate names"),
    FeatureSpec("name_token_count_s1", "name", "float32", (0.0, None), False, ("business_name",), "Number of alphanumeric tokens in S1 business name"),
    FeatureSpec("name_token_count_cand", "name", "float32", (0.0, None), False, ("business_name",), "Number of alphanumeric tokens in candidate business name"),
    FeatureSpec("name_token_count_diff", "name", "float32", (0.0, None), False, ("business_name",), "Absolute difference in token counts between S1 and candidate names"),
    FeatureSpec("name_token_jaccard", "name", "float32", (0.0, 1.0), False, ("business_name",), "Jaccard similarity between S1 and candidate name token sets"),
    FeatureSpec("name_token_overlap", "name", "float32", (0.0, 1.0), False, ("business_name",), "Overlap containment coefficient between S1 and candidate name tokens"),
    FeatureSpec("name_char_3gram_jaccard", "name", "float32", (0.0, 1.0), False, ("business_name",), "Jaccard similarity between character 3-grams of S1 and candidate names"),
    FeatureSpec("name_levenshtein_sim", "name", "float32", (0.0, 1.0), False, ("business_name",), "Normalized Levenshtein edit-distance similarity in [0, 1]"),
    FeatureSpec("name_jaro_winkler", "name", "float32", (0.0, 1.0), False, ("business_name",), "Jaro-Winkler string similarity with prefix weight in [0, 1]"),
    FeatureSpec("name_prefix_similarity", "name", "float32", (0.0, 1.0), False, ("business_name",), "Length of longest common prefix divided by max name length"),
    FeatureSpec("name_suffix_similarity", "name", "float32", (0.0, 1.0), False, ("business_name",), "Length of longest common suffix divided by max name length"),

    # Group B: Address Features
    FeatureSpec("address_exact_match", "address", "float32", (0.0, 1.0), False, ("business_address",), "Normalized business address exact equality indicator"),
    FeatureSpec("address_sorted_exact_match", "address", "float32", (0.0, 1.0), False, ("business_address",), "Component-sorted business address exact equality indicator"),
    FeatureSpec("address_length_s1", "address", "float32", (0.0, None), False, ("business_address",), "Character length of S1 business address"),
    FeatureSpec("address_length_cand", "address", "float32", (0.0, None), False, ("business_address",), "Character length of candidate business address"),
    FeatureSpec("address_length_diff", "address", "float32", (0.0, None), False, ("business_address",), "Absolute difference between S1 and candidate address lengths"),
    FeatureSpec("address_length_ratio", "address", "float32", (0.0, 1.0), False, ("business_address",), "Ratio of min to max address character length"),
    FeatureSpec("address_token_count_s1", "address", "float32", (0.0, None), False, ("business_address",), "Number of address tokens in S1 record"),
    FeatureSpec("address_token_count_cand", "address", "float32", (0.0, None), False, ("business_address",), "Number of address tokens in candidate record"),
    FeatureSpec("address_token_count_diff", "address", "float32", (0.0, None), False, ("business_address",), "Absolute difference in address token counts"),
    FeatureSpec("address_token_jaccard", "address", "float32", (0.0, 1.0), False, ("business_address",), "Jaccard similarity between address token sets"),
    FeatureSpec("address_token_overlap", "address", "float32", (0.0, 1.0), False, ("business_address",), "Overlap containment coefficient between address token sets"),
    FeatureSpec("address_char_3gram_jaccard", "address", "float32", (0.0, 1.0), False, ("business_address",), "Jaccard similarity of character 3-grams of addresses"),
    FeatureSpec("address_levenshtein_sim", "address", "float32", (0.0, 1.0), False, ("business_address",), "Normalized Levenshtein edit distance similarity of addresses"),

    # Group C: Structured Fields
    FeatureSpec("postal_exact_match", "structured", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that postal code is present in both and matches exactly"),
    FeatureSpec("postal_both_present", "structured", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that postal code is present in both records"),
    FeatureSpec("postal_one_missing", "structured", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that postal code is present in exactly one record"),
    FeatureSpec("postal_both_missing", "structured", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that postal code is missing from both records"),
    FeatureSpec("house_number_exact_match", "structured", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that street house number matches exactly"),
    FeatureSpec("house_number_both_present", "structured", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that house number is present in both records"),
    FeatureSpec("house_number_one_missing", "structured", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that house number is present in exactly one record"),
    FeatureSpec("house_number_both_missing", "structured", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that house number is missing in both records"),
    FeatureSpec("unit_exact_match", "structured", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that apartment/suite unit designator matches exactly"),
    FeatureSpec("unit_both_present", "structured", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that apartment/suite unit is present in both records"),

    # Group D: Field Missingness
    FeatureSpec("missing_s1_name", "missingness", "float32", (0.0, 1.0), False, ("business_name",), "Indicator that S1 business name is blank/missing"),
    FeatureSpec("missing_cand_name", "missingness", "float32", (0.0, 1.0), False, ("business_name",), "Indicator that candidate business name is blank/missing"),
    FeatureSpec("missing_both_name", "missingness", "float32", (0.0, 1.0), False, ("business_name",), "Indicator that both business names are blank/missing"),
    FeatureSpec("missing_either_name", "missingness", "float32", (0.0, 1.0), False, ("business_name",), "Indicator that either business name is blank/missing"),
    FeatureSpec("missing_s1_address", "missingness", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that S1 address is blank/missing"),
    FeatureSpec("missing_cand_address", "missingness", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that candidate address is blank/missing"),
    FeatureSpec("missing_both_address", "missingness", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that both addresses are blank/missing"),
    FeatureSpec("missing_either_address", "missingness", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that either address is blank/missing"),
    FeatureSpec("missing_s1_country", "missingness", "float32", (0.0, 1.0), False, ("country",), "Indicator that S1 country is blank/missing"),
    FeatureSpec("missing_cand_country", "missingness", "float32", (0.0, 1.0), False, ("country",), "Indicator that candidate country is blank/missing"),
    FeatureSpec("missing_s1_postal", "missingness", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that S1 postal code is missing"),
    FeatureSpec("missing_cand_postal", "missingness", "float32", (0.0, 1.0), False, ("business_address",), "Indicator that candidate postal code is missing"),
    FeatureSpec("missing_field_count_total", "missingness", "float32", (0.0, 8.0), False, ("business_name", "business_address", "country"), "Sum of missing attribute indicators across both records"),

    # Group E: Cross-Field Consistency
    FeatureSpec("cross_name_match_and_address_match", "cross_field", "float32", (0.0, 1.0), False, ("business_name", "business_address"), "Both name exact match and address exact match hold"),
    FeatureSpec("cross_name_match_and_postal_match", "cross_field", "float32", (0.0, 1.0), False, ("business_name", "business_address"), "Both name exact match and postal exact match hold"),
    FeatureSpec("cross_name_sim_x_country_match", "cross_field", "float32", (0.0, 1.0), False, ("business_name", "country"), "Interaction term: name token Jaccard multiplied by country match"),
    FeatureSpec("cross_name_sim_x_address_sim", "cross_field", "float32", (0.0, 1.0), False, ("business_name", "business_address"), "Interaction term: name token Jaccard multiplied by address token Jaccard"),
    FeatureSpec("cross_address_sim_x_postal_match", "cross_field", "float32", (0.0, 1.0), False, ("business_address",), "Interaction term: address token Jaccard multiplied by postal match"),
    FeatureSpec("cross_country_and_postal_match", "cross_field", "float32", (0.0, 1.0), False, ("country", "business_address"), "Both country match and postal match hold"),

    # Group F: Generic Open-Set Country Features (Zero country hard-coding)
    FeatureSpec("country_exact_match", "country", "float32", (0.0, 1.0), False, ("country",), "Indicator that country labels are both present and match case-insensitively"),
    FeatureSpec("country_conflict", "country", "float32", (0.0, 1.0), False, ("country",), "Indicator that both country labels are present but differ (contradiction)"),
    FeatureSpec("country_both_present", "country", "float32", (0.0, 1.0), False, ("country",), "Indicator that country is declared in both records"),
    FeatureSpec("country_missing", "country", "float32", (0.0, 1.0), False, ("country",), "Indicator that country is missing in either record"),

    # Group G: Candidate Provenance Features (Phase 3 Retrieval Signals)
    FeatureSpec("blocking_exact_name", "provenance", "float32", (0.0, 1.0), False, ("provenance",), "Candidate was retrieved by exact normalized name strategy"),
    FeatureSpec("blocking_exact_name_sorted", "provenance", "float32", (0.0, 1.0), False, ("provenance",), "Candidate was retrieved by exact token-sorted name strategy"),
    FeatureSpec("blocking_exact_country_name", "provenance", "float32", (0.0, 1.0), False, ("provenance",), "Candidate was retrieved by country + name strategy"),
    FeatureSpec("blocking_exact_country_postal_name", "provenance", "float32", (0.0, 1.0), False, ("provenance",), "Candidate was retrieved by country + postal + name strategy"),
    FeatureSpec("blocking_exact_country_postal_house", "provenance", "float32", (0.0, 1.0), False, ("provenance",), "Candidate was retrieved by country + postal + house number"),
    FeatureSpec("blocking_exact_country_house_token", "provenance", "float32", (0.0, 1.0), False, ("provenance",), "Candidate was retrieved by country + house + name token"),
    FeatureSpec("blocking_name_token", "provenance", "float32", (0.0, 1.0), False, ("provenance",), "Candidate was retrieved by name token inverted index"),
    FeatureSpec("blocking_address_token", "provenance", "float32", (0.0, 1.0), False, ("provenance",), "Candidate was retrieved by address token inverted index"),
    FeatureSpec("blocking_char_ngram", "provenance", "float32", (0.0, 1.0), False, ("provenance",), "Candidate was retrieved by character n-gram overlap"),
    FeatureSpec("blocking_tfidf_name", "provenance", "float32", (0.0, 1.0), False, ("provenance",), "Candidate was retrieved by TF-IDF name sparse retrieval"),
    FeatureSpec("blocking_tfidf_addr", "provenance", "float32", (0.0, 1.0), False, ("provenance",), "Candidate was retrieved by TF-IDF address sparse retrieval"),
    FeatureSpec("blocking_strategy_count", "provenance", "float32", (1.0, None), False, ("provenance",), "Total number of Phase 3 blocking strategies that retrieved this pair"),
    FeatureSpec("blocking_multi_strategy", "provenance", "float32", (0.0, 1.0), False, ("provenance",), "Indicator that pair was retrieved by 2 or more distinct strategies"),

    # Group H: Source Indicators
    FeatureSpec("source_cand_is_s2", "source", "float32", (0.0, 1.0), False, ("candidate_source",), "Indicator that candidate entity originates from Source 2"),
    FeatureSpec("source_cand_is_s3", "source", "float32", (0.0, 1.0), False, ("candidate_source",), "Indicator that candidate entity originates from Source 3"),
    FeatureSpec("source_s1_is_s1", "source", "float32", (0.0, 1.0), False, ("source1_entity_id",), "Constant indicator verifying anchor is Source 1"),

    # Group I: Baseline Signals
    FeatureSpec("baseline_score", "baseline", "float32", (0.0, 1.0), False, ("baseline_score",), "Phase 4 deterministic baseline composite matching score"),
    FeatureSpec("baseline_rank", "baseline", "float32", (1.0, None), False, ("baseline_rank",), "Rank of candidate entity under Phase 4 baseline score (1 = best)"),
    FeatureSpec("baseline_name_sim", "baseline", "float32", (0.0, 1.0), False, ("name_similarity_score",), "Phase 4 baseline name similarity component"),
    FeatureSpec("baseline_address_sim", "baseline", "float32", (0.0, 1.0), False, ("address_similarity_score",), "Phase 4 baseline address similarity component"),
    FeatureSpec("baseline_structured_sim", "baseline", "float32", (0.0, 1.0), False, ("structured_similarity_score",), "Phase 4 baseline structured similarity component"),
)

FEATURE_REGISTRY: dict[str, FeatureSpec] = {spec.name: spec for spec in FEATURE_SPECS}
FEATURE_NAMES: list[str] = [spec.name for spec in FEATURE_SPECS]


def export_feature_metadata(output_path: Path) -> dict[str, Any]:
    """Export complete machine-readable feature schema to JSON."""
    payload = {
        "feature_count": len(FEATURE_SPECS),
        "feature_groups": sorted(list({s.group for s in FEATURE_SPECS})),
        "features": {s.name: s.to_dict() for s in FEATURE_SPECS},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload
