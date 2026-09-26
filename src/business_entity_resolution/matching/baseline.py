"""Deterministic baseline matching for Business Entity Resolution.

Computes explainable, transparent similarity scores between anchor S1 entities
and candidate S2/S3 entities without machine learning. Ranks candidate pairs
per anchor while preserving candidate source provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from business_entity_resolution.matching.similarity import (
    char_ngram_jaccard,
    compare_structured_field,
    exact_equality,
    is_blank,
    jaro_winkler_similarity,
    levenshtein_similarity,
    token_jaccard,
    token_overlap,
)
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BaselineScoreWeights:
    """Configurable weights for transparent baseline similarity calculation."""

    # High-level component weights (sum = 1.0)
    name_weight: float = 0.50
    address_weight: float = 0.35
    structured_weight: float = 0.15

    # Name internal weights (sum = 1.0)
    w_name_exact: float = 0.20
    w_name_core_exact: float = 0.20
    w_name_sorted_exact: float = 0.10
    w_name_token_jaccard: float = 0.20
    w_name_levenshtein: float = 0.15
    w_name_jaro_winkler: float = 0.15

    # Address internal weights (sum = 1.0)
    w_addr_exact: float = 0.25
    w_addr_sorted_exact: float = 0.15
    w_addr_token_jaccard: float = 0.30
    w_addr_char_ngram: float = 0.15
    w_addr_postal_match: float = 0.15

    # Structured internal weights (sum = 1.0)
    w_struct_country: float = 0.50
    w_struct_postal: float = 0.30
    w_struct_house: float = 0.20


@dataclass(frozen=True, slots=True)
class BaselinePairScore:
    """Individual similarity signals and composite baseline score."""

    baseline_score: float
    name_similarity: float
    address_similarity: float
    structured_similarity: float


class DeterministicBaselineMatcher:
    """Explainable, deterministic entity similarity matcher."""

    def __init__(self, weights: BaselineScoreWeights | None = None) -> None:
        self.weights = weights or BaselineScoreWeights()

    def score_pair(
        self,
        s1: Mapping[str, Any],
        cand: Mapping[str, Any],
    ) -> BaselinePairScore:
        """Calculate composite baseline score and sub-scores for a single pair."""
        w = self.weights

        # 1. Name Similarity
        s1_name_clean = s1.get("business_name__cleaned") or s1.get("business_name")
        cand_name_clean = cand.get("business_name__cleaned") or cand.get("business_name")

        s1_name_core = s1.get("business_name__name_core") or s1_name_clean
        cand_name_core = cand.get("business_name__name_core") or cand_name_clean

        s1_name_sorted = s1.get("business_name__name_sorted") or s1_name_clean
        cand_name_sorted = cand.get("business_name__name_sorted") or cand_name_clean

        sim_name_exact = exact_equality(s1_name_clean, cand_name_clean)
        sim_name_core = exact_equality(s1_name_core, cand_name_core)
        sim_name_sorted = exact_equality(s1_name_sorted, cand_name_sorted)
        sim_name_jaccard = token_jaccard(s1_name_clean, cand_name_clean)
        sim_name_lev = levenshtein_similarity(s1_name_clean, cand_name_clean)
        sim_name_jw = jaro_winkler_similarity(s1_name_clean, cand_name_clean)

        name_similarity = (
            w.w_name_exact * sim_name_exact
            + w.w_name_core_exact * sim_name_core
            + w.w_name_sorted_exact * sim_name_sorted
            + w.w_name_token_jaccard * sim_name_jaccard
            + w.w_name_levenshtein * sim_name_lev
            + w.w_name_jaro_winkler * sim_name_jw
        )

        # 2. Address Similarity
        s1_addr = s1.get("business_address__cleaned") or s1.get("business_address")
        cand_addr = cand.get("business_address__cleaned") or cand.get("business_address")

        s1_addr_sorted = s1.get("business_address__address_sorted") or s1_addr
        cand_addr_sorted = cand.get("business_address__address_sorted") or cand_addr

        s1_postal = s1.get("business_address__address_postal")
        cand_postal = cand.get("business_address__address_postal")

        s1_house = s1.get("business_address__address_house_number")
        cand_house = cand.get("business_address__address_house_number")

        addr_missing = is_blank(s1_addr) or is_blank(cand_addr)
        if addr_missing:
            address_similarity = 0.0
        else:
            sim_addr_exact = exact_equality(s1_addr, cand_addr)
            sim_addr_sorted = exact_equality(s1_addr_sorted, cand_addr_sorted)
            sim_addr_jaccard = token_jaccard(s1_addr, cand_addr)
            sim_addr_char = char_ngram_jaccard(s1_addr, cand_addr, n=3)
            sim_addr_postal = exact_equality(s1_postal, cand_postal) if (not is_blank(s1_postal) or not is_blank(cand_postal)) else sim_addr_exact

            address_similarity = (
                w.w_addr_exact * sim_addr_exact
                + w.w_addr_sorted_exact * sim_addr_sorted
                + w.w_addr_token_jaccard * sim_addr_jaccard
                + w.w_addr_char_ngram * sim_addr_char
                + w.w_addr_postal_match * sim_addr_postal
            )

        # 3. Structured Field Similarity
        s1_country = s1.get("country__cleaned") or s1.get("country")
        cand_country = cand.get("country__cleaned") or cand.get("country")

        country_comp = compare_structured_field(s1_country, cand_country)
        postal_comp = compare_structured_field(s1_postal, cand_postal)
        house_comp = compare_structured_field(s1_house, cand_house)

        struct_weight_sum = 0.0
        struct_score_sum = 0.0

        if country_comp.both_present == 1.0 or country_comp.one_missing == 1.0:
            struct_weight_sum += w.w_struct_country
            struct_score_sum += w.w_struct_country * country_comp.exact_match

        if postal_comp.both_present == 1.0 or postal_comp.one_missing == 1.0:
            struct_weight_sum += w.w_struct_postal
            struct_score_sum += w.w_struct_postal * postal_comp.exact_match

        if house_comp.both_present == 1.0 or house_comp.one_missing == 1.0:
            struct_weight_sum += w.w_struct_house
            struct_score_sum += w.w_struct_house * house_comp.exact_match

        if struct_weight_sum > 0:
            structured_similarity = struct_score_sum / struct_weight_sum
        else:
            structured_similarity = 0.0

        # 4. Composite Baseline Score Construction
        # Country conflict: If both countries present and unequal, score is 0.0
        if country_comp.both_present == 1.0 and country_comp.exact_match == 0.0:
            return BaselinePairScore(
                baseline_score=0.0,
                name_similarity=name_similarity,
                address_similarity=address_similarity,
                structured_similarity=0.0,
            )

        # If candidate address is missing, redistribute address weight to name
        if addr_missing:
            eff_name_w = w.name_weight + w.address_weight * 0.75
            eff_struct_w = w.structured_weight + w.address_weight * 0.25
            raw_score = eff_name_w * name_similarity + eff_struct_w * structured_similarity
        else:
            raw_score = (
                w.name_weight * name_similarity
                + w.address_weight * address_similarity
                + w.structured_weight * structured_similarity
            )

        baseline_score = max(0.0, min(1.0, float(raw_score)))
        return BaselinePairScore(
            baseline_score=baseline_score,
            name_similarity=name_similarity,
            address_similarity=address_similarity,
            structured_similarity=structured_similarity,
        )

    def rank_candidates(
        self,
        candidate_pairs_df: pd.DataFrame,
        s1_lookup: Mapping[str, Mapping[str, Any]],
        cand_lookup: Mapping[str, Mapping[str, Any]],
    ) -> pd.DataFrame:
        """Compute scores and per-anchor rank for all candidate pairs.
        
        Preserves source1_entity_id, candidate_entity_id, candidate_source,
        and adds baseline_score, baseline_rank, and similarity components.
        """
        n_pairs = len(candidate_pairs_df)
        LOGGER.info("Scoring and ranking %d candidate pairs...", n_pairs)

        scores = np.zeros(n_pairs, dtype=np.float32)
        name_sims = np.zeros(n_pairs, dtype=np.float32)
        addr_sims = np.zeros(n_pairs, dtype=np.float32)
        struct_sims = np.zeros(n_pairs, dtype=np.float32)

        s1_ids = candidate_pairs_df["source1_entity_id"].values
        cand_ids = candidate_pairs_df["candidate_entity_id"].values

        empty_dict: dict[str, Any] = {}
        for idx in range(n_pairs):
            s1_rec = s1_lookup.get(s1_ids[idx], empty_dict)
            cand_rec = cand_lookup.get(cand_ids[idx], empty_dict)

            res = self.score_pair(s1_rec, cand_rec)
            scores[idx] = res.baseline_score
            name_sims[idx] = res.name_similarity
            addr_sims[idx] = res.address_similarity
            struct_sims[idx] = res.structured_similarity

        scored_df = candidate_pairs_df.copy()
        scored_df["baseline_score"] = scores
        scored_df["name_similarity_score"] = name_sims
        scored_df["address_similarity_score"] = addr_sims
        scored_df["structured_similarity_score"] = struct_sims

        # Rank per anchor: descending score, ascending candidate ID for stable tie-breaking
        scored_df.sort_values(
            by=["source1_entity_id", "baseline_score", "candidate_entity_id"],
            ascending=[True, False, True],
            inplace=True,
        )
        scored_df["baseline_rank"] = (
            scored_df.groupby("source1_entity_id").cumcount() + 1
        ).astype(int)

        return scored_df
