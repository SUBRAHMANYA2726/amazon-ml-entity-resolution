"""Pairwise feature generator for Business Entity Resolution.

Extracts deterministic, reproducible features across 9 distinct groups:
- Group A: Business Name similarities & length features
- Group B: Business Address similarities & length features
- Group C: Structured field comparisons (postal, house, unit)
- Group D: Missingness indicators
- Group E: Cross-field consistency & interaction features
- Group F: Generic open-set country features (zero hard-coding)
- Group G: Candidate blocking provenance features
- Group H: Candidate source indicators (S2 vs S3)
- Group I: Phase 4 Baseline matching signals
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from business_entity_resolution.contracts import CandidatePair
from business_entity_resolution.features.base import PairFeatureGenerator
from business_entity_resolution.features.metadata import FEATURE_NAMES, FEATURE_REGISTRY
from business_entity_resolution.matching.baseline import DeterministicBaselineMatcher
from business_entity_resolution.matching.similarity import (
    char_ngram_jaccard,
    common_prefix_ratio,
    common_suffix_ratio,
    compare_structured_field,
    exact_equality,
    extract_tokens,
    is_blank,
    jaro_winkler_similarity,
    levenshtein_similarity,
    token_jaccard,
    token_overlap,
)
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)

# Strategy flags recognized in Phase 3 candidate generation provenance
STRATEGY_COL_MAP: dict[str, str] = {
    "exact_name": "blocking_exact_name",
    "exact_name_sorted": "blocking_exact_name_sorted",
    "exact_country_name": "blocking_exact_country_name",
    "exact_country_postal_name": "blocking_exact_country_postal_name",
    "exact_country_postal_house": "blocking_exact_country_postal_house",
    "exact_country_house_token": "blocking_exact_country_house_token",
    "name_token": "blocking_name_token",
    "address_token": "blocking_address_token",
    "char_ngram": "blocking_char_ngram",
    "tfidf_name": "blocking_tfidf_name",
    "tfidf_addr": "blocking_tfidf_addr",
}


class PairwiseFeatureGenerator(PairFeatureGenerator):
    """Deterministic pairwise feature engineer."""

    def __init__(self, baseline_matcher: DeterministicBaselineMatcher | None = None) -> None:
        self.baseline_matcher = baseline_matcher or DeterministicBaselineMatcher()

    def generate(self, pairs: Iterable[CandidatePair], records: object) -> pd.DataFrame:
        """PairFeatureGenerator interface implementation."""
        records_map = records if isinstance(records, Mapping) else {}
        pairs_list = list(pairs)
        pairs_df = pd.DataFrame([
            {
                "source1_entity_id": p.anchor_ref,
                "candidate_entity_id": p.candidate_ref,
                "candidate_source": "S2" if p.candidate_ref.startswith("S2-") else "S3",
                "strategies": ";".join(sorted(p.provenance)),
                "strategy_count": len(p.provenance),
            }
            for p in pairs_list
        ])
        feat_df, _ = self.generate_features(
            candidate_pairs_df=pairs_df,
            s1_records=records_map,
            cand_records=records_map,
        )
        return feat_df

    def generate_features(
        self,
        candidate_pairs_df: pd.DataFrame,
        s1_records: Mapping[str, Mapping[str, Any]],
        cand_records: Mapping[str, Mapping[str, Any]],
        baseline_scored_df: pd.DataFrame | None = None,
        ground_truth: Mapping[str, set[str]] | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Compute complete pairwise feature matrix.
        
        Returns:
            (feature_matrix_df, list_of_feature_column_names)
        """
        n_pairs = len(candidate_pairs_df)
        LOGGER.info("Generating pairwise feature matrix for %d candidate pairs...", n_pairs)

        # Ensure baseline scores exist
        if baseline_scored_df is None or "baseline_score" not in baseline_scored_df.columns:
            scored_df = self.baseline_matcher.rank_candidates(
                candidate_pairs_df=candidate_pairs_df,
                s1_lookup=s1_records,
                cand_lookup=cand_records,
            )
        else:
            scored_df = baseline_scored_df

        # Pre-allocate feature columns dict with float32 arrays
        f_arrays: dict[str, np.ndarray] = {
            fname: np.zeros(n_pairs, dtype=np.float32) for fname in FEATURE_NAMES
        }

        s1_ids = scored_df["source1_entity_id"].values
        cand_ids = scored_df["candidate_entity_id"].values
        cand_sources = scored_df["candidate_source"].values

        # Candidate provenance strategies
        has_strategies = "strategies" in scored_df.columns
        strat_col = scored_df["strategies"].values if has_strategies else None
        strat_counts = scored_df["strategy_count"].values if "strategy_count" in scored_df.columns else None

        # Pre-computed baseline columns
        base_scores = scored_df["baseline_score"].values
        base_ranks = scored_df["baseline_rank"].values
        base_name_sims = scored_df["name_similarity_score"].values
        base_addr_sims = scored_df["address_similarity_score"].values
        base_struct_sims = scored_df["structured_similarity_score"].values

        empty_dict: dict[str, Any] = {}

        for i in range(n_pairs):
            aid = s1_ids[i]
            cid = cand_ids[i]
            c_src = cand_sources[i]

            s1 = s1_records.get(aid, empty_dict)
            cand = cand_records.get(cid, empty_dict)

            # Record fields
            s1_name = s1.get("business_name__cleaned") or s1.get("business_name")
            c_name = cand.get("business_name__cleaned") or cand.get("business_name")
            s1_name_core = s1.get("business_name__name_core")
            c_name_core = cand.get("business_name__name_core")
            s1_name_sorted = s1.get("business_name__name_sorted")
            c_name_sorted = cand.get("business_name__name_sorted")

            s1_addr = s1.get("business_address__cleaned") or s1.get("business_address")
            c_addr = cand.get("business_address__cleaned") or cand.get("business_address")
            s1_addr_sorted = s1.get("business_address__address_sorted")
            c_addr_sorted = cand.get("business_address__address_sorted")

            s1_country = s1.get("country__cleaned") or s1.get("country")
            c_country = cand.get("country__cleaned") or cand.get("country")

            s1_postal = s1.get("business_address__address_postal")
            c_postal = cand.get("business_address__address_postal")
            s1_house = s1.get("business_address__address_house_number")
            c_house = cand.get("business_address__address_house_number")
            s1_unit = s1.get("business_address__address_unit")
            c_unit = cand.get("business_address__address_unit")

            # Missingness checks
            m_s1_name = 1.0 if is_blank(s1_name) else 0.0
            m_c_name = 1.0 if is_blank(c_name) else 0.0
            m_s1_addr = 1.0 if is_blank(s1_addr) else 0.0
            m_c_addr = 1.0 if is_blank(c_addr) else 0.0
            m_s1_country = 1.0 if is_blank(s1_country) else 0.0
            m_c_country = 1.0 if is_blank(c_country) else 0.0
            m_s1_postal = 1.0 if is_blank(s1_postal) else 0.0
            m_c_postal = 1.0 if is_blank(c_postal) else 0.0

            # -------------------------------------------------------------
            # Group A: Name Features
            # -------------------------------------------------------------
            s1_name_len = float(len(str(s1_name))) if not m_s1_name else 0.0
            c_name_len = float(len(str(c_name))) if not m_c_name else 0.0
            max_name_len = max(s1_name_len, c_name_len)
            min_name_len = min(s1_name_len, c_name_len)
            name_len_ratio = (min_name_len / max_name_len) if max_name_len > 0 else 0.0

            s1_name_toks = extract_tokens(s1_name)
            c_name_toks = extract_tokens(c_name)
            s1_tok_cnt = float(len(s1_name_toks))
            c_tok_cnt = float(len(c_name_toks))

            n_exact = exact_equality(s1_name, c_name)
            n_core_exact = exact_equality(s1_name_core, c_name_core)
            n_sorted_exact = exact_equality(s1_name_sorted, c_name_sorted)
            n_tok_jaccard = token_jaccard(s1_name_toks, c_name_toks)
            n_tok_overlap = token_overlap(s1_name_toks, c_name_toks)
            n_char_3g = char_ngram_jaccard(s1_name, c_name, n=3)
            n_lev = levenshtein_similarity(s1_name, c_name)
            n_jw = jaro_winkler_similarity(s1_name, c_name)
            n_pfx = common_prefix_ratio(s1_name, c_name)
            n_sfx = common_suffix_ratio(s1_name, c_name)

            f_arrays["name_exact_match"][i] = n_exact
            f_arrays["name_core_exact_match"][i] = n_core_exact
            f_arrays["name_sorted_exact_match"][i] = n_sorted_exact
            f_arrays["name_length_s1"][i] = s1_name_len
            f_arrays["name_length_cand"][i] = c_name_len
            f_arrays["name_length_diff"][i] = abs(s1_name_len - c_name_len)
            f_arrays["name_length_ratio"][i] = name_len_ratio
            f_arrays["name_token_count_s1"][i] = s1_tok_cnt
            f_arrays["name_token_count_cand"][i] = c_tok_cnt
            f_arrays["name_token_count_diff"][i] = abs(s1_tok_cnt - c_tok_cnt)
            f_arrays["name_token_jaccard"][i] = n_tok_jaccard
            f_arrays["name_token_overlap"][i] = n_tok_overlap
            f_arrays["name_char_3gram_jaccard"][i] = n_char_3g
            f_arrays["name_levenshtein_sim"][i] = n_lev
            f_arrays["name_jaro_winkler"][i] = n_jw
            f_arrays["name_prefix_similarity"][i] = n_pfx
            f_arrays["name_suffix_similarity"][i] = n_sfx

            # -------------------------------------------------------------
            # Group B: Address Features
            # -------------------------------------------------------------
            s1_addr_len = float(len(str(s1_addr))) if not m_s1_addr else 0.0
            c_addr_len = float(len(str(c_addr))) if not m_c_addr else 0.0
            max_addr_len = max(s1_addr_len, c_addr_len)
            min_addr_len = min(s1_addr_len, c_addr_len)
            addr_len_ratio = (min_addr_len / max_addr_len) if max_addr_len > 0 else 0.0

            s1_addr_toks = extract_tokens(s1_addr)
            c_addr_toks = extract_tokens(c_addr)
            s1_a_tok_cnt = float(len(s1_addr_toks))
            c_a_tok_cnt = float(len(c_addr_toks))

            a_exact = exact_equality(s1_addr, c_addr)
            a_sorted_exact = exact_equality(s1_addr_sorted, c_addr_sorted)
            a_tok_jaccard = token_jaccard(s1_addr_toks, c_addr_toks)
            a_tok_overlap = token_overlap(s1_addr_toks, c_addr_toks)
            a_char_3g = char_ngram_jaccard(s1_addr, c_addr, n=3)
            a_lev = levenshtein_similarity(s1_addr, c_addr)

            f_arrays["address_exact_match"][i] = a_exact
            f_arrays["address_sorted_exact_match"][i] = a_sorted_exact
            f_arrays["address_length_s1"][i] = s1_addr_len
            f_arrays["address_length_cand"][i] = c_addr_len
            f_arrays["address_length_diff"][i] = abs(s1_addr_len - c_addr_len)
            f_arrays["address_length_ratio"][i] = addr_len_ratio
            f_arrays["address_token_count_s1"][i] = s1_a_tok_cnt
            f_arrays["address_token_count_cand"][i] = c_a_tok_cnt
            f_arrays["address_token_count_diff"][i] = abs(s1_a_tok_cnt - c_a_tok_cnt)
            f_arrays["address_token_jaccard"][i] = a_tok_jaccard
            f_arrays["address_token_overlap"][i] = a_tok_overlap
            f_arrays["address_char_3gram_jaccard"][i] = a_char_3g
            f_arrays["address_levenshtein_sim"][i] = a_lev

            # -------------------------------------------------------------
            # Group C: Structured Fields
            # -------------------------------------------------------------
            postal_res = compare_structured_field(s1_postal, c_postal)
            house_res = compare_structured_field(s1_house, c_house)
            unit_res = compare_structured_field(s1_unit, c_unit)

            f_arrays["postal_exact_match"][i] = postal_res.exact_match
            f_arrays["postal_both_present"][i] = postal_res.both_present
            f_arrays["postal_one_missing"][i] = postal_res.one_missing
            f_arrays["postal_both_missing"][i] = postal_res.both_missing

            f_arrays["house_number_exact_match"][i] = house_res.exact_match
            f_arrays["house_number_both_present"][i] = house_res.both_present
            f_arrays["house_number_one_missing"][i] = house_res.one_missing
            f_arrays["house_number_both_missing"][i] = house_res.both_missing

            f_arrays["unit_exact_match"][i] = unit_res.exact_match
            f_arrays["unit_both_present"][i] = unit_res.both_present

            # -------------------------------------------------------------
            # Group D: Field Missingness
            # -------------------------------------------------------------
            f_arrays["missing_s1_name"][i] = m_s1_name
            f_arrays["missing_cand_name"][i] = m_c_name
            f_arrays["missing_both_name"][i] = 1.0 if (m_s1_name and m_c_name) else 0.0
            f_arrays["missing_either_name"][i] = 1.0 if (m_s1_name or m_c_name) else 0.0

            f_arrays["missing_s1_address"][i] = m_s1_addr
            f_arrays["missing_cand_address"][i] = m_c_addr
            f_arrays["missing_both_address"][i] = 1.0 if (m_s1_addr and m_c_addr) else 0.0
            f_arrays["missing_either_address"][i] = 1.0 if (m_s1_addr or m_c_addr) else 0.0

            f_arrays["missing_s1_country"][i] = m_s1_country
            f_arrays["missing_cand_country"][i] = m_c_country
            f_arrays["missing_s1_postal"][i] = m_s1_postal
            f_arrays["missing_cand_postal"][i] = m_c_postal

            f_arrays["missing_field_count_total"][i] = (
                m_s1_name + m_c_name + m_s1_addr + m_c_addr + m_s1_country + m_c_country + m_s1_postal + m_c_postal
            )

            # -------------------------------------------------------------
            # Group E: Cross-Field Consistency
            # -------------------------------------------------------------
            country_res = compare_structured_field(s1_country, c_country)
            c_exact = country_res.exact_match
            c_conflict = 1.0 if (country_res.both_present == 1.0 and c_exact == 0.0) else 0.0

            f_arrays["cross_name_match_and_address_match"][i] = 1.0 if (n_exact == 1.0 and a_exact == 1.0) else 0.0
            f_arrays["cross_name_match_and_postal_match"][i] = 1.0 if (n_exact == 1.0 and postal_res.exact_match == 1.0) else 0.0
            f_arrays["cross_name_sim_x_country_match"][i] = n_tok_jaccard * c_exact
            f_arrays["cross_name_sim_x_address_sim"][i] = n_tok_jaccard * a_tok_jaccard
            f_arrays["cross_address_sim_x_postal_match"][i] = a_tok_jaccard * postal_res.exact_match
            f_arrays["cross_country_and_postal_match"][i] = 1.0 if (c_exact == 1.0 and postal_res.exact_match == 1.0) else 0.0

            # -------------------------------------------------------------
            # Group F: Country Features (Open-Set Generic)
            # -------------------------------------------------------------
            f_arrays["country_exact_match"][i] = c_exact
            f_arrays["country_conflict"][i] = c_conflict
            f_arrays["country_both_present"][i] = country_res.both_present
            f_arrays["country_missing"][i] = 1.0 if (m_s1_country or m_c_country) else 0.0

            # -------------------------------------------------------------
            # Group G: Candidate Provenance
            # -------------------------------------------------------------
            if strat_col is not None and not is_blank(strat_col[i]):
                strats = set(str(strat_col[i]).split(";"))
                for st_key, feat_name in STRATEGY_COL_MAP.items():
                    f_arrays[feat_name][i] = 1.0 if st_key in strats else 0.0
                scnt = float(strat_counts[i]) if strat_counts is not None else float(len(strats))
                f_arrays["blocking_strategy_count"][i] = scnt
                f_arrays["blocking_multi_strategy"][i] = 1.0 if scnt >= 2 else 0.0
            else:
                f_arrays["blocking_strategy_count"][i] = 1.0

            # -------------------------------------------------------------
            # Group H: Source Indicators
            # -------------------------------------------------------------
            f_arrays["source_cand_is_s2"][i] = 1.0 if c_src == "S2" or cid.startswith("S2-") else 0.0
            f_arrays["source_cand_is_s3"][i] = 1.0 if c_src == "S3" or cid.startswith("S3-") else 0.0
            f_arrays["source_s1_is_s1"][i] = 1.0

            # -------------------------------------------------------------
            # Group I: Baseline Signals
            # -------------------------------------------------------------
            f_arrays["baseline_score"][i] = float(base_scores[i])
            f_arrays["baseline_rank"][i] = float(base_ranks[i])
            f_arrays["baseline_name_sim"][i] = float(base_name_sims[i])
            f_arrays["baseline_address_sim"][i] = float(base_addr_sims[i])
            f_arrays["baseline_structured_sim"][i] = float(base_struct_sims[i])

        # Assemble full dataframe: Identifiers + Features (+ Optional Label)
        out_df = pd.DataFrame({
            "source1_entity_id": s1_ids,
            "candidate_entity_id": cand_ids,
            "candidate_source": cand_sources,
        })
        for fname in FEATURE_NAMES:
            out_df[fname] = f_arrays[fname]

        # Attach ground truth label ONLY if provided (never leak into FEATURE_NAMES)
        if ground_truth is not None:
            labels = [
                1 if cid in ground_truth.get(aid, set()) else 0
                for aid, cid in zip(s1_ids, cand_ids)
            ]
            out_df["ground_truth_label"] = np.array(labels, dtype=np.int32)
            LOGGER.info("Attached ground truth label: %d positive pairs (%.2f%%)", sum(labels), 100 * sum(labels) / n_pairs)

        LOGGER.info("Completed pairwise feature matrix generation: %d rows x %d features", len(out_df), len(FEATURE_NAMES))
        return out_df, FEATURE_NAMES
