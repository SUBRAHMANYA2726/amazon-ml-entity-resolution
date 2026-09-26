"""Unit tests for Phase 4 Baseline Matching and Evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from business_entity_resolution.matching.baseline import (
    BaselineScoreWeights,
    DeterministicBaselineMatcher,
)
from business_entity_resolution.matching.evaluation import (
    BaselineEvaluator,
    compute_f_beta,
)
from business_entity_resolution.matching.similarity import (
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


class TestSimilarityFunctions:
    """Test core string, token, and structured similarity functions."""

    def test_exact_equality(self) -> None:
        assert exact_equality("Acme Corp", "acme corp") == 1.0
        assert exact_equality("Acme Corp", "Acme Inc") == 0.0
        # Missing values must NEVER match
        assert exact_equality("", "") == 0.0
        assert exact_equality(None, None) == 0.0
        assert exact_equality("Acme", None) == 0.0
        assert exact_equality("nan", "nan") == 0.0

    def test_token_jaccard_and_overlap(self) -> None:
        toks1 = ["acme", "global", "services"]
        toks2 = ["services", "acme", "corp"]
        # Intersection = {acme, services} (2), Union = {acme, global, services, corp} (4)
        assert token_jaccard(toks1, toks2) == pytest.approx(0.5)
        # Min size = 3, Overlap = 2/3
        assert token_overlap(toks1, toks2) == pytest.approx(2.0 / 3.0)

        # Empty handling
        assert token_jaccard([], ["acme"]) == 0.0
        assert token_overlap([], []) == 0.0

    def test_levenshtein_similarity(self) -> None:
        assert levenshtein_similarity("kitten", "sitting") == pytest.approx(1.0 - 3.0 / 7.0)
        assert levenshtein_similarity("exact", "exact") == 1.0
        assert levenshtein_similarity("", "test") == 0.0
        assert levenshtein_similarity(None, "test") == 0.0

    def test_jaro_winkler_similarity(self) -> None:
        jw = jaro_winkler_similarity("martha", "marhta")
        assert 0.90 <= jw <= 1.0
        assert jaro_winkler_similarity("same", "same") == 1.0
        assert jaro_winkler_similarity(None, "test") == 0.0

    def test_char_ngram_jaccard(self) -> None:
        sim = char_ngram_jaccard("Novent Owl", "Novent Owl PLLC")
        assert 0.5 <= sim <= 1.0
        assert char_ngram_jaccard("", "test") == 0.0

    def test_prefix_and_suffix_similarity(self) -> None:
        assert common_prefix_ratio("Supermarket", "Superstore") == pytest.approx(5.0 / 11.0)
        assert common_suffix_ratio("Green Light", "Red Light") == pytest.approx(6.0 / 11.0)

    def test_compare_structured_field(self) -> None:
        # Both present and equal
        res1 = compare_structured_field("60016", "60016")
        assert res1.exact_match == 1.0
        assert res1.both_present == 1.0
        assert res1.both_missing == 0.0

        # Both present and unequal
        res2 = compare_structured_field("60016", "90210")
        assert res2.exact_match == 0.0
        assert res2.both_present == 1.0

        # One missing
        res3 = compare_structured_field("60016", None)
        assert res3.exact_match == 0.0
        assert res3.one_missing == 1.0
        assert res3.both_present == 0.0

        # Both missing - MUST NOT match!
        res4 = compare_structured_field(None, "")
        assert res4.exact_match == 0.0
        assert res4.both_missing == 1.0
        assert res4.both_present == 0.0


class TestDeterministicBaselineMatcher:
    """Test deterministic scoring, missing value handling, and candidate ranking."""

    def test_exact_match_score(self) -> None:
        matcher = DeterministicBaselineMatcher()
        s1 = {
            "business_name": "Novent Owl",
            "business_name__cleaned": "novent owl",
            "business_name__name_core": "novent owl",
            "business_name__name_sorted": "novent owl",
            "business_address": "100 Main St, Des Plaines, IL 60016",
            "business_address__cleaned": "100 main st des plaines il 60016",
            "business_address__address_sorted": "100 60016 des il main plaines st",
            "business_address__address_postal": "60016",
            "business_address__address_house_number": "100",
            "country": "US",
            "country__cleaned": "us",
        }
        cand = dict(s1)
        res = matcher.score_pair(s1, cand)
        assert res.baseline_score >= 0.99
        assert res.name_similarity >= 0.99
        assert res.address_similarity >= 0.99
        assert res.structured_similarity >= 0.99

    def test_country_conflict_zeros_score(self) -> None:
        matcher = DeterministicBaselineMatcher()
        s1 = {"business_name": "Acme Corp", "country": "US", "country__cleaned": "us"}
        cand = {"business_name": "Acme Corp", "country": "India", "country__cleaned": "india"}
        res = matcher.score_pair(s1, cand)
        # Physical impossibility: same business in US and India cannot match
        assert res.baseline_score == 0.0

    def test_missing_address_redistributes_weight(self) -> None:
        matcher = DeterministicBaselineMatcher()
        s1 = {
            "business_name": "Novent Owl",
            "business_name__cleaned": "novent owl",
            "business_address": "100 Main St, IL 60016",
            "country": "US",
            "country__cleaned": "us",
        }
        cand_no_addr = {
            "business_name": "Novent Owl",
            "business_name__cleaned": "novent owl",
            "business_address": None,
            "country": "US",
            "country__cleaned": "us",
        }
        res = matcher.score_pair(s1, cand_no_addr)
        # Should not drop to zero; name similarity carries the score
        assert res.baseline_score > 0.60
        assert res.address_similarity == 0.0

    def test_candidate_ranking(self) -> None:
        matcher = DeterministicBaselineMatcher()
        s1_lookup = {
            "S1-1": {
                "business_name": "Novent Owl",
                "business_name__cleaned": "novent owl",
                "business_address": "100 Main St, IL 60016",
                "country": "US",
                "country__cleaned": "us",
            }
        }
        cand_lookup = {
            "S2-1": {
                "business_name": "Novent Owl",
                "business_name__cleaned": "novent owl",
                "business_address": "100 Main St, IL 60016",
                "country": "US",
                "country__cleaned": "us",
            },
            "S2-2": {
                "business_name": "Unrelated Bakery",
                "business_name__cleaned": "unrelated bakery",
                "business_address": "500 Oak St, IL 60016",
                "country": "US",
                "country__cleaned": "us",
            },
        }
        pairs_df = pd.DataFrame([
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-2", "candidate_source": "S2"},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "candidate_source": "S2"},
        ])
        ranked = matcher.rank_candidates(pairs_df, s1_lookup, cand_lookup)
        assert len(ranked) == 2
        # S2-1 should be rank 1
        top = ranked.iloc[0]
        assert top["candidate_entity_id"] == "S2-1"
        assert top["baseline_rank"] == 1
        assert top["baseline_score"] > 0.90


class TestBaselineEvaluation:
    """Test precision, recall, Top-K recall, and F0.5 computation."""

    def test_compute_f_beta(self) -> None:
        # F0.5 weights precision more heavily than recall
        # At P=0.8, R=0.4: F0.5 = (1.25 * 0.32) / (0.25 * 0.8 + 0.4) = 0.4 / 0.6 = 0.6667
        f05 = compute_f_beta(0.8, 0.4, beta=0.5)
        assert f05 == pytest.approx(0.666666, rel=1e-3)

    def test_baseline_evaluator_top_k(self) -> None:
        gt = {"S1-1": {"S2-1"}, "S1-2": {"S3-1"}}
        ranked_df = pd.DataFrame([
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "candidate_source": "S2", "baseline_score": 0.95, "baseline_rank": 1, "name_similarity_score": 0.95, "address_similarity_score": 0.95, "structured_similarity_score": 1.0},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-2", "candidate_source": "S2", "baseline_score": 0.40, "baseline_rank": 2, "name_similarity_score": 0.4, "address_similarity_score": 0.4, "structured_similarity_score": 0.0},
            {"source1_entity_id": "S1-2", "candidate_entity_id": "S3-2", "candidate_source": "S3", "baseline_score": 0.80, "baseline_rank": 1, "name_similarity_score": 0.8, "address_similarity_score": 0.8, "structured_similarity_score": 1.0},
            {"source1_entity_id": "S1-2", "candidate_entity_id": "S3-1", "candidate_source": "S3", "baseline_score": 0.70, "baseline_rank": 2, "name_similarity_score": 0.7, "address_similarity_score": 0.7, "structured_similarity_score": 1.0},
        ])
        evaluator = BaselineEvaluator(gt)
        report = evaluator.evaluate(ranked_df)

        assert report.total_candidate_pairs == 4
        assert report.total_true_pairs_gt == 2
        assert report.true_pairs_in_candidate_pool == 2
        # Top-1 recall: S1-1 has S2-1 at rank 1, S1-2 has S3-1 at rank 2
        # So 1 out of 2 true pairs found at rank 1
        assert report.overall_top_k.top_1_recall == pytest.approx(0.5)
        # Top-3 recall: both found at rank <= 3
        assert report.overall_top_k.top_3_recall == pytest.approx(1.0)
