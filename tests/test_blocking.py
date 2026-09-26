"""Unit and integration tests for Phase 3 Multi-Strategy Blocking and Recall Evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from business_entity_resolution.blocking.evaluation import (
    CandidateRecallEvaluator,
    parse_ground_truth,
)
from business_entity_resolution.blocking.exact import ExactBlocker
from business_entity_resolution.blocking.ngram import CharNgramBlocker
from business_entity_resolution.blocking.pipeline import (
    BlockingConfig,
    MultiStrategyBlockingPipeline,
)
from business_entity_resolution.blocking.tfidf import TfidfBlocker
from business_entity_resolution.blocking.token import AddressTokenBlocker, NameTokenBlocker
from business_entity_resolution.blocking.union import CandidateUnion
from business_entity_resolution.contracts import CandidatePair


@pytest.fixture
def sample_candidates() -> pd.DataFrame:
    """Synthetic candidate DataFrame with known variations and noise."""
    return pd.DataFrame([
        {
            "entity_id": "S2-101",
            "business_name": "Novent Owl PLLC",
            "business_address": "100 Main St, Des Plaines, IL 60016",
            "country": "US",
        },
        {
            "entity_id": "S2-102",
            "business_name": "Novent [Owl]",
            "business_address": "100 Main Street, Des Plaines, IL 60016",
            "country": "US",
        },
        {
            "entity_id": "S3-201",
            "business_name": "Owl Novent Inc",
            "business_address": "100 Main Street, Des Plaines, IL 60016",
            "country": "US",
        },
        {
            "entity_id": "S3-202",
            "business_name": "Spicer-Star Environmental Services LLC",
            "business_address": "9308 Home Court, Des Plaines, IL 60016",
            "country": "US",
        },
        {
            "entity_id": "S2-103",
            "business_name": "Shakti Agro Limited",
            "business_address": "Beside Zudio, Kultapara, Sambalpur, 768001 Orissa",
            "country": "India",
        },
        {
            "entity_id": "S3-203",
            "business_name": "Shakthi Agro Private Limited",
            "business_address": "Near Zudio, Sambalpur, 768001 Odisha",
            "country": "India",
        },
    ])


@pytest.fixture
def sample_anchors() -> pd.DataFrame:
    """Synthetic anchor DataFrame with queries matching sample_candidates."""
    return pd.DataFrame([
        {
            "entity_id": "S1-1",
            "business_name": "Novent Owl PLLC",
            "business_address": "Des Plaines, IL 60016, 100 Main St",
            "country": "US",
        },
        {
            "entity_id": "S1-2",
            "business_name": "Spicer Star Environmental Seraices LLC",  # Typo: Seraices
            "business_address": "9308 Home Ct, Des Plaines, IL 60016",
            "country": "US",
        },
        {
            "entity_id": "S1-3",
            "business_name": "Shakti Agro Ltd",
            "business_address": "Kultapara, Sambalpur, 768001 Orissa",
            "country": "India",
        },
        {
            "entity_id": "S1-4",  # Singleton anchor (no true matches)
            "business_name": "Completely Unmatched Global Corp",
            "business_address": "555 Nowhere Rd, Seattle, WA 98101",
            "country": "US",
        },
    ])


@pytest.fixture
def sample_ground_truth() -> dict[str, set[str]]:
    """Synthetic ground truth matches."""
    return {
        "S1-1": {"S2-101", "S2-102", "S3-201"},
        "S1-2": {"S3-202"},
        "S1-3": {"S2-103", "S3-203"},
        "S1-4": set(),  # Singleton
    }


# 1. Exact Blocking Tests
def test_exact_blocking(sample_candidates: pd.DataFrame, sample_anchors: pd.DataFrame) -> None:
    blocker = ExactBlocker()
    blocker.fit(sample_candidates)

    pairs = blocker.generate_pairs(sample_anchors)
    # S1-1 exact match on "Novent Owl PLLC" should retrieve S2-101
    assert "S2-101" in pairs["S1-1"]
    assert "exact_name" in pairs["S1-1"]["S2-101"]

    # S1-1 word-order invariant match with S3-201 ("Owl Novent Inc")
    assert "S3-201" in pairs["S1-1"]
    assert "exact_name_sorted" in pairs["S1-1"]["S3-201"]


# 2. Name Token Blocking Tests
def test_name_token_blocking(sample_candidates: pd.DataFrame, sample_anchors: pd.DataFrame) -> None:
    blocker = NameTokenBlocker(min_token_length=3)
    blocker.fit(sample_candidates)

    pairs = blocker.generate_pairs(sample_anchors)
    # Token "novent" should retrieve S2-101, S2-102, S3-201
    s1_1_cands = set(pairs["S1-1"].keys())
    assert "S2-101" in s1_1_cands
    assert "S2-102" in s1_1_cands
    assert "S3-201" in s1_1_cands


# 3. Address Token Blocking Tests
def test_address_token_blocking(sample_candidates: pd.DataFrame, sample_anchors: pd.DataFrame) -> None:
    blocker = AddressTokenBlocker()
    blocker.fit(sample_candidates)

    pairs = blocker.generate_pairs(sample_anchors)
    # S1-2 has house "9308", postal "60016" -> matches S3-202
    assert "S3-202" in pairs["S1-2"]
    assert "address_token" in pairs["S1-2"]["S3-202"]


# 4. Character N-Gram Blocking Tests
def test_char_ngram_blocking(sample_candidates: pd.DataFrame, sample_anchors: pd.DataFrame) -> None:
    blocker = CharNgramBlocker(ngram_size=3, min_overlap_ratio=0.25)
    blocker.fit(sample_candidates)

    pairs = blocker.generate_pairs(sample_anchors)
    # S1-2 has typo "Seraices" vs candidate S3-202 "Services"
    # Char n-gram must retrieve S3-202 despite the typo!
    assert "S3-202" in pairs["S1-2"]
    assert "char_ngram" in pairs["S1-2"]["S3-202"]


# 5. TF-IDF Retrieval Tests
def test_tfidf_retrieval(sample_candidates: pd.DataFrame, sample_anchors: pd.DataFrame) -> None:
    blocker = TfidfBlocker(top_k_name=5, min_similarity=0.10)
    blocker.fit(sample_candidates)

    pairs = blocker.generate_pairs(sample_anchors)
    # S1-3 "Shakti Agro Ltd" should retrieve Indian candidates S2-103 and S3-203
    s1_3_cands = set(pairs["S1-3"].keys())
    assert "S2-103" in s1_3_cands
    assert "S3-203" in s1_3_cands


# 6 & 7. Candidate Union & Deduplication Tests
def test_candidate_union_and_deduplication() -> None:
    union = CandidateUnion()
    strategy_map_1 = ("exact", {"S1-1": {"S2-101": {"exact_name"}, "S2-102": {"exact_name"}}})
    strategy_map_2 = ("tfidf", {"S1-1": {"S2-101": {"tfidf_name"}, "S3-201": {"tfidf_name"}}})

    merged = union.merge_strategy_results([strategy_map_1, strategy_map_2])
    assert "S1-1" in merged
    acands = merged["S1-1"]

    # Candidate IDs must be unique
    cand_ids = acands.candidate_ids()
    assert len(cand_ids) == len(set(cand_ids))
    assert len(cand_ids) == 3

    # Provenance tracking: S2-101 was retrieved by BOTH exact and tfidf!
    s2_101 = [c for c in acands.candidates if c.candidate_id == "S2-101"][0]
    assert "exact_name" in s2_101.strategies
    assert "tfidf_name" in s2_101.strategies


# 8. Invalid Candidate Removal Tests
def test_invalid_candidate_removal() -> None:
    union = CandidateUnion()
    invalid_map = (
        "strategy",
        {
            "S1-1": {
                "S1-1": {"self_match"},        # Invalid: self match
                "S1-999": {"s1_match"},        # Invalid: S1 as candidate
                "INVALID-ID": {"bad_prefix"},  # Invalid: not S2 or S3
                "S2-101": {"valid"},           # Valid!
            }
        },
    )
    merged = union.merge_strategy_results([invalid_map])
    cands = merged["S1-1"].candidate_ids()
    assert cands == ["S2-101"]


# 9. Ground Truth Recall Calculation Tests
def test_ground_truth_recall_calculation(
    sample_candidates: pd.DataFrame,
    sample_anchors: pd.DataFrame,
    sample_ground_truth: dict[str, set[str]],
) -> None:
    pipeline = MultiStrategyBlockingPipeline()
    merged, report = pipeline.run_with_evaluation(
        anchors_df=sample_anchors,
        candidates_df=sample_candidates,
        ground_truth=sample_ground_truth,
    )

    # 100% of true pairs should be retrieved in our synthetic dataset
    assert report.total_true_pairs == 6
    assert report.true_pairs_retrieved == 6
    assert report.pair_recall == 1.0
    assert report.s2_recall == 1.0
    assert report.s3_recall == 1.0
    assert report.entity_coverage == 1.0
    assert report.total_singleton_anchors == 1
    assert report.pct_zero_candidates > 0  # S1-4 has 0 true matches and low candidate count


# 10. Empty and Missing Fields Safety Tests
def test_empty_missing_fields_handling() -> None:
    anchors = pd.DataFrame([
        {"entity_id": "S1-NULL", "business_name": None, "business_address": None, "country": None},
        {"entity_id": "S1-EMPTY", "business_name": "", "business_address": "", "country": ""},
    ])
    candidates = pd.DataFrame([
        {"entity_id": "S2-1", "business_name": None, "business_address": None, "country": None},
        {"entity_id": "S2-2", "business_name": "", "business_address": "", "country": ""},
    ])

    pipeline = MultiStrategyBlockingPipeline()
    # Must run without crashing, throwing KeyError, or creating massive Cartesian blocks
    merged, _ = pipeline.run_with_evaluation(
        anchors_df=anchors,
        candidates_df=candidates,
        ground_truth={"S1-NULL": set(), "S1-EMPTY": set()},
    )
    assert len(merged["S1-NULL"].candidate_ids()) == 0
    assert len(merged["S1-EMPTY"].candidate_ids()) == 0


# 11. Duplicate IDs Handling Tests
def test_duplicate_ids_handling() -> None:
    candidates = pd.DataFrame([
        {"entity_id": "S2-1", "business_name": "Duplicate Alpha", "business_address": "1 Main St", "country": "US"},
        {"entity_id": "S2-1", "business_name": "Duplicate Alpha", "business_address": "1 Main St", "country": "US"},
    ])
    anchors = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Duplicate Alpha", "business_address": "1 Main St", "country": "US"},
    ])

    pipeline = MultiStrategyBlockingPipeline()
    merged, _ = pipeline.run_with_evaluation(
        anchors_df=anchors,
        candidates_df=candidates,
        ground_truth={"S1-1": {"S2-1"}},
    )
    # The candidate ID must only be retrieved once (deduplicated)
    assert merged["S1-1"].candidate_ids() == ["S2-1"]


# 12. Provenance Tracking Tests
def test_provenance_tracking(sample_candidates: pd.DataFrame, sample_anchors: pd.DataFrame) -> None:
    pipeline = MultiStrategyBlockingPipeline()
    merged, _ = pipeline.run_with_evaluation(
        anchors_df=sample_anchors,
        candidates_df=sample_candidates,
        ground_truth={},
    )

    s1_1_candidates = merged["S1-1"].candidates
    s2_101 = [c for c in s1_1_candidates if c.candidate_id == "S2-101"][0]
    # S2-101 shares exact name, name tokens, character ngrams, and TF-IDF
    assert len(s2_101.strategies) >= 2
    assert isinstance(s2_101.strategies, frozenset)
    assert "exact_name" in s2_101.strategies or "exact" in s2_101.strategies
