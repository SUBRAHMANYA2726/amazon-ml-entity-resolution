"""Multi-Strategy Blocking Pipeline Coordinator.

Orchestrates:
1. Exact Blocking (Strategy A)
2. Name Token Inverted Index Blocking (Strategy B)
3. Address Token Blocking (Strategy C)
4. Character N-Gram Overlap Blocking (Strategy D)
5. Sparse TF-IDF Top-K Retrieval (Strategy E)
6. Country Partitioning (strict zero cross-country matching)
7. Candidate Union and Deduplication
8. Candidate Pair Validation and Provenance Tracking
9. Ground Truth Candidate Recall Evaluation
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from business_entity_resolution.blocking.base import Blocker
from business_entity_resolution.blocking.evaluation import (
    CandidateRecallEvaluator,
    CandidateRecallReport,
)
from business_entity_resolution.blocking.exact import ExactBlocker
from business_entity_resolution.blocking.ngram import CharNgramBlocker
from business_entity_resolution.blocking.tfidf import TfidfBlocker
from business_entity_resolution.blocking.token import AddressTokenBlocker, NameTokenBlocker
from business_entity_resolution.blocking.union import AnchorCandidates, CandidateUnion
from business_entity_resolution.contracts import CandidatePair
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BlockingConfig:
    """Configurable hyperparameters for multi-strategy candidate generation."""

    # Strategy toggles
    enable_exact: bool = True
    enable_name_token: bool = True
    enable_address_token: bool = True
    enable_char_ngram: bool = True
    enable_tfidf: bool = True
    partition_by_country: bool = True

    # Exact blocking settings
    exact_max_block_size: int = 5000

    # Name token settings
    min_token_length: int = 3
    max_token_frequency: int = 10000
    name_token_max_cands: int = 50

    # Address token settings
    address_max_block_size: int = 5000
    address_max_cands: int = 50

    # Char ngram settings
    char_ngram_size: int = 3
    char_ngram_min_length: int = 3
    char_ngram_top_k: int = 20
    char_ngram_min_overlap: float = 0.35

    # TF-IDF settings
    tfidf_top_k_name: int = 10
    tfidf_top_k_address: int = 5
    tfidf_min_similarity: float = 0.15
    tfidf_batch_size: int = 2000

    # Candidate cap per anchor
    max_candidates_per_anchor: int | None = 100


class MultiStrategyBlockingPipeline(Blocker):
    """End-to-end multi-strategy candidate generation pipeline."""

    def __init__(self, config: BlockingConfig | None = None) -> None:
        self.config = config or BlockingConfig()
        self.union_engine = CandidateUnion(
            max_candidates_per_anchor=self.config.max_candidates_per_anchor
        )

        # Blockers per country partition: country -> strategy_name -> blocker
        self._partition_blockers: dict[str, dict[str, Any]] = defaultdict(dict)
        self._fitted = False
        self._candidate_pool_df: pd.DataFrame | None = None

    def fit(self, candidates_df: pd.DataFrame, id_col: str = "entity_id") -> MultiStrategyBlockingPipeline:
        """Fit all active blocking strategies on candidate pool."""
        LOGGER.info("Fitting MultiStrategyBlockingPipeline on %d candidates...", len(candidates_df))
        self._candidate_pool_df = candidates_df

        country_col = None
        for col in ("country__country", "country"):
            if col in candidates_df.columns:
                country_col = col
                break

        if self.config.partition_by_country and country_col:
            countries = candidates_df[country_col].fillna("").astype(str).str.lower().unique()
            for cty in countries:
                if not cty:
                    continue
                sub_df = candidates_df[candidates_df[country_col].fillna("").astype(str).str.lower() == cty]
                LOGGER.info("Fitting partition for country: '%s' (%d candidates)", cty, len(sub_df))
                self._fit_partition(cty, sub_df, id_col)
        else:
            self._fit_partition("all", candidates_df, id_col)

        self._fitted = True
        return self

    def _fit_partition(self, partition_key: str, sub_df: pd.DataFrame, id_col: str) -> None:
        """Fit blockers for a single partition."""
        blockers: dict[str, Any] = {}

        if self.config.enable_exact:
            exact = ExactBlocker(max_block_size=self.config.exact_max_block_size)
            exact.fit(sub_df, id_col=id_col)
            blockers["exact"] = exact

        if self.config.enable_name_token:
            name_tok = NameTokenBlocker(
                min_token_length=self.config.min_token_length,
                max_token_frequency=self.config.max_token_frequency,
                max_candidates_per_query=self.config.name_token_max_cands,
            )
            name_tok.fit(sub_df, id_col=id_col)
            blockers["name_token"] = name_tok

        if self.config.enable_address_token:
            addr_tok = AddressTokenBlocker(
                max_block_size=self.config.address_max_block_size,
                max_candidates_per_query=self.config.address_max_cands,
            )
            addr_tok.fit(sub_df, id_col=id_col)
            blockers["address_token"] = addr_tok

        if self.config.enable_char_ngram:
            ngram = CharNgramBlocker(
                ngram_size=self.config.char_ngram_size,
                min_string_length=self.config.char_ngram_min_length,
                min_overlap_ratio=self.config.char_ngram_min_overlap,
                max_candidates_per_query=self.config.char_ngram_top_k,
            )
            ngram.fit(sub_df, id_col=id_col)
            blockers["char_ngram"] = ngram

        if self.config.enable_tfidf:
            tfidf = TfidfBlocker(
                top_k_name=self.config.tfidf_top_k_name,
                top_k_address=self.config.tfidf_top_k_address,
                min_similarity=self.config.tfidf_min_similarity,
                batch_size=self.config.tfidf_batch_size,
            )
            tfidf.fit(sub_df, id_col=id_col)
            blockers["tfidf"] = tfidf

        self._partition_blockers[partition_key] = blockers

    def block(
        self,
        anchors_df: pd.DataFrame,
        id_col: str = "entity_id",
    ) -> tuple[dict[str, AnchorCandidates], list[tuple[str, dict[str, dict[str, set[str]]]]]]:
        """Run candidate generation for all anchors.

        Returns:
            Tuple of (merged_anchor_candidates, strategy_candidate_maps).
        """
        if not self._fitted:
            raise RuntimeError("MultiStrategyBlockingPipeline must be fitted before calling block()")

        country_col = None
        for col in ("country__country", "country"):
            if col in anchors_df.columns:
                country_col = col
                break

        # Accumulate strategy maps: strategy_name -> anchor_id -> {cand_id -> set_of_strategies}
        combined_strategy_maps: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(lambda: defaultdict(dict))

        if self.config.partition_by_country and country_col:
            countries = anchors_df[country_col].fillna("").astype(str).str.lower().unique()
            for cty in countries:
                sub_anchors = anchors_df[anchors_df[country_col].fillna("").astype(str).str.lower() == cty]
                if sub_anchors.empty:
                    continue
                blockers = self._partition_blockers.get(cty)
                if not blockers:
                    # Fallback to all partition if specific country blocker not found
                    blockers = self._partition_blockers.get("all", {})

                for strat_name, blocker in blockers.items():
                    cands = blocker.generate_pairs(sub_anchors, id_col=id_col)
                    for aid, cand_map in cands.items():
                        combined_strategy_maps[strat_name][aid].update(cand_map)
        else:
            blockers = self._partition_blockers.get("all", {})
            for strat_name, blocker in blockers.items():
                cands = blocker.generate_pairs(anchors_df, id_col=id_col)
                for aid, cand_map in cands.items():
                    combined_strategy_maps[strat_name][aid].update(cand_map)

        strategy_list = list(combined_strategy_maps.items())
        merged = self.union_engine.merge_strategy_results(strategy_list)

        # Ensure all anchors are present in merged result even if 0 candidates were found
        for aid in anchors_df[id_col].astype(str):
            if aid not in merged:
                merged[aid] = AnchorCandidates(anchor_id=aid, candidates=())

        return merged, strategy_list

    def generate(self, anchors: object, candidates: object) -> Iterable[CandidatePair]:
        """Implement Blocker abstract base class contract."""
        if not isinstance(anchors, pd.DataFrame):
            anchors_df = pd.DataFrame(anchors)  # type: ignore[arg-type]
        else:
            anchors_df = anchors

        if not isinstance(candidates, pd.DataFrame):
            candidates_df = pd.DataFrame(candidates)  # type: ignore[arg-type]
        else:
            candidates_df = candidates

        self.fit(candidates_df)
        merged, _ = self.block(anchors_df)
        return CandidateUnion.to_candidate_pairs(merged)

    def run_with_evaluation(
        self,
        anchors_df: pd.DataFrame,
        candidates_df: pd.DataFrame,
        ground_truth: Mapping[str, set[str]] | pd.DataFrame,
        id_col: str = "entity_id",
    ) -> tuple[dict[str, AnchorCandidates], CandidateRecallReport]:
        """Fit, block, and run candidate recall evaluation against ground truth."""
        self.fit(candidates_df, id_col=id_col)
        merged, strategy_list = self.block(anchors_df, id_col=id_col)

        evaluator = CandidateRecallEvaluator(ground_truth)
        report = evaluator.evaluate(
            anchor_candidates=merged,
            strategy_candidate_maps=strategy_list,
            anchor_records_df=anchors_df,
            candidate_records_df=candidates_df,
        )
        return merged, report
