"""Strategy D: Character N-Gram Blocking.

Tolerates spelling errors, typos, punctuation changes, and transliteration differences.
Uses character n-grams derived from normalized business names.
Implements an inverted index or sparse retrieval mechanism over character n-grams
without computing full dense pairwise similarity.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import pandas as pd

from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)

ALPHANUM_ONLY_RE = re.compile(r"[a-z0-9]+")


def extract_char_ngrams(text: str | None, n: int = 3, min_length: int = 3) -> list[str]:
    """Generate padded character n-grams from cleaned text."""
    if not text or not isinstance(text, str):
        return []
    words = ALPHANUM_ONLY_RE.findall(text.lower())
    if not words:
        return []
    clean = " ".join(words)
    if len(clean) < min_length:
        return []
    # Pad string with leading/trailing markers to anchor boundaries
    padded = f"_{clean}_"
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


class CharNgramBlocker:
    """Strategy D: Character n-gram blocking for typo and spelling tolerance."""

    def __init__(
        self,
        *,
        ngram_size: int = 3,
        min_string_length: int = 3,
        min_overlap_ratio: float = 0.35,
        max_candidates_per_query: int = 25,
        max_posting_size: int = 5000,
    ) -> None:
        self.ngram_size = ngram_size
        self.min_string_length = min_string_length
        self.min_overlap_ratio = min_overlap_ratio
        self.max_candidates_per_query = max_candidates_per_query
        self.max_posting_size = max_posting_size

        # Inverted index: ngram -> list of candidate entity IDs
        self.index: dict[str, list[str]] = defaultdict(list)
        # Length cache: candidate_id -> total number of unique ngrams
        self.candidate_ngram_counts: dict[str, int] = {}
        self._pruned_ngrams: set[str] = set()

    def fit(self, candidates_df: pd.DataFrame, id_col: str = "entity_id") -> CharNgramBlocker:
        """Build character n-gram inverted index."""
        name_col = None
        for col in ("business_name__name_core", "business_name__name", "business_name"):
            if col in candidates_df.columns:
                name_col = col
                break

        names = candidates_df[name_col].fillna("").astype(str).tolist() if name_col else [""] * len(candidates_df)
        ids = candidates_df[id_col].astype(str).tolist()

        raw_index = defaultdict(list)
        for i in range(len(ids)):
            cand_id = ids[i]
            ngrams = set(extract_char_ngrams(names[i], n=self.ngram_size, min_length=self.min_string_length))
            self.candidate_ngram_counts[cand_id] = len(ngrams)
            for ng in ngrams:
                raw_index[ng].append(cand_id)

        for ng, postings in raw_index.items():
            if len(postings) > self.max_posting_size:
                self._pruned_ngrams.add(ng)
            else:
                self.index[ng] = postings

        LOGGER.info(
            "CharNgramBlocker indexed %d ngrams across %d candidates (%d pruned as too frequent)",
            len(self.index),
            len(self.candidate_ngram_counts),
            len(self._pruned_ngrams),
        )
        return self

    def block_query(self, anchor_id: str, name: str | None) -> dict[str, set[str]]:
        """Retrieve candidate IDs for an anchor using character n-gram overlap."""
        ngrams = set(extract_char_ngrams(name, n=self.ngram_size, min_length=self.min_string_length))
        if not ngrams:
            return {}

        query_len = len(ngrams)
        overlap_counts: Counter[str] = Counter()

        for ng in ngrams:
            if ng in self._pruned_ngrams:
                continue
            postings = self.index.get(ng, [])
            for cand_id in postings:
                if cand_id != anchor_id:
                    overlap_counts[cand_id] += 1

        results: dict[str, set[str]] = {}
        for cand_id, common_count in overlap_counts.most_common(self.max_candidates_per_query * 2):
            cand_len = self.candidate_ngram_counts.get(cand_id, query_len)
            # Jaccard overlap approximation on ngrams
            union_len = query_len + cand_len - common_count
            jaccard = common_count / union_len if union_len > 0 else 0.0

            if jaccard >= self.min_overlap_ratio or (common_count >= 5 and jaccard >= 0.25):
                results[cand_id] = {"char_ngram"}
                if len(results) >= self.max_candidates_per_query:
                    break

        return results

    def generate_pairs(
        self,
        anchors_df: pd.DataFrame,
        id_col: str = "entity_id",
    ) -> dict[str, dict[str, set[str]]]:
        """Generate candidates for all anchors in DataFrame."""
        name_col = None
        for col in ("business_name__name_core", "business_name__name", "business_name"):
            if col in anchors_df.columns:
                name_col = col
                break

        names = anchors_df[name_col].fillna("").astype(str).tolist() if name_col else [""] * len(anchors_df)
        ids = anchors_df[id_col].astype(str).tolist()

        all_cands: dict[str, dict[str, set[str]]] = {}
        for i in range(len(ids)):
            aid = ids[i]
            all_cands[aid] = self.block_query(aid, names[i])
        return all_cands
