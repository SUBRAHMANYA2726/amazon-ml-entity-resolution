"""Strategy E: TF-IDF Sparse Top-K Retrieval Blocking.

Implements sparse TF-IDF retrieval over normalized business names and addresses:
- Fits TfidfVectorizer on candidate corpus using sparse CSR representation.
- Computes query-candidate sparse dot products in memory-bounded batches.
- Retrieves top-K candidates from Source 2 and top-K candidates from Source 3.
- Enforces configurable minimum similarity thresholds and top-K limits.
- Avoids dense pairwise similarity matrices entirely.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)


class TfidfBlocker:
    """Strategy E: Sparse TF-IDF Top-K candidate retrieval."""

    def __init__(
        self,
        *,
        top_k_name: int = 10,
        top_k_address: int = 5,
        min_similarity: float = 0.15,
        batch_size: int = 1000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.85,
        sublinear_tf: bool = True,
    ) -> None:
        self.top_k_name = top_k_name
        self.top_k_address = top_k_address
        self.min_similarity = min_similarity
        self.batch_size = batch_size
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        # Vectorizers
        self.name_vectorizer: TfidfVectorizer | None = None
        self.addr_vectorizer: TfidfVectorizer | None = None

        # Candidate representations
        self.candidate_ids: list[str] = []
        self.candidate_sources: list[str] = []  # "S2", "S3", etc.
        self.name_matrix: csr_matrix | None = None
        self.addr_matrix: csr_matrix | None = None

        # Index partitioning by source prefix for fast source-specific top-K
        self.s2_indices: np.ndarray = np.array([], dtype=int)
        self.s3_indices: np.ndarray = np.array([], dtype=int)

    def fit(
        self,
        candidates_df: pd.DataFrame,
        id_col: str = "entity_id",
    ) -> TfidfBlocker:
        """Fit TF-IDF vectorizers and transform candidate texts into sparse CSR matrices."""
        if candidates_df.empty:
            return self

        name_col = None
        for col in ("business_name__name_core", "business_name__name", "business_name"):
            if col in candidates_df.columns:
                name_col = col
                break

        addr_col = None
        for col in ("business_address__address", "business_address"):
            if col in candidates_df.columns:
                addr_col = col
                break

        self.candidate_ids = candidates_df[id_col].astype(str).tolist()
        self.candidate_sources = [cid.split("-")[0] if "-" in cid else "unknown" for cid in self.candidate_ids]

        s2_idx = [i for i, s in enumerate(self.candidate_sources) if s == "S2"]
        s3_idx = [i for i, s in enumerate(self.candidate_sources) if s == "S3"]
        self.s2_indices = np.array(s2_idx, dtype=int)
        self.s3_indices = np.array(s3_idx, dtype=int)

        cand_names = candidates_df[name_col].fillna("").astype(str).tolist() if name_col else [""] * len(candidates_df)

        # Adapt min_df and max_df for small candidate pools
        n_cands = len(cand_names)
        eff_min_df = 1 if n_cands < 10 else self.min_df
        eff_max_df = 1.0 if n_cands < 10 else self.max_df

        LOGGER.info("Fitting TF-IDF Name Vectorizer on %d candidates...", n_cands)
        try:
            self.name_vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=self.ngram_range,
                min_df=eff_min_df,
                max_df=eff_max_df,
                sublinear_tf=self.sublinear_tf,
                token_pattern=r"(?u)\b\w+\b",
            )
            self.name_matrix = self.name_vectorizer.fit_transform(cand_names)
            LOGGER.info(
                "Name TF-IDF matrix shape: %s, non-zeros: %d",
                self.name_matrix.shape,
                self.name_matrix.nnz,
            )
        except ValueError as err:
            LOGGER.warning("Could not fit TF-IDF name vectorizer: %s", err)
            self.name_vectorizer = None
            self.name_matrix = None

        if addr_col and self.top_k_address > 0:
            cand_addrs = candidates_df[addr_col].fillna("").astype(str).tolist()
            LOGGER.info("Fitting TF-IDF Address Vectorizer on %d candidates...", len(cand_addrs))
            try:
                self.addr_vectorizer = TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=eff_min_df,
                    max_df=eff_max_df,
                    sublinear_tf=self.sublinear_tf,
                    token_pattern=r"(?u)\b\w+\b",
                )
                self.addr_matrix = self.addr_vectorizer.fit_transform(cand_addrs)
            except ValueError as err:
                LOGGER.warning("Could not fit TF-IDF address vectorizer: %s", err)
                self.addr_vectorizer = None
                self.addr_matrix = None

        return self

    def generate_pairs(
        self,
        anchors_df: pd.DataFrame,
        id_col: str = "entity_id",
    ) -> dict[str, dict[str, set[str]]]:
        """Retrieve top-K S2 and S3 candidates for each anchor in anchors_df.

        Returns:
            Mapping of anchor_id -> {candidate_id -> set of strategies (e.g. {'tfidf_name'})}
        """
        if self.name_vectorizer is None or self.name_matrix is None or anchors_df.empty:
            return {}

        name_col = None
        for col in ("business_name__name_core", "business_name__name", "business_name"):
            if col in anchors_df.columns:
                name_col = col
                break

        addr_col = None
        for col in ("business_address__address", "business_address"):
            if col in anchors_df.columns:
                addr_col = col
                break

        anchor_ids = anchors_df[id_col].astype(str).tolist()
        anchor_names = anchors_df[name_col].fillna("").astype(str).tolist() if name_col else [""] * len(anchors_df)
        anchor_addrs = anchors_df[addr_col].fillna("").astype(str).tolist() if addr_col else [""] * len(anchors_df)

        all_candidates: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        num_anchors = len(anchor_ids)

        # Process in memory-bounded batches
        for start_idx in range(0, num_anchors, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_anchors)
            batch_names = anchor_names[start_idx:end_idx]
            batch_ids = anchor_ids[start_idx:end_idx]

            # 1. Name TF-IDF similarity
            query_name_mat = self.name_vectorizer.transform(batch_names)
            # Dot product: (batch_size, num_features) x (num_candidates, num_features).T -> (batch_size, num_candidates)
            sim_matrix = query_name_mat.dot(self.name_matrix.T)

            for b_idx in range(len(batch_ids)):
                aid = batch_ids[b_idx]
                row = sim_matrix.getrow(b_idx)
                if row.nnz == 0:
                    continue

                col_indices = row.indices
                data_scores = row.data

                # Retrieve top-K S2 and top-K S3 separately
                self._extract_top_k(
                    aid=aid,
                    col_indices=col_indices,
                    data_scores=data_scores,
                    strategy_name="tfidf_name",
                    out_dict=all_candidates[aid],
                )

            # 2. Address TF-IDF similarity (if enabled)
            if self.addr_vectorizer is not None and self.addr_matrix is not None and self.top_k_address > 0:
                batch_addrs = anchor_addrs[start_idx:end_idx]
                # Filter out empty queries to save computation
                non_empty = [addr for addr in batch_addrs if addr.strip()]
                if non_empty:
                    query_addr_mat = self.addr_vectorizer.transform(batch_addrs)
                    addr_sim_matrix = query_addr_mat.dot(self.addr_matrix.T)

                    for b_idx in range(len(batch_ids)):
                        aid = batch_ids[b_idx]
                        if not batch_addrs[b_idx].strip():
                            continue
                        row = addr_sim_matrix.getrow(b_idx)
                        if row.nnz == 0:
                            continue

                        self._extract_top_k(
                            aid=aid,
                            col_indices=row.indices,
                            data_scores=row.data,
                            strategy_name="tfidf_addr",
                            out_dict=all_candidates[aid],
                            top_k_limit=self.top_k_address,
                        )

        LOGGER.info(
            "TfidfBlocker completed retrieval for %d anchors",
            num_anchors,
        )
        return dict(all_candidates)

    def _extract_top_k(
        self,
        aid: str,
        col_indices: np.ndarray,
        data_scores: np.ndarray,
        strategy_name: str,
        out_dict: dict[str, set[str]],
        top_k_limit: int | None = None,
    ) -> None:
        """Extract top-K candidates separately for S2 and S3."""
        k = top_k_limit or self.top_k_name

        # Filter by minimum similarity
        valid_mask = data_scores >= self.min_similarity
        if not np.any(valid_mask):
            return

        valid_cols = col_indices[valid_mask]
        valid_scores = data_scores[valid_mask]

        # Separate candidates by S2 and S3
        s2_cand_ids: list[tuple[float, str]] = []
        s3_cand_ids: list[tuple[float, str]] = []
        other_cand_ids: list[tuple[float, str]] = []

        for idx, score in zip(valid_cols, valid_scores):
            cid = self.candidate_ids[idx]
            if cid == aid:
                continue
            src = self.candidate_sources[idx]
            if src == "S2":
                s2_cand_ids.append((score, cid))
            elif src == "S3":
                s3_cand_ids.append((score, cid))
            else:
                other_cand_ids.append((score, cid))

        # Sort descending and take top-K for S2
        if s2_cand_ids:
            s2_cand_ids.sort(key=lambda x: x[0], reverse=True)
            for _, cid in s2_cand_ids[:k]:
                out_dict[cid].add(strategy_name)

        # Sort descending and take top-K for S3
        if s3_cand_ids:
            s3_cand_ids.sort(key=lambda x: x[0], reverse=True)
            for _, cid in s3_cand_ids[:k]:
                out_dict[cid].add(strategy_name)

        if other_cand_ids and not (s2_cand_ids or s3_cand_ids):
            other_cand_ids.sort(key=lambda x: x[0], reverse=True)
            for _, cid in other_cand_ids[:k]:
                out_dict[cid].add(strategy_name)
