"""Exact MaxScore-pruned TF-IDF top-K retrieval.

Problem: on million-scale candidate pools, the address TF-IDF query-candidate
dot product is up to ~63% dense (ultra-common bigrams such as French "de"/"la"
appear in most addresses), so the standard batched sparse dot materializes
tens of GB per anchor batch and per-anchor Python loops crawl.

This module provides:

- MaxScoreTfidfIndex: exact top-K-per-source retrieval over one
  source-restricted TF-IDF matrix via MaxScore pruning (same top-K,
  same min_similarity; exact-score ties broken by candidate column order).
- MaxScoreTfidfBlocker: wraps a FITTED TfidfBlocker andreplaces both name and
  address retrieval with the MaxScore engine.
- HybridTfidfBlocker (production path for Phase 11a): standard batched NAME
  retrieval copied line-for-line from TfidfBlocker.generate_pairs (bitwise
  identical, including tie order) + MaxScore ADDRESS retrieval (identical sets
  except exact-score ties at the K boundary, which follow candidate column
  order instead of scipy's internal row order -- recall-neutral: tied scores
  are interchangeable evidence and the reference order is itself an arbitrary
  but deterministic scipy artifact).

All engines reuse the fitted vectorizer/matrices/params and emit identical
('tfidf_name'/'tfidf_addr') provenance tokens.
"""

from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)


class MaxScoreTfidfIndex:
    """Exact top-K retrieval over one source-restricted TF-IDF matrix."""

    def __init__(self, cand_matrix: csr_matrix, top_k: int, min_similarity: float) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = int(top_k)
        self.min_similarity = float(min_similarity)
        self.csr = cand_matrix.tocsr(copy=False)
        self.csc = cand_matrix.tocsc(copy=False)
        self.n_docs, self.n_terms = self.csr.shape
        # Per-term max candidate weight M_t (max over docs of C[d,t]).
        m = self.csr.max(axis=0)
        self.term_max = np.asarray(m.todense()).ravel().astype(np.float64)
        self.indptr = self.csc.indptr
        self.indices = self.csc.indices
        self._seen = np.zeros(self.n_docs, dtype=np.int32)
        self._clock = 0

    def query_top_k(self, q_indices: np.ndarray, q_weights: np.ndarray) -> list[tuple[float, int]]:
        """Return [(score, doc_position)] top-K with score >= min_similarity.

        doc_position indexes rows of the source-restricted matrix.
        """
        m = len(q_indices)
        if m == 0 or self.n_docs == 0:
            return []
        potentials = q_weights * self.term_max[q_indices]
        order = np.argsort(-potentials, kind="stable")
        terms = q_indices[order]
        qw = q_weights[order]
        pots = potentials[order]
        suffix = np.cumsum(pots[::-1])[::-1]

        self._clock += 1
        clock = self._clock
        heap: list[tuple[float, int]] = []  # (score, -col) min-heap of size <= K
        theta = self.min_similarity
        K = self.top_k

        for k in range(m):
            if suffix[k] < theta:
                break
            t = int(terms[k])
            start, end = int(self.indptr[t]), int(self.indptr[t + 1])
            if end <= start:
                continue
            post_docs = self.indices[start:end]
            unseen_mask = self._seen[post_docs] != clock
            if not np.any(unseen_mask):
                continue
            todo = post_docs[unseen_mask]
            self._seen[todo] = clock
            # Exact scores for unseen docs: restricted sparse dot.
            sub = self.csr[todo]
            q_row = csr_matrix((qw, (np.zeros(m, dtype=np.int32), terms)),
                               shape=(1, self.n_terms))
            scores = np.asarray((sub.dot(q_row.T)).todense()).ravel()
            for doc, s in zip(todo.tolist(), scores.tolist()):
                if s >= theta:
                    entry = (float(s), -int(doc))
                    if len(heap) < K:
                        heapq.heappush(heap, entry)
                        if len(heap) == K:
                            theta = max(theta, heap[0][0])
                    elif entry > heap[0]:
                        heapq.heapreplace(heap, entry)
                        theta = max(theta, heap[0][0])
        return [(s, -neg) for (s, neg) in heap]


class MaxScoreTfidfBlocker:
    """MaxScore replacement for both name and address TF-IDF retrieval.

    Wraps a FITTED TfidfBlocker (reuses vectorizer, matrices, candidate
    ids/sources, top_k, min_similarity).
    """

    def __init__(self, fitted: object) -> None:
        self._f = fitted
        sources = np.array(fitted.candidate_sources)
        self._s2_pos = np.flatnonzero(sources == "S2")
        self._s3_pos = np.flatnonzero(sources == "S3")
        self._name_index: dict[str, MaxScoreTfidfIndex] = {}
        self._addr_index: dict[str, MaxScoreTfidfIndex] = {}
        if fitted.name_matrix is not None:
            mat = fitted.name_matrix.tocsr(copy=False)
            if len(self._s2_pos):
                self._name_index["S2"] = MaxScoreTfidfIndex(mat[self._s2_pos], fitted.top_k_name, fitted.min_similarity)
            if len(self._s3_pos):
                self._name_index["S3"] = MaxScoreTfidfIndex(mat[self._s3_pos], fitted.top_k_name, fitted.min_similarity)
        if getattr(fitted, "addr_matrix", None) is not None and fitted.top_k_address > 0:
            mat = fitted.addr_matrix.tocsr(copy=False)
            if len(self._s2_pos):
                self._addr_index["S2"] = MaxScoreTfidfIndex(mat[self._s2_pos], fitted.top_k_address, fitted.min_similarity)
            if len(self._s3_pos):
                self._addr_index["S3"] = MaxScoreTfidfIndex(mat[self._s3_pos], fitted.top_k_address, fitted.min_similarity)

    def _query_index(self, index: MaxScoreTfidfIndex, q_row) -> list[tuple[float, int]]:
        coo = q_row.tocoo()
        return index.query_top_k(coo.col.astype(np.int64), coo.data.astype(np.float64))

    def generate_pairs(self, anchors_df, id_col: str = "entity_id", only: str | None = None):
        fitted = self._f
        if fitted.name_vectorizer is None or fitted.name_matrix is None or anchors_df.empty:
            return {}
        do_name = only in (None, "name")
        do_addr = only in (None, "addr")
        name_col = next((c for c in ("business_name__name_core", "business_name__name", "business_name") if c in anchors_df.columns), None)
        addr_col = next((c for c in ("business_address__address", "business_address") if c in anchors_df.columns), None)

        anchor_ids = anchors_df[id_col].astype(str).tolist()
        anchor_names = anchors_df[name_col].fillna("").astype(str).tolist() if name_col else [""] * len(anchors_df)
        anchor_addrs = anchors_df[addr_col].fillna("").astype(str).tolist() if addr_col else [""] * len(anchors_df)

        out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        batch = int(getattr(fitted, "batch_size", 2000))
        for start in range(0, len(anchor_ids), batch):
            end = min(start + batch, len(anchor_ids))
            qn = fitted.name_vectorizer.transform(anchor_names[start:end]) if do_name else None
            qa = fitted.addr_vectorizer.transform(anchor_addrs[start:end]) if (do_addr and fitted.addr_vectorizer is not None) else None
            for b in range(end - start):
                aid = anchor_ids[start + b]
                if aid in (None, ""):
                    continue
                if do_name and qn.indptr[b] != qn.indptr[b + 1]:
                    for src, idx in (("S2", self._name_index.get("S2")), ("S3", self._name_index.get("S3"))):
                        if idx is None:
                            continue
                        hits = self._query_index(idx, qn.getrow(b))
                        pos = self._s2_pos if src == "S2" else self._s3_pos
                        for _, d in hits:
                            cid = fitted.candidate_ids[int(pos[d])]
                            if cid == aid:
                                continue
                            out[aid][cid].add("tfidf_name")
                if do_addr and qa is not None and anchor_addrs[start + b].strip() and qa.indptr[b] != qa.indptr[b + 1]:
                    for src, idx in (("S2", self._addr_index.get("S2")), ("S3", self._addr_index.get("S3"))):
                        if idx is None:
                            continue
                        hits = self._query_index(idx, qa.getrow(b))
                        pos = self._s2_pos if src == "S2" else self._s3_pos
                        for _, d in hits:
                            cid = fitted.candidate_ids[int(pos[d])]
                            if cid == aid:
                                continue
                            out[aid][cid].add("tfidf_addr")
        return dict(out)


class HybridTfidfBlocker:
    """Production hybrid: standard batched NAME retrieval + MaxScore ADDRESS retrieval.

    The NAME section copies TfidfBlocker.generate_pairs exactly (same
    vectorizer/matrix, same batching, same _extract_top_k tie order), so name
    results are bitwise identical to the validated implementation. The ADDRESS
    section delegates to MaxScoreTfidfBlocker (identical top-K per source above
    min_similarity; exact-score ties follow candidate column order).
    """

    def __init__(self, fitted: object, name_batch_size: int = 500) -> None:
        self._f = fitted
        self._name_batch = int(name_batch_size)
        self._addr = MaxScoreTfidfBlocker(fitted)

    def generate_pairs(self, anchors_df, id_col: str = "entity_id"):
        fitted = self._f
        if fitted.name_vectorizer is None or fitted.name_matrix is None or anchors_df.empty:
            return {}
        name_col = next((c for c in ("business_name__name_core", "business_name__name", "business_name") if c in anchors_df.columns), None)

        anchor_ids = anchors_df[id_col].astype(str).tolist()
        anchor_names = anchors_df[name_col].fillna("").astype(str).tolist() if name_col else [""] * len(anchors_df)

        out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        batch = self._name_batch
        for start in range(0, len(anchor_ids), batch):
            end = min(start + batch, len(anchor_ids))
            bench_names = anchor_names[start:end]
            bench_ids = anchor_ids[start:end]
            query_name_mat = fitted.name_vectorizer.transform(bench_names)
            sim_matrix = query_name_mat.dot(fitted.name_matrix.T)
            for b_idx in range(len(bench_ids)):
                aid = bench_ids[b_idx]
                row = sim_matrix.getrow(b_idx)
                if row.nnz == 0:
                    continue
                fitted._extract_top_k(
                    aid=aid,
                    col_indices=row.indices,
                    data_scores=row.data,
                    strategy_name="tfidf_name",
                    out_dict=out[aid],
                )
        # Address section via the exact MaxScore engine (addr tokens only).
        addr_hits = self._addr.generate_pairs(anchors_df, id_col=id_col, only="addr")
        for aid, cmap in addr_hits.items():
            for cid, strats in cmap.items():
                if "tfidf_addr" in strats:
                    out[aid][cid].add("tfidf_addr")
        return dict(out)
