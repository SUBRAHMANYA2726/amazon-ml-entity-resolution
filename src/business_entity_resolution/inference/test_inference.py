"""Test-inference support for Phase 11 (final test inference + submission).

Design constraints (8 GB RAM, 12 CPUs, Windows spawn):
- No test labels exist or are used anywhere.
- Blocking runs per country partition in its own process (11a) and exits,
  freeing the multi-GB blocker indexes before scoring starts.
- Scoring (11b) uses ONE read-only SharedRecordStore per country in
  multiprocessing.shared_memory; workers materialize per-pair record dicts
  with the exact same keys the training-time code expects, then call the
  UNMODIFIED DeterministicBaselineMatcher.rank_candidates,
  PairwiseFeatureGenerator.generate_features, and LightGBMMatcher.score paths.
  Results are therefore bitwise-equivalent to single-process execution for the
  same inputs (verified by tests/test_phase11_inference.py).
- Chunking is always by whole S1 anchors, so per-anchor ranks are exact.

Slim record schema (13 columns): the exact record fields consumed by
score_pair / generate_features, nothing more.
"""

from __future__ import annotations

import io
import json
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SLIM_RECORD_FIELDS: tuple[str, ...] = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
    "business_name__cleaned",
    "business_name__name_core",
    "business_name__name_sorted",
    "business_address__cleaned",
    "business_address__address_sorted",
    "business_address__address_postal",
    "business_address__address_house_number",
    "business_address__address_unit",
    "country__cleaned",
)

# Columns the Phase 3 blockers read (subset of the slim schema).
BLOCK_SLIM_FIELDS: tuple[str, ...] = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
    "business_name__cleaned",
    "business_name__name_core",
    "business_address__cleaned",
    "business_address__address",
    "country__cleaned",
    "country__country",
)

# Extra normalized columns produced by normalize_frame that blockers may fall
# back to; kept only if present (cheap insurance, still slim).
BLOCK_EXTRA_FALLBACKS: tuple[str, ...] = (
    "business_name__name",
)

THRESHOLD_DEFAULT = 0.922


def slim_frame(norm_df: pd.DataFrame, for_blocking: bool = False) -> pd.DataFrame:
    """Project a normalized frame to the slim inference schema.

    - Blocking mode (for_blocking=True): fillna("").astype(str), which is
      EXACT because every blocker does `.fillna("").astype(str)` internally
      before reading these columns.
    - Scoring mode (default): values preserved VERBATIM (None stays None, NaN
      stays NaN, str stays str). This is load-bearing: training-time records
      flow through `x or fallback` chains where NaN is truthy but None/""
      are falsy, so any normalization would silently change features.
      Missing columns materialize as None (equivalent to dict.get default).
    """
    import math

    fields = BLOCK_SLIM_FIELDS + BLOCK_EXTRA_FALLBACKS if for_blocking else SLIM_RECORD_FIELDS
    out = pd.DataFrame()
    for col in fields:
        if col in norm_df.columns:
            out[col] = norm_df[col] if not for_blocking else norm_df[col].fillna("").astype(str)
        else:
            out[col] = None if not for_blocking else ""
    return out


def is_nan_value(value: Any) -> bool:
    return isinstance(value, float) and value != value


class SharedRecordStore:
    """Read-only shared-memory record store for one country's S1 + candidates.

    Each field is stored as one bytes blob plus one int64 offsets array
    (len n+1) plus one int8 kinds array (len n):
      kind 0 = None, 1 = NaN (float), 2 = str (blob slice), 3 = pickle payload.
    Records are addressed by integer index; S1 rows come first, then
    candidates. Decoding reproduces the EXACT training-time objects so that
    `x or fallback` chains and similarity functions behave identically.
    """

    KIND_NONE = 0
    KIND_NAN = 1
    KIND_STR = 2
    KIND_PICKLE = 3

    def __init__(self) -> None:
        self.fields: list[str] = []
        self.n_s1: int = 0
        self.n_cand: int = 0
        self._segs: list[shared_memory.SharedMemory] = []
        self._blob_arrs: list[np.ndarray] = []
        self._off_arrs: list[np.ndarray] = []
        self._kind_arrs: list[np.ndarray] = []
        self._owned = False

    @property
    def n_total(self) -> int:
        return self.n_s1 + self.n_cand

    @staticmethod
    def _enc(values: Sequence[Any]) -> tuple[bytes, np.ndarray, np.ndarray]:
        import math
        import pickle

        parts: list[bytes] = []
        offs = np.zeros(len(values) + 1, dtype=np.int64)
        kinds = np.zeros(len(values), dtype=np.int8)
        pos = 0
        for i, v in enumerate(values):
            if v is None:
                kinds[i] = SharedRecordStore.KIND_NONE
            elif isinstance(v, float) and math.isnan(v):
                kinds[i] = SharedRecordStore.KIND_NAN
            elif isinstance(v, str):
                kinds[i] = SharedRecordStore.KIND_STR
                b = v.encode("utf-8")
                parts.append(b)
                pos += len(b)
            else:
                kinds[i] = SharedRecordStore.KIND_PICKLE
                b = pickle.dumps(v, protocol=4)
                parts.append(b)
                pos += len(b)
            offs[i + 1] = pos
        return b"".join(parts), offs, kinds

    def build(
        self,
        s1_df: pd.DataFrame,
        cand_df: pd.DataFrame,
        fields: Sequence[str] = SLIM_RECORD_FIELDS,
    ) -> dict[str, Any]:
        """Create shared segments from slim frames; return attach descriptor."""
        self.fields = list(fields)
        self.n_s1 = len(s1_df)
        self.n_cand = len(cand_df)
        descriptor: dict[str, Any] = {
            "fields": self.fields, "n_s1": self.n_s1, "n_cand": self.n_cand,
            "segments": [],
        }
        for c in self.fields:
            s1_vals = s1_df[c].tolist() if c in s1_df.columns else [None] * self.n_s1
            c_vals = cand_df[c].tolist() if c in cand_df.columns else [None] * self.n_cand
            blob, offs, kinds = self._enc(list(s1_vals) + list(c_vals))
            shm_b = shared_memory.SharedMemory(create=True, size=max(len(blob), 1))
            if blob:
                shm_b.buf[: len(blob)] = blob
            shm_o = shared_memory.SharedMemory(create=True, size=offs.nbytes)
            shm_o.buf[:] = offs.tobytes()
            shm_k = shared_memory.SharedMemory(create=True, size=kinds.nbytes)
            shm_k.buf[:] = kinds.tobytes()
            self._segs.extend([shm_b, shm_o, shm_k])
            self._blob_arrs.append(np.ndarray((len(blob),), dtype=np.uint8, buffer=shm_b.buf))
            self._off_arrs.append(np.ndarray((len(offs),), dtype=np.int64, buffer=shm_o.buf))
            self._kind_arrs.append(np.ndarray((len(kinds),), dtype=np.int8, buffer=shm_k.buf))
            descriptor["segments"].append({"blob": shm_b.name, "offs": shm_o.name,
                                           "kinds": shm_k.name, "blob_n": len(blob),
                                           "offs_n": len(offs), "kinds_n": len(kinds)})
        self._owned = True
        return descriptor

    def attach(self, descriptor: Mapping[str, Any]) -> None:
        """Attach to existing segments (worker side; read-only use)."""
        self.fields = list(descriptor["fields"])
        self.n_s1 = int(descriptor["n_s1"])
        self.n_cand = int(descriptor["n_cand"])
        for seg in descriptor["segments"]:
            shm_b = shared_memory.SharedMemory(name=seg["blob"])
            shm_o = shared_memory.SharedMemory(name=seg["offs"])
            shm_k = shared_memory.SharedMemory(name=seg["kinds"])
            self._segs.extend([shm_b, shm_o, shm_k])
            self._blob_arrs.append(np.ndarray((seg["blob_n"],), dtype=np.uint8, buffer=shm_b.buf))
            self._off_arrs.append(np.ndarray((seg["offs_n"],), dtype=np.int64, buffer=shm_o.buf))
            self._kind_arrs.append(np.ndarray((seg["kinds_n"],), dtype=np.int8, buffer=shm_k.buf))
        self._owned = False

    def close(self) -> None:
        for shm in self._segs:
            shm.close()

    def unlink(self) -> None:
        if self._owned:
            for shm in self._segs:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass

    def get_field(self, field_idx: int, idx: int) -> Any:
        import pickle

        kind = int(self._kind_arrs[field_idx][idx])
        if kind == self.KIND_NONE:
            return None
        if kind == self.KIND_NAN:
            return float("nan")
        offs = self._off_arrs[field_idx]
        blob = self._blob_arrs[field_idx]
        s, e = int(offs[idx]), int(offs[idx + 1])
        raw = bytes(blob[s:e])
        if kind == self.KIND_STR:
            return raw.decode("utf-8")
        return pickle.loads(raw)

    def get_record(self, idx: int) -> dict[str, Any]:
        return {f: self.get_field(j, idx) for j, f in enumerate(self.fields)}

    def get_records(self, idxs: np.ndarray) -> dict[int, dict[str, Any]]:
        uniq = np.unique(idxs)
        return {int(i): self.get_record(int(i)) for i in uniq}


# --------------------------------------------------------------------------
# Scoring worker (module-level for Windows spawn pickling)
# --------------------------------------------------------------------------

_WORKER_CTX: dict[str, Any] = {}


def score_worker_init(descriptor: dict[str, Any], model_path: str,
                      feature_order: list[str]) -> None:
    """Worker initializer: attach store + load model once per process."""
    from business_entity_resolution.features import PairwiseFeatureGenerator
    from business_entity_resolution.matching.baseline import DeterministicBaselineMatcher
    from business_entity_resolution.models.pairwise import LightGBMMatcher

    store = SharedRecordStore()
    store.attach(descriptor)
    matcher = LightGBMMatcher.load(model_path)
    _WORKER_CTX["store"] = store
    _WORKER_CTX["matcher"] = DeterministicBaselineMatcher()
    _WORKER_CTX["generator"] = PairwiseFeatureGenerator()
    _WORKER_CTX["model"] = matcher
    _WORKER_CTX["feature_order"] = list(feature_order)


def score_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Score one anchor-chunk; return integer-keyed probabilities.

    payload keys: s1_idx (int32), cand_idx (int32), cand_source (list[str]),
    strategies (list[str]), strategy_count (list[int]), s1_base (int, global
    offset added to s1_idx for candidate-index translation).
    """
    store: SharedRecordStore = _WORKER_CTX["store"]
    s1_idx = np.asarray(payload["s1_idx"], dtype=np.int64)
    cand_idx = np.asarray(payload["cand_idx"], dtype=np.int64)
    n = len(s1_idx)
    if n == 0:
        return {"s1_idx": s1_idx, "cand_idx": cand_idx,
                "proba": np.zeros(0, dtype=np.float32)}

    # Materialize unique records once per chunk.
    uniq_s1 = np.unique(s1_idx)
    uniq_cand = np.unique(cand_idx)
    s1_recs = {int(i): store.get_record(int(i)) for i in uniq_s1}
    cand_recs = {int(i): store.get_record(store.n_s1 + int(i)) for i in uniq_cand}

    cand_ids = [cand_recs[int(i)]["entity_id"] for i in cand_idx]
    s1_ids = [s1_recs[int(i)]["entity_id"] for i in s1_idx]

    pairs_df = pd.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": cand_ids,
        "candidate_source": list(payload["cand_source"]),
        "strategies": list(payload["strategies"]),
        "strategy_count": np.asarray(payload["strategy_count"], dtype=np.int64),
        "_wpos": np.arange(n, dtype=np.int64),
    })
    scored = _WORKER_CTX["matcher"].rank_candidates(
        pairs_df,
        {rec["entity_id"]: rec for rec in s1_recs.values()},
        {rec["entity_id"]: rec for rec in cand_recs.values()},
    )
    # NOTE: rank_candidates sorts by (source1_entity_id, baseline_score DESC,
    # candidate_entity_id ASC). The worker chunk always contains whole anchors,
    # so per-anchor ranks are identical to full-frame execution. _wpos restores
    # input order for the returned probabilities.
    feat_df, _ = _WORKER_CTX["generator"].generate_features(
        candidate_pairs_df=scored,
        s1_records={rec["entity_id"]: rec for rec in s1_recs.values()},
        cand_records={rec["entity_id"]: rec for rec in cand_recs.values()},
        baseline_scored_df=scored,
        ground_truth=None,
    )
    proba = _WORKER_CTX["model"].score(feat_df[_WORKER_CTX["feature_order"]])
    # Restore input order (float64: no precision loss vs direct path).
    order = np.asarray(scored["_wpos"].to_numpy(dtype=np.int64))
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    return {"s1_idx": np.asarray(s1_idx), "cand_idx": np.asarray(cand_idx),
            "proba": np.asarray(proba, dtype=np.float64)[inv]}


def apply_threshold_per_entity(
    s1_ids: list[str],
    cand_ids: list[str],
    proba: np.ndarray,
    threshold: float = THRESHOLD_DEFAULT,
) -> dict[str, list[str]]:
    """Threshold + intra-S1 dedup + deterministic ordering (proba DESC, id ASC)."""
    order = np.lexsort((np.array(cand_ids, dtype=object),
                        -np.asarray(proba, dtype=np.float64)))
    result: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for pos in order:
        s, c, p = s1_ids[int(pos)], cand_ids[int(pos)], float(proba[int(pos)])
        bucket = seen.setdefault(s, set())
        if p >= threshold and c not in bucket:
            bucket.add(c)
            result.setdefault(s, []).append(c)
    return result
