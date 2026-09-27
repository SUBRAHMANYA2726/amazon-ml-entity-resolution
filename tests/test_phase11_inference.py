"""Phase 11 inference equivalence tests.

Proves that the shared-memory worker scoring path produces EXACTLY the same
probabilities as the direct single-process path (baseline ranking +
pairwise features + LightGBMMatcher.score) on real data, and that the
SharedRecordStore round-trips slim records losslessly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from business_entity_resolution.blocking import BlockingConfig, MultiStrategyBlockingPipeline
from business_entity_resolution.config.settings import load_settings
from business_entity_resolution.features import PairwiseFeatureGenerator
from business_entity_resolution.inference.test_inference import (
    SharedRecordStore,
    score_task,
    score_worker_init,
    slim_frame,
)
from business_entity_resolution.matching.baseline import DeterministicBaselineMatcher
from business_entity_resolution.models.pairwise import LightGBMMatcher
from business_entity_resolution.normalization.pipeline import NormalizationPipeline

TRAIN = Path("dataset/student_resource/dataset/train")
MODEL_PATH = "output/phase 8/model_b_lightgbm.joblib"


def _sample_norm():
    import json

    settings = load_settings("config.yaml")
    pipe = NormalizationPipeline(settings.normalization)
    s1raw = pd.read_csv(TRAIN / "train_source1.tsv", sep="\t", nrows=120, dtype=str)
    s2raw = pd.read_csv(TRAIN / "train_source2.tsv", sep="\t", nrows=4000, dtype=str)
    s3raw = pd.read_csv(TRAIN / "train_source3.tsv", sep="\t", nrows=4000, dtype=str)
    s1n = pipe.normalize_frame(s1raw)
    cands = pd.concat([pipe.normalize_frame(s2raw), pipe.normalize_frame(s3raw)],
                      ignore_index=True)
    meta = json.load(open("output/phase 8/model_b_metadata.json"))
    return s1n, cands, meta["feature_column_order"]


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float) and (a != a or b != b):
        return (a != a) and (b != b)
    return type(a) is type(b) and a == b


def test_slim_roundtrip_preserves_fields():
    s1n, cands, _ = _sample_norm()
    slim = slim_frame(cands)
    store = SharedRecordStore()
    desc = store.build(slim_frame(s1n).head(10), slim.head(50))
    try:
        for i in [0, 7, 49]:
            rec = store.get_record(store.n_s1 + i)
            row = slim.iloc[i]
            for col in slim.columns:
                assert _same(rec[col], row[col]), (col, i, rec[col], row[col])
    finally:
        store.close()
        store.unlink()


def test_worker_path_matches_direct_path():
    import json

    s1n, cands, feature_order = _sample_norm()
    cfg = BlockingConfig(partition_by_country=False)
    bl = MultiStrategyBlockingPipeline(cfg)
    bl.fit(cands)
    merged, _ = bl.block(s1n)
    assert len(merged) > 10
    # pairs dataframe exactly as Phase 5 consumes it
    from business_entity_resolution.blocking.union import CandidateUnion

    pairs_df = CandidateUnion.to_dataframe(merged)
    assert len(pairs_df) > 200

    s1_lookup = {r["entity_id"]: r.to_dict() for _, r in s1n.iterrows()}
    cand_lookup = {r["entity_id"]: r.to_dict() for _, r in cands.iterrows()}

    # Direct path
    matcher = DeterministicBaselineMatcher()
    gen = PairwiseFeatureGenerator(matcher)
    scored = matcher.rank_candidates(pairs_df, s1_lookup, cand_lookup)
    feat, _ = gen.generate_features(pairs_df, s1_lookup, cand_lookup,
                                    baseline_scored_df=scored, ground_truth=None)
    model = LightGBMMatcher.load(MODEL_PATH)
    direct_proba = model.score(feat[feature_order])

    # Worker path (in-process)
    s1_slim = slim_frame(s1n).reset_index(drop=True)
    cand_slim = slim_frame(cands).reset_index(drop=True)
    store = SharedRecordStore()
    desc = store.build(s1_slim, cand_slim)
    try:
        score_worker_init(desc, MODEL_PATH, feature_order)
        s1_id2idx = {s: i for i, s in enumerate(s1_slim["entity_id"].astype(str))}
        cand_id2idx = {s: i for i, s in enumerate(cand_slim["entity_id"].astype(str))}
        payload = {
            "s1_idx": np.array([s1_id2idx[s] for s in pairs_df["source1_entity_id"].astype(str)],
                               dtype=np.int64),
            "cand_idx": np.array([cand_id2idx[s] for s in pairs_df["candidate_entity_id"].astype(str)],
                                 dtype=np.int64),
            "cand_source": pairs_df["candidate_source"].astype(str).tolist(),
            "strategies": pairs_df["strategies"].astype(str).tolist(),
            "strategy_count": pairs_df["strategy_count"].astype(int).tolist(),
        }
        res = score_task(payload)
        worker_proba = np.asarray(res["proba"], dtype=np.float64)
    finally:
        store.close()
        store.unlink()

    assert len(worker_proba) == len(direct_proba)
    # Direct path returns rows in baseline-rank-sorted order (rank_candidates
    # sorts); the worker restores input order. Align by pair key for comparison.
    s1_list = pairs_df["source1_entity_id"].astype(str).tolist()
    c_list = pairs_df["candidate_entity_id"].astype(str).tolist()
    scored_order = np.asarray(scored["_wpos"].to_numpy()) if "_wpos" in scored.columns else None
    # scored has no _wpos here; rebuild: rank_candidates sorted by
    # (s1 ASC, score DESC, cand ASC). Re-derive permutation via mergesort on keys.
    keys = np.array([s + "\x00" + c for s, c in zip(s1_list, c_list)], dtype=object)
    sort_perm = np.argsort(keys, kind="stable")
    # direct_proba is in scored (sorted) order; worker_proba in input order.
    # Map: for input position i, find its rank position via key equality.
    scored_keys = np.array(
        [s + "\x00" + c for s, c in zip(scored["source1_entity_id"].astype(str),
                                        scored["candidate_entity_id"].astype(str))],
        dtype=object)
    rank_of_input = np.empty(len(keys), dtype=np.int64)
    pos_of_key = {k: j for j, k in enumerate(scored_keys)}
    for i, k in enumerate(keys):
        rank_of_input[i] = pos_of_key[k]
    np.testing.assert_array_equal(worker_proba, np.asarray(direct_proba)[rank_of_input])
