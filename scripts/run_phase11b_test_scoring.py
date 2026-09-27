"""Phase 11b — Test pair scoring (per country).

Loads Phase 11a pair checkpoints, normalizes raw test records (exact
None/NaN/str preservation), builds one read-only SharedRecordStore, and
scores all pairs with the Phase 7 Model B artifact in a process pool.
Outputs per-chunk probabilities. No labels of any kind are used.

Usage:
    python scripts/run_phase11b_test_scoring.py --country india [--workers 8]
    python scripts/run_phase11b_test_scoring.py --all [--workers 8]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from business_entity_resolution.config.settings import load_settings
from business_entity_resolution.inference.test_inference import (
    SLIM_RECORD_FIELDS,
    SharedRecordStore,
    score_task,
    score_worker_init,
    slim_frame,
)
from business_entity_resolution.normalization.pipeline import NormalizationPipeline
from business_entity_resolution.utils.logging import configure_logging, get_logger
from business_entity_resolution.utils.seed import set_random_seed

LOGGER = get_logger("phase11b_runner")

TEST_DIR = Path("dataset/student_resource/dataset/test")
WORK_DIR = Path("output/phase11/work")
MODEL_PATH = "output/phase 8/model_b_lightgbm.joblib"
MODEL_META = "output/phase 8/model_b_metadata.json"
COUNTRIES = ("india", "us", "france")
READ_KWARGS = {"sep": "\t", "dtype": str, "na_filter": False}
CACHE_READ_KWARGS = {"sep": ",", "dtype": str, "na_filter": False}
NORM_CHUNKSIZE = 200_000
TASK_ANCHORS = 5_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 11b test pair scoring.")
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--country", type=str, default=None, choices=list(COUNTRIES))
    p.add_argument("--all", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--task-anchors", type=int, default=TASK_ANCHORS)
    return p.parse_args()


def run_country(country: str, workers: int, task_anchors: int) -> dict:
    t_start = time.time()
    pairs_dir = WORK_DIR / f"pairs_{country}"
    chunk_files = sorted(pairs_dir.glob("chunk_*.csv.gz"))
    if not chunk_files:
        raise FileNotFoundError(f"No pair checkpoints for {country}; run Phase 11a first.")
    roster = pd.read_csv(WORK_DIR / f"anchors_{country}.csv.gz", **CACHE_READ_KWARGS)["entity_id"].astype(str)
    s1_ids_all = roster.tolist()
    LOGGER.info("[%s] anchors=%d pair chunks=%d", country, len(s1_ids_all), len(chunk_files))

    # Needed candidate IDs across all pair checkpoints.
    LOGGER.info("[%s] scanning pair checkpoints for needed IDs...", country)
    need_cands: set[str] = set()
    pairs_per_chunk: list[int] = []
    anchors_with_pairs: set[str] = set()
    for cf in chunk_files:
        df = pd.read_csv(cf, usecols=["source1_entity_id", "candidate_entity_id"], **CACHE_READ_KWARGS)
        need_cands.update(df["candidate_entity_id"].astype(str).unique().tolist())
        anchors_with_pairs.update(df["source1_entity_id"].astype(str).unique().tolist())
        pairs_per_chunk.append(len(df))
        del df
    total_pairs = sum(pairs_per_chunk)
    LOGGER.info("[%s] total pairs=%d unique cands=%d anchors w/ pairs=%d",
                country, total_pairs, len(need_cands), len(anchors_with_pairs))

    # Normalize raw test records, keeping only needed IDs (exact objects).
    from business_entity_resolution.config.settings import load_settings as _ls

    settings = _ls("config.yaml")
    pipe = NormalizationPipeline(settings.normalization)
    need_s1 = set(s1_ids_all) & anchors_with_pairs
    s1_parts, c2_parts, c3_parts = [], [], []
    for kind, path, keep in (("s1", TEST_DIR / "test_source1.tsv", need_s1),
                             ("s2", TEST_DIR / "test_source2.tsv", need_cands),
                             ("s3", TEST_DIR / "test_source3.tsv", need_cands)):
        LOGGER.info("[%s] normalizing %s (keeping needed IDs)...", country, path.name)
        for chunk in pd.read_csv(path, chunksize=NORM_CHUNKSIZE, **READ_KWARGS):
            # Country pre-filter on the raw label (same open-set semantics as
            # the pipeline's lowercased partition key) to skip irrelevant rows.
            sub = chunk[chunk["country"].fillna("").astype(str).str.lower() == country]
            if sub.empty:
                continue
            sub = sub[sub["entity_id"].isin(keep)]
            if sub.empty:
                continue
            normed = pipe.normalize_frame(sub)
            slim = slim_frame(normed)  # exact preservation mode
            (s1_parts if kind == "s1" else c2_parts if kind == "s2" else c3_parts).append(slim)
            del sub, normed, slim
        gc.collect()
    s1_slim = pd.concat(s1_parts, ignore_index=True) if s1_parts else pd.DataFrame(columns=list(SLIM_RECORD_FIELDS))
    cand_slim = pd.concat(c2_parts + c3_parts, ignore_index=True) if (c2_parts or c3_parts) else pd.DataFrame(columns=list(SLIM_RECORD_FIELDS))
    del s1_parts, c2_parts, c3_parts
    gc.collect()
    LOGGER.info("[%s] slim records: s1=%d cands=%d", country, len(s1_slim), len(cand_slim))

    # ID <-> index maps (parent side only).
    s1_id_list = s1_slim["entity_id"].tolist()
    cand_id_list = cand_slim["entity_id"].tolist()
    assert len(set(s1_id_list)) == len(s1_id_list), "duplicate S1 ids in store"
    assert len(set(cand_id_list)) == len(cand_id_list), "duplicate cand ids in store"
    s1_id2idx = {s: i for i, s in enumerate(s1_id_list)}
    cand_id2idx = {s: i for i, s in enumerate(cand_id_list)}
    missing = need_cands - set(cand_id2idx)
    if missing:
        raise RuntimeError(f"[{country}] {len(missing)} pair candidates missing from store")

    # Persist id lists for assembly.
    np.save(WORK_DIR / f"store_s1_ids_{country}.npy", np.array(s1_id_list, dtype=object))
    np.save(WORK_DIR / f"store_cand_ids_{country}.npy", np.array(cand_id_list, dtype=object))

    store = SharedRecordStore()
    descriptor = store.build(s1_slim, cand_slim)
    del s1_slim, cand_slim
    gc.collect()
    LOGGER.info("[%s] shared store built (s1=%d cands=%d)", country, store.n_s1, store.n_cand)

    with open(MODEL_META, encoding="utf-8") as f:
        feature_order = json.load(f)["feature_column_order"]

    # Build tasks per pair-chunk (split into task_anchors-anchor groups).
    proba_dir = WORK_DIR / f"proba_{country}"
    proba_dir.mkdir(parents=True, exist_ok=True)
    tasks_meta = []
    for ci, cf in enumerate(chunk_files):
        out_npz = proba_dir / f"chunk_{ci:04d}.npz"
        if out_npz.exists():
            continue
        df = pd.read_csv(cf, **CACHE_READ_KWARGS)
        anchors_here = sorted(df["source1_entity_id"].astype(str).unique().tolist())
        for a0 in range(0, len(anchors_here), task_anchors):
            agroup = set(anchors_here[a0:a0 + task_anchors])
            sub = df[df["source1_entity_id"].isin(agroup)]
            payload = {
                "chunk": ci,
                "s1_idx": np.array([s1_id2idx[s] for s in sub["source1_entity_id"].astype(str)], dtype=np.int64),
                "cand_idx": np.array([cand_id2idx[s] for s in sub["candidate_entity_id"].astype(str)], dtype=np.int64),
                "cand_source": sub["candidate_source"].astype(str).tolist(),
                "strategies": sub["strategies"].astype(str).tolist(),
                "strategy_count": sub["strategy_count"].astype(int).tolist(),
            }
            tasks_meta.append((ci, out_npz, payload))
            del sub
        del df
        gc.collect()
    LOGGER.info("[%s] scoring tasks: %d (workers=%d)", country, len(tasks_meta), workers)

    results: dict[int, list] = {}
    if tasks_meta:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=score_worker_init,
                                 initargs=(descriptor, str(MODEL_PATH), feature_order)) as ex:
            futs = {ex.submit(score_task, p): (ci, p) for ci, _, p in tasks_meta}
            done = 0
            for fut in futs:
                ci, _ = futs[fut]
                res = fut.result()
                results.setdefault(ci, []).append(res)
                done += 1
                if done % 20 == 0:
                    LOGGER.info("[%s] scored %d/%d tasks", country, done, len(tasks_meta))
        # Write per-chunk npz.
        for ci, parts in results.items():
            out_npz = proba_dir / f"chunk_{ci:04d}.npz"
            s1_all = np.concatenate([r["s1_idx"] for r in parts])
            c_all = np.concatenate([r["cand_idx"] for r in parts])
            p_all = np.concatenate([r["proba"] for r in parts])
            np.savez_compressed(out_npz, s1_idx=s1_all, cand_idx=c_all, proba=p_all)
            LOGGER.info("[%s] wrote %s (%d pairs)", country, out_npz.name, len(p_all))

    store.close()
    store.unlink()
    audit = {"country": country, "total_pairs": total_pairs,
             "anchors_with_pairs": len(anchors_with_pairs),
             "unique_candidates": len(cand_id_list),
             "workers": workers, "elapsed_seconds": round(time.time() - t_start, 2)}
    with open(WORK_DIR / f"scoring_audit_{country}.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    LOGGER.info("[%s] done: %s", country, audit)
    return audit


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    configure_logging(settings.logging)
    set_random_seed(settings.random_seed)
    targets = list(COUNTRIES) if args.all else ([args.country] if args.country else [])
    if not targets:
        print("Specify --country or --all")
        return 2
    for c in targets:
        run_country(c, args.workers, args.task_anchors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
