"""Phase 11a — Test candidate generation (per country).

TEST S1/S2/S3 -> Phase 2 normalization -> Phase 3 multi-strategy blocking
-> per-anchor-chunk pair checkpoints. No labels, no ground truth, no test
supervision of any kind.

Usage:
    python scripts/run_phase11a_test_blocking.py --country india
    python scripts/run_phase11a_test_blocking.py --country us
    python scripts/run_phase11a_test_blocking.py --country france
    python scripts/run_phase11a_test_blocking.py --normalize-only   # step 0 cache
    python scripts/run_phase11a_test_blocking.py --all             # everything

Step 0 normalizes all test files once (chunked) into per-country slim-block
caches. Blocking then runs per country in its own process so the multi-GB
blocker indexes are freed on exit. BlockingConfig is identical to Phase 3.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd

from business_entity_resolution.blocking import BlockingConfig, MultiStrategyBlockingPipeline
from business_entity_resolution.blocking.union import AnchorCandidates, CandidateUnion
from business_entity_resolution.config.settings import load_settings
from business_entity_resolution.inference.test_inference import BLOCK_SLIM_FIELDS, slim_frame
from business_entity_resolution.normalization.pipeline import NormalizationPipeline
from business_entity_resolution.utils.logging import configure_logging, get_logger
from business_entity_resolution.utils.seed import set_random_seed

LOGGER = get_logger("phase11a_runner")

TEST_DIR = Path("dataset/student_resource/dataset/test")
WORK_DIR = Path("output/phase11/work")
COUNTRIES = ("india", "us", "france")
READ_KWARGS = {"sep": "\t", "dtype": str, "na_filter": False}
CACHE_READ_KWARGS = {"sep": ",", "dtype": str, "na_filter": False}
NORM_CHUNKSIZE = 200_000
QUERY_CHUNK = 20_000


def blocking_config() -> BlockingConfig:
    """Identical hyperparameters to the Phase 3 validated pipeline."""
    return BlockingConfig(
        enable_exact=True,
        enable_name_token=True,
        enable_address_token=True,
        enable_char_ngram=True,
        enable_tfidf=True,
        partition_by_country=True,
        exact_max_block_size=5000,
        min_token_length=3,
        max_token_frequency=10000,
        name_token_max_cands=50,
        address_max_block_size=5000,
        address_max_cands=50,
        char_ngram_size=3,
        char_ngram_min_length=3,
        char_ngram_top_k=20,
        char_ngram_min_overlap=0.35,
        tfidf_top_k_name=10,
        tfidf_top_k_address=5,
        tfidf_min_similarity=0.15,
        tfidf_batch_size=2000,
        max_candidates_per_anchor=100,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 11a test candidate generation.")
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--country", type=str, default=None, choices=list(COUNTRIES))
    p.add_argument("--normalize-only", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--query-chunk", type=int, default=QUERY_CHUNK)
    p.add_argument("--tfidf-mode", type=str, default="hybrid", choices=["standard", "hybrid"],
                   help="standard = validated TfidfBlocker path; hybrid = standard name "
                        "retrieval + exact MaxScore address retrieval (same params, "
                        "address exact-ties by column order)")
    p.add_argument("--name-batch-size", type=int, default=500,
                   help="Batch size for the (sparse-safe) name TF-IDF dot.")
    return p.parse_args()


def step0_normalize_cache(settings) -> dict:
    """Normalize all test files once; write per-country slim-block caches."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    done_marker = WORK_DIR / "norm_cache_done.json"
    if done_marker.exists():
        LOGGER.info("Normalization cache already complete, skipping.")
        return json.load(open(done_marker, encoding="utf-8"))

    pipe = NormalizationPipeline(settings.normalization)
    writers: dict[str, dict[str, list]] = {}
    counts = {"s1": {}, "s2": {}, "s3": {}}
    missing_country = {"s1": 0, "s2": 0, "s3": 0}
    t_start = time.time()

    files = {"s1": TEST_DIR / "test_source1.tsv",
             "s2": TEST_DIR / "test_source2.tsv",
             "s3": TEST_DIR / "test_source3.tsv"}
    for kind, path in files.items():
        LOGGER.info("Normalizing %s ...", path.name)
        buffers: dict[str, list] = {}
        for chunk in pd.read_csv(path, chunksize=NORM_CHUNKSIZE, **READ_KWARGS):
            normed = pipe.normalize_frame(chunk)
            cty = normed["country__country"].fillna("").astype(str).str.lower()
            missing_country[kind] += int((cty == "").sum())
            slim = slim_frame(normed, for_blocking=True)
            slim["_cty"] = cty.values
            for c, sub in slim.groupby("_cty"):
                c = str(c)
                buffers.setdefault(c, []).append(sub.drop(columns=["_cty"]))
            counts[kind] = {c: counts[kind].get(c, 0) + len(v)
                            for c, v in slim.groupby("_cty", observed=True)
                            for v in [v]}
            del normed, slim
            gc.collect()
        # flush per-country caches
        for c, parts in buffers.items():
            out = WORK_DIR / f"normblock_{kind}_{c}.csv.gz"
            df = pd.concat(parts, ignore_index=True)
            df.to_csv(out, index=False, compression="gzip")
            LOGGER.info("Wrote %s (%d rows)", out, len(df))
            del df
        del buffers
        gc.collect()

    summary = {"counts": counts, "missing_country_records": missing_country,
               "elapsed_seconds": round(time.time() - t_start, 2)}
    with open(done_marker, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Normalization cache done: %s", summary)
    return summary


def load_cache(kind: str, country: str) -> pd.DataFrame:
    path = WORK_DIR / f"normblock_{kind}_{country}.csv.gz"
    if not path.exists():
        return pd.DataFrame(columns=list(BLOCK_SLIM_FIELDS))
    return pd.read_csv(path, **CACHE_READ_KWARGS)


def run_country(country: str, query_chunk: int, tfidf_mode: str = "hybrid",
                name_batch_size: int = 500) -> dict:
    t_start = time.time()
    out_dir = WORK_DIR / f"pairs_{country}"
    out_dir.mkdir(parents=True, exist_ok=True)

    anchors = load_cache("s1", country)
    s2 = load_cache("s2", country)
    s3 = load_cache("s3", country)
    LOGGER.info("[%s] anchors=%d s2=%d s3=%d", country, len(anchors), len(s2), len(s3))
    if len(anchors) == 0:
        return {"country": country, "anchors": 0, "pairs": 0}

    # Roster of all anchors (chunking key + assembly roster)
    roster_path = WORK_DIR / f"anchors_{country}.csv.gz"
    anchors[["entity_id"]].to_csv(roster_path, index=False, compression="gzip")

    cands = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    pipeline = MultiStrategyBlockingPipeline(blocking_config())
    t0 = time.time()
    pipeline.fit(cands)
    LOGGER.info("[%s] fit %d candidates in %.1fs", country, len(cands), time.time() - t0)
    del cands
    gc.collect()

    if tfidf_mode == "hybrid":
        from business_entity_resolution.blocking.maxscore import HybridTfidfBlocker
        for part, blockers in pipeline._partition_blockers.items():
            if "tfidf" in blockers:
                blockers["tfidf"] = HybridTfidfBlocker(blockers["tfidf"],
                                                       name_batch_size=name_batch_size)
        LOGGER.info("[%s] TF-IDF engine: hybrid (standard name + MaxScore addr)", country)
    else:
        LOGGER.info("[%s] TF-IDF engine: standard (validated path)", country)

    # Fallback partitions for anchors whose country has no fitted blockers.
    # (No-op on this test set: all S1 countries have candidate partitions.)
    anchor_cty = anchors["country__country"].fillna("").astype(str).str.lower()
    known = set(pipeline._partition_blockers.keys())
    fallback_mask = ~anchor_cty.isin(known)

    total_pairs = 0
    zero_cand = 0
    n_chunks = (len(anchors) + query_chunk - 1) // query_chunk
    chunk_files = []
    for i in range(n_chunks):
        ckpt = out_dir / f"chunk_{i:04d}.csv.gz"
        if ckpt.exists():
            df_old = pd.read_csv(ckpt, **CACHE_READ_KWARGS)
            total_pairs += len(df_old)
            zero_cand += int((df_old.groupby("source1_entity_id").size() == 0).sum())
            chunk_files.append(str(ckpt))
            del df_old
            continue
        sub = anchors.iloc[i * query_chunk:(i + 1) * query_chunk]
        t0 = time.time()
        merged, _ = pipeline.block(sub)
        # Fallback: anchors whose country has no fitted partition query every
        # partition and merge via the same union engine (same cap/validation).
        # No-op on this test set (all S1 countries have candidate partitions).
        fb_idx = fallback_mask.iloc[i * query_chunk:(i + 1) * query_chunk]
        if bool(fb_idx.any()):
            from collections import defaultdict as _dd
            fb = sub[fb_idx.values]
            strat_maps: dict[str, dict[str, dict[str, set[str]]]] = _dd(lambda: _dd(dict))
            for _cty, blockers in pipeline._partition_blockers.items():
                for strat_name, blocker in blockers.items():
                    extra = blocker.generate_pairs(fb, id_col="entity_id")
                    for aid, cand_map in extra.items():
                        strat_maps[strat_name][aid].update(cand_map)
            fb_merged = pipeline.union_engine.merge_strategy_results(list(strat_maps.items()))
            for aid in fb["entity_id"].astype(str):
                if aid not in fb_merged:
                    fb_merged[aid] = AnchorCandidates(anchor_id=aid, candidates=())
            merged.update(fb_merged)
            LOGGER.warning("[%s] chunk %d: %d fallback anchors merged across partitions",
                           country, i, len(fb))
        rows = []
        for aid, ac in merged.items():
            if ac.total_candidates == 0:
                zero_cand += 1
            for c in ac.candidates:
                rows.append((aid, c.candidate_id, c.source,
                             ";".join(sorted(c.strategies)), len(c.strategies)))
        df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id",
                                         "candidate_source", "strategies", "strategy_count"])
        df.to_csv(ckpt, index=False, compression="gzip")
        total_pairs += len(df)
        chunk_files.append(str(ckpt))
        LOGGER.info("[%s] chunk %d/%d: %d anchors -> %d pairs (%.1fs)",
                    country, i + 1, n_chunks, len(sub), len(df), time.time() - t0)
        del df, merged, rows
        gc.collect()

    audit = {"country": country, "anchors": len(anchors), "total_pairs": total_pairs,
             "avg_pairs_per_anchor": round(total_pairs / max(len(anchors), 1), 3),
             "zero_candidate_anchors": zero_cand,
             "chunks": n_chunks, "chunk_files": chunk_files,
             "fallback_anchors": int(fallback_mask.sum()),
             "elapsed_seconds": round(time.time() - t_start, 2)}
    with open(WORK_DIR / f"blocking_audit_{country}.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    LOGGER.info("[%s] done: %s", country, audit)
    return audit


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    configure_logging(settings.logging)
    set_random_seed(settings.random_seed)

    step0_normalize_cache(settings)
    if args.normalize_only:
        return 0
    targets = list(COUNTRIES) if args.all else ([args.country] if args.country else [])
    if not targets:
        print("Specify --country, --all, or --normalize-only")
        return 2
    for c in targets:
        run_country(c, args.query_chunk, args.tfidf_mode, args.name_batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
