"""Phase 3 Candidate Generation & Ground Truth Recall Evaluation Runner.

Discovers training files dynamically from the configured dataset root.
Uses Phase 2 NormalizationPipeline to produce comparison representations.
Runs MultiStrategyBlockingPipeline across:
- Strategy A: Exact Keys
- Strategy B: Name Token Blocking
- Strategy C: Address Token Blocking
- Strategy D: Character N-Gram Overlap
- Strategy E: TF-IDF Sparse Retrieval
Evaluates candidate recall against training ground truth.
Outputs candidate_pairs.tsv and detailed diagnostics reports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from business_entity_resolution.blocking import (
    BlockingConfig,
    CandidateRecallEvaluator,
    CandidateUnion,
    MultiStrategyBlockingPipeline,
    parse_ground_truth,
)
from business_entity_resolution.config.settings import load_settings
from business_entity_resolution.ingestion.discovery import discover_dataset
from business_entity_resolution.normalization.pipeline import NormalizationPipeline
from business_entity_resolution.utils.logging import configure_logging, get_logger

LOGGER = get_logger("phase3_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 candidate generation and evaluation.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Config file path")
    parser.add_argument("--num-anchors", type=int, default=5000, help="Number of S1 anchors to evaluate (default: 5000)")
    parser.add_argument("--num-candidates-per-source", type=int, default=100000, help="Number of candidates to index per source (default: 100000)")
    parser.add_argument("--output-dir", type=Path, default=Path("output/phase3"), help="Output directory")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for sampling")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    configure_logging(settings.logging)
    LOGGER.info("Starting Phase 3 Multi-Strategy Blocking Evaluation")
    t_start = time.time()

    # 1. Discover training files dynamically
    project_root = Path.cwd()
    discovered = discover_dataset(settings.dataset, project_root)
    train_sources = discovered.train_sources
    gt_file = discovered.train_ground_truth

    if not train_sources or len(train_sources) < 3:
        LOGGER.error("Failed to discover 3 training source files under %s", settings.dataset.root)
        return 1
    if not gt_file:
        LOGGER.error("Failed to discover training ground truth file under %s", settings.dataset.root)
        return 1

    s1_file = [f for f in train_sources if f.source_index == 1 or "source1" in f.filename][0]
    s2_file = [f for f in train_sources if f.source_index == 2 or "source2" in f.filename][0]
    s3_file = [f for f in train_sources if f.source_index == 3 or "source3" in f.filename][0]

    LOGGER.info("Discovered S1: %s (%s)", s1_file.filename, s1_file.relative_path)
    LOGGER.info("Discovered S2: %s (%s)", s2_file.filename, s2_file.relative_path)
    LOGGER.info("Discovered S3: %s (%s)", s3_file.filename, s3_file.relative_path)
    LOGGER.info("Discovered GT: %s (%s)", gt_file.filename, gt_file.relative_path)

    # 2. Load Ground Truth
    LOGGER.info("Loading training ground truth from %s...", gt_file.path)
    t0 = time.time()
    gt_df = pd.read_csv(gt_file.path, sep=settings.dataset.delimiter, dtype=str)
    ground_truth = parse_ground_truth(gt_df)
    LOGGER.info("Loaded %d ground truth anchors in %.2fs", len(ground_truth), time.time() - t0)

    # 3. Sample or Load S1 Anchors
    LOGGER.info("Loading S1 anchors from %s...", s1_file.path)
    t0 = time.time()
    num_anchors = args.num_anchors
    if num_anchors > 0:
        # Load sample of S1 anchors deterministically
        s1_df = pd.read_csv(
            s1_file.path,
            sep=settings.dataset.delimiter,
            nrows=num_anchors,
            dtype=str,
        )
    else:
        s1_df = pd.read_csv(s1_file.path, sep=settings.dataset.delimiter, dtype=str)
    LOGGER.info("Loaded %d S1 anchors in %.2fs", len(s1_df), time.time() - t0)

    # Collect all true matched IDs for these S1 anchors to guarantee target coverage
    eval_s1_ids = set(s1_df["entity_id"])
    needed_candidate_ids: set[str] = set()
    for aid in eval_s1_ids:
        needed_candidate_ids.update(ground_truth.get(aid, set()))
    needed_s2 = {cid for cid in needed_candidate_ids if cid.startswith("S2-")}
    needed_s3 = {cid for cid in needed_candidate_ids if cid.startswith("S3-")}
    LOGGER.info("Sample S1 anchors require %d true S2 matches and %d true S3 matches", len(needed_s2), len(needed_s3))

    # 4. Load Candidate Pool (S2 and S3)
    # To ensure evaluation validity, we load background candidates + all true target candidates
    LOGGER.info("Loading S2 candidate pool from %s...", s2_file.path)
    t0 = time.time()
    cand_limit = args.num_candidates_per_source
    s2_chunks = []
    # Read background rows
    if cand_limit > 0:
        s2_bg = pd.read_csv(s2_file.path, sep=settings.dataset.delimiter, nrows=cand_limit, dtype=str)
        s2_chunks.append(s2_bg)
    else:
        s2_bg = pd.read_csv(s2_file.path, sep=settings.dataset.delimiter, dtype=str)
        s2_chunks.append(s2_bg)

    # Check if any needed S2 matches are missing from s2_bg, and stream to find them
    s2_existing_ids = set(s2_bg["entity_id"])
    missing_s2 = needed_s2 - s2_existing_ids
    if missing_s2:
        LOGGER.info("Streaming S2 to locate %d target matches not in leading rows...", len(missing_s2))
        found_s2 = []
        for chunk in pd.read_csv(s2_file.path, sep=settings.dataset.delimiter, chunksize=250000, dtype=str):
            sub = chunk[chunk["entity_id"].isin(missing_s2)]
            if not sub.empty:
                found_s2.append(sub)
                missing_s2 -= set(sub["entity_id"])
            if not missing_s2:
                break
        if found_s2:
            s2_chunks.extend(found_s2)
    s2_df = pd.concat(s2_chunks, ignore_index=True).drop_duplicates(subset=["entity_id"])
    LOGGER.info("Prepared S2 candidate pool (%d records) in %.2fs", len(s2_df), time.time() - t0)

    LOGGER.info("Loading S3 candidate pool from %s...", s3_file.path)
    t0 = time.time()
    s3_chunks = []
    if cand_limit > 0:
        s3_bg = pd.read_csv(s3_file.path, sep=settings.dataset.delimiter, nrows=cand_limit, dtype=str)
        s3_chunks.append(s3_bg)
    else:
        s3_bg = pd.read_csv(s3_file.path, sep=settings.dataset.delimiter, dtype=str)
        s3_chunks.append(s3_bg)

    s3_existing_ids = set(s3_bg["entity_id"])
    missing_s3 = needed_s3 - s3_existing_ids
    if missing_s3:
        LOGGER.info("Streaming S3 to locate %d target matches not in leading rows...", len(missing_s3))
        found_s3 = []
        for chunk in pd.read_csv(s3_file.path, sep=settings.dataset.delimiter, chunksize=250000, dtype=str):
            sub = chunk[chunk["entity_id"].isin(missing_s3)]
            if not sub.empty:
                found_s3.append(sub)
                missing_s3 -= set(sub["entity_id"])
            if not missing_s3:
                break
        if found_s3:
            s3_chunks.extend(found_s3)
    s3_df = pd.concat(s3_chunks, ignore_index=True).drop_duplicates(subset=["entity_id"])
    LOGGER.info("Prepared S3 candidate pool (%d records) in %.2fs", len(s3_df), time.time() - t0)

    # Combine S2 and S3 into candidate pool
    candidates_df = pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates(subset=["entity_id"])
    LOGGER.info("Total candidate pool: %d records (S2: %d, S3: %d)", len(candidates_df), len(s2_df), len(s3_df))

    # 5. Apply Phase 2 Normalization Pipeline to anchors
    LOGGER.info("Applying Phase 2 NormalizationPipeline to anchors...")
    t0 = time.time()
    norm_pipe = NormalizationPipeline(settings.normalization)
    s1_norm = norm_pipe.normalize_frame(s1_df)
    LOGGER.info("Anchor normalization complete in %.2fs", time.time() - t0)

    # 6. Configure & Run Multi-Strategy Blocking Pipeline
    blocking_cfg = BlockingConfig(
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

    pipeline = MultiStrategyBlockingPipeline(blocking_cfg)
    LOGGER.info("Executing MultiStrategyBlockingPipeline with Ground Truth Evaluation...")
    t0 = time.time()
    merged_candidates, report = pipeline.run_with_evaluation(
        anchors_df=s1_norm,
        candidates_df=candidates_df,
        ground_truth=ground_truth,
    )
    t_blocking = time.time() - t0
    LOGGER.info("Candidate generation & evaluation completed in %.2fs", t_blocking)

    # 7. Print and Save Results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_dict = report.to_dict()
    report_dict["execution_time_seconds"] = round(time.time() - t_start, 2)
    report_dict["blocking_time_seconds"] = round(t_blocking, 2)

    # Save JSON report
    json_path = args.output_dir / "candidate_recall_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    LOGGER.info("Saved JSON report to %s", json_path)

    # Export candidate_pairs.tsv
    tsv_path = args.output_dir / "candidate_pairs.tsv"
    CandidateUnion.export_candidate_pairs_tsv(merged_candidates, tsv_path)
    LOGGER.info("Exported candidate_pairs.tsv to %s", tsv_path)

    # Export candidate pairs with provenance table
    pairs_df = CandidateUnion.to_dataframe(merged_candidates)
    pairs_csv = args.output_dir / "candidate_pairs_provenance.csv"
    pairs_df.to_csv(pairs_csv, index=False)
    LOGGER.info("Exported candidate pairs provenance to %s", pairs_csv)

    # Print summary scoreboard
    print("\n" + "=" * 65)
    print("PHASE 3 CANDIDATE GENERATION & RECALL SCOREBOARD")
    print("=" * 65)
    print(f"Total Anchors Evaluated:            {report.total_anchors_evaluated:,}")
    print(f"Anchors with Matches (Non-Single): {report.total_true_anchors_with_matches:,}")
    print(f"Singletons (0 true matches):        {report.total_singleton_anchors:,}")
    print("-" * 65)
    print(f"Total True Pairs in GT:             {report.total_true_pairs:,}")
    print(f"True Pairs Retrieved:               {report.true_pairs_retrieved:,}")
    print(f"PAIR-LEVEL CANDIDATE RECALL:        {report.pair_recall * 100:.2f}%")
    print(f"  - Source 2 True-Pair Recall:      {report.s2_recall * 100:.2f}% ({report.true_s2_retrieved:,} / {report.total_true_s2_pairs:,})")
    print(f"  - Source 3 True-Pair Recall:      {report.s3_recall * 100:.2f}% ({report.true_s3_retrieved:,} / {report.total_true_s3_pairs:,})")
    print(f"ENTITY-LEVEL COVERAGE (100% pairs): {report.entity_coverage * 100:.2f}% ({report.anchors_with_full_coverage:,} / {report.total_true_anchors_with_matches:,})")
    print("-" * 65)
    print(f"Total Candidate Pairs Generated:    {report.total_candidate_pairs:,}")
    print(f"Average Candidates per Anchor:      {report.avg_candidates_per_anchor:.2f}")
    print(f"Median Candidates per Anchor:       {report.median_candidates_per_anchor:.1f}")
    print(f"p90 Candidates per Anchor:          {report.p90_candidates_per_anchor:.1f}")
    print(f"p95 Candidates per Anchor:          {report.p95_candidates_per_anchor:.1f}")
    print(f"p99 Candidates per Anchor:          {report.p99_candidates_per_anchor:.1f}")
    print(f"Maximum Candidates for an Anchor:   {report.max_candidates_per_anchor}")
    print("-" * 65)
    print("CANDIDATE DISTRIBUTION:")
    print(f"  0 candidates:      {report.pct_zero_candidates:.2f}%")
    print(f"  1 candidate:       {report.pct_one_candidate:.2f}%")
    print(f"  2 - 10 candidates: {report.pct_two_to_ten:.2f}%")
    print(f"  11 - 50 cands:     {report.pct_eleven_to_fifty:.2f}%")
    print(f"  51 - 100 cands:    {report.pct_fifty_one_to_hundred:.2f}%")
    print(f"  > 100 candidates:  {report.pct_over_hundred:.2f}%")
    print("-" * 65)
    print("STRATEGY-BY-STRATEGY METRICS:")
    print(f"{'Strategy':<18} {'Cand Pairs':<12} {'True Found':<12} {'Recall':<8} {'Avg Cands/S1'}")
    for sm in report.strategy_metrics:
        print(f"{sm['strategy']:<18} {sm['candidate_pairs']:<12,d} {sm['true_pairs_found']:<12,d} {sm['recall']*100:.2f}%   {sm['avg_candidates_per_anchor']}")
    print("-" * 65)
    print("ABLATION ANALYSIS (CUMULATIVE UNION):")
    print(f"{'Stage':<24} {'Cand Pairs':<12} {'True Found':<12} {'Recall':<8} {'Avg Cands/S1'}")
    for am in report.ablation_metrics:
        print(f"{am['stage']:<24} {am['total_candidate_pairs']:<12,d} {am['true_pairs_found']:<12,d} {am['recall']*100:.2f}%   {am['avg_candidates_per_anchor']}")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
