"""Phase 4 Baseline Matching, Retrieval, and Diagnostic Evaluation Runner.

Loads Phase 3 candidate pairs and corresponding normalized S1/S2/S3 entity records.
Executes DeterministicBaselineMatcher to compute explainable similarity scores and rankings.
Evaluates Top-K recall, diagnostic threshold precision/recall/F0.5, and error analysis.
Exports baseline ranked pairs and evaluation reports to output/phase4/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from business_entity_resolution.blocking.evaluation import parse_ground_truth
from business_entity_resolution.config.settings import load_settings
from business_entity_resolution.ingestion.discovery import discover_dataset
from business_entity_resolution.matching import (
    BaselineEvaluationReport,
    BaselineEvaluator,
    BaselineScoreWeights,
    DeterministicBaselineMatcher,
)
from business_entity_resolution.normalization.pipeline import NormalizationPipeline
from business_entity_resolution.utils.logging import configure_logging, get_logger

LOGGER = get_logger("phase4_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 Baseline Matching and Evaluation.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Config file path")
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=Path("output/phase3/candidate_pairs_provenance.csv"),
        help="Path to Phase 3 candidate pairs provenance CSV",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/phase4"), help="Output directory")
    return parser.parse_args()


def load_candidate_records(
    file_path: Path,
    needed_ids: set[str],
    sep: str = "\t",
    chunksize: int = 500000,
) -> pd.DataFrame:
    """Stream file in chunks to extract only records present in needed_ids."""
    LOGGER.info("Scanning %s for %d needed records...", file_path.name, len(needed_ids))
    found_chunks = []
    remaining = set(needed_ids)

    for chunk in pd.read_csv(file_path, sep=sep, chunksize=chunksize, dtype=str):
        sub = chunk[chunk["entity_id"].isin(remaining)]
        if not sub.empty:
            found_chunks.append(sub)
            remaining -= set(sub["entity_id"])
        if not remaining:
            break

    if not found_chunks:
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    res = pd.concat(found_chunks, ignore_index=True).drop_duplicates(subset=["entity_id"])
    LOGGER.info("Retrieved %d / %d records from %s", len(res), len(needed_ids), file_path.name)
    return res


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    configure_logging(settings.logging)
    LOGGER.info("Starting Phase 4 Baseline Matching and Retrieval Evaluation")
    t_start = time.time()

    # 1. Discover training files
    project_root = Path.cwd()
    discovered = discover_dataset(settings.dataset, project_root)
    train_sources = discovered.train_sources
    gt_file = discovered.train_ground_truth

    if not train_sources or len(train_sources) < 3 or not gt_file:
        LOGGER.error("Failed to discover training dataset files")
        return 1

    s1_file = [f for f in train_sources if f.source_index == 1 or "source1" in f.filename][0]
    s2_file = [f for f in train_sources if f.source_index == 2 or "source2" in f.filename][0]
    s3_file = [f for f in train_sources if f.source_index == 3 or "source3" in f.filename][0]

    # 2. Load Ground Truth
    LOGGER.info("Loading ground truth from %s...", gt_file.path)
    gt_df = pd.read_csv(gt_file.path, sep=settings.dataset.delimiter, dtype=str)
    ground_truth = parse_ground_truth(gt_df)

    # 3. Load Phase 3 Candidate Pairs
    if not args.candidates_path.exists():
        LOGGER.error("Candidate pairs file not found at %s. Run Phase 3 first!", args.candidates_path)
        return 1

    LOGGER.info("Loading Phase 3 candidate pairs from %s...", args.candidates_path)
    pairs_df = pd.read_csv(args.candidates_path, dtype=str)
    LOGGER.info("Loaded %d candidate pairs across %d anchors", len(pairs_df), pairs_df["source1_entity_id"].nunique())

    # 4. Retrieve and normalize raw records for candidates
    needed_s1_ids = set(pairs_df["source1_entity_id"])
    cand_ids = set(pairs_df["candidate_entity_id"])
    needed_s2_ids = {cid for cid in cand_ids if cid.startswith("S2-") or str(cid).startswith("S2")}
    needed_s3_ids = {cid for cid in cand_ids if cid.startswith("S3-") or str(cid).startswith("S3")}

    norm_pipe = NormalizationPipeline(settings.normalization)

    # S1 anchors
    LOGGER.info("Loading and normalizing S1 anchors...")
    t0 = time.time()
    s1_raw = load_candidate_records(s1_file.path, needed_s1_ids, sep=settings.dataset.delimiter)
    s1_norm = norm_pipe.normalize_frame(s1_raw)
    s1_lookup = {row["entity_id"]: row.to_dict() for _, row in s1_norm.iterrows()}
    LOGGER.info("Normalized %d S1 anchors in %.2fs", len(s1_lookup), time.time() - t0)

    # S2 candidates
    LOGGER.info("Loading and normalizing S2 candidates...")
    t0 = time.time()
    s2_raw = load_candidate_records(s2_file.path, needed_s2_ids, sep=settings.dataset.delimiter)
    s2_norm = norm_pipe.normalize_frame(s2_raw)
    cand_lookup = {row["entity_id"]: row.to_dict() for _, row in s2_norm.iterrows()}
    LOGGER.info("Normalized %d S2 records in %.2fs", len(s2_norm), time.time() - t0)

    # S3 candidates
    LOGGER.info("Loading and normalizing S3 candidates...")
    t0 = time.time()
    s3_raw = load_candidate_records(s3_file.path, needed_s3_ids, sep=settings.dataset.delimiter)
    s3_norm = norm_pipe.normalize_frame(s3_raw)
    for _, row in s3_norm.iterrows():
        cand_lookup[row["entity_id"]] = row.to_dict()
    LOGGER.info("Normalized %d S3 records in %.2fs. Total candidate pool lookup: %d", len(s3_norm), time.time() - t0, len(cand_lookup))

    # 5. Execute Deterministic Baseline Matching and Ranking
    LOGGER.info("Executing Deterministic Baseline Matching...")
    weights = BaselineScoreWeights(
        name_weight=0.50,
        address_weight=0.35,
        structured_weight=0.15,
    )
    matcher = DeterministicBaselineMatcher(weights=weights)
    t0 = time.time()
    ranked_df = matcher.rank_candidates(
        candidate_pairs_df=pairs_df,
        s1_lookup=s1_lookup,
        cand_lookup=cand_lookup,
    )
    t_match = time.time() - t0
    LOGGER.info("Baseline scoring & ranking completed in %.2fs (%.0f pairs/sec)", t_match, len(ranked_df) / max(t_match, 0.001))

    # 6. Evaluate Baseline Matcher
    LOGGER.info("Evaluating baseline matcher against ground truth...")
    evaluator = BaselineEvaluator(ground_truth=ground_truth)
    report = evaluator.evaluate(
        ranked_df=ranked_df,
        s1_lookup=s1_lookup,
        cand_lookup=cand_lookup,
    )

    # 7. Export Outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Export ranked candidate pairs CSV
    ranked_csv = args.output_dir / "baseline_ranked_candidates.csv"
    ranked_df.to_csv(ranked_csv, index=False)
    LOGGER.info("Exported ranked candidate pairs to %s", ranked_csv)

    # Export evaluation report JSON
    report_dict = report.to_dict()
    report_dict["execution_time_seconds"] = round(time.time() - t_start, 2)
    report_dict["scoring_time_seconds"] = round(t_match, 2)
    json_path = args.output_dir / "baseline_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    LOGGER.info("Exported baseline evaluation report to %s", json_path)

    # Export diagnostic threshold sweep CSV
    thresh_df = pd.DataFrame(report_dict["diagnostic_thresholds"])
    thresh_csv = args.output_dir / "baseline_diagnostic_thresholds.csv"
    thresh_df.to_csv(thresh_csv, index=False)
    LOGGER.info("Exported diagnostic threshold sweep to %s", thresh_csv)

    # Export error analysis JSON
    error_analysis = {
        "false_negative_count": report.false_negative_count,
        "false_positive_count": report.false_positive_count,
        "error_patterns": report.error_patterns,
        "top_false_negatives": report.top_false_negatives,
        "top_false_positives": report.top_false_positives,
    }
    error_json = args.output_dir / "baseline_error_analysis.json"
    with open(error_json, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2)
    LOGGER.info("Exported baseline error analysis to %s", error_json)

    # Print Scoreboard
    print("\n" + "=" * 70)
    print("PHASE 4 BASELINE MATCHING & RETRIEVAL SCOREBOARD")
    print("=" * 70)
    print(f"Total Candidate Pairs Evaluated:    {report.total_candidate_pairs:,}")
    print(f"Total True Pairs in Ground Truth:   {report.total_true_pairs_gt:,}")
    print(f"True Pairs in Candidate Pool:       {report.true_pairs_in_candidate_pool:,}")
    print(f"Phase 3 Candidate Recall:           {report.phase3_candidate_recall * 100:.2f}%")
    print("-" * 70)
    print("BASELINE RANKING RECALL (TOP-K):")
    print(f"  Top-1 Recall:   {report.overall_top_k.top_1_recall * 100:.2f}% ({report.overall_top_k.true_at_1:,} true pairs ranked #1)")
    print(f"  Top-3 Recall:   {report.overall_top_k.top_3_recall * 100:.2f}% ({report.overall_top_k.true_at_3:,} true pairs ranked <= 3)")
    print(f"  Top-5 Recall:   {report.overall_top_k.top_5_recall * 100:.2f}% ({report.overall_top_k.true_at_5:,} true pairs ranked <= 5)")
    print(f"  Top-10 Recall:  {report.overall_top_k.top_10_recall * 100:.2f}% ({report.overall_top_k.true_at_10:,} true pairs ranked <= 10)")
    print("-" * 70)
    print("SOURCE-SPECIFIC TOP-1 / TOP-5 RECALL:")
    print(f"  Source 2 Top-1 Recall:  {report.s2_top_k.top_1_recall * 100:.2f}%  |  Top-5: {report.s2_top_k.top_5_recall * 100:.2f}%")
    print(f"  Source 3 Top-1 Recall:  {report.s3_top_k.top_1_recall * 100:.2f}%  |  Top-5: {report.s3_top_k.top_5_recall * 100:.2f}%")
    print("-" * 70)
    print("DIAGNOSTIC THRESHOLD SWEEP (F0.5 DIAGNOSTIC):")
    print(f"{'Threshold':<11} {'Pred Pairs':<12} {'True Pos':<10} {'Precision':<11} {'Recall':<10} {'F0.5':<8}")
    for tm in report.diagnostic_thresholds:
        print(f"{tm.threshold:<11.2f} {tm.predicted_pairs:<12,d} {tm.true_positives:<10,d} {tm.precision*100:<10.2f}% {tm.recall*100:<9.2f}% {tm.f0_5:.4f}")
    print("-" * 70)
    print("ERROR ANALYSIS SUMMARY:")
    print(f"  False Negatives (True Pairs with Score < 0.60): {report.false_negative_count:,}")
    print(f"  False Positives (Non-Matches with Score >= 0.70): {report.false_positive_count:,}")
    print("  Error Pattern Breakdown:")
    for pat, count in sorted(report.error_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {pat}: {count:,}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
