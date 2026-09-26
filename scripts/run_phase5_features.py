"""Phase 5 Pairwise Feature Engineering and Diagnostics Runner.

Takes Phase 3 candidates and Phase 4 baseline rankings.
Computes 84 deterministic pairwise features across all 9 feature groups:
A: Name Features
B: Address Features
C: Structured Fields
D: Field Missingness
E: Cross-Field Consistency
F: Country Features (Open-Set Generic)
G: Candidate Blocking Provenance
H: Candidate Source Indicators
I: Phase 4 Baseline Signals

Attaches ground_truth_label ONLY for training artifacts (strictly isolated from feature columns).
Performs data quality validation, redundancy analysis, and statistical separation diagnostics.
Exports artifacts to output/phase5/.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from business_entity_resolution.blocking.evaluation import parse_ground_truth
from business_entity_resolution.config.settings import load_settings
from business_entity_resolution.features import (
    FEATURE_NAMES,
    PairwiseFeatureGenerator,
    compute_redundancy_diagnostics,
    compute_separation_diagnostics,
    export_feature_metadata,
    summarize_class_distribution,
    validate_feature_matrix,
)
from business_entity_resolution.ingestion.discovery import discover_dataset
from business_entity_resolution.normalization.pipeline import NormalizationPipeline
from business_entity_resolution.utils.logging import configure_logging, get_logger

LOGGER = get_logger("phase5_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 5 Pairwise Feature Engineering.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Config file path")
    parser.add_argument(
        "--ranked-candidates-path",
        type=Path,
        default=Path("output/phase4/baseline_ranked_candidates.csv"),
        help="Path to Phase 4 baseline ranked candidate pairs CSV",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/phase5"), help="Output directory")
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
    LOGGER.info("Starting Phase 5 Pairwise Feature Engineering")
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

    # 3. Load Phase 4 Ranked Candidate Pairs
    if not args.ranked_candidates_path.exists():
        LOGGER.error("Ranked candidates not found at %s. Run Phase 4 first!", args.ranked_candidates_path)
        return 1

    LOGGER.info("Loading Phase 4 ranked pairs from %s...", args.ranked_candidates_path)
    pairs_df = pd.read_csv(args.ranked_candidates_path)
    LOGGER.info("Loaded %d candidate pairs across %d anchors", len(pairs_df), pairs_df["source1_entity_id"].nunique())

    # 4. Retrieve and normalize raw records
    needed_s1_ids = set(pairs_df["source1_entity_id"])
    cand_ids = set(pairs_df["candidate_entity_id"])
    needed_s2_ids = {cid for cid in cand_ids if str(cid).startswith("S2")}
    needed_s3_ids = {cid for cid in cand_ids if str(cid).startswith("S3")}

    norm_pipe = NormalizationPipeline(settings.normalization)

    # S1 anchors
    t0 = time.time()
    s1_raw = load_candidate_records(s1_file.path, needed_s1_ids, sep=settings.dataset.delimiter)
    s1_norm = norm_pipe.normalize_frame(s1_raw)
    s1_lookup = {row["entity_id"]: row.to_dict() for _, row in s1_norm.iterrows()}
    LOGGER.info("Normalized %d S1 anchors in %.2fs", len(s1_lookup), time.time() - t0)

    # S2 candidates
    t0 = time.time()
    s2_raw = load_candidate_records(s2_file.path, needed_s2_ids, sep=settings.dataset.delimiter)
    s2_norm = norm_pipe.normalize_frame(s2_raw)
    cand_lookup = {row["entity_id"]: row.to_dict() for _, row in s2_norm.iterrows()}
    LOGGER.info("Normalized %d S2 records in %.2fs", len(s2_norm), time.time() - t0)

    # S3 candidates
    t0 = time.time()
    s3_raw = load_candidate_records(s3_file.path, needed_s3_ids, sep=settings.dataset.delimiter)
    s3_norm = norm_pipe.normalize_frame(s3_raw)
    for _, row in s3_norm.iterrows():
        cand_lookup[row["entity_id"]] = row.to_dict()
    LOGGER.info("Normalized %d S3 records in %.2fs. Total candidate lookup pool: %d", len(s3_norm), time.time() - t0, len(cand_lookup))

    # 5. Execute Pairwise Feature Generation
    LOGGER.info("Generating complete pairwise feature matrix...")
    generator = PairwiseFeatureGenerator()
    t0 = time.time()
    feat_df, feature_cols = generator.generate_features(
        candidate_pairs_df=pairs_df,
        s1_records=s1_lookup,
        cand_records=cand_lookup,
        baseline_scored_df=pairs_df,
        ground_truth=ground_truth,
    )
    t_feat = time.time() - t0
    LOGGER.info("Feature engineering completed in %.2fs (%.0f pairs/sec)", t_feat, len(feat_df) / max(t_feat, 0.001))

    # 6. Feature Validation & Diagnostics
    LOGGER.info("Running feature quality validation...")
    quality_report = validate_feature_matrix(feat_df, feature_cols)

    LOGGER.info("Running redundancy and collinearity diagnostics...")
    redundancy_report = compute_redundancy_diagnostics(feat_df, feature_cols, corr_threshold=0.95)

    LOGGER.info("Running statistical separation diagnostics...")
    separation_diags = compute_separation_diagnostics(feat_df, feature_cols, label_col="ground_truth_label")

    LOGGER.info("Summarizing class distribution...")
    class_dist = summarize_class_distribution(feat_df, label_col="ground_truth_label")

    # 7. Export Outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Export Feature Metadata JSON
    meta_path = args.output_dir / "feature_metadata.json"
    export_feature_metadata(meta_path)
    LOGGER.info("Exported feature metadata schema to %s", meta_path)

    # Export Quality Report JSON
    quality_json = args.output_dir / "feature_quality_report.json"
    with open(quality_json, "w", encoding="utf-8") as f:
        json.dump(quality_report.to_dict(), f, indent=2)
    LOGGER.info("Exported feature quality report to %s", quality_json)

    # Export Redundancy Report JSON
    redundancy_json = args.output_dir / "feature_redundancy_report.json"
    with open(redundancy_json, "w", encoding="utf-8") as f:
        json.dump(redundancy_report, f, indent=2)
    LOGGER.info("Exported feature redundancy report to %s", redundancy_json)

    # Export Separation Diagnostics CSV & JSON
    sep_df = pd.DataFrame(separation_diags)
    sep_csv = args.output_dir / "feature_separation_diagnostics.csv"
    sep_df.to_csv(sep_csv, index=False)
    sep_json = args.output_dir / "feature_separation_diagnostics.json"
    with open(sep_json, "w", encoding="utf-8") as f:
        json.dump(separation_diags, f, indent=2)
    LOGGER.info("Exported feature separation diagnostics to %s and %s", sep_csv, sep_json)

    # Export Class Distribution JSON
    dist_json = args.output_dir / "class_distribution.json"
    with open(dist_json, "w", encoding="utf-8") as f:
        json.dump(class_dist, f, indent=2)
    LOGGER.info("Exported class distribution to %s", dist_json)

    # Export Feature Matrix CSV (compressed gzip to save disk, plus leading 1000 sample CSV)
    matrix_csv_gz = args.output_dir / "train_feature_matrix.csv.gz"
    LOGGER.info("Exporting compressed feature matrix to %s...", matrix_csv_gz)
    feat_df.to_csv(matrix_csv_gz, index=False, compression="gzip")
    LOGGER.info("Exported compressed feature matrix (%d rows x %d cols) to %s", len(feat_df), len(feat_df.columns), matrix_csv_gz)

    matrix_sample_csv = args.output_dir / "train_feature_matrix_sample.csv"
    feat_df.head(1000).to_csv(matrix_sample_csv, index=False)
    LOGGER.info("Exported 1,000 row feature matrix sample to %s", matrix_sample_csv)

    # Print Scoreboard
    print("\n" + "=" * 75)
    print("PHASE 5 PAIRWISE FEATURE ENGINEERING SCOREBOARD")
    print("=" * 75)
    print(f"Total Candidate Pairs Processed:    {class_dist['total_candidate_pairs']:,}")
    print(f"Total Positive Match Pairs (1):     {class_dist['positive_pairs']:,} ({class_dist['positive_percentage']:.2f}%)")
    print(f"Total Negative Pairs (0):           {class_dist['negative_pairs']:,} ({class_dist['negative_percentage']:.2f}%)")
    print("-" * 75)
    print("SOURCE-SPECIFIC CLASS DISTRIBUTION:")
    print(f"  Source 2: {class_dist['s2_distribution']['total_pairs']:,} pairs | {class_dist['s2_distribution']['positives']:,} positives ({class_dist['s2_distribution']['positive_rate']:.2f}%) | {class_dist['s2_distribution']['negatives']:,} negatives")
    print(f"  Source 3: {class_dist['s3_distribution']['total_pairs']:,} pairs | {class_dist['s3_distribution']['positives']:,} positives ({class_dist['s3_distribution']['positive_rate']:.2f}%) | {class_dist['s3_distribution']['negatives']:,} negatives")
    print("-" * 75)
    print("FEATURE MATRIX VALIDATION:")
    print(f"  Total Features Defined:            {quality_report.total_features}")
    print(f"  NaN Feature Columns:               {len(quality_report.nan_features)}")
    print(f"  Inf Feature Columns:               {len(quality_report.inf_features)}")
    print(f"  Invalid Numeric Ranges:            {len(quality_report.invalid_range_features)}")
    print(f"  Constant Feature Columns:          {len(quality_report.constant_features)} ({', '.join(quality_report.constant_features) if quality_report.constant_features else 'None'})")
    print(f"  Collinear Pairs (|r| >= 0.95):     {redundancy_report['total_collinear_pairs']}")
    print("-" * 75)
    print("TOP 15 FEATURES BY STATISTICAL SEPARATION (COHEN'S D):")
    print(f"{'Feature Name':<34} {'Group':<14} {'Cohen d':<10} {'Pos Mean':<10} {'Neg Mean'}")
    for item in separation_diags[:15]:
        print(f"{item['feature_name']:<34} {item['feature_group']:<14} {item['cohens_d_separation']:<10.3f} {item['pos_mean']:<10.3f} {item['neg_mean']:.3f}")
    print("=" * 75 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
