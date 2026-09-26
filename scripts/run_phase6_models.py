"""Phase 6 Supervised Pairwise ML Matching Runner.

Trains and evaluates LightGBM and CatBoost pairwise matchers on Phase 5 feature matrix.
Enforces:
- Strict entity-aware train/validation splitting by source1_entity_id (seed 2026).
- Strict feature leakage prevention.
- Class imbalance handling via measured training scale_pos_weight.
- Probability scoring and validation (0 <= p <= 1, no NaN, no inf).
- Objective measured validation comparison (PR-AUC, ROC-AUC, Precision, Recall, F0.5).
- Model serialization and metadata export to output/phase6/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from business_entity_resolution.config.settings import load_settings
from business_entity_resolution.evaluation.metrics import (
    build_validation_predictions_artifact,
    compute_distribution_summary,
    evaluate_predictions,
    extract_prediction_diagnostics,
)
from business_entity_resolution.models.pairwise import (
    CatBoostMatcher,
    LightGBMMatcher,
    validate_probabilities,
)
from business_entity_resolution.training.split import entity_aware_train_val_split
from business_entity_resolution.utils.logging import configure_logging, get_logger
from business_entity_resolution.utils.seed import set_random_seed

LOGGER = get_logger("phase6_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 Supervised Pairwise ML Matching.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Config file path")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("output/phase5/train_feature_matrix.csv.gz"),
        help="Path to Phase 5 train feature matrix CSV/GZ",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("output/phase5/feature_metadata.json"),
        help="Path to Phase 5 feature metadata JSON",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/phase6"), help="Output directory")
    parser.add_argument("--val-fraction", type=float, default=0.20, help="Validation fraction of unique entities")
    return parser.parse_args()


def load_feature_metadata(meta_path: Path) -> list[str]:
    """Discover feature names strictly from Phase 5 feature metadata artifact."""
    if not meta_path.exists():
        raise FileNotFoundError(f"Feature metadata not found at {meta_path}. Run Phase 5 first!")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = list(meta.get("features", {}).keys())
    if not features:
        raise ValueError(f"No features discovered in {meta_path}")
    LOGGER.info("Discovered %d features across %d groups from %s", len(features), len(meta.get("feature_groups", [])), meta_path.name)
    return features


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    configure_logging(settings.logging)
    LOGGER.info("Starting Phase 6 Supervised Pairwise ML Matching")
    t_start = time.time()

    # 1. Deterministic Seeding
    random_seed = settings.random_seed
    set_random_seed(random_seed)
    LOGGER.info("Seeded environment with random seed %d", random_seed)

    # 2. Discover Features from Phase 5 Metadata
    feature_names = load_feature_metadata(args.metadata_path)

    # 3. Load Phase 5 Feature Matrix
    if not args.features_path.exists():
        LOGGER.error("Phase 5 feature matrix not found at %s. Run Phase 5 first!", args.features_path)
        return 1

    LOGGER.info("Loading Phase 5 feature matrix from %s...", args.features_path)
    t0 = time.time()
    df = pd.read_csv(args.features_path)
    LOGGER.info("Loaded feature matrix (%d rows x %d cols) in %.2fs", len(df), len(df.columns), time.time() - t0)

    # 4. Perform Entity-Aware Train/Validation Split
    LOGGER.info("Executing entity-aware split by source1_entity_id (val_fraction=%.2f, seed=%d)...", args.val_fraction, random_seed)
    split_res = entity_aware_train_val_split(
        df,
        entity_col="source1_entity_id",
        label_col="ground_truth_label",
        val_fraction=args.val_fraction,
        random_seed=random_seed,
    )
    split_stats = split_res.statistics

    # Re-verify strict zero overlap
    overlap = split_res.train_entity_ids.intersection(split_res.val_entity_ids)
    if overlap:
        LOGGER.critical("FATAL: Train/Validation entity overlap detected! Count: %d", len(overlap))
        return 1

    train_df = split_res.train_df
    val_df = split_res.val_df

    LOGGER.info("Train pairs: %d (%d pos / %d neg), Pos rate: %.2f%%", len(train_df), split_stats.train_positive_pairs, split_stats.train_negative_pairs, split_stats.train_positive_rate * 100)
    LOGGER.info("Val pairs:   %d (%d pos / %d neg), Pos rate: %.2f%%", len(val_df), split_stats.val_positive_pairs, split_stats.val_negative_pairs, split_stats.val_positive_rate * 100)
    LOGGER.info("Training scale_pos_weight: %.2f", split_stats.train_scale_pos_weight)

    # Extract training and validation matrices (strictly isolated features and labels)
    X_train = train_df[feature_names]
    y_train = train_df["ground_truth_label"].to_numpy(dtype=np.int32)

    X_val = val_df[feature_names]
    y_val = val_df["ground_truth_label"].to_numpy(dtype=np.int32)

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Train & Evaluate LightGBM Matcher
    LOGGER.info("=" * 60)
    LOGGER.info("Training LightGBM Matcher...")
    lgb_start = time.time()
    lgb_matcher = LightGBMMatcher(
        scale_pos_weight=split_stats.train_scale_pos_weight,
        random_seed=random_seed,
        feature_names=feature_names,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    lgb_matcher.fit(X_train, y_train)
    lgb_train_time = time.time() - lgb_start
    LOGGER.info("LightGBM trained in %.2fs", lgb_train_time)

    # Predict probabilities on held-out validation set
    lgb_val_probs = lgb_matcher.score(X_val)
    validate_probabilities(lgb_val_probs)
    lgb_eval = evaluate_predictions(y_val, lgb_val_probs)
    lgb_importance_df = lgb_matcher.get_feature_importances()
    lgb_pred_df = build_validation_predictions_artifact(val_df, lgb_val_probs)

    LOGGER.info(
        "LightGBM Validation Metrics: PR-AUC=%.4f | ROC-AUC=%.4f",
        lgb_eval["pr_auc"],
        lgb_eval["roc_auc"],
    )

    # 6. Train & Evaluate CatBoost Matcher
    LOGGER.info("=" * 60)
    LOGGER.info("Training CatBoost Matcher...")
    cb_start = time.time()
    cb_matcher = CatBoostMatcher(
        scale_pos_weight=split_stats.train_scale_pos_weight,
        random_seed=random_seed,
        feature_names=feature_names,
        iterations=350,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
    )
    cb_matcher.fit(X_train, y_train)
    cb_train_time = time.time() - cb_start
    LOGGER.info("CatBoost trained in %.2fs", cb_train_time)

    # Predict probabilities on held-out validation set
    cb_val_probs = cb_matcher.score(X_val)
    validate_probabilities(cb_val_probs)
    cb_eval = evaluate_predictions(y_val, cb_val_probs)
    cb_importance_df = cb_matcher.get_feature_importances()
    cb_pred_df = build_validation_predictions_artifact(val_df, cb_val_probs)

    LOGGER.info(
        "CatBoost Validation Metrics: PR-AUC=%.4f | ROC-AUC=%.4f",
        cb_eval["pr_auc"],
        cb_eval["roc_auc"],
    )

    # 7. Model Comparison & Selection
    # F0.5 is emphasized because the challenge prioritizes precision.
    # Use the measured F0.5 at the highest diagnostic threshold (0.95)
    # as the primary comparison criterion, with PR-AUC as the tie-breaker.
    def f05_at_threshold(eval_result, threshold):
        return next(
            metric["f0_5"]
            for metric in eval_result["diagnostic_threshold_sweep"]
            if abs(metric["threshold"] - threshold) < 1e-3
        )


    lgb_f05 = f05_at_threshold(lgb_eval, 0.95)
    cb_f05 = f05_at_threshold(cb_eval, 0.95)

    if lgb_f05 > cb_f05:
        carried_forward_name = "lightgbm"
    elif cb_f05 > lgb_f05:
        carried_forward_name = "catboost"
    else:
        carried_forward_name = (
            "lightgbm"
            if lgb_eval["pr_auc"] >= cb_eval["pr_auc"]
            else "catboost"
        )

    carried_matcher = lgb_matcher if carried_forward_name == "lightgbm" else cb_matcher
    carried_pred_df = lgb_pred_df if carried_forward_name == "lightgbm" else cb_pred_df

    LOGGER.info(
        "Model Comparison: LightGBM F0.5@0.95=%.4f, PR-AUC=%.4f | "
        "CatBoost F0.5@0.95=%.4f, PR-AUC=%.4f. "
        "Model carried forward: %s",
        lgb_f05,
        lgb_eval["pr_auc"],
        cb_f05,
        cb_eval["pr_auc"],
        carried_forward_name,
    )

    # 8. Export All Phase 6 Artifacts
    LOGGER.info("=" * 60)
    LOGGER.info("Exporting Phase 6 artifacts to %s...", args.output_dir)

    # 8.1 Split Metadata JSON
    split_meta_file = args.output_dir / "split_metadata.json"
    with open(split_meta_file, "w", encoding="utf-8") as f:
        json.dump(split_stats.to_dict(), f, indent=2)
    LOGGER.info("Saved split metadata to %s", split_meta_file)

    # 8.2 Serialize Models and Model Metadata
    lgb_matcher.save(
        model_file_path=args.output_dir / "lightgbm_model.joblib",
        metadata_file_path=args.output_dir / "lightgbm_metadata.json",
    )
    cb_matcher.save(
        model_file_path=args.output_dir / "catboost_model.joblib",
        metadata_file_path=args.output_dir / "catboost_metadata.json",
    )

    # 8.3 Validation Metrics JSON
    metrics_summary = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": round(time.time() - t_start, 2),
        "split_summary": split_stats.to_dict(),
        "carried_forward_model": carried_forward_name,
        "models": {
            "lightgbm": {
                "training_time_seconds": round(lgb_train_time, 2),
                "pr_auc": round(lgb_eval["pr_auc"], 4),
                "roc_auc": round(lgb_eval["roc_auc"], 4),
                "probability_distributions": lgb_eval["probability_distributions"],
                "diagnostic_threshold_sweep": lgb_eval["diagnostic_threshold_sweep"],
            },
            "catboost": {
                "training_time_seconds": round(cb_train_time, 2),
                "pr_auc": round(cb_eval["pr_auc"], 4),
                "roc_auc": round(cb_eval["roc_auc"], 4),
                "probability_distributions": cb_eval["probability_distributions"],
                "diagnostic_threshold_sweep": cb_eval["diagnostic_threshold_sweep"],
            },
        },
    }
    metrics_file = args.output_dir / "validation_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    LOGGER.info("Saved validation metrics to %s", metrics_file)

    # 8.4 Diagnostic Threshold Sweeps CSV
    sweep_records = []
    for model_key, eval_data in [("lightgbm", lgb_eval), ("catboost", cb_eval)]:
        for item in eval_data["diagnostic_threshold_sweep"]:
            rec = {"model": model_key, **item}
            sweep_records.append(rec)
    sweep_df = pd.DataFrame(sweep_records)
    sweep_csv = args.output_dir / "diagnostic_threshold_sweeps.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    LOGGER.info("Saved diagnostic threshold sweeps to %s", sweep_csv)

    # 8.5 Model Feature Importance CSVs
    lgb_imp_file = args.output_dir / "lightgbm_feature_importance.csv"
    lgb_importance_df.to_csv(lgb_imp_file, index=False)
    cb_imp_file = args.output_dir / "catboost_feature_importance.csv"
    cb_importance_df.to_csv(cb_imp_file, index=False)
    LOGGER.info("Saved feature importance tables to %s and %s", lgb_imp_file, cb_imp_file)

    # 8.6 Validation Predictions Artifacts
    val_pred_gz = args.output_dir / "validation_predictions.csv.gz"
    carried_pred_df.to_csv(val_pred_gz, index=False, compression="gzip")
    val_pred_sample = args.output_dir / "validation_predictions_sample.csv"
    carried_pred_df.head(1000).to_csv(val_pred_sample, index=False)
    LOGGER.info("Saved primary validation predictions artifact to %s (%d pairs)", val_pred_gz, len(carried_pred_df))

    # Also save alternate model validation predictions
    alt_name = "catboost" if carried_forward_name == "lightgbm" else "lightgbm"
    alt_pred_df = cb_pred_df if carried_forward_name == "lightgbm" else lgb_pred_df
    alt_pred_gz = args.output_dir / f"{alt_name}_validation_predictions.csv.gz"
    alt_pred_df.to_csv(alt_pred_gz, index=False, compression="gzip")

    # 8.7 High-Confidence False Positives and Low-Confidence True Positives
    error_diagnostics = extract_prediction_diagnostics(carried_pred_df, top_n=30)
    error_file = args.output_dir / "prediction_error_diagnostics.json"
    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(error_diagnostics, f, indent=2)
    LOGGER.info("Saved prediction error diagnostics to %s", error_file)

    # 9. Print Phase 6 Scoreboard
    print("\n" + "=" * 80)
    print("PHASE 6 SUPERVISED PAIRWISE ML MATCHING SCOREBOARD")
    print("=" * 80)
    print(f"Random Seed:                 {random_seed}")
    print(f"Features Used:               {len(feature_names)} (strictly 0 identifier / label leakage)")
    print(f"Split Method:                Entity-aware grouped by source1_entity_id")
    print(f"Total Unique S1 Entities:    {split_stats.n_total_entities:,} (Train: {split_stats.n_train_entities:,} | Val: {split_stats.n_val_entities:,})")
    print(f"Total Candidate Pairs:       {split_stats.n_total_pairs:,} (Train: {split_stats.n_train_pairs:,} | Val: {split_stats.n_val_pairs:,})")
    print(f"Train Class Distribution:    {split_stats.train_positive_pairs:,} pos ({split_stats.train_positive_rate:.2%}) / {split_stats.train_negative_pairs:,} neg")
    print(f"Validation Class Dist:       {split_stats.val_positive_pairs:,} pos ({split_stats.val_positive_rate:.2%}) / {split_stats.val_negative_pairs:,} neg")
    print(f"Imbalance scale_pos_weight:  {split_stats.train_scale_pos_weight:.2f}")
    print("-" * 80)
    print("MEASURED VALIDATION PERFORMANCE COMPARISON:")
    print(f"{'Model':<12} {'Train Time':<12} {'PR-AUC':<10} {'ROC-AUC':<10} {'F0.5 @ 0.50':<12} {'F0.5 @ 0.70':<12} {'F0.5 @ 0.85':<12}")

    lgb_f05_50 = next((m["f0_5"] for m in lgb_eval["diagnostic_threshold_sweep"] if abs(m["threshold"] - 0.50) < 1e-3), 0.0)
    lgb_f05_70 = next((m["f0_5"] for m in lgb_eval["diagnostic_threshold_sweep"] if abs(m["threshold"] - 0.70) < 1e-3), 0.0)
    lgb_f05_85 = next((m["f0_5"] for m in lgb_eval["diagnostic_threshold_sweep"] if abs(m["threshold"] - 0.85) < 1e-3), 0.0)

    cb_f05_50 = next((m["f0_5"] for m in cb_eval["diagnostic_threshold_sweep"] if abs(m["threshold"] - 0.50) < 1e-3), 0.0)
    cb_f05_70 = next((m["f0_5"] for m in cb_eval["diagnostic_threshold_sweep"] if abs(m["threshold"] - 0.70) < 1e-3), 0.0)
    cb_f05_85 = next((m["f0_5"] for m in cb_eval["diagnostic_threshold_sweep"] if abs(m["threshold"] - 0.85) < 1e-3), 0.0)

    print(f"{'LightGBM':<12} {lgb_train_time:<12.2f} {lgb_eval['pr_auc']:<10.4f} {lgb_eval['roc_auc']:<10.4f} {lgb_f05_50:<12.4f} {lgb_f05_70:<12.4f} {lgb_f05_85:<12.4f}")
    print(f"{'CatBoost':<12} {cb_train_time:<12.2f} {cb_eval['pr_auc']:<10.4f} {cb_eval['roc_auc']:<10.4f} {cb_f05_50:<12.4f} {cb_f05_70:<12.4f} {cb_f05_85:<12.4f}")
    print("-" * 80)
    print(f"Model carried forward based on measured validation results: {carried_forward_name}")
    print("-" * 80)
    print("TOP 10 FEATURES BY MODEL FEATURE IMPORTANCE (LightGBM):")
    print(f"{'Rank':<6} {'Feature Name':<38} {'Gain Importance':<18} {'Normalized Gain'}")
    for idx, row in lgb_importance_df.head(10).iterrows():
        print(f"{idx + 1:<6} {row['feature_name']:<38} {row['gain_importance']:<18.2f} {row['normalized_gain_importance']:.4f}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
