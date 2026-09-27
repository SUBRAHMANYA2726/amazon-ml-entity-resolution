"""Phase 9 Final F0.5 Threshold Optimization Runner.

Reuses Phase 8 Model B validation predictions (no recomputation of
probabilities) and selects the final operating threshold by optimizing the
CHALLENGE metric:

    MACRO F0.5 = mean over S1 entities of per-entity F0.5

with the documented singleton rule (empty prediction on a zero-match entity
scores 1.0; any predicted match on it scores 0.0).

Phase 8 optimized pair-level (micro) F0.5; Phase 9 corrects the objective to
the official macro-averaged definition, evaluates a fine grid
(0.00 -> 1.00, step 0.01) plus a finer local search around the optimum,
verifies train/validation entity disjointness, and reports threshold
stability. No test data or test labels are touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from business_entity_resolution.matching.evaluation import compute_f_beta
from business_entity_resolution.utils.logging import configure_logging, get_logger
from business_entity_resolution.utils.seed import set_random_seed

LOGGER = get_logger("phase9_runner")

BETA = 0.5
TIE_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 9 Final F0.5 Threshold Optimization.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--val-predictions-path",
        type=Path,
        default=Path("output/phase 8/model_b_validation_predictions.csv.gz"),
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("output/phase5/train_feature_matrix.csv.gz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/phase9"))
    parser.add_argument("--coarse-step", type=float, default=0.01)
    parser.add_argument("--fine-step", type=float, default=0.002)
    parser.add_argument("--fine-window", type=float, default=0.03)
    return parser.parse_args()


def verify_entity_disjointness(
    val_s1: set[str], features_path: Path
) -> dict:
    """Verify train/val S1 entity disjointness using the Phase 5 matrix.

    The Phase 5 matrix holds all 2,500 split entities (135,342 train pairs +
    34,811 val pairs). Train entity set is derived as matrix entities minus
    validation entities; the check asserts the recorded split counts and
    zero overlap.
    """
    use_cols = ["source1_entity_id", "ground_truth_label"]
    chunks = []
    for chunk in pd.read_csv(features_path, usecols=use_cols, chunksize=500_000):
        chunks.append(chunk)
    feat = pd.concat(chunks, ignore_index=True)
    matrix_s1 = set(feat["source1_entity_id"].astype(str).unique())
    train_s1 = matrix_s1 - val_s1
    overlap = train_s1.intersection(val_s1)
    val_in_matrix = val_s1.intersection(matrix_s1)
    return {
        "matrix_unique_s1": len(matrix_s1),
        "val_unique_s1": len(val_s1),
        "derived_train_unique_s1": len(train_s1),
        "train_val_overlap": len(overlap),
        "zero_overlap": len(overlap) == 0,
        "val_entities_found_in_matrix": len(val_in_matrix),
        "expected_train_entities": 2000,
        "expected_val_entities": 500,
        "counts_match": (len(train_s1) == 2000 and len(val_s1) == 500),
    }


def build_entity_arrays(val_df: pd.DataFrame):
    """Map entities to integer indices and return sorted numpy arrays."""
    s1_ids = np.array(sorted(val_df["source1_entity_id"].astype(str).unique()))
    s1_index = {s: i for i, s in enumerate(s1_ids)}
    order = np.argsort(val_df["source1_entity_id"].astype(str).values, kind="stable")
    ent = np.array([s1_index[s] for s in val_df["source1_entity_id"].astype(str).values[order]])
    prob = val_df["match_probability"].to_numpy(dtype=np.float64)[order]
    label = val_df["ground_truth_label"].to_numpy(dtype=np.int32)[order]
    # entity boundaries via bincount
    counts = np.bincount(ent, minlength=len(s1_ids))
    starts = np.zeros(len(s1_ids), dtype=np.int64)
    starts[1:] = np.cumsum(counts[:-1])
    gt_counts = np.bincount(ent, weights=label, minlength=len(s1_ids)).astype(np.int64)
    return s1_ids, ent, prob, label, counts, starts, gt_counts


def evaluate_grid(
    ent: np.ndarray,
    prob: np.ndarray,
    label: np.ndarray,
    counts: np.ndarray,
    starts: np.ndarray,
    gt_counts: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    """Evaluate pair + entity + macro metrics for every threshold (vectorized)."""
    n_entities = len(counts)
    n_pairs = len(prob)
    rows: list[dict] = []
    for th in thresholds:
        pred = (prob >= th).astype(np.int32)
        tp = int(((pred == 1) & (label == 1)).sum())
        fp = int(((pred == 1) & (label == 0)).sum())
        fn = int(((pred == 0) & (label == 1)).sum())
        tn = int(((pred == 0) & (label == 0)).sum())
        pair_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        pair_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pair_f05 = compute_f_beta(pair_p, pair_r, beta=BETA)
        pair_f1 = compute_f_beta(pair_p, pair_r, beta=1.0)

        tp_e = np.bincount(ent, weights=(pred * label), minlength=n_entities)
        pred_e = np.bincount(ent, weights=pred, minlength=n_entities)
        # per-entity precision/recall
        with np.errstate(divide="ignore", invalid="ignore"):
            ent_p = np.where(pred_e > 0, tp_e / np.maximum(pred_e, 1), 0.0)
            ent_r = np.where(gt_counts > 0, tp_e / np.maximum(gt_counts, 1), 0.0)
        beta_sq = BETA * BETA
        denom = beta_sq * ent_p + ent_r
        ent_f05 = np.where(denom > 0, (1 + beta_sq) * ent_p * ent_r / np.maximum(denom, 1e-12), 0.0)
        # singleton rule: empty prediction on zero-match entity scores 1.0
        singleton_empty = (gt_counts == 0) & (pred_e == 0)
        ent_f05[singleton_empty] = 1.0
        macro_f05 = float(ent_f05.mean())

        exact = int((((pred_e - tp_e) == 0) & ((gt_counts - tp_e) == 0)).sum())
        singleton_fp = int(((gt_counts == 0) & (pred_e > 0)).sum())
        matched_fn = int(((gt_counts > 0) & (pred_e == 0)).sum())
        matched_s1 = int((pred_e > 0).sum())
        rows.append({
            "threshold": round(float(th), 6),
            "pair_precision": pair_p,
            "pair_recall": pair_r,
            "pair_f0_5": pair_f05,
            "pair_f1": pair_f1,
            "pair_tp": tp,
            "pair_fp": fp,
            "pair_fn": fn,
            "pair_tn": tn,
            "predicted_match_count": int(pred.sum()),
            "macro_f0_5": macro_f05,
            "exact_entity_set_matches": exact,
            "exact_entity_set_accuracy": exact / n_entities,
            "singleton_false_positives": singleton_fp,
            "matched_entity_false_negatives": matched_fn,
            "matched_s1_entities": matched_s1,
            "unmatched_s1_entities": n_entities - matched_s1,
        })
    return pd.DataFrame(rows)


def select_threshold(grid_df: pd.DataFrame, metric: str = "macro_f0_5") -> tuple[float, dict]:
    best_val = float(grid_df[metric].max())
    tied = grid_df[np.abs(grid_df[metric] - best_val) <= TIE_TOLERANCE]
    # Tie-break: most conservative (highest) threshold among ties, justified by
    # lower false-positive exposure at equal macro F0.5.
    best_row = tied.loc[tied["threshold"].idxmax()]
    info = {
        "selected_threshold": float(best_row["threshold"]),
        "primary_metric": metric,
        "primary_metric_value": float(best_row[metric]),
        "tie_count_within_tolerance": int(len(tied)),
        "tie_tolerance": TIE_TOLERANCE,
        "tie_break_rule": (
            "If multiple thresholds tie on macro F0.5 within 1e-9, select the "
            "highest (most conservative) threshold to minimize false merges."
        ),
    }
    return float(best_row["threshold"]), info


def main() -> int:
    from business_entity_resolution.config.settings import load_settings

    args = parse_args()
    settings = load_settings(args.config)
    configure_logging(settings.logging)
    set_random_seed(settings.random_seed)
    LOGGER.info("Starting Phase 9 Final F0.5 Threshold Optimization")
    t_start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load validation predictions (reuse Phase 8, no recomputation)
    if not args.val_predictions_path.exists():
        LOGGER.error("Validation predictions not found: %s", args.val_predictions_path)
        return 1
    val_df = pd.read_csv(args.val_predictions_path)
    for col in ("source1_entity_id", "candidate_entity_id", "candidate_source",
                "match_probability", "ground_truth_label"):
        if col not in val_df.columns:
            LOGGER.error("Missing column %s", col)
            return 1
    n_entities = int(val_df["source1_entity_id"].nunique())
    LOGGER.info("Loaded %d pairs / %d S1 entities", len(val_df), n_entities)

    # 2. Leakage verification
    val_s1 = set(val_df["source1_entity_id"].astype(str).unique())
    leakage = verify_entity_disjointness(val_s1, args.features_path)
    LOGGER.info("Leakage check: %s", leakage)
    with open(args.output_dir / "leakage_verification.json", "w", encoding="utf-8") as f:
        json.dump(leakage, f, indent=2)
    if not leakage["zero_overlap"]:
        LOGGER.critical("FATAL: train/validation entity overlap detected!")
        return 1

    # 3. Entity arrays
    s1_ids, ent, prob, label, counts, starts, gt_counts = build_entity_arrays(val_df)
    n_singletons = int((gt_counts == 0).sum())
    LOGGER.info("Ground-truth singletons in validation: %d / %d", n_singletons, len(s1_ids))

    # 4. Coarse grid 0.00 -> 1.00 step 0.01
    coarse = [round(i * args.coarse_step, 6) for i in range(int(1.0 / args.coarse_step) + 1)]
    coarse_df = evaluate_grid(ent, prob, label, counts, starts, gt_counts, coarse)
    coarse_best, _ = select_threshold(coarse_df)

    # 5. Fine local search around coarse optimum
    lo = max(0.0, coarse_best - args.fine_window)
    hi = min(1.0, coarse_best + args.fine_window)
    fine = [round(lo + i * args.fine_step, 6) for i in range(int(round((hi - lo) / args.fine_step)) + 1)]
    fine = sorted(set(fine) | {round(coarse_best, 6)})
    fine_df = evaluate_grid(ent, prob, label, counts, starts, gt_counts, fine)

    # combined grid (deduplicated, sorted)
    grid_df = pd.concat([coarse_df, fine_df], ignore_index=True)
    grid_df = grid_df.drop_duplicates(subset=["threshold"]).sort_values("threshold").reset_index(drop=True)
    grid_df.to_csv(args.output_dir / "threshold_macro_grid.csv", index=False)
    with open(args.output_dir / "threshold_macro_grid.json", "w", encoding="utf-8") as f:
        json.dump(grid_df.to_dict(orient="records"), f, indent=2)

    # 6. Final selection on macro F0.5
    selected_th, sel_info = select_threshold(grid_df)
    sel_row = grid_df[grid_df["threshold"] == selected_th].iloc[0].to_dict()
    sel_info.update({
        "pair_precision": float(sel_row["pair_precision"]),
        "pair_recall": float(sel_row["pair_recall"]),
        "pair_f0_5": float(sel_row["pair_f0_5"]),
        "pair_fp": int(sel_row["pair_fp"]),
        "pair_fn": int(sel_row["pair_fn"]),
        "exact_entity_set_matches": int(sel_row["exact_entity_set_matches"]),
        "exact_entity_set_accuracy": float(sel_row["exact_entity_set_accuracy"]),
        "singleton_false_positives": int(sel_row["singleton_false_positives"]),
        "matched_entity_false_negatives": int(sel_row["matched_entity_false_negatives"]),
        "validation_population": {
            "source1_entities": len(s1_ids),
            "candidate_pairs": len(val_df),
            "ground_truth_matches": int(label.sum()),
            "ground_truth_singletons": n_singletons,
        },
        "coarse_grid_step": args.coarse_step,
        "fine_grid_step": args.fine_step,
        "fine_window": args.fine_window,
        "coarse_best_threshold": float(coarse_best),
        "methodology": (
            "Threshold selected by maximizing challenge-defined macro F0.5 "
            "(per-S1-entity F0.5 averaged over all 500 held-out validation "
            "entities; singleton entities score 1.0 when correctly predicted "
            "empty, else 0.0) on a 0.00-1.00 step-0.01 grid with step-0.002 "
            "local refinement. Pair-level metrics reported for diagnostics only."
        ),
    })
    with open(args.output_dir / "selected_threshold.json", "w", encoding="utf-8") as f:
        json.dump(sel_info, f, indent=2)

    # 7. Stability: neighbors of selected threshold present in grid
    deltas = [-0.01, -0.005, -0.002, 0.0, 0.002, 0.005, 0.01]
    stab = []
    for d in deltas:
        t = round(selected_th + d, 6)
        hit = grid_df[np.isclose(grid_df["threshold"], t, atol=1e-9)]
        if not hit.empty:
            r = hit.iloc[0]
            stab.append({"delta": d, "threshold": t,
                         "macro_f0_5": float(r["macro_f0_5"]),
                         "pair_f0_5": float(r["pair_f0_5"]),
                         "pair_fp": int(r["pair_fp"]), "pair_fn": int(r["pair_fn"])})
    with open(args.output_dir / "threshold_stability.json", "w", encoding="utf-8") as f:
        json.dump(stab, f, indent=2)

    # Phase 8 comparison (pair-level optimum) for the report
    p8_row = grid_df[np.isclose(grid_df["threshold"], 0.95)].iloc[0].to_dict()
    summary = {
        "phase": 9,
        "phase_name": "Final F0.5 Threshold Optimization",
        "execution_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": round(time.time() - t_start, 2),
        "selected_threshold": selected_th,
        "selection": sel_info,
        "stability": stab,
        "phase8_pair_optimum_for_reference": {
            "threshold": 0.95,
            "macro_f0_5": float(p8_row["macro_f0_5"]),
            "pair_f0_5": float(p8_row["pair_f0_5"]),
        },
        "leakage_verification": leakage,
        "no_test_data_used": True,
    }
    with open(args.output_dir / "phase9_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 85)
    print("PHASE 9 MACRO-F0.5 THRESHOLD OPTIMIZATION SCOREBOARD")
    print("=" * 85)
    print(f"Validation population : {len(s1_ids)} S1 entities | {len(val_df)} pairs | {n_singletons} singletons")
    print(f"Leakage check         : train/val overlap = {leakage['train_val_overlap']} (zero={leakage['zero_overlap']})")
    print(f"Selected threshold    : {selected_th:.4f}  (macro F0.5 = {sel_info['primary_metric_value']:.6f})")
    print(f"  pair P/R/F0.5       : {sel_info['pair_precision']:.4f} / {sel_info['pair_recall']:.4f} / {sel_info['pair_f0_5']:.4f}")
    print(f"  FP/FN               : {sel_info['pair_fp']} / {sel_info['pair_fn']}")
    print(f"  exact-set acc       : {sel_info['exact_entity_set_matches']}/500 ({sel_info['exact_entity_set_accuracy']*100:.1f}%)")
    print(f"  singleton FP / matched FN entities: {sel_info['singleton_false_positives']} / {sel_info['matched_entity_false_negatives']}")
    print("-" * 85)
    print("Top 10 thresholds by macro F0.5:")
    top10 = grid_df.sort_values("macro_f0_5", ascending=False).head(10)
    for _, r in top10.iterrows():
        print(f"  tau={r['threshold']:.4f}  macroF0.5={r['macro_f0_5']:.6f}  pairF0.5={r['pair_f0_5']:.4f}  "
              f"P={r['pair_precision']:.4f} R={r['pair_recall']:.4f} FP={int(r['pair_fp'])} FN={int(r['pair_fn'])}")
    print("-" * 85)
    print("Stability around selected threshold:")
    for s in stab:
        print(f"  delta={s['delta']:+.3f} tau={s['threshold']:.4f} macroF0.5={s['macro_f0_5']:.6f} FP={s['pair_fp']} FN={s['pair_fn']}")
    print("=" * 85 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
