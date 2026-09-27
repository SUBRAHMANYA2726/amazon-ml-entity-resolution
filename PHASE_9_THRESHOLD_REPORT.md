# Phase 9 — Final F0.5 Threshold Optimization Report

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution
**Phase:** 9 (Final F0.5 Threshold Optimization)
**Execution Timestamp:** 2026-09-27
**Carried-Forward Model:** Phase 7 Model B (Hard-Negative Retrained LightGBM, `output/phase 8/model_b_lightgbm.joblib`, 84 features)
**Runner:** `scripts/run_phase9_threshold_optimization.py`

---

## 1. Relationship to Phase 8 (what was reused vs what is new)

Phase 8 evaluated a 16-point threshold grid and selected τ = 0.95 by maximizing
**pair-level (micro) F0.5** (0.9785). The official challenge metric, however, is
**macro F0.5: per-Source-1-entity F0.5 averaged over all S1 entities**, with
singletons scoring 1.0 when correctly predicted empty and 0.0 otherwise
(`dataset/student_resource/README.md`, Evaluation Criteria).

Phase 9 therefore:

- **Reused:** Model B artifact, feature schema, validation probabilities
  (`output/phase 8/model_b_validation_predictions.csv.gz`, 34,811 pairs /
  500 held-out S1 entities). No probabilities were recomputed.
- **Verified:** train/val entity disjointness (see §3) and Phase 8's reported
  pair-level numbers at τ = 0.95 (reproduced exactly: P = 0.9856, R = 0.9512,
  pair F0.5 = 0.9785, FP = 24, FN = 84).
- **New work:** re-optimized the threshold against the **challenge-defined macro
  F0.5** on a fine grid (0.00 → 1.00, step 0.01, 101 points) plus a step-0.002
  local refinement (±0.03 around the coarse optimum). 125 unique thresholds
  evaluated in total.

## 2. Metric definition used for selection

For each S1 entity *e* with ground-truth set *G(e)* and predicted set *P(e)* at
threshold τ:

- Precision(e) = |P∩G| / |P| (0.0 when P non-empty and disjoint; undefined→1.0 case below)
- Recall(e) = |P∩G| / |G| (0.0 when P empty and G non-empty)
- F0.5(e) = 1.25·P·R / (0.25·P + R), 0.0 when P+R = 0
- **Singleton rule:** G(e) = ∅ and P(e) = ∅ → F0.5(e) = **1.0**; G(e) = ∅ and
  P(e) ≠ ∅ → F0.5(e) = **0.0**
- **Macro F0.5(τ)** = mean of F0.5(e) over all 500 validation entities

Pair-level precision/recall/F0.5 are reported for diagnostics only and were
**not** used for selection.

## 3. Leakage verification

| Check | Result |
|---|---|
| Validation S1 entities | 500 |
| Derived train S1 entities (Phase 5 matrix minus validation set) | 2000 |
| train_S1 ∩ validation_S1 | **0 (empty)** |
| Validation entities found in Phase 5 matrix | 500 / 500 |
| Expected 2000 / 500 split counts | match |
| Test data / test labels used | none |

Artifact: `output/phase9/leakage_verification.json`. No stop condition triggered.

## 4. Threshold search results (macro F0.5)

Full grid: `output/phase9/threshold_macro_grid.csv` (125 rows) + JSON.

| τ | Macro F0.5 | Pair P | Pair R | Pair F0.5 | FP | FN | Exact-set | Singleton FP |
|---|---|---|---|---|---|---|---|---|
| 0.85 | 0.967937 | 0.9778 | 0.9698 | 0.9762 | 38 | 52 | 421 | 3 |
| 0.87 | 0.970389 | 0.9794 | 0.9675 | 0.9770 | 35 | 56 | 422 | 2 |
| 0.90 | 0.969607 | 0.9805 | 0.9617 | 0.9766 | 33 | 66 | 418 | 2 |
| **0.922** | **0.970997** | 0.9828 | 0.9599 | 0.9781 | 29 | 69 | 420 | 2 |
| 0.93 | 0.970024 | 0.9827 | 0.9570 | 0.9775 | 29 | 74 | 416 | 2 |
| 0.95 (Phase 8) | 0.969820 | 0.9856 | 0.9512 | 0.9785 | 24 | 84 | 412 | 2 |
| 0.975 | — | 0.9913 | 0.9233 | 0.9769 | 14 | 132 | 386 | 1 |

Top of grid by macro F0.5 (measured):

- τ = 0.9220 → 0.970997 (selected)
- τ = 0.9200 → 0.970997 (tie)
- τ = 0.9180 → 0.970997 (tie)
- τ = 0.9240 → 0.970816
- τ = 0.9160 → 0.970721

The three tied thresholds (0.918/0.920/0.922) have **identical predictions**:
no validation pair scores in (0.918, 0.922], so FP/FN are equal (29/69).

## 5. Selected threshold

**Final operating threshold: τ = 0.922**

- Tie-breaking rule (documented in `selected_threshold.json`): among thresholds
  tied on macro F0.5 within 1e-9, select the **highest (most conservative)** to
  minimize false-merge exposure. Applied to the {0.918, 0.920, 0.922} plateau.
- Validation macro F0.5 at τ = 0.922: **0.970997**
- Pair P/R/F0.5: 0.9828 / 0.9599 / 0.9781; FP = 29, FN = 69
- Exact entity-set accuracy: 420/500 (84.0%)
- Singleton false positives: 2 entities; matched-entity false negatives: 2 entities
- Predicted match count: 1682 pairs across 473 matched / 27 unmatched S1 entities
  (ground truth: 473 matched / 27 singletons — exact split reproduced)

### Why not 0.95 (Phase 8)?

τ = 0.95 maximizes **pair-level** F0.5 (0.9785 vs 0.9781) but scores lower on the
challenge metric: macro F0.5 0.969820 vs 0.970997 (**−0.00118**), with 15 more
false negatives (84 vs 69) and 8 fewer exact entity-set matches (412 vs 420).
Under macro averaging, the recall loss at 0.95 outweighs its precision gain.

## 6. Threshold stability

| Δ | τ | Macro F0.5 | FP | FN |
|---|---|---|---|---|
| −0.010 | 0.9120 | 0.970234 (−0.00076) | 31 | 68 |
| −0.002 | 0.9200 | 0.970997 (±0) | 29 | 69 |
| 0 | **0.9220** | **0.970997** | 29 | 69 |
| +0.002 | 0.9240 | 0.970816 (−0.00018) | 29 | 70 |
| +0.010 | 0.9320 | 0.969899 (−0.00110) | 29 | 75 |

The optimum sits on a flat plateau (±0.002 identical) and degrades by ≤0.0011
at ±0.01. The selection is **stable**, not a sharp peak.
Artifact: `output/phase9/threshold_stability.json`.

## 7. Artifacts

| Artifact | Description |
|---|---|
| `output/phase9/threshold_macro_grid.csv` / `.json` | 125 thresholds: pair P/R/F0.5/F1/FP/FN, macro F0.5, exact-set, singleton FP, matched FN |
| `output/phase9/selected_threshold.json` | Selected τ = 0.922 with rationale, tie-break rule, validation population |
| `output/phase9/threshold_stability.json` | Neighbor-threshold stability table |
| `output/phase9/leakage_verification.json` | Entity disjointness evidence |
| `output/phase9/phase9_summary.json` | Executive summary incl. Phase 8 reference comparison |

## 8. Conclusion

**Phase 9 complete. Final operating threshold τ = 0.922**, selected by the
challenge-defined macro F0.5 on held-out validation entities with verified zero
leakage. This refines — not duplicates — Phase 8: the pair-level optimum
(0.95) is measurably suboptimal under macro averaging.
