# Phase 6 — Supervised Pairwise ML Matching Report

> **Project:** Amazon ML Challenge 2026 — Business Entity Resolution
> **Pipeline Phase:** Phase 6 — Supervised Pairwise ML Entity Matcher
> **Input Feature Matrix:** Phase 5 Pairwise Features (`output/phase5/train_feature_matrix.csv.gz`, 170,153 pairs $\times$ 88 columns)
> **Feature Schema & Metadata:** `output/phase5/feature_metadata.json` (84 pairwise feature columns across 9 groups)
> **Random Seed:** 2026
> **Primary Artifacts Produced:** `output/phase6/`

---

## 1. Executive Summary & Objective

Phase 6 constructs a supervised pairwise machine learning entity matcher to learn the conditional match probability:

$$P(\text{candidate is true match} \mid S_1, \text{Cand}, \vec{x}_{\text{Phase 5}})$$

The resulting probabilities provide calibrated ranking scores for downstream candidate thresholding and decision-making. In strict adherence to challenge design and Phase 6 task instructions:
* **No final production threshold is selected in Phase 6.** Diagnostic threshold sweeps are reported for analytical evaluation, while candidate ranking probabilities are preserved across the entire score spectrum.
* **Strict leakage boundaries are maintained:** Model inputs are exclusively restricted to the 84 Phase 5 numeric pairwise features. Identifier and provenance metadata (`source1_entity_id`, `candidate_entity_id`, `candidate_source`) and label target (`ground_truth_label`) are strictly barred from model training and scoring.
* **Mandatory entity-aware train/validation splitting:** Partitioning is performed strictly at the anchor entity level (`source1_entity_id`), guaranteeing zero anchor overlap ($\text{Train} \cap \text{Val} = \emptyset$).
* **Model exploration & comparison:** Both LightGBM and CatBoost were trained with deterministic configurations using the project seed (2026) and class imbalance weighting (`scale_pos_weight = 18.94`).

### Key Validation Results Highlights

| Metric | LightGBM | CatBoost |
| :--- | :---: | :---: |
| **Validation PR-AUC** | **0.9957** | **0.9948** |
| **Validation ROC-AUC** | **0.9997** | **0.9996** |
| **Diagnostic F0.5 @ 0.50** | **0.9640** (P: 0.9592, R: 0.9837) | **0.9513** (P: 0.9428, R: 0.9866) |
| **Diagnostic F0.5 @ 0.70** | **0.9691** (P: 0.9672, R: 0.9768) | **0.9629** (P: 0.9581, R: 0.9826) |
| **Diagnostic F0.5 @ 0.85** | **0.9740** (P: 0.9749, R: 0.9704) | **0.9694** (P: 0.9682, R: 0.9739) |
| **Diagnostic F0.5 @ 0.95** | **0.9756** (P: 0.9804, R: 0.9570) | **0.9731** (P: 0.9769, R: 0.9582) |
| **Training Time** | **3.06s** | **9.20s** |
| **Carried Forward** | **Yes** (higher PR-AUC and lower FP rate) | Reloadable artifact preserved |

---

## 2. Entity-Aware Train / Validation Split

### Splitting Methodology
Individual candidate pairs must never be randomly split across partitions, as candidate pairs for the same anchor share structural context and could leak identity-specific attributes.

Splitting is strictly executed by partitioning the set of unique `source1_entity_id` values:
1. Extract unique $S_1$ entity IDs ($N = 2,500$).
2. Deterministically permute unique IDs using `np.random.RandomState(2026)`.
3. Assign 80% of anchor entities ($N = 2,000$) to the training set and 20% ($N = 500$) to the validation set.
4. Filter all candidate pairs associated with each entity partition.
5. Explicit programmatic verification: $\text{set}(S_{1, \text{train}}) \cap \text{set}(S_{1, \text{val}}) = \emptyset$. A fail-loud assertion terminates execution if any intersection is detected.

### Split Statistics (`output/phase6/split_metadata.json`)

```json
{
  "random_seed": 2026,
  "val_fraction": 0.2,
  "split_method": "entity_group_split_by_source1_entity_id",
  "n_train_entities": 2000,
  "n_val_entities": 500,
  "n_total_entities": 2500,
  "n_train_pairs": 135342,
  "n_val_pairs": 34811,
  "n_total_pairs": 170153,
  "train_positive_pairs": 6786,
  "train_negative_pairs": 128556,
  "train_positive_rate": 0.0501396,
  "val_positive_pairs": 1722,
  "val_negative_pairs": 33089,
  "val_positive_rate": 0.0494671,
  "train_scale_pos_weight": 18.944297
}
```

The positive match rate is exceptionally stable across the boundary: **5.01%** in training and **4.95%** in validation.

---

## 3. Class Imbalance Strategy

The dataset exhibits severe natural class imbalance (~1 positive pair per 19 negative candidate pairs).
* **Strategy:** Cost-sensitive gradient boosting via class weighting (`scale_pos_weight`).
* **Calculation:** Derived strictly from the training partition:
  $$\text{scale\_pos\_weight} = \frac{N_{\text{train, negative}}}{N_{\text{train, positive}}} = \frac{128,556}{6,786} \approx 18.9443$$
* Oversampling (e.g. SMOTE) was avoided to preserve the true negative density geometry and prevent synthetic artifact distortion.

---

## 4. Models & Training Specifications

### LightGBM Configuration
* **Estimator:** `lightgbm.LGBMClassifier` (version 4.7.0)
* **Hyperparameters:**
  * `n_estimators`: 300
  * `learning_rate`: 0.05
  * `num_leaves`: 31
  * `max_depth`: -1
  * `min_child_samples`: 20
  * `subsample`: 0.8
  * `colsample_bytree`: 0.8
  * `scale_pos_weight`: 18.9443
  * `random_state`: 2026
  * `deterministic`: True
* **Training Time:** 3.06s on 135,342 pairs $\times$ 84 features.

### CatBoost Configuration
* **Estimator:** `catboost.CatBoostClassifier` (version 1.2.10)
* **Hyperparameters:**
  * `iterations`: 350
  * `learning_rate`: 0.05
  * `depth`: 6
  * `l2_leaf_reg`: 3.0
  * `scale_pos_weight`: 18.9443
  * `random_seed`: 2026
  * `thread_count`: -1
  * `verbose`: False
* **Training Time:** 9.20s on 135,342 pairs $\times$ 84 features.

---

## 5. Measured Validation Performance Comparison

Evaluation was performed on the held-out validation set of 34,811 candidate pairs (1,722 true positive matches, 33,089 negatives).

### Overall Ranking Metrics

$$\begin{aligned}
\text{LightGBM PR-AUC} &= \mathbf{0.9957} \quad &\text{LightGBM ROC-AUC} &= \mathbf{0.9997} \\
\text{CatBoost PR-AUC} &= \mathbf{0.9948} \quad &\text{CatBoost ROC-AUC} &= \mathbf{0.9996}
\end{aligned}$$

### Diagnostic Threshold Sweep Table (`output/phase6/diagnostic_threshold_sweeps.csv`)

| Threshold | LightGBM TP | LightGBM FP | LightGBM FN | LightGBM Prec | LightGBM Rec | LightGBM F0.5 | CatBoost TP | CatBoost FP | CatBoost FN | CatBoost Prec | CatBoost Rec | CatBoost F0.5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 1,707 | 150 | 15 | 0.9192 | 0.9913 | 0.9328 | 1,716 | 307 | 6 | 0.8482 | 0.9965 | 0.8743 |
| **0.20** | 1,704 | 117 | 18 | 0.9357 | 0.9895 | 0.9460 | 1,712 | 201 | 10 | 0.8949 | 0.9942 | 0.9132 |
| **0.30** | 1,700 | 95 | 22 | 0.9471 | 0.9872 | 0.9548 | 1,709 | 144 | 13 | 0.9223 | 0.9925 | 0.9355 |
| **0.40** | 1,696 | 79 | 26 | 0.9555 | 0.9849 | 0.9612 | 1,702 | 122 | 20 | 0.9331 | 0.9884 | 0.9437 |
| **0.50** | 1,694 | 72 | 28 | 0.9592 | 0.9837 | 0.9640 | 1,699 | 103 | 23 | 0.9428 | 0.9866 | 0.9513 |
| **0.55** | 1,692 | 65 | 30 | 0.9630 | 0.9826 | 0.9669 | 1,698 | 93 | 24 | 0.9481 | 0.9861 | 0.9554 |
| **0.60** | 1,690 | 65 | 32 | 0.9630 | 0.9814 | 0.9666 | 1,695 | 89 | 27 | 0.9501 | 0.9843 | 0.9568 |
| **0.65** | 1,686 | 61 | 36 | 0.9651 | 0.9791 | 0.9679 | 1,695 | 79 | 27 | 0.9555 | 0.9843 | 0.9611 |
| **0.70** | 1,682 | 57 | 40 | 0.9672 | 0.9768 | 0.9691 | 1,692 | 74 | 30 | 0.9581 | 0.9826 | 0.9629 |
| **0.75** | 1,681 | 51 | 41 | 0.9706 | 0.9762 | 0.9717 | 1,688 | 67 | 34 | 0.9618 | 0.9803 | 0.9655 |
| **0.80** | 1,676 | 46 | 46 | 0.9733 | 0.9733 | 0.9733 | 1,683 | 61 | 39 | 0.9650 | 0.9774 | 0.9675 |
| **0.85** | 1,671 | 43 | 51 | 0.9749 | 0.9704 | 0.9740 | 1,677 | 55 | 45 | 0.9682 | 0.9739 | 0.9694 |
| **0.90** | 1,667 | 39 | 55 | 0.9771 | 0.9681 | 0.9753 | 1,668 | 50 | 54 | 0.9709 | 0.9686 | 0.9704 |
| **0.95** | 1,648 | 33 | 74 | 0.9804 | 0.9570 | 0.9756 | 1,650 | 39 | 72 | 0.9769 | 0.9582 | 0.9731 |


### Model Selection Decision
* **Model selection:** LightGBM was selected using the measured diagnostic F0.5 at threshold 0.95 as the primary comparison criterion, with PR-AUC as the tie-breaker.
* **Measured comparison:** LightGBM achieved F0.5 = **0.9756** at 0.95 versus **0.9731** for CatBoost. LightGBM also achieved higher PR-AUC (**0.9957** vs **0.9948**) and fewer false positives across the diagnostic thresholds.
* **Carried Forward Model:** `LightGBMMatcher` is carried forward as the primary model. Both models, their metadata, feature importances, and validation predictions are persisted in `output/phase6/`.
---

## 6. Model Feature Importance

> **IMPORTANT:** The following metrics represent **MODEL FEATURE IMPORTANCE** (information gain in tree split decisions). They reflect association in the learned tree ensemble and **MUST NOT be interpreted as causal importance**.

### Top 15 Features by LightGBM Information Gain (`output/phase6/lightgbm_feature_importance.csv`)

| Rank | Feature Name | Feature Group | Gain Importance | Normalized Gain | Description |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | `baseline_score` | baseline | 1,331,329.01 | 0.4534 | Phase 4 deterministic multi-component baseline similarity score |
| **2** | `address_char_3gram_jaccard` | address | 730,497.27 | 0.2488 | Character 3-gram Jaccard similarity of normalized addresses |
| **3** | `address_token_overlap` | address | 387,448.19 | 0.1319 | Asymmetric token overlap ratio between business addresses |
| **4** | `blocking_strategy_count` | provenance | 82,502.69 | 0.0281 | Number of distinct blocking rules retrieving candidate pair |
| **5** | `cross_name_sim_x_address_sim` | cross_field | 59,987.61 | 0.0204 | Cross-field interaction product between name and address scores |
| **6** | `address_length_ratio` | address | 39,345.99 | 0.0134 | Length ratio between anchor and candidate addresses |
| **7** | `name_jaro_winkler` | name | 31,630.78 | 0.0108 | Jaro-Winkler string similarity of normalized names |
| **8** | `baseline_name_sim` | baseline | 25,915.49 | 0.0088 | Phase 4 composite name similarity score |
| **9** | `baseline_rank` | baseline | 23,030.77 | 0.0078 | Phase 4 candidate rank within anchor |
| **10** | `address_token_jaccard` | address | 20,228.28 | 0.0069 | Token Jaccard similarity of normalized business addresses |
| **11** | `blocking_multi_strategy` | provenance | 19,888.21 | 0.0068 | Binary indicator for candidate retrieval by $\ge 2$ strategies |
| **12** | `name_char_3gram_jaccard` | name | 18,517.35 | 0.0063 | Character 3-gram Jaccard similarity of normalized names |
| **13** | `name_length_cand` | name | 15,544.22 | 0.0053 | Character length of candidate business name |
| **14** | `baseline_address_sim` | baseline | 13,289.59 | 0.0045 | Phase 4 composite address similarity score |
| **15** | `name_token_count_diff` | name | 12,591.61 | 0.0043 | Difference in token count between names |

The top 3 features (`baseline_score`, `address_char_3gram_jaccard`, and `address_token_overlap`) account for **83.41%** of total model information gain.

---

## 7. Probability Distribution Diagnostics

Predicted match probabilities $p = P(\text{match} = 1)$ were verified to strictly satisfy $0 \le p \le 1$ with zero NaN, infinite, or missing values.

### Probability Distribution Statistics (`output/phase6/validation_metrics.json`)

| Statistic | Overall (34,811 pairs) | True Positives (1,722 pairs) | True Negatives (33,089 pairs) |
| :--- | :---: | :---: | :---: |
| **Mean** | 0.0511 | **0.9796** | **0.0027** |
| **Standard Dev** | 0.2173 | 0.1145 | 0.0422 |
| **Min** | $5.20 \times 10^{-9}$ | $5.38 \times 10^{-4}$ | $5.20 \times 10^{-9}$ |
| **p10** | $2.02 \times 10^{-6}$ | 0.9925 | $1.96 \times 10^{-6}$ |
| **p25** | $3.67 \times 10^{-6}$ | 0.9992 | $3.57 \times 10^{-6}$ |
| **Median (p50)** | $7.41 \times 10^{-6}$ | **0.9999** | **$7.18 \times 10^{-6}$** |
| **p75** | $2.38 \times 10^{-5}$ | 0.9999 | $2.14 \times 10^{-5}$ |
| **p90** | $3.11 \times 10^{-4}$ | 0.9999 | $1.15 \times 10^{-4}$ |
| **p95** | 0.6933 | 0.9999 | $6.91 \times 10^{-4}$ |
| **p99** | 0.9999 | 0.9999 | 0.0381 |
| **Max** | 0.9999 | 0.9999 | 0.9998 |

* **Sharp Separation:** The median true negative probability is **$0.000007$**, while the median true positive probability is **$0.9999$**.
* 99% of all true negatives receive a predicted probability below **0.0381**.

---

## 8. Prediction Error Analysis (`prediction_error_diagnostics.json`)

### High-Confidence False Positives ($y = 0$, $p \approx 0.999$)
* **S1-631152952 $\leftrightarrow$ S3-532359746 ($p = 0.9998$, rank 5):**
  * Anchor had multiple candidates from the same organization (e.g. branch offices with identical corporate names and near-identical regional addresses). The model scored both highly, but ground truth labeled only one specific record as the matched anchor.
* **Root Cause:** Sibling branch ambiguity and entity splitting across multi-store chains.

### Low-Confidence True Positives ($y = 1$, $p \le 0.0026$)
* **S1-697294833 $\leftrightarrow$ S3-527682590 ($p = 0.0005$, rank 10):**
  * Single blocking strategy retrieval (`blocking_strategy_count = 1`, `blocking_multi_strategy = 0`).
  * Severe typographical divergence in both business name and street address (`baseline_score = 0.3368`).
* **Root Cause:** Heavy name abbreviation combined with street address relocation.

---

## 9. Validation Predictions Artifact Structure

The validation predictions artifact (`output/phase6/validation_predictions.csv.gz`) adheres strictly to deterministic ordering:
1. `source1_entity_id` ascending
2. `match_probability` descending
3. `candidate_entity_id` ascending

### Column Schema:
1. `source1_entity_id`: Anchor entity ID
2. `candidate_entity_id`: Candidate entity ID
3. `candidate_source`: Source identifier (`S2` or `S3`)
4. `match_probability`: Predicted probability $P(\text{match} = 1)$ in $[0, 1]$
5. `ground_truth_label`: Validation label ($1$ or $0$)
6. `baseline_score`: Phase 4 baseline similarity score
7. `blocking_strategy_count`: Candidate provenance blocking strategy count
8. `blocking_multi_strategy`: Candidate provenance multi-strategy indicator
9. `rank_within_source1`: Candidate rank within the anchor entity ($1, 2, \dots, K$)

---

## 10. Phase 6 Artifact Inventory

All Phase 6 outputs are safely stored in `output/phase6/`:

| Artifact | Size | Description |
| :--- | :---: | :--- |
| `split_metadata.json` | 550 B | Detailed entity-aware train/val split counts, positive rates, and seed |
| `lightgbm_model.joblib` | 430 KB | Serialized reloadable LightGBM matcher artifact |
| `lightgbm_metadata.json` | 3.6 KB | LightGBM feature column order, hyperparameters, and imbalance strategy |
| `catboost_model.joblib` | 226 KB | Serialized reloadable CatBoost matcher artifact |
| `catboost_metadata.json` | 3.5 KB | CatBoost feature column order, hyperparameters, and imbalance strategy |
| `validation_predictions.csv.gz` | 855 KB | Full validation predictions (34,811 pairs) sorted deterministically |
| `validation_predictions_sample.csv` | 76 KB | Leading 1,000 validation prediction rows for quick inspection |
| `catboost_validation_predictions.csv.gz` | 860 KB | CatBoost validation predictions (34,811 pairs) |
| `validation_metrics.json` | 14 KB | Comprehensive PR-AUC, ROC-AUC, threshold sweep, and distributions |
| `diagnostic_threshold_sweeps.csv` | 3.1 KB | Tabular precision, recall, F0.5, TP, FP, FN across 14 thresholds |
| `lightgbm_feature_importance.csv` | 6.7 KB | Complete MODEL FEATURE IMPORTANCE table (split & gain) |
| `catboost_feature_importance.csv` | 6.5 KB | CatBoost MODEL FEATURE IMPORTANCE table |
| `prediction_error_diagnostics.json` | 22 KB | High-confidence false positives and low-confidence true positives |

---

## 11. Verification & Test Suite

The Phase 6 test suite was added to `tests/test_phase6_models.py` without modifying or regressing earlier phases.
* **Phase 6 Tests:** 15 unit tests covering feature leakage rejection, entity-aware split determinism and zero-overlap assertion, probability validation, feature order alignment, model scoring shapes, and serialization/reload.
* **Full Regression Suite:** **126 passed, 3 failed (129 collected)**.
* The 3 failures are dataset-dependent ingestion tests because the local challenge dataset is not present under `dataset/`; they are not Phase 6 model-test failures.
