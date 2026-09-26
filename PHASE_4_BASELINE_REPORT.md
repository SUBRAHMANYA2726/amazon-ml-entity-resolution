# Phase 4 — Baseline Matching & Retrieval Report

> **Project:** Amazon ML Challenge 2026 — Business Entity Resolution  
> **Pipeline Phase:** Phase 4 — Deterministic Baseline Matching, Ranking, and Retrieval Evaluation  
> **Anchor Dataset:** `train_source1.tsv` (Sample: 2,500 anchor records)  
> **Candidate Input:** Phase 3 Multi-Strategy Candidate Pairs (`output/phase3/candidate_pairs_provenance.csv`, 170,153 pairs)  
> **Evaluation Ground Truth:** `train_ground_truth.tsv` (8,674 true match pairs across evaluated anchors)  

---

## 1. Executive Summary & Objective

Phase 4 constructs an explainable, deterministic baseline matching system on top of Phase 3 candidate generation. The fundamental question answered by Phase 4 is:

$$\text{"Given an } S_1 \text{ entity and a Phase 3 candidate } S_2/S_3 \text{ entity, how similar are they?"}$$

This phase operates strictly without machine learning models. It computes multi-signal string, token, character, edit-distance, and structured similarities, forms a calibrated composite baseline score, ranks candidates per anchor, and conducts diagnostic evaluations against training ground truth.

### Key Results Highlights:
- **Total Candidate Pairs Scored:** **170,153 pairs** across 2,500 anchors.
- **Scoring Throughput:** **19,790 pairs/second** (complete scoring in 8.60s).
- **Ranking Recall (Top-K):**
  - **Top-1 Recall:** **24.06%** (2,087 true matches ranked #1)
  - **Top-3 Recall:** **62.20%** (5,395 true matches ranked $\le 3$)
  - **Top-5 Recall:** **81.32%** (7,054 true matches ranked $\le 5$)
  - **Top-10 Recall:** **90.71%** (7,868 true matches ranked $\le 10$)
- **Source Breakdown:**
  - $S_2$ Top-5 Recall: **81.79%**
  - $S_3$ Top-5 Recall: **80.89%**
- **Diagnostic Threshold Performance:**
  - At threshold $\tau = 0.50$: **Precision 76.32%**, **Recall 48.17%**, **$F_{0.5} = 0.6833$**.
  - At threshold $\tau = 0.70$: **Precision 98.24%**, **Recall 19.92%**, **$F_{0.5} = 0.5500$**.
- **Candidate Recall vs. Matching Recall:** Phase 3 candidate generation retrieved **98.09%** of all true pairs. At Top-10 ranking, Phase 4 captures **90.71%** of ground truth pairs, demonstrating that the deterministic baseline successfully concentrates true matches at the very top of the candidate list.

---

## 2. Phase 4 Architecture

```
PHASE 3 CANDIDATE PAIRS (source1_entity_id, candidate_entity_id, candidate_source, provenance)
                             │
                             ▼
         Record Retrieval & Phase 2 Normalization Lookup
            ├── Anchor S1 (name, address, country, postal, house, core/sorted)
            └── Candidate S2/S3 (name, address, country, postal, house, core/sorted)
                             │
                             ▼
         Deterministic Multi-Signal Similarity Calculator
            ├── Name Similarity: exact, core_exact, sorted_exact, token_jaccard, Levenshtein, Jaro-Winkler
            ├── Address Similarity: exact, sorted_exact, token_jaccard, char_3gram, postal_match
            └── Structured Comparison: country_match, postal_match, house_match (4-way indicator)
                             │
                             ▼
         Composite Baseline Score Construction & Country Conflict Gating
            ├── Country Conflict Rule: if both present and unequal → baseline_score = 0.0
            └── Missing Address Dynamic Weight Redistribution
                             │
                             ▼
         Deterministic Candidate Ranking per S1 Anchor (descending score, stable tie-break)
                             │
                             ▼
         Evaluation against Training Ground Truth & Error Analysis
            ├── Top-1, Top-3, Top-5, Top-10 Recall (Overall, S2, S3)
            ├── Diagnostic Threshold Sweep (Precision, Recall, F0.5)
            └── False Negative / False Positive Categorization
```

---

## 3. Implemented Similarity Methods & Missing-Value Policy

All similarity calculations adhere strictly to defensive missingness rules:
> [!IMPORTANT]
> Missing values (null, None, empty string, or whitespace) **NEVER** produce a match. If both values are missing, similarity is strictly **0.0**, preventing sparse records from becoming false exact matches.

### Name Similarity ($S_{\text{name}} \in [0.0, 1.0]$)
Composed of six distinct deterministic signals:
1. **Normalized Exact Equality ($w=0.20$):** Case-folded, whitespace-normalized equality.
2. **Core Name Exact Equality ($w=0.20$):** Equality after corporate suffix stripping (e.g. *LLC, Inc, Corp, Pvt Ltd*).
3. **Sorted Token Exact Equality ($w=0.10$):** Word-order invariant token equality (e.g. *"Owl Novent"* $\leftrightarrow$ *"Novent Owl"*).
4. **Token Jaccard Similarity ($w=0.20$):** Ratio of shared alphanumeric words to union of words.
5. **Normalized Levenshtein Similarity ($w=0.15$):** $1.0 - \frac{\text{edit\_distance}}{\max(L_1, L_2)}$.
6. **Jaro-Winkler Similarity ($w=0.15$):** Character transposition and prefix-weighted similarity.

### Address Similarity ($S_{\text{addr}} \in [0.0, 1.0]$)
1. **Normalized Exact Address Equality ($w=0.25$):** Exact match across cleaned address string.
2. **Component-Sorted Address Equality ($w=0.15$):** Invariant to comma-separated component order swaps.
3. **Address Token Jaccard ($w=0.30$):** Word-level overlap across address text.
4. **Character 3-Gram Jaccard ($w=0.15$):** Boundary-padded character tri-gram overlap.
5. **Postal Code Equality ($w=0.15$):** Exact match of extracted 5-digit US ZIP or 6-digit India PIN.

*Dynamic Weight Redistribution:* If candidate address is missing (observed in ~3.3% of $S_2$ and $S_3$), address weight is safely redistributed (75% to name, 25% to structured fields), preventing artificial score collapse.

### Structured Field Comparison ($S_{\text{struct}} \in [0.0, 1.0]$)
Computed using generic 4-way indicators: `(exact_match, both_present, one_missing, both_missing)`:
1. **Country Match ($w=0.50$):** Open-set case-insensitive comparison.
2. **Postal Code Match ($w=0.30$):** Extracted postal identifier comparison.
3. **House Number Match ($w=0.20$):** Leading street house number comparison.

### Country Conflict Hard Gating
If both $S_1$ and candidate declare country labels and they differ (e.g. $S_1$ is `US` and candidate is `India`), $\text{baseline\_score} \equiv 0.0$. Same commercial entities cannot be simultaneously located in two different nations under this challenge setting.

---

## 4. Baseline Score Formula & Weights

The overall baseline score is a convex combination:

$$\text{baseline\_score} = 0.50 \cdot S_{\text{name}} + 0.35 \cdot S_{\text{addr}} + 0.15 \cdot S_{\text{struct}}$$

Weights were selected from inspecting the Phase 1 training variation profiles: business names provide the primary discriminating signal, while address spatial alignment validates physical identity.

---

## 5. Baseline Evaluation Metrics

### 5.1 Candidate Ranking Recall (Top-K)

| Metric | Overall Recall | S2 Recall | S3 Recall | True Pairs Found |
| :--- | :--- | :--- | :--- | :--- |
| **Top-1 Recall** | **24.06%** | **25.71%** | **22.51%** | 2,087 / 8,674 |
| **Top-3 Recall** | **62.20%** | **63.60%** | **60.89%** | 5,395 / 8,674 |
| **Top-5 Recall** | **81.32%** | **81.79%** | **80.89%** | 7,054 / 8,674 |
| **Top-10 Recall** | **90.71%** | **90.10%** | **91.28%** | 7,868 / 8,674 |

> [!NOTE]
> Recall is reported against total ground truth pairs (8,674). At Top-5 candidates per anchor, the deterministic baseline captures **81.32%** of all true matches, compressing the candidate pool by $13.6\times$ (from average 68 candidates to 5 candidates).

---

### 5.2 Phase 4 Diagnostic Threshold Sweep

Diagnostic sweep across baseline score thresholds $\tau \in [0.50, 0.95]$:

| Threshold ($\tau$) | Predicted Pairs | True Positives | Pair Precision | Pair Recall | Diagnostic $F_{0.5}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | 5,474 | 4,178 | **76.32%** | **48.17%** | **0.6833** |
| **0.55** | 3,561 | 2,559 | **71.86%** | **29.50%** | **0.5583** |
| **0.60** | 3,219 | 2,277 | **70.74%** | **26.25%** | **0.5283** |
| **0.65** | 2,890 | 2,095 | **72.49%** | **24.15%** | **0.5177** |
| **0.70** | 1,759 | 1,728 | **98.24%** | **19.92%** | **0.5500** |
| **0.75** | 1,119 | 1,091 | **97.50%** | **12.58%** | **0.4148** |
| **0.80** | 313 | 289 | **92.33%** | **3.33%** | **0.1456** |
| **0.85** | 213 | 189 | **88.73%** | **2.18%** | **0.0992** |
| **0.90** | 185 | 161 | **87.03%** | **1.86%** | **0.0855** |
| **0.95** | 77 | 77 | **100.00%** | **0.89%** | **0.0429** |

*Note:* This sweep is strictly diagnostic. Final threshold optimization occurs in Phase 9.

---

## 6. Baseline Error Analysis

### 6.1 Categorized Error Breakdown

Analysis of true pairs with baseline score $< 0.60$ (False Negatives) and non-matches with score $\ge 0.70$ (False Positives):

```
Error Patterns Discovered:
├── False Negatives (Score < 0.60): 6,231
│   ├── Address formatting / reordering variations: 3,567 (57.2%)
│   ├── Abbreviation or acronym in business name:   1,245 (20.0%)
│   ├── Address house / unit number discrepancy:      835 (13.4%)
│   ├── Word order inversion in business name:        327  (5.2%)
│   └── Missing candidate address in S2/S3:           257  (4.1%)
└── False Positives (Score >= 0.70): 31
    ├── High token overlap collision (chains/stores):  18 (58.1%)
    └── Identical name at distinct physical location:  13 (41.9%)
```

### 6.2 Key Error Analysis Findings & Handoff to Phase 5
1. **Severe Address Formatting Noise:** Address strings have heavily rearranged parts (city before street, zip before locality). Phase 5 must compute token Jaccard and character 3-gram features that are invariant to formatting.
2. **Abbreviation Discrepancies:** Name abbreviations (e.g. *"FedEx"* vs *"Federal Express"*, *"St."* vs *"Saint"*) cause sharp drops in edit-distance similarity. Phase 5 must include prefix similarity and token overlap to compensate.
3. **Chain Stores / Same Name:** Businesses with identical names in different locations need strong street/postal penalty features to avoid false positives.

---

## 7. Artifact Manifest (Phase 4)

| Artifact Path | Format | Description |
| :--- | :--- | :--- |
| `output/phase4/baseline_ranked_candidates.csv` | CSV (15.7 MB) | 170,153 candidate pairs with baseline scores, ranks, and similarity components |
| `output/phase4/baseline_evaluation_report.json` | JSON (2.9 KB) | Complete Top-K recall, threshold metrics, and summary |
| `output/phase4/baseline_diagnostic_thresholds.csv` | CSV (412 B) | Detailed precision, recall, and F0.5 across threshold grid |
| `output/phase4/baseline_error_analysis.json` | JSON (37.2 KB) | Top false negative and false positive pairs with diagnosed causes |
