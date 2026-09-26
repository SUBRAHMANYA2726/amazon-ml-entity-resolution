# Business Entity Resolution — Dataset Profiling & Architecture Report

> **Dataset Reference:** Amazon ML Challenge 2026 — Business Entity Resolution  
> **Profile Date:** 2026-09-26  
> **Source Directory:** `dataset/student_resource/dataset/`

---

## 📁 1. Exact Filenames & Storage Metrics

| File Role | Relative Path | File Size | Format |
| :--- | :--- | :--- | :--- |
| **Train Source 1 (Reference Anchor)** | `dataset/student_resource/dataset/train/train_source1.tsv` | **200.34 MB** | TSV (`sep="\t"`) |
| **Train Source 2 (Target)** | `dataset/student_resource/dataset/train/train_source2.tsv` | **466.63 MB** | TSV (`sep="\t"`) |
| **Train Source 3 (Target)** | `dataset/student_resource/dataset/train/train_source3.tsv` | **480.37 MB** | TSV (`sep="\t"`) |
| **Train Ground Truth** | `dataset/student_resource/dataset/train/train_ground_truth.tsv` | **121.13 MB** | TSV (`sep="\t"`) |
| **Test Source 1 (Test Anchor)** | `dataset/student_resource/dataset/test/test_source1.tsv` | **166.91 MB** | TSV (`sep="\t"`) |
| **Test Source 2 (Test Target)** | `dataset/student_resource/dataset/test/test_source2.tsv` | **485.86 MB** | TSV (`sep="\t"`) |
| **Test Source 3 (Test Target)** | `dataset/student_resource/dataset/test/test_source3.tsv` | **482.56 MB** | TSV (`sep="\t"`) |

---

## 📊 2. Exact Row Counts

| Dataset Split | Source 1 (`S1`) | Source 2 (`S2`) | Source 3 (`S3`) | Ground Truth Links | Split Total Records |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **2,206,821** | **5,034,616** | **5,285,603** | **2,206,821** (7,638,365 match edges) | **12,527,040** |
| **Test** | **1,732,544** | **4,887,273** | **5,082,316** | *Hidden (Evaluation Set)* | **11,702,133** |
| **Total** | **3,939,365** | **9,921,889** | **10,367,919** | — | **24,229,173** |

---

## 🧱 3. Exact Columns and Data Types

### Source Files Schema (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`)
- **Format:** Tab-separated values (`sep="\t"`).
- **Columns:**
  1. `entity_id` (`string`): Unique record ID formatted as `S1-<id>`, `S2-<id>`, or `S3-<id>`.
  2. `business_name` (`string`): Raw commercial business name.
  3. `business_address` (`string`, nullable in S2/S3): Physical/postal business address.
  4. `country` (`string`): Country categorical identifier (`US`, `India`, `France`).

### Ground Truth Schema (`train_ground_truth.tsv`)
- **Columns:**
  1. `source1_entity_id` (`string`): Anchor entity ID matching `train_source1.tsv`.
  2. `matched_entity_ids` (`string`, nullable/empty): Comma-separated list of matched `S2-` and `S3-` IDs (e.g. `S2-83796020,S3-945547250`) or empty string for singletons.

---

## 🕳️ 4. Missing-Value Percentages

| Dataset File | `entity_id` | `business_name` | `business_address` | `country` | `matched_entity_ids` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | `0 (0.0000%)` | `0 (0.0000%)` | `0 (0.0000%)` | `0 (0.0000%)` | — |
| `train_source2.tsv` | `0 (0.0000%)` | `0 (0.0000%)` | **168,967 (3.3561%)** | `0 (0.0000%)` | — |
| `train_source3.tsv` | `0 (0.0000%)` | `0 (0.0000%)` | **175,916 (3.3282%)** | `0 (0.0000%)` | — |
| `train_ground_truth.tsv`| `0 (0.0000%)` | — | — | — | **123,247 (5.5848%)** *(Singletons)* |
| `test_source1.tsv` | `0 (0.0000%)` | `0 (0.0000%)` | `0 (0.0000%)` | `0 (0.0000%)` | — |
| `test_source2.tsv` | `0 (0.0000%)` | `0 (0.0000%)` | **129,408 (2.6479%)** | `0 (0.0000%)` | — |
| `test_source3.tsv` | `0 (0.0000%)` | `0 (0.0000%)` | **136,098 (2.6779%)** | `0 (0.0000%)` | — |

---

## 🆔 5. S1 / S2 / S3 Distributions & Ratios

- **Source 1 (Canonical Anchor):** 100% unique entities. There are exactly 2,206,821 anchor records in train and 1,732,544 in test.
- **Source 2 Volume Ratio:** ~2.28× relative to S1 in train, ~2.82× in test.
- **Source 3 Volume Ratio:** ~2.40× relative to S1 in train, ~2.93× in test.
- **ID Overlap Between Train & Test:** Exactly **0 overlapping entity IDs** across all sources. Every entity ID in test is unseen.

---

## 🌎 6. Country Distribution

```
TRAIN SPLIT (Total S1: 2,206,821)
├── US:     1,323,633 (59.98%)
└── India:    883,188 (40.02%)

TEST SPLIT (Total S1: 1,732,544) — CRITICAL DISTRIBUTION SHIFT
├── India:    809,986 (46.75%)  [+6.73% shift]
├── US:       663,106 (38.27%)  [-21.71% shift]
└── France:   259,452 (14.98%)  [ZERO-SHOT / UNSEEN COUNTRY]
```

### Full Source-by-Source Country Breakdown:
- **`train_source1`**: US = 1,323,633 (59.98%), India = 883,188 (40.02%)
- **`train_source2`**: US = 3,016,817 (59.92%), India = 2,017,799 (40.08%)
- **`train_source3`**: US = 3,170,056 (59.98%), India = 2,115,547 (40.02%)
- **`test_source1`**: India = 809,986 (46.75%), US = 663,106 (38.27%), France = 259,452 (14.98%)
- **`test_source2`**: India = 2,312,565 (47.32%), US = 1,871,330 (38.29%), France = 703,378 (14.39%)
- **`test_source3`**: India = 2,405,000 (47.32%), US = 1,945,701 (38.28%), France = 731,615 (14.40%)

---

## 🔗 7. Ground-Truth Structure

- **Total Ground Truth Anchors (`S1` rows):** 2,206,821 (100% alignment with `train_source1.tsv`)
- **Singletons (0 matches):** **123,247 (5.5848%)**
- **Resolved Entities (≥1 matches):** **2,083,574 (94.4152%)**
- **Total Matched Pairs (Edges):** **7,638,365**
  - Matched to `S2`: 3,693,619 edges (48.36%)
  - Matched to `S3`: 3,944,746 edges (51.64%)

### Match Count per Anchor ($k$ matches per S1):
| Match Count | S1 Entities | Percentage |
| :--- | :--- | :--- |
| **0 (Singleton)** | 123,247 | 5.58% |
| **1 match** | 119,157 | 5.40% |
| **2 matches** | 375,212 | 17.00% |
| **3 matches** | 530,841 | 24.05% |
| **4 matches** | 484,115 | 21.94% |
| **5 matches** | 321,957 | 14.59% |
| **6 matches** | 164,868 | 7.47% |
| **7 matches** | 63,968 | 2.90% |
| **8 matches** | 18,680 | 0.85% |
| **9 matches** | 4,205 | 0.19% |
| **10 matches** | 554 | 0.025% |
| **11 matches (Max)**| 17 | <0.001% |

### Multi-Source Matching Profile:
- **Both S2 and S3 matched:** 1,776,047 entities (**80.48%**)
- **S3 only matched:** 164,498 entities (**7.45%**)
- **S2 only matched:** 143,029 entities (**6.48%**)
- **No matches (Singletons):** 123,247 entities (**5.58%**)

---

## 🏢 8. Entity Cardinality & Duplicate Structure

- **Many-to-One Conflicts in Target Sources:** **0!**
  - Exactly **7,638,365 unique target IDs** (`S2`/`S3`) are matched across all 7,638,365 edges.
  - **No target record ($S_2$ or $S_3$) ever maps to more than one $S_1$ anchor.**
- **Structural Invariant:** 
  $$\text{Card}(S_1 \to S_2) = [0 \dots m], \quad \text{Card}(S_1 \to S_3) = [0 \dots n], \quad \text{Card}(S_2 \to S_1) \le 1, \quad \text{Card}(S_3 \to S_1) \le 1$$
  This allows **mutual-exclusion constraint enforcement** or 1-to-many bipartite assignment during final post-processing.

---

## 🔍 9. Name & Address Noise Patterns (Profiled from Real Data)

1. **Name Transpositions & Noise:**
   - **Anchor:** `Novent Owl PLLC`
     - $S_2$: `Novent [Owl]` (bracket injection, casing)
     - $S_3$: `PLLC Novent wl` (prefix legal form permutation, encoding corruption `\ufffd`, word order flip)
     - $S_3$: `Beloavi d/b/a Novent Owl PLLC` (DBA / trade-name prefix additions)
   - **Typo variations:** `Spicer Star Environmental Services LLC` $\to$ `Spicer-Star Environmental Seraices LLC` (`Seraices` typo, hyphenation).

2. **Address Variations & Component Transpositions:**
   - **Anchor:** `Des Plaines, IL, 9308 Home Court`
     - $S_2$: `9308 Home Court, DES PLAINES CITY, IL` (street before city, full capitalization, suffix addition `CITY`)
     - $S_3$: `9308 Home Ct, DES Plainescity, Illinois` (abbreviation `Ct` vs `Court`, word boundary concatenation `Plainescity`, state expansion `Illinois` vs `IL`)
   - **Floor permutations:** `Fl 0, MO, Saint Louis, 9327 Atwood Drive` vs `932 Atwood Dr, Fl 0, Stlouis, Missouri` (number truncation/typo `932` vs `9327`, `Fl.` vs `Fl 0`).

3. **Regional & Indian Address Characteristics:**
   - **Anchor:** `Shakti Agro Limited`
   - **Address:** `C/O Gurnav Singh Saluja, Beside Zudio, Kultapara, Sadar, Sambalpur, Orissa`
   - Landmarks (`Beside Zudio`, `Near SBI ATM`), Care-of names (`C/O ...`), sub-district/taluka tokens (`Sadar`), and historical state names (`Orissa` vs `Odisha`).

---

## 🧩 10. Blocking Opportunities (Candidate Generation)

Given ~1.73M test S1 anchors against ~4.89M S2 and ~5.08M S3 records, an exhaustive cross-product is $\approx 1.7 \times 10^{13}$ pairs. Multi-strategy blocking is mandatory:

1. **Hard Blocking on `country`:**
   - Strict partition by country reduces search space by ~60%. Records from `US` only match `US`, `India` only `India`, `France` only `France`.
2. **Standardized Token Inverted Index / TF-IDF / BM25:**
   - Stripping legal suffixes (`LLC`, `INC`, `PVT`, `LTD`, `CORP`, `PLLC`, `GMBH`, `SAS`, `SARL`).
   - Token overlap on normalized business name stem tokens.
3. **Character 3-gram & MinHash / LSH:**
   - Captures typos (`Seraices` $\leftrightarrow$ `Services`, `wl` $\leftrightarrow$ `Owl`).
4. **Address Spatial & Postal Extraction Keys:**
   - 5-digit US ZIP code, 6-digit Indian PIN code, 5-digit French Code Postal.
   - Normalized city + street number blocking keys.
5. **Acronym / First-Token Blocking:**
   - Blocking on the first significant name token or acronym letters.

---

## ⚠️ 11. Leakage Risks

1. **Zero-Shot Country Generalization (`France` in Test):**
   - Training has **only** `US` and `India`. Test introduces `France` (14.98% of test anchors).
   - *Risk:* Hardcoded country lists, country one-hot encodings, country-specific frequency dictionaries, or language-specific regex rules will fail on French addresses/names.
2. **Entity-Level Information Leakage:**
   - S1 entities must **never** be split across train and validation folds.
3. **Target Encoding & Frequency Encoding Leakage:**
   - Computing out-of-fold statistics on business names/addresses must be done strictly within the training folds.
4. **Metric Sensitivity (Macro $F_{0.5}$):**
   - Since $F_{0.5}$ weights precision 2× over recall, false positives heavily degrade the macro-score.

---

## 🧪 12. Correct Train / Validation Strategy

1. **Entity-Grouped Split:**
   - Split strictly on `source1_entity_id`. All corresponding $S_2$ and $S_3$ ground truth matches stay with their anchor entity.
2. **Stratified by Country & Match Cardinality:**
   - Maintain the ~60/40 US/India ratio in train/validation.
   - Stratify to maintain the 5.58% singleton ratio.
3. **Leave-One-Country-Out / Zero-Shot Simulation:**
   - To validate French performance, create a synthetic zero-shot validation partition (e.g. train on US, validate on India, or test language-agnostic features).
4. **Exact Metric Implementation:**
   - Evaluate using macro-averaged $F_{0.5}$ over all $S_1$ validation entities (including singletons scoring 1.0 for true empty and 0.0 for false merges).

---

## 🏗️ 13. Current Repository State vs. Next Phase

### What is Currently Implemented:
- **Phase 0 Scaffolding:** Safe configuration loading (`config.yaml`), random seeding (`utils/seed.py`), structured logging (`utils/logging.py`), and abstract contract interfaces (`contracts.py`, `ingestion/base.py`, `blocking/base.py`, `normalization/base.py`, etc.).
- **Current Phase:** `phase: 0` (non-data execution check).

### What is Next (Phase 1):
- **Phase 1 Goal:** Ingestion pipeline, rule-based text/address normalization, multi-key candidate blocking generator, candidate pairs exporter (`candidate_pairs.tsv`), ground-truth leak-free validation splitter, and Macro $F_{0.5}$ baseline evaluator.

---

## 🚀 14. Exact Phase 1 OpenCode Prompt

```markdown
You are an expert ML Engineer specializing in high-performance Entity Resolution at scale. 

### Context & Dataset Audit
We are building a scalable, leak-free Business Entity Resolution pipeline for the Amazon ML Challenge 2026.
The dataset has been audited with the following locked specifications:
- Training: 2,206,821 Source 1 anchors (59.98% US, 40.02% India), 5,034,616 S2 records, 5,285,603 S3 records, and 2,206,821 ground truth entries (7,638,365 match pairs, 5.58% singletons).
- Test: 1,732,544 Source 1 anchors (46.75% India, 38.27% US, 14.98% France - zero-shot country), 4,887,273 S2 records, 5,082,316 S3 records.
- Invariant: Ground truth is 1-to-Many (S1 -> {S2, S3}); no S2 or S3 record maps to multiple S1 anchors.
- Primary Evaluation Metric: Macro-averaged F_0.5 score over all S1 entities (including singletons).

### Phase 1 Objective: Ingestion, Normalization, Multi-Key Blocking, and Validation Pipeline
Implement Phase 1 in `src/business_entity_resolution/` adhering to existing contracts without altering Phase 0 design patterns.

#### 1. Ingestion Module (`src/business_entity_resolution/ingestion/`)
- Implement streaming/chunked TSV loaders (`sep='\t'`, utf-8 with error handling) for all train and test files.
- Ensure strict zero-null validation on `entity_id`, `business_name`, and `country`, while safely handling nullable `business_address` (approx 3% nulls in S2/S3).
- Implement schema and format verification against competition specifications.

#### 2. Normalization Module (`src/business_entity_resolution/normalization/`)
- Implement multi-lingual, country-agnostic text normalizers:
  - Unicode normalization (NFKD, handling corruption tokens like `\ufffd`).
  - Legal suffix stripping (US: LLC, INC, CORP, PLLC; India: PVT LTD, LIMITED; France: SAS, SARL, SA, EURL).
  - Address abbreviation standardizer (St -> Street, Rd -> Road, Ct -> Court, Fl -> Floor, Ste -> Suite).
  - Postal code and token extraction (US 5-digit ZIP, India 6-digit PIN, France 5-digit postal code).

#### 3. High-Recall Multi-Strategy Blocking Engine (`src/business_entity_resolution/blocking/`)
- Implement a candidate generation engine that achieves ≥95% candidate recall ceiling while keeping candidates/anchor ≤ 20:
  - Partition strictly by `country` (zero cross-country candidate generation).
  - Exact cleaned name match.
  - Multi-token inverted index with BM25 / TF-IDF top-k candidate retrieval.
  - Character n-gram MinHash / LSH index for typo tolerance.
  - Address key blocking (normalized postal code / city prefix + first name token).
  - Candidate deduplication and ranking.

#### 4. Candidate Export & Formatting (`src/business_entity_resolution/submission/`)
- Generate `candidate_pairs.tsv` meeting the competition submission format:
  - Header: `source1_entity_id\tcandidate_entity_ids`
  - Exactly 1 row per S1 entity in test (or validation).
  - Comma-separated list of candidate S2/S3 IDs with no self-references and no duplicates.
- Integrate validation using `dataset/student_resource/utils/validate_submission.py`.

#### 5. Leak-Free Validation Splitter & Evaluation (`src/business_entity_resolution/validation/`, `evaluation/`)
- Implement GroupKFold / Stratified Group Splitter on `source1_entity_id`:
  - Preserve country proportion and singleton ratio (5.58%).
- Implement exact Macro-averaged F_0.5 evaluator computing precision, recall, and F_0.5 per S1 entity, correctly handling singletons (1.0 for true empty, 0.0 for false merge).

#### 6. Pipeline CLI Integration & Config Update
- Update `config.yaml` to `phase: 1` with blocking parameter grids and paths.
- Update `cli.py` to support `--stage profile`, `--stage blocking`, and `--stage evaluate`.
- Provide unit and integration tests under `tests/` verifying candidate recall and formatting compliance.
```
