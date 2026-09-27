# FINAL PROJECT REPORT
## Amazon ML Challenge 2026 — Business Entity Resolution

**Project Status:** Phase 11 — Test Inference In Progress
**Branch:** master
**Last Updated:** 2026-09-27

---

## Executive Summary

This project implements a complete business entity resolution pipeline for the Amazon ML Challenge 2026. The pipeline resolves entities from Source 1 (S1) to Sources 2 and 3 (S2, S3) across multiple countries (France, India, US) using a multi-stage approach: normalization → blocking → pairwise feature generation → supervised LightGBM matching → threshold optimization → consistency rules → final submission assembly.

**Current Status:** Phase 11 test inference is in progress. All non-test-inference preparation work is complete. Final submission files will be generated once test scoring completes.

---

## Phase-by-Phase Summary

### Phase 0 — Project Initialization
- **Status:** ✅ Complete
- **Deliverables:** Project structure, config.yaml, dataset placeholders
- **Key Artifacts:** `config.yaml`, `src/business_entity_resolution/` package structure

### Phase 1 — Data Ingestion & Exploration
- **Status:** ✅ Complete
- **Deliverables:** Dataset diagnostics, schema inspection, country distribution
- **Key Artifacts:** `scripts/inspect_dataset.py`, `scripts/inspect_quick.py`, `DATASET.md`

### Phase 2 — Normalization Pipeline
- **Status:** ✅ Complete
- **Deliverables:** Comprehensive normalization for business_name, business_address, country
- **Key Features:**
  - Unicode normalization (NFKC)
  - Whitespace collapse
  - Name core extraction (remove legal suffixes: LLC, INC, LTD, etc.)
  - Address parsing (house number, unit, postal code)
  - Country standardization (ISO alpha-2)
  - Sorted token representations for blocking
- **Key Artifacts:** `src/business_entity_resolution/normalization/`, `scripts/run_diagnostics.py`

### Phase 3 — Multi-Strategy Blocking
- **Status:** ✅ Complete
- **Deliverables:** 5 blocking strategies with country partitioning
- **Strategies:**
  1. Exact match (name + address + country)
  2. Name token overlap (Jaccard ≥ 0.35)
  3. Address token overlap (Jaccard ≥ 0.35)
  4. Character 3-gram (min overlap 0.35)
  5. TF-IDF cosine similarity (name ≥ 0.15, address ≥ 0.15)
- **Config:** Max 100 candidates per anchor, partition by country
- **Key Artifacts:** `src/business_entity_resolution/blocking/`, `scripts/run_phase3_blocking.py`

### Phase 4 — Baseline Matching
- **Status:** ✅ Complete
- **Deliverables:** Deterministic baseline matcher for candidate ranking
- **Features:** Token Jaccard, fuzzy ratios, edit distance, phonetic, address components
- **Key Artifacts:** `src/business_entity_resolution/matching/baseline.py`, `scripts/run_phase4_baseline.py`

### Phase 5 — Pairwise Feature Engineering
- **Status:** ✅ Complete
- **Deliverables:** 84-dimensional feature vector per candidate pair
- **Feature Groups:** Name, Address, Country, Strategy, Cross-field
- **Key Artifacts:** `src/business_entity_resolution/features/`, `scripts/run_phase5_features.py`

### Phase 6 — Supervised Pairwise Models
- **Status:** ✅ Complete
- **Deliverables:** LightGBM models (Model A: baseline features; Model B: +hard negatives)
- **Training:** 2000 train S1 entities, 500 validation S1 entities
- **Key Artifacts:** `src/business_entity_resolution/models/`, `scripts/run_phase6_models.py`, `output/phase 6/`

### Phase 7 — Hard Negative Mining
- **Status:** ✅ Complete
- **Deliverables:** Model B trained on hard negatives (high-scoring false positives)
- **Impact:** Improved precision on difficult cases
- **Key Artifacts:** `scripts/run_phase7_hard_negatives.py`, `output/phase 7/`

### Phase 8 — Entity-Level Validation
- **Status:** ✅ Complete
- **Deliverables:** Entity-level metrics (macro F0.5, exact set accuracy, singleton analysis)
- **Validation Population:** 500 held-out S1 entities, 34,811 candidate pairs
- **Key Artifacts:** `scripts/run_phase8_entity_validation.py`, `output/phase 8/`

### Phase 9 — Final F0.5 Threshold Optimization
- **Status:** ✅ Complete
- **Selected Threshold:** **0.922**
- **Selection Method:** Coarse grid (0.01) → fine grid (0.002, ±0.015 window)
- **Metric:** Macro F0.5 (challenge-defined: per-S1 F0.5 averaged over 500 entities)
- **Result:** 0.970997 macro F0.5, 84% exact set accuracy, 2 singleton FPs, 2 matched FNs
- **Stability:** Verified ±0.01 window
- **Leakage Check:** Train/val S1 overlap = 0 ✅
- **Key Artifacts:** `output/phase9/selected_threshold.json`, `output/phase9/phase9_summary.json`, `scripts/run_phase9_threshold_optimization.py`

### Phase 10 — Entity Consistency + Singleton Audit
- **Status:** ✅ Complete
- **Retained Rules:**
  1. tau=0.922 thresholding
  2. Intra-S1 duplicate removal + deterministic ordering (prob DESC, ID ASC)
  3. Fan-in safety check (0 conflicts on validation)
- **Rejected Rules:**
  - Top-1-only (macro Δ -0.280)
  - Country-conflict veto (macro Δ 0.000)
  - Margin/gap veto
  - Extra singleton veto
- **Key Artifacts:** `output/phase10/phase10_audit.json`, `scripts/run_phase10_consistency_audit.py`

### Phase 11 — Test Inference & Submission Assembly
- **Status:** 🔄 IN PROGRESS
- **Sub-phases:**
  - **11a — Test Blocking:** ✅ Normalization cache complete; France blocking complete; India/US in progress
  - **11b — Test Scoring:** 🔄 Pending blocking completion
  - **11c — Assemble Submission:** ✅ Assembler ready (`scripts/run_phase11c_assemble_submission.py`)
  - **11d — Official Validator Wrapper:** ✅ Ready (`scripts/run_phase11d_validate_submission.py`)
  - **11e — Submission Manifest:** ✅ Ready (`scripts/run_phase11e_manifest.py`)

#### Phase 11 Preparation Complete (Non-Test Work)
| Component | Status | Location |
|-----------|--------|----------|
| Submission Assembler | ✅ Ready | `scripts/run_phase11c_assemble_submission.py` |
| Matching Results Validator | ✅ Ready | `src/business_entity_resolution/validation/submission_validation.py` |
| Candidate Pairs Validator | ✅ Ready | `src/business_entity_resolution/validation/submission_validation.py` |
| Candidate-Subset Checker | ✅ Ready | `validate_subset()` in submission_validation.py |
| Official Validator Wrapper | ✅ Ready | `scripts/run_phase11d_validate_submission.py` |
| Validation Result Handling | ✅ Ready | Outputs to `output/phase11/validation_result.txt` |
| Submission Manifest | ✅ Ready | `scripts/run_phase11e_manifest.py` |
| Final Inference Report | ✅ Template Ready | `output/phase11/final_inference_report.md` |
| Final Project Report | ✅ This Document | `FINAL_PROJECT_REPORT.md` |
| Unit Tests | ✅ Ready | `tests/test_phase11_submission.py` (new) |
| Git Safety Checks | ✅ Verified | See below |

#### Phase 11 Test Inference Status
| Sub-phase | Status |
|-----------|--------|
| 11a Normalization Cache | ✅ Complete |
| 11a France Blocking | ✅ Complete |
| 11a India Blocking | 🔄 In Progress |
| 11a US Blocking | 🔄 In Progress |
| 11b France Scoring | ⏳ Pending |
| 11b India Scoring | ⏳ Pending |
| 11b US Scoring | ⏳ Pending |
| 11c Assembly | ⏳ Pending |
| 11d Validation | ⏳ Pending |
| 11e Manifest | ⏳ Pending |

---

## Key Technical Decisions

1. **Country Partitioning:** All blocking, scoring, and assembly done per-country to manage memory (8 GB limit)
2. **Deterministic Ordering:** Probability DESC, Candidate ID ASC — ensures reproducible submissions
3. **No Test Leakage:** Verified at every phase; threshold selected on validation split only
4. **Open-Set Countries:** No hard-coded country restrictions; pipeline handles any country value
5. **Shared Memory Scoring:** `SharedRecordStore` uses `multiprocessing.shared_memory` for zero-copy worker access
6. **Chunked Processing:** All stages use fixed-size anchor chunks for memory predictability

---

## File Structure (Output)

```
output/
├── phase9/
│   ├── selected_threshold.json       # {"selected_threshold": 0.922}
│   ├── phase9_summary.json           # Full threshold optimization report
│   ├── leakage_verification.json     # Train/val overlap = 0
│   ├── threshold_macro_grid.csv      # Grid search results
│   └── threshold_stability.json      # ±0.01 stability check
├── phase10/
│   ├── phase10_audit.json            # Consistency audit results
│   ├── margin_analysis.csv           # Margin/gap analysis
│   ├── margin_analysis.json
│   ├── singleton_analysis.csv        # Singleton error analysis
│   └── singleton_analysis.json
└── phase11/
    ├── work/                         # Intermediate checkpoints (per country)
    │   ├── normblock_s1_*.csv.gz     # Normalized blocking caches
    │   ├── normblock_s2_*.csv.gz
    │   ├── normblock_s3_*.csv.gz
    │   ├── anchors_*.csv.gz          # S1 rosters
    │   ├── pairs_*/chunk_*.csv.gz    # Candidate pairs
    │   ├── proba_*/chunk_*.npz       # Model probabilities
    │   ├── blocking_audit_*.json
    │   └── scoring_audit_*.json
    ├── matching_results.tsv          # PENDING — generated by 11c
    ├── candidate_pairs.tsv           # PENDING — generated by 11c
    ├── validation_result.txt         # PENDING — generated by 11d
    ├── submission_manifest.json      # PENDING — generated by 11e
    ├── submission_audit.json         # PENDING — generated by 11c
    └── final_inference_report.md     # ✅ Template ready
```

---

## Git Safety Verification

| Check | Status |
|-------|--------|
| Dataset ignored (`.gitignore`) | ✅ `dataset/` in .gitignore |
| No raw dataset staged | ✅ `git status` shows only code/config |
| No credentials in repo | ✅ Verified |
| No API keys in repo | ✅ Verified |
| No temporary cache files staged | ✅ `output/phase11/work/` in .gitignore |
| No accidental huge files | ✅ Largest tracked < 1 MB |
| Output phase11 structure correct | ✅ Verified |

**Large files in `.gitignore`:**
- `dataset/`
- `output/phase11/work/`
- `output/phase 6/`, `output/phase 7/`, `output/phase 8/`
- `*.joblib`, `*.npy`, `*.npz`

---

## Reproducibility Checklist

- [x] Random seed fixed (42) across all phases
- [x] LightGBM seed fixed (42)
- [x] Deterministic blocking (no randomness)
- [x] Deterministic feature generation
- [x] Fixed threshold (0.922)
- [x] Deterministic ordering (proba DESC, ID ASC)
- [x] Fixed chunk sizes
- [x] Model artifact versioned (`output/phase 8/model_b_lightgbm.joblib`)
- [x] Feature schema versioned (`output/phase 8/model_b_metadata.json`)
- [x] Config versioned (`config.yaml`)

---

## Next Steps

1. **Complete Phase 11a** — Finish India and US blocking
2. **Run Phase 11b** — Score all country pairs (France → India → US)
3. **Run Phase 11c** — Assemble final submission files
4. **Run Phase 11d** — Execute official validator
5. **Run Phase 11e** — Generate final submission manifest
6. **Verify** — All validation checks PASS
7. **Package** — Create submission zip with `matching_results.tsv` and `candidate_pairs.tsv`

---

## Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Test inference OOM | Medium | High | Chunked processing, per-country isolation, shared memory |
| Validator FAIL | Low | High | Comprehensive pre-validator audit in 11c |
| Missing test S1 coverage | Low | High | Coverage check in assembler (fails fast) |
| Non-deterministic output | Low | Medium | Fixed seeds, deterministic ordering verified in tests |
| Country partition missing | Low | High | Fallback logic in blocking (queries all partitions) |

---

*Report generated automatically. Update after Phase 11 completion.*