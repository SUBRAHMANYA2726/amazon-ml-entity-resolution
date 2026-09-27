# Amazon ML Challenge 2026 — Business Entity Resolution

## Project Overview

This repository contains a complete, production-ready entity resolution pipeline for the **Amazon ML Challenge 2026**. The challenge requires matching business entities across three data sources (S1 reference, S2 vendor, S3 vendor) using only business name, address, and country fields.

## Problem Statement

Given three sources of business entity records:
- **S1**: Reference entities (anchors to be matched)
- **S2**: Vendor catalog 1
- **S3**: Vendor catalog 2

Find all matching entities in S2 and S3 for each S1 entity. Matches are evaluated using **Macro F0.5** (per-S1-entity F0.5 averaged over all S1 entities).

## Objective

Build a scalable, reproducible entity resolution pipeline that:
1. Normalizes text conservatively (no match decisions during normalization)
2. Generates high-recall candidate pairs via multi-strategy blocking
3. Extracts 84 deterministic pairwise features
4. Trains a LightGBM matcher with hard-negative mining
5. Selects an operating threshold via validation-set Macro F0.5 optimization
6. Applies consistency rules for final predictions
7. Supports memory-safe test inference for large datasets

## Architecture / Pipeline

```
S1, S2, S3 raw data
       │
       ▼
Normalization (Phase 2) ──► raw + derived columns preserved
       │
       ▼
Multi-Strategy Blocking (Phase 3)
  ├─ Exact name/address
  ├─ Name token inverted index
  ├─ Address token inverted index
  ├─ Character 3-gram overlap
  └─ TF-IDF sparse retrieval
       │
       ▼
Candidate Union & Deduplication
       │
       ▼
Pairwise Feature Generation (Phase 5) ──► 84 features / 9 groups
       │
       ▼
LightGBM Matcher (Phase 6)
       │
       ▼
Hard-Negative Mining (Phase 7) ──► Model B (selected)
       │
       ▼
Entity-Level Validation (Phase 8)
       │
       ▼
Threshold Optimization (Phase 9) ──► 0.922 (Macro F0.5 = 0.971)
       │
       ▼
Consistency Rules (Phase 10)
       │
       ▼
Production Inference (Phase 11)
```

## Phase 0–11 Overview

| Phase | Description | Key Artifact |
|-------|-------------|--------------|
| 0 | Project foundation, config, logging | `config.yaml` |
| 1 | Data profiling & discovery | `dataset_profile.json` |
| 2 | Text normalization (conservative) | `derived_normalization_rules.json` |
| 3 | Multi-strategy blocking | `candidate_pairs.tsv` |
| 4 | Baseline deterministic matching | `baseline_ranked_candidates.csv` |
| 5 | Pairwise feature engineering | `train_feature_matrix.csv.gz` (84 features) |
| 6 | LightGBM / CatBoost training | `lightgbm_model.joblib` |
| 7 | Hard-negative mining & retrain | `model_b_lightgbm.joblib` |
| 8 | Entity-level validation & analysis | `entity_level_validation.csv` |
| 9 | Macro F0.5 threshold selection | `selected_threshold.json` (0.922) |
| 10 | Consistency & singleton configuration | `consistency_config.json` |
| 11 | Test inference & submission | `matching_results.tsv` |

## Data Sources

| Source | Role | Fields |
|--------|------|--------|
| **S1** | Reference (anchors) | `entity_id`, `business_name`, `business_address`, `country` |
| **S2** | Vendor catalog 1 | `entity_id`, `business_name`, `business_address`, `country` |
| **S3** | Vendor catalog 2 | `entity_id`, `business_name`, `business_address`, `country` |

All sources share the same schema. Country is treated as an **open set** (no hard-coded country list).

## Entity Resolution Methodology

### Normalization (Phase 2)
- Unicode NFKC, casefold, whitespace collapse
- Punctuation → space, symbol stripping (preserving `&'-.#,/`)
- Ampersand → "and"
- Conservative token mappings derived from training data only
- **Never decides matches** — only builds comparison representations

### Blocking & Candidate Generation (Phase 3)
Five independent strategies, partitioned by country:
1. **Exact** — normalized name + country
2. **Name Token** — inverted index on name tokens (min length 3, max freq 10k)
3. **Address Token** — inverted index on address tokens
4. **Char 3-gram** — Jaccard overlap ≥ 0.35, top-20
5. **TF-IDF** — sparse retrieval on name/address, top-k per anchor

Candidates are unioned, deduplicated, capped at 100 per anchor.

### Pairwise Features (Phase 5)
84 features across 9 groups:
- **Name (14)**: exact, core, sorted, lengths, token Jaccard/overlap, char 3-gram, Levenshtein, Jaro-Winkler, prefix/suffix
- **Address (11)**: exact, sorted, lengths, token Jaccard/overlap, char 3-gram, Levenshtein
- **Structured (10)**: postal (4), house number (4), unit (2)
- **Missingness (12)**: field-level missing indicators
- **Cross-field (5)**: interaction terms (name×address, name×postal, etc.)
- **Country (4)**: exact match, conflict, both present, missing
- **Provenance (13)**: 11 strategy flags + count + multi-strategy
- **Source (3)**: S2, S3, S1 indicators
- **Baseline (5)**: Phase 4 score, rank, and component similarities

### LightGBM Matcher (Phase 6–7)
- **Model A**: Phase 6 baseline (135,342 pairs)
- **Model B**: Phase 7 hard-negative retrained (136,965 pairs, +1,623 mined negatives)
- **Selected**: Model B (superior precision, 27% FP reduction at threshold 0.95)
- Hyperparameters: 300 estimators, lr=0.05, num_leaves=31, scale_pos_weight≈18.94
- Deterministic training (`deterministic=true`, `random_state=2026`)

### Entity-Level Validation (Phase 8)
- Per-S1-entity candidate grouping
- Prediction status: confident / ambiguous / unmatched / conflicting
- Ambiguity analysis (margin-based)
- Conflict analysis (fan-in, multi-source, ties)
- Categorized error analysis (FP/FN)

### Threshold Optimization (Phase 9)
- Grid search on validation set (500 held-out S1 entities)
- Coarse 0.01 step, fine 0.002 step around best
- **Selected: 0.922** (Macro F0.5 = 0.970997)
- Tie-break: highest threshold within 1e-9 tolerance

### Consistency Rules (Phase 10)
- Operating threshold: 0.922
- Duplicate removal: enabled
- Deterministic ordering: probability DESC, candidate_id ASC
- Fan-in resolution: keep all above threshold
- Top-1 only: **disabled** (multi-match allowed)
- Country veto: **disabled**
- Margin veto: **disabled**

### Test Inference (Phase 11)
- Per-country blocking (process isolation, indexes freed)
- Shared-memory record store for scoring
- Chunked by whole S1 anchors (exact per-anchor ranks)
- No full candidate matrix in RAM
- Produces `matching_results.tsv` and `candidate_pairs.tsv`

## Evaluation Metric

**Macro F0.5** — Official challenge metric:
- Per-S1-entity F0.5 (β=0.5, precision-weighted)
- Averaged over all S1 entities
- Singleton S1 entities: score 1.0 if correctly predicted empty, else 0.0
- Pair-level metrics reported for diagnostics only

## Repository Structure

```
.
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CITATION.cff
├── BUILD_GUIDE.md
├── Documentation_template.md
├── requirements.txt
├── pyproject.toml
├── config.yaml
├── data/
│   ├── raw/           # (git-ignored) Challenge files
│   ├── processed/     # (git-ignored)
│   ├── train/         # (git-ignored)
│   ├── validation/    # (git-ignored)
│   └── test/          # (git-ignored)
├── dataset/           # (git-ignored) Challenge archive
├── output/            # (git-ignored) Generated artifacts
├── experiments/       # Experiment tracking
├── notebooks/         # Jupyter exploration
├── scripts/           # Phase execution scripts
├── src/
│   └── business_entity_resolution/  # Main package
│       ├── blocking/
│       ├── config/
│       ├── entity/
│       ├── evaluation/
│       ├── features/
│       ├── inference/
│       ├── ingestion/
│       ├── matching/
│       ├── models/
│       ├── normalization/
│       ├── retrieval/
│       ├── submission/
│       ├── training/
│       ├── validation/
│       └── utils/
├── tests/             # 193 tests (unit + integration)
└── .github/
    └── ISSUE_TEMPLATE/
```

## Installation

```bash
# Clone
git clone https://github.com/SUBRAHMANYA2726/amazon-ml-entity-resolution.git
cd amazon-ml-entity-resolution

# Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# Or with uv (recommended)
uv sync
```

### Dependencies
- Python ≥ 3.11
- Core: `pandas`, `PyYAML`, `rapidfuzz`, `scikit-learn`, `scipy`
- ML: `lightgbm`, `catboost`, `joblib`
- Test: `pytest`

## Usage

### Build Stage (Phases 0–10)
```bash
# Validate configuration
python -m business_entity_resolution --stage validate

# Run individual phases
python scripts/run_phase3_blocking.py
python scripts/run_phase4_baseline.py
python scripts/run_phase5_features.py
python scripts/run_phase6_models.py
python scripts/run_phase7_hard_negatives.py
python scripts/run_phase8_entity_validation.py
python scripts/run_phase9_threshold.py
python scripts/run_phase10_consistency.py
```

### Programmatic Inference
```python
from business_entity_resolution.inference.pipeline import create_inference_pipeline

pipeline = create_inference_pipeline()
result = pipeline.run_batch(s1_df, s2_df, s3_df)
predictions = result.predictions  # dict[s1_id] -> list[candidate_ids]
```

### Test Inference Stage (Phase 11) — Separate Operation
```bash
# Blocking (per-country)
python scripts/run_phase11_inference.py --stage blocking

# Scoring (shared memory)
python scripts/run_phase11_inference.py --stage scoring

# Threshold & consistency
python scripts/run_phase11_inference.py --stage threshold

# Submission
python scripts/run_phase11_submission.py
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run integration test only
python -m pytest tests/test_integration_pipeline.py -v

# Run specific phase tests
python -m pytest tests/test_phase6_models.py -v
python -m pytest tests/test_phase8_entity_validation.py -v
```

All 193 tests pass (7 skipped for artifacts not yet generated in CI).

## Submission Generation

After Phase 11 test inference completes:
```bash
python scripts/run_phase11_submission.py
```

Outputs:
- `output/phase11/matching_results.tsv` — Primary submission (S1_id → matched candidate_ids)
- `output/phase11/candidate_pairs.tsv` — Candidate pairs for validator

## Dataset Setup

**The dataset is NOT included in this repository** due to:
- Size (~2.52 GB)
- Challenge data restrictions

To run the pipeline, obtain the challenge data from the official Amazon ML Challenge 2026 distribution and place it under:
```
dataset/
├── train/
│   ├── source1_train.tsv
│   ├── source2_train.tsv
│   ├── source3_train.tsv
│   └── ground_truth_train.tsv
└── test/
    ├── source1_test.tsv
    ├── source2_test.tsv
    └── source3_test.tsv
```

The pipeline discovers files automatically via `config.yaml` (no hard-coded filenames).

## Reproducibility

- **Global seed**: 2026 (set in `config.yaml`)
- **Model seed**: 2026 (LightGBM `random_state`)
- **Split seed**: 2026 (entity-aware train/val split)
- **Deterministic guarantees**:
  - Feature generation: deterministic
  - Blocking: sorted outputs
  - Model training: `deterministic=true`
  - Inference ordering: probability DESC, candidate_id ASC

## Limitations

- **Validation metrics only**: Test ground truth is unavailable; all reported metrics are on the 500-entity validation partition
- **Candidate ceiling**: Blocking recall upper-bounds final recall; threshold optimization operates only on blocked candidates
- **Residual false positives**: 24 FP pairs at threshold 0.922 (precision 0.9828)
- **False negatives**: 69 FN pairs at threshold 0.922 (recall 0.9599), primarily missing addresses or severe OCR noise
- **Open-set country**: No country canonicalization; relies on model to learn cross-country patterns
- **Memory**: Full test inference requires ~8 GB RAM with streaming; not suitable for extremely limited environments

## License

MIT License — see [LICENSE](LICENSE) for details.

Copyright (c) 2026 Subrahmanya Manjunatha Bhat

## Acknowledgements

- Amazon ML Challenge 2026 organizers
- LightGBM and CatBoost communities
- RapidFuzz for string similarity
- scikit-learn, pandas, NumPy, SciPy ecosystems