# BUILD_GUIDE.md

## Amazon ML Challenge 2026 — Business Entity Resolution
### Production Build Documentation

---

## 1. Repository Structure

```
AMZ ML/
├── config.yaml                    # Centralized production configuration
├── pyproject.toml                 # Project metadata and dependencies
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
├── .gitignore                     # Git ignore rules
├── data/                          # Data directory (git-ignored raw data)
│   ├── raw/                       # Raw challenge files
│   ├── processed/                 # Processed intermediate data
│   ├── train/                     # Training split
│   ├── validation/                # Validation split
│   └── test/                      # Test split (inference target)
├── dataset/                       # Challenge archive (git-ignored)
├── experiments/                   # Experiment tracking
├── notebooks/                     # Jupyter notebooks for exploration
├── output/                        # Generated artifacts (git-ignored large files)
│   ├── phase 3/                   # Phase 3 blocking outputs
│   ├── phase 4/                   # Phase 4 baseline outputs
│   ├── phase5/                    # Phase 5 feature outputs
│   ├── phase 6/                   # Phase 6 model outputs
│   ├── phase 7/                   # Phase 7 hard-negative outputs
│   ├── phase 8/                   # Phase 8 validation outputs
│   ├── phase9/                    # Phase 9 threshold outputs
│   ├── phase10/                   # Phase 10 consistency outputs
│   └── phase11/                   # Phase 11 test inference workspace
├── scripts/                       # Executable phase scripts
│   ├── run_phase3_blocking.py
│   ├── run_phase4_baseline.py
│   ├── run_phase5_features.py
│   ├── run_phase6_models.py
│   ├── run_phase7_hard_negatives.py
│   ├── run_phase8_entity_validation.py
│   ├── run_phase9_threshold.py
│   ├── run_phase10_consistency.py
│   ├── run_phase11_inference.py
│   └── run_phase11_submission.py
├── src/
│   ├── amazon_ml/                 # Legacy module (Phase 0-2)
│   └── business_entity_resolution/ # Main production package
│       ├── blocking/              # Phase 3: Multi-strategy blocking
│       ├── config/                # Configuration management
│       ├── entity/                # Entity consistency contracts
│       ├── evaluation/            # Evaluation metrics
│       ├── features/              # Phase 5: Pairwise features
│       ├── inference/             # Production inference pipeline
│       ├── ingestion/             # Phase 0-1: Data loading & profiling
│       ├── matching/              # Phase 4: Baseline matching
│       ├── models/                # Phase 6-7: ML models
│       ├── normalization/         # Phase 2: Text normalization
│       ├── retrieval/             # Retrieval interfaces
│       ├── submission/            # Submission generation
│       ├── training/              # Training utilities
│       ├── validation/            # Phase 8: Entity validation
│       ├── utils/                 # Utilities (logging, seeding)
│       ├── cli.py                 # Command-line interface
│       ├── contracts.py           # Data contracts
│       └── exceptions.py          # Custom exceptions
└── tests/                         # Comprehensive test suite
```

---

## 2. Environment Setup

### Prerequisites
- Python 3.11+
- 8 GB RAM minimum (12 GB recommended for full test inference)
- Windows/Linux/macOS

### Installation
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or use uv (faster)
uv sync
```

### Key Dependencies
- `pandas`, `numpy` — Data manipulation
- `lightgbm`, `catboost` — Gradient boosted models
- `scikit-learn` — Metrics, preprocessing
- `pyyaml` — Configuration
- `joblib` — Model serialization
- `pytest` — Testing
- `rapidfuzz` — String similarity

---

## 3. Phase 0–10 Pipeline Overview

| Phase | Description | Key Artifact | Script |
|-------|-------------|--------------|--------|
| 0 | Project foundation, config, logging | `config.yaml` | — |
| 1 | Data profiling & discovery | `dataset_profile.json` | `scripts/run_phase1_profiling.py` |
| 2 | Text normalization | `derived_normalization_rules.json` | `scripts/run_phase2_normalization.py` |
| 3 | Multi-strategy blocking | `candidate_pairs.tsv` | `scripts/run_phase3_blocking.py` |
| 4 | Baseline matching | `baseline_ranked_candidates.csv` | `scripts/run_phase4_baseline.py` |
| 5 | Pairwise features | `train_feature_matrix.csv.gz` | `scripts/run_phase5_features.py` |
| 6 | ML model (LightGBM/CatBoost) | `lightgbm_model.joblib` | `scripts/run_phase6_models.py` |
| 7 | Hard-negative mining | `model_b_lightgbm.joblib` | `scripts/run_phase7_hard_negatives.py` |
| 8 | Entity-level validation | `entity_level_validation.csv` | `scripts/run_phase8_entity_validation.py` |
| 9 | F0.5 threshold selection | `selected_threshold.json` | `scripts/run_phase9_threshold.py` |
| 10 | Consistency rules | `consistency_config.json` | `scripts/run_phase10_consistency.py` |

---

## 4. Model Artifact

**Selected Model**: Model B — LightGBM (Phase 7 hard-negative retrained)

- **Location**: `output/phase 8/model_b_lightgbm.joblib`
- **Features**: 84 (exact schema match with `FEATURE_NAMES`)
- **Training pairs**: 136,965 (135,342 Phase 6 + 1,623 hard negatives)
- **Validation pairs**: 34,811 (500 held-out S1 entities)
- **Hyperparameters**:
  - `n_estimators`: 300
  - `learning_rate`: 0.05
  - `num_leaves`: 31
  - `max_depth`: -1
  - `min_child_samples`: 20
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `scale_pos_weight`: 18.944
  - `random_state`: 2026
  - `deterministic`: true

### Loading the Model
```python
from business_entity_resolution.models.pairwise import LightGBMMatcher
matcher = LightGBMMatcher.load("output/phase 8/model_b_lightgbm.joblib")
```

---

## 5. Feature Schema

**84 Features** across 9 groups (defined in `src/business_entity_resolution/features/metadata.py`):

| Group | Features | Count |
|-------|----------|-------|
| A: Name | exact, core, sorted, lengths, token Jaccard, overlap, char 3-gram, Levenshtein, Jaro-Winkler, prefix, suffix | 14 |
| B: Address | exact, sorted, lengths, token Jaccard, overlap, char 3-gram, Levenshtein | 11 |
| C: Structured | postal (4), house_number (4), unit (2) | 10 |
| D: Missingness | name/address/country/postal missing indicators | 12 |
| E: Cross-field | name×address, name×postal, name×country, address×postal, country×postal | 5 |
| F: Country | exact_match, conflict, both_present, missing | 4 |
| G: Provenance | 11 blocking strategies + count + multi-strategy | 13 |
| H: Source | S2, S3, S1 indicators | 3 |
| I: Baseline | score, rank, name_sim, address_sim, structured_sim | 5 |
| **Total** | | **84** |

### Export Feature Metadata
```python
from business_entity_resolution.features.metadata import export_feature_metadata
from pathlib import Path
export_feature_metadata(Path("feature_schema.json"))
```

---

## 6. Threshold

**Final Operating Threshold**: **0.922**

- **Source**: Phase 9 validation on 500 held-out entities
- **Metric**: Macro F0.5 = 0.970997
- **Pair Precision**: 0.98276
- **Pair Recall**: 0.95993
- **Pair F0.5**: 0.97811
- **Exact Entity Set Matches**: 420/500 (84%)
- **Singleton False Positives**: 2
- **Matched Entity False Negatives**: 2

### Loading the Threshold
```python
import json
with open("output/phase9/selected_threshold.json") as f:
    threshold = json.load(f)["selected_threshold"]  # 0.922
```

---

## 7. Consistency Configuration

**Source**: Phase 10 `output/phase10/consistency_config.json`

```json
{
  "operating_threshold": 0.922,
  "duplicate_removal": true,
  "deterministic_ordering": "match_probability DESC, candidate_entity_id ASC",
  "fan_in_resolution": "keep_all_above_threshold_with_safety_check",
  "top1_only": false,
  "country_conflict_veto": false,
  "margin_veto": false,
  "singleton_extra_veto": false,
  "per_source_quota": null
}
```

### Key Behaviors
- **Multi-match allowed**: S1 entities can match multiple candidates if all exceed threshold
- **Deterministic ordering**: Probability descending, candidate ID ascending for tie-breaking
- **No country veto**: Cross-country matches allowed if model scores above threshold
- **No top-1 forcing**: All above-threshold candidates retained

---

## 8. Production Inference Command

### Programmatic Usage
```python
from business_entity_resolution.inference.pipeline import create_inference_pipeline

# Create pipeline (loads all artifacts automatically)
pipeline = create_inference_pipeline(
    model_path="output/phase 8/model_b_lightgbm.joblib",
    threshold_path="output/phase9/selected_threshold.json",
    consistency_path="output/phase10/consistency_config.json",
    config_path="config.yaml",
    batch_size=1000,
)

# Run inference on batch
result = pipeline.run_batch(
    s1_df=source1_dataframe,
    s2_df=source2_dataframe,
    s3_df=source3_dataframe,
)

# Access predictions
predictions = result.predictions  # dict[s1_entity_id] -> list[candidate_entity_id]
```

### CLI Usage (Build Mode)
```bash
# Validate configuration
python -m business_entity_resolution --stage validate

# Run blocking (Phase 3)
python -m business_entity_resolution --stage blocking --num-anchors 2500
```

### Streaming for Large Data
```python
results = pipeline.run_streaming(
    s1_iterator=s1_chunk_iterator,
    s2_iterator=s2_chunk_iterator,
    s3_iterator=s3_chunk_iterator,
    chunk_size=1000,
)
```

---

## 9. Test Inference Command (TEST/INFERENCE STAGE)

**⚠️ SEPARATE OPERATION — NOT PART OF BUILD**

Full test inference is executed separately AFTER build completion:

```bash
# Phase 11a: Blocking (per-country, multiprocessing)
python scripts/run_phase11_inference.py --stage blocking

# Phase 11b: Scoring (shared memory, multiprocessing)
python scripts/run_phase11_inference.py --stage scoring

# Phase 11c: Threshold & consistency
python scripts/run_phase11_inference.py --stage threshold
```

### Memory-Safe Design
- **Blocking**: Per-country process isolation, indexes freed before scoring
- **Scoring**: Single read-only `SharedRecordStore` per country in `multiprocessing.shared_memory`
- **Chunking**: By whole S1 anchors (exact per-anchor ranks)
- **No full candidate matrix in RAM**: Streaming chunk processing

---

## 10. Submission Generation Command (TEST/INFERENCE STAGE)

**⚠️ SEPARATE OPERATION — NOT PART OF BUILD**

```bash
# Generate submission files from test inference results
python scripts/run_phase11_submission.py
```

Outputs:
- `output/phase11/matching_results.tsv` — Primary submission
- `output/phase11/candidate_pairs.tsv` — Candidate pairs for validator

---

## 11. Official Validator Command (TEST/INFERENCE STAGE)

**⚠️ SEPARATE OPERATION — NOT PART OF BUILD**

```bash
# Run official challenge validator
python -m amazon_ml_validator \
    --submission output/phase11/matching_results.tsv \
    --candidates output/phase11/candidate_pairs.tsv \
    --ground-truth dataset/test/ground_truth.tsv
```

---

## 12. BUILD STAGE vs TEST/INFERENCE STAGE

| Aspect | BUILD STAGE | TEST/INFERENCE STAGE |
|--------|-------------|---------------------|
| **Data** | Training/validation only | Full test dataset (S1/S2/S3) |
| **Labels** | Used for training/validation | NEVER used (no test labels exist) |
| **Phases** | 0–10 complete | 11 (inference) → submission → validate |
| **Output** | Model, threshold, config artifacts | `matching_results.tsv`, `candidate_pairs.tsv` |
| **Commands** | `scripts/run_phase*.py` (0-10) | `scripts/run_phase11_*.py` |
| **Memory** | Single-machine fit | Multi-process, shared memory, streaming |

### Build Stage Checklist
- [x] All Phase 0–10 artifacts integrated
- [x] Model B LightGBM loads successfully (84 features)
- [x] Feature schema verified (84 features exact match)
- [x] Threshold 0.922 loaded from Phase 9 artifact
- [x] Consistency config loaded from Phase 10 artifact
- [x] Production inference pipeline exists and tested
- [x] Arbitrary countries supported (open-set)
- [x] Batching supported (configurable batch_size)
- [x] Missing data handled (empty strings, NaN, None)
- [x] Zero-candidate handling implemented
- [x] Multiple-match handling implemented
- [x] Small end-to-end integration test passes
- [x] All 193 existing tests pass
- [x] Documentation complete
- [x] No full TEST inference executed
- [x] No final submission generated
- [x] No official validator executed against test outputs

---

## 13. Git Safety Verification

```bash
# Verify clean state
git status
git diff --stat

# Confirm:
# - dataset/ directory is ignored
# - No credentials or API keys in repo
# - No temporary files tracked
# - No huge generated test artifacts tracked
# - Only source code, configs, tests, docs tracked
```

### .gitignore Key Entries
```
dataset/
output/
*.joblib
*.csv.gz
*.pyc
__pycache__/
.pytest_cache/
.venv/
.env
*.log
```

---

## 14. Reproducibility

### Random Seed
- **Global seed**: 2026 (set in `config.yaml`)
- **Model seed**: 2026 (LightGBM `random_state`)
- **Split seed**: 2026 (entity-aware train/val split)

### Deterministic Guarantees
- All pipelines use fixed seeds
- Feature generation is deterministic
- Blocking is deterministic (sorted outputs)
- Model training is deterministic (`deterministic: true`, `n_jobs: -1`)
- Inference ordering: probability DESC, candidate_id ASC

---

## 15. Troubleshooting

### Common Issues

| Issue | Resolution |
|-------|------------|
| Model feature mismatch | Verify `FEATURE_NAMES` matches model metadata |
| Threshold not loading | Check `output/phase9/selected_threshold.json` exists |
| Blocking returns zero candidates | Check country partitioning, normalization |
| Memory error on test inference | Use streaming mode, reduce batch_size |
| Import errors | Ensure `pip install -e .` or `uv sync` |

### Debug Commands
```bash
# Verify model loads
python -c "from business_entity_resolution.models.pairwise import LightGBMMatcher; m=LightGBMMatcher.load('output/phase 8/model_b_lightgbm.joblib'); print(len(m.feature_names))"

# Verify feature schema
python -c "from business_entity_resolution.features.metadata import FEATURE_NAMES; print(len(FEATURE_NAMES), FEATURE_NAMES[:5])"

# Verify threshold
python -c "import json; print(json.load(open('output/phase9/selected_threshold.json'))['selected_threshold'])"

# Run integration test
python -m pytest tests/test_integration_pipeline.py -v
```

---

## 16. Final Build Status

| Component | Status |
|-----------|--------|
| **BUILD STATUS** | ✅ COMPLETE |
| **MODEL** | Model B LightGBM (Phase 7 hard-negative retrained) |
| **FEATURES** | 84 (verified schema match) |
| **THRESHOLD** | 0.922 (Phase 9 validated) |
| **INTEGRATION TEST** | ✅ PASS (12/12 tests) |
| **FULL TEST INFERENCE** | ⏸️ NOT RUN — intentionally deferred |
| **SUBMISSION** | ⏸️ NOT GENERATED — intentionally deferred |
| **VALIDATOR** | ⏸️ NOT RUN — intentionally deferred |
| **DOCUMENTATION** | `BUILD_GUIDE.md` (this file) |
| **GIT STATUS** | Clean — only source, config, tests tracked |

---

*Build completed: 2026-09-27*
*Pipeline version: Phase 10 production-ready*
*Next stage: Test inference (Phase 11) — execute separately*