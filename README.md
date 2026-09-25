# Amazon ML Challenge 2026: Business Entity Resolution

This repository is the engineering foundation for a high-precision business
entity-resolution solution for the Amazon ML Challenge 2026. The eventual
system will map noisy records from the challenge's candidate sources to the
deduplicated reference source and optimize the official Macro F0.5 metric.

## Current Status

**Phase 0 is the only implemented phase.** The repository contains a runnable
package skeleton, immutable configuration, logging, generic pipeline
interfaces, and tests. It does not contain challenge data, schema assumptions,
normalization rules, blocking implementations, model training, metrics, or
submission files.

Dataset-specific work begins only after the official files are available and
their filenames, schemas, types, missingness, identifiers, source structure,
ground-truth format, train/test split, and submission contract have been
inspected and recorded.

## Architecture

The future pipeline is intentionally staged:

```text
Ingestion -> Profiling -> Normalization -> Multi-strategy blocking
-> Candidate union/deduplication -> Pairwise features -> ML matcher
-> Hard-negative mining -> Threshold selection -> Entity consistency
-> Singleton audit -> Submission generation -> Official validation
```

The architecture protects candidate recall before model tuning, then optimizes
precision-aware Macro F0.5 at the entity level. It will use only supplied
challenge data. External business lookups, APIs, geocoding, registries, and
data enrichment are prohibited. Country handling remains open-set; no fixed
country universe is encoded in Phase 0.

## Repository Layout

```text
.
├── config.yaml                         # Generic Phase 0 settings
├── configs/                            # Future reviewed configuration variants
├── data/                               # Local challenge data; not tracked by Git
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
├── src/business_entity_resolution/
│   ├── config/                         # Typed configuration model and loader
│   ├── ingestion/                      # Ingestion, profiling, schema contracts
│   ├── normalization/                  # Raw-preserving normalization contract
│   ├── blocking/                       # Bounded candidate blocking contract
│   ├── retrieval/                      # Candidate union/retrieval contract
│   ├── features/                       # Pairwise feature contract
│   ├── models/                         # Generic matcher contract
│   ├── training/                       # Training and hard-negative contracts
│   ├── evaluation/                     # Evaluation and threshold contracts
│   ├── entity/                         # Consistency and singleton contracts
│   ├── inference/                      # Future inference orchestration contract
│   ├── submission/                     # Future official-format generation contract
│   ├── validation/                     # Official validator integration contract
│   └── utils/                          # Logging and deterministic seed utilities
├── tests/                              # Dataset-independent pytest coverage
├── notebooks/                          # Empty Phase 0 notebook placeholders
├── experiments/                        # Future experiment metadata; no results yet
├── output/                             # Reserved for future validated artifacts
├── code/                               # Reserved for the final self-contained package
├── scripts/                            # Reserved for reviewed operational scripts
└── utils/                              # Reserved for challenge-supplied utilities
```

## Installation

Python 3.11 or later is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate the virtual environment with
`source .venv/bin/activate`.

## Phase 0 Commands

Validate the generic configuration without reading data or creating artifacts:

```powershell
ber-phase0 --config config.yaml
# or
python -m business_entity_resolution --config config.yaml
```

Run the foundation tests:

```powershell
python -m pytest
```

## Development Workflow

1. Create a focused feature branch from the agreed integration branch.
2. Keep data, generated artifacts, and experiment outputs out of Git.
3. Add tests with each reusable component.
4. Run `python -m pytest` and inspect `git diff` before requesting review.
5. Use clear commits such as `feat:`, `fix:`, `test:`, `docs:`, or `perf:`.
6. Merge only through the team's review process; do not push or merge from an
   automation session without explicit approval.

## Dataset Arrival Gate

When the official challenge data arrives, stop before implementing matching
logic and create a deterministic audit that reports:

1. Exact filenames and sizes.
2. Exact columns and data types.
3. Record counts and missingness.
4. Identifier uniqueness and source structure.
5. Ground-truth and train/test format.
6. Entity cardinality, candidate-retrieval opportunities, and leakage risks.
7. Official output and validator requirements.

Only after that audit is reviewed should Phase 1 begin.
