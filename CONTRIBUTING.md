# Contributing to Amazon ML Challenge 2026 — Business Entity Resolution

Thank you for your interest in contributing! This document outlines the guidelines for contributing to this project.

## Project Setup

1. **Fork and clone** the repository
2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # Or with uv:
   uv sync
   ```
4. **Run tests** to verify setup:
   ```bash
   python -m pytest tests/ -v
   ```

## Branch Workflow

- **main/master**: Stable, production-ready code
- **feature/**: New features or improvements (`feature/description`)
- **fix/**: Bug fixes (`fix/description`)
- **docs/**: Documentation updates (`docs/description`)

### Pull Request Process

1. Create a feature branch from `master`
2. Make focused, atomic commits
3. Ensure all tests pass (`python -m pytest tests/`)
4. Update documentation if behavior changes
5. Open a Pull Request against `master`
6. Address review feedback
7. Squash and merge after approval

## Coding Standards

- **Python ≥ 3.11** syntax
- **Type hints** on all public functions and classes
- **Docstrings** for all public modules, classes, and functions (Google/NumPy style)
- **Line length**: 100 characters (soft), 120 (hard)
- **Imports**: grouped (stdlib, third-party, local), sorted alphabetically
- **No dead code** — remove unused imports, functions, variables

### Tools

- Formatting: `black` (if configured) or consistent manual style
- Linting: `ruff` or `flake8` (if configured)
- Type checking: `mypy` (if configured)

Run checks before committing:
```bash
python -m pytest tests/ -v
```

## Testing Requirements

- **All new code must have tests**
- Unit tests for individual functions/classes
- Integration tests for pipeline stages
- Tests must be deterministic (fixed seeds)
- No test should depend on external services or the challenge dataset

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_integration_pipeline.py -v

# With coverage (if configured)
python -m pytest tests/ --cov=src/business_entity_resolution
```

## Pull Request Expectations

- **Clear title** and description explaining the change
- **Linked issue** (if applicable)
- **Tests added/updated** for the change
- **Documentation updated** (README, docstrings, BUILD_GUIDE.md if relevant)
- **No breaking changes** without discussion
- **CI passes** (all tests, linting, type checks)

## Security & Privacy

- **No secrets** in commits (API keys, tokens, passwords, credentials)
- **No dataset uploads** — the challenge dataset (~2.52 GB) is excluded via `.gitignore`
- **No large generated artifacts** — `output/`, model files, caches are git-ignored
- Use `.env` for local configuration (never committed)

## Reproducibility

- Preserve the **global random seed** (2026 in `config.yaml`)
- Document any changes affecting reproducibility
- Maintain deterministic pipeline behavior
- Schema contracts (`contracts.py`) should remain stable

## Reporting Issues

- Use GitHub Issues for bugs and feature requests
- Provide minimal reproduction steps
- Include relevant logs/error messages
- Specify Python version and OS

## Questions?

Open a GitHub Discussion or Issue for questions about the codebase, architecture, or contribution process.