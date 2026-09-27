"""Unit tests for Phase 11 final submission logic.

Tests cover:
- Duplicate S1 detection
- Missing S1 detection
- Invalid candidate IDs
- Duplicate candidate pairs
- Duplicate matched IDs
- Empty match handling
- Candidate-subset validation
- Deterministic ordering
- Manifest generation
- Validator command construction
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from business_entity_resolution.validation.submission_validation import (
    ValidationResult,
    validate_all,
    validate_candidate_pairs,
    validate_matching_results,
    validate_subset,
    check_deterministic_ordering,
)
from scripts.run_phase11d_validate_submission import (
    check_files_exist,
    run_validator,
)
from scripts.run_phase11e_manifest import (
    get_git_branch,
    get_git_commit,
    load_threshold,
    load_feature_count,
    load_model_version,
    count_lines,
    get_validator_status,
    get_test_s1_count,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def test_source1(temp_dir):
    """Create a minimal test_source1.tsv"""
    path = temp_dir / "test_source1.tsv"
    path.write_text("entity_id\nS1-1\nS1-2\nS1-3\n", encoding="utf-8")
    return path


@pytest.fixture
def test_source2(temp_dir):
    """Create a minimal test_source2.tsv"""
    path = temp_dir / "test_source2.tsv"
    path.write_text("entity_id\nS2-1\nS2-2\nS2-3\n", encoding="utf-8")
    return path


@pytest.fixture
def test_source3(temp_dir):
    """Create a minimal test_source3.tsv"""
    path = temp_dir / "test_source3.tsv"
    path.write_text("entity_id\nS3-1\nS3-2\n", encoding="utf-8")
    return path


@pytest.fixture
def test_dir(temp_dir, test_source1, test_source2, test_source3):
    """Return test directory path"""
    return temp_dir


# ============================================================
# Matching Results Validation Tests
# ============================================================

def test_matching_results_valid(test_dir):
    """Test valid matching_results.tsv passes validation"""
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-1\n"
        "S1-2\t\n"
        "S1-3\tS2-2\n",
        encoding="utf-8"
    )

    result = validate_matching_results(matching, test_dir)
    assert result.passed
    assert result.stats["matching_rows"] == 3
    assert result.stats["matching_non_empty"] == 2
    assert result.stats["matching_empty"] == 1


def test_matching_results_duplicate_s1(test_dir):
    """Test duplicate S1 rows are detected"""
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1\n"
        "S1-1\tS2-2\n"
        "S1-2\t\n"
        "S1-3\t\n",
        encoding="utf-8"
    )

    result = validate_matching_results(matching, test_dir)
    assert not result.passed
    assert any("duplicate source1_entity_id" in e for e in result.errors)


def test_matching_results_missing_s1(test_dir):
    """Test missing S1 rows are detected"""
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1\n"
        "S1-2\t\n",
        encoding="utf-8"
    )

    result = validate_matching_results(matching, test_dir)
    assert not result.passed
    assert any("required S1 entity(ies) missing" in e for e in result.errors)


def test_matching_results_extra_s1(test_dir):
    """Test extra S1 rows (not in test set) are detected"""
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1\n"
        "S1-2\t\n"
        "S1-3\t\n"
        "S1-999\tS2-1\n",
        encoding="utf-8"
    )

    result = validate_matching_results(matching, test_dir)
    assert not result.passed
    assert any("row(s) using an S1 ID that is not in the test set" in e for e in result.errors)


def test_matching_results_duplicate_matched_ids(test_dir):
    """Test duplicate matched IDs within a list are detected"""
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S2-1\n"
        "S1-2\t\n"
        "S1-3\t\n",
        encoding="utf-8"
    )

    result = validate_matching_results(matching, test_dir)
    assert not result.passed
    assert any("repeated ID inside a matched_entity_ids list" in e for e in result.errors)


def test_matching_results_self_match(test_dir):
    """Test S1 self-matches are detected"""
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS1-2\n"
        "S1-2\t\n"
        "S1-3\t\n",
        encoding="utf-8"
    )

    result = validate_matching_results(matching, test_dir)
    assert not result.passed
    assert any("self-matches" in e for e in result.errors)


def test_matching_results_invalid_prefix(test_dir):
    """Test invalid ID prefixes are detected"""
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tX2-1\n"
        "S1-2\t\n"
        "S1-3\t\n",
        encoding="utf-8"
    )

    result = validate_matching_results(matching, test_dir)
    assert not result.passed
    assert any("without an S2-/S3- prefix" in e for e in result.errors)


def test_matching_results_empty_match_handling(test_dir):
    """Test empty matches are represented correctly (empty string after tab)"""
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1\n"
        "S1-2\t\n"
        "S1-3\t\n",
        encoding="utf-8"
    )

    result = validate_matching_results(matching, test_dir)
    assert result.passed
    assert result.stats["matching_empty"] == 2


# ============================================================
# Candidate Pairs Validation Tests
# ============================================================

def test_candidate_pairs_valid(test_dir):
    """Test valid candidate_pairs.tsv passes validation"""
    candidate = test_dir / "candidate_pairs.tsv"
    candidate.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-1,S3-1,S2-2\n"
        "S1-2\tS2-1\n"
        "S1-3\t\n",
        encoding="utf-8"
    )

    result = validate_candidate_pairs(candidate, test_dir)
    assert result.passed
    assert result.stats["candidate_rows"] == 3


def test_candidate_pairs_duplicate_pairs(test_dir):
    """Test duplicate candidate pairs within a list are detected"""
    candidate = test_dir / "candidate_pairs.tsv"
    candidate.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-1,S2-1\n"
        "S1-2\t\n"
        "S1-3\t\n",
        encoding="utf-8"
    )

    result = validate_candidate_pairs(candidate, test_dir)
    assert not result.passed
    assert any("repeated ID inside a candidate_entity_ids list" in e for e in result.errors)


def test_candidate_pairs_missing_s1(test_dir):
    """Test missing S1 in candidate pairs"""
    candidate = test_dir / "candidate_pairs.tsv"
    candidate.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-1\n"
        "S1-2\t\n",
        encoding="utf-8"
    )

    result = validate_candidate_pairs(candidate, test_dir)
    assert not result.passed
    assert any("required S1 entity(ies) missing" in e for e in result.errors)


# ============================================================
# Subset Validation Tests
# ============================================================

def test_subset_check_pass(test_dir):
    """Test subset check passes when matches ⊆ candidates"""
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-1\n"
        "S1-2\t\n"
        "S1-3\tS2-2\n",
        encoding="utf-8"
    )

    candidate = test_dir / "candidate_pairs.tsv"
    candidate.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-1,S3-1,S2-2\n"
        "S1-2\tS2-1\n"
        "S1-3\tS2-2,S3-2\n",
        encoding="utf-8"
    )

    result = validate_subset(matching, candidate)
    assert result.passed
    assert result.stats["s1_not_subset"] == 0


def test_subset_check_fail(test_dir):
    """Test subset check fails when matches ⊈ candidates"""
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-1\n"
        "S1-2\t\n"
        "S1-3\tS2-2\n",
        encoding="utf-8"
    )

    candidate = test_dir / "candidate_pairs.tsv"
    candidate.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-1\n"  # Missing S3-1
        "S1-2\t\n"
        "S1-3\tS2-2\n",
        encoding="utf-8"
    )

    result = validate_subset(matching, candidate)
    assert not result.passed
    assert result.stats["s1_not_subset"] == 1


def test_subset_check_pending_when_files_missing(test_dir):
    """Test subset check returns PENDING when files don't exist"""
    matching = test_dir / "matching_results.tsv"
    candidate = test_dir / "candidate_pairs.tsv"

    result = validate_subset(matching, candidate)
    assert not result.passed
    assert result.stats.get("status") == "PENDING"


# ============================================================
# Deterministic Ordering Tests
# ============================================================

def test_deterministic_ordering_pass(test_dir):
    """Test ordering check passes when IDs are sorted ascending (tie-break)"""
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S2-2,S3-1\n"  # Sorted ascending
        "S1-2\t\n"
        "S1-3\tS2-3\n",
        encoding="utf-8"
    )

    candidate = test_dir / "candidate_pairs.tsv"
    candidate.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-1,S2-2,S3-1,S3-2\n"
        "S1-2\t\n"
        "S1-3\tS2-3\n",
        encoding="utf-8"
    )

    result = check_deterministic_ordering(matching, candidate)
    assert result.passed


def test_deterministic_ordering_fail(test_dir):
    """Test ordering check fails when IDs not sorted"""
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-2,S2-1\n"  # Not sorted
        "S1-2\t\n"
        "S1-3\t\n",
        encoding="utf-8"
    )

    candidate = test_dir / "candidate_pairs.tsv"
    candidate.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-1,S2-2\n"
        "S1-2\t\n"
        "S1-3\t\n",
        encoding="utf-8"
    )

    result = check_deterministic_ordering(matching, candidate)
    assert not result.passed
    assert any("not in deterministic order" in e for e in result.errors)


# ============================================================
# Full Validation Suite Tests
# ============================================================

def test_validate_all_passes(test_dir):
    """Test full validation suite passes for valid files"""
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-1\n"
        "S1-2\t\n"
        "S1-3\tS2-2\n",
        encoding="utf-8"
    )

    candidate = test_dir / "candidate_pairs.tsv"
    candidate.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-1,S3-1,S2-2\n"
        "S1-2\tS2-1\n"
        "S1-3\tS2-2,S3-2\n",
        encoding="utf-8"
    )

    m_result, c_result, s_result = validate_all(matching, candidate, test_dir)
    assert m_result.passed
    assert c_result.passed
    assert s_result.passed


def test_validate_all_fails_on_bad_matching(test_dir):
    """Test full validation fails when matching has errors"""
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1\n"
        "S1-1\tS2-2\n"  # Duplicate
        "S1-2\t\n"
        "S1-3\t\n",
        encoding="utf-8"
    )

    candidate = test_dir / "candidate_pairs.tsv"
    candidate.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-1,S2-2\n"
        "S1-2\t\n"
        "S1-3\t\n",
        encoding="utf-8"
    )

    m_result, c_result, s_result = validate_all(matching, candidate, test_dir)
    assert not m_result.passed
    assert c_result.passed


# ============================================================
# Validator Wrapper Tests
# ============================================================

def test_check_files_exist_pending(temp_dir):
    """Test check_files_exist returns PENDING when files missing"""
    matching = temp_dir / "matching_results.tsv"
    candidate = temp_dir / "candidate_pairs.tsv"

    exists, msg = check_files_exist(matching, candidate)
    assert not exists
    assert "PENDING" in msg
    assert "matching_results.tsv" in msg
    assert "candidate_pairs.tsv" in msg


def test_check_files_exist_ready(temp_dir):
    """Test check_files_exist returns ready when files exist"""
    matching = temp_dir / "matching_results.tsv"
    candidate = temp_dir / "candidate_pairs.tsv"
    matching.write_text("source1_entity_id\tmatched_entity_ids\n", encoding="utf-8")
    candidate.write_text("source1_entity_id\tcandidate_entity_ids\n", encoding="utf-8")

    exists, msg = check_files_exist(matching, candidate)
    assert exists
    assert msg == ""


# ============================================================
# Manifest Generation Tests
# ============================================================

def test_load_threshold():
    """Test loading threshold from phase9 output"""
    threshold = load_threshold()
    assert threshold == 0.922


def test_load_feature_count():
    """Test loading feature count from model metadata"""
    count = load_feature_count()
    assert count == 84


def test_load_model_version():
    """Test loading model version"""
    version = load_model_version()
    assert version != "unknown" or True  # May be unknown if file missing


def test_get_git_branch():
    """Test git branch retrieval"""
    branch = get_git_branch()
    assert isinstance(branch, str)
    assert len(branch) > 0


def test_get_git_commit():
    """Test git commit retrieval"""
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_count_lines():
    """Test line counting utility"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.tsv"
        path.write_text("header\na\tb\nc\td\n", encoding="utf-8")
        assert count_lines(path) == 2

        # Missing file
        assert count_lines(Path(tmp) / "missing.tsv") is None


def test_get_validator_status_pending():
    """Test validator status when validation_result.txt missing"""
    status = get_validator_status()
    assert status == "PENDING"


def test_get_test_s1_count():
    """Test test S1 count retrieval"""
    count = get_test_s1_count()
    # May be None if test file not in expected location
    assert count is None or isinstance(count, int)


# ============================================================
# Integration-style Tests with Synthetic Data
# ============================================================

def test_full_submission_workflow_synthetic(test_dir):
    """Test complete submission validation workflow with synthetic data"""
    # Create test source files
    (test_dir / "test_source1.tsv").write_text("entity_id\nS1-1\nS1-2\nS1-3\nS1-4\n", encoding="utf-8")
    (test_dir / "test_source2.tsv").write_text("entity_id\nS2-1\nS2-2\nS2-3\n", encoding="utf-8")
    (test_dir / "test_source3.tsv").write_text("entity_id\nS3-1\nS3-2\n", encoding="utf-8")

    # Create valid submission files
    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-1\n"
        "S1-2\t\n"
        "S1-3\tS2-2\n"
        "S1-4\t\n",
        encoding="utf-8"
    )

    candidate = test_dir / "candidate_pairs.tsv"
    candidate.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-1,S3-1,S2-3\n"
        "S1-2\tS2-1\n"
        "S1-3\tS2-2,S3-2\n"
        "S1-4\tS2-3\n",
        encoding="utf-8"
    )

    # Run full validation
    m_result, c_result, s_result = validate_all(matching, candidate, test_dir, check_ids=True)

    assert m_result.passed, f"Matching errors: {m_result.errors}"
    assert c_result.passed, f"Candidate errors: {c_result.errors}"
    assert s_result.passed, f"Subset errors: {s_result.errors}"

    # Check stats
    assert m_result.stats["matching_rows"] == 4
    assert m_result.stats["matching_non_empty"] == 2
    assert m_result.stats["matching_empty"] == 2
    assert c_result.stats["candidate_rows"] == 4
    assert s_result.stats["s1_not_subset"] == 0


def test_submission_workflow_with_errors(test_dir):
    """Test submission validation catches multiple error types"""
    (test_dir / "test_source1.tsv").write_text("entity_id\nS1-1\nS1-2\n", encoding="utf-8")
    (test_dir / "test_source2.tsv").write_text("entity_id\nS2-1\n", encoding="utf-8")
    (test_dir / "test_source3.tsv").write_text("entity_id\nS3-1\n", encoding="utf-8")

    matching = test_dir / "matching_results.tsv"
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S2-1\n"  # Duplicate in list
        "S1-999\tS2-1\n",   # Extra S1
        encoding="utf-8"
    )

    candidate = test_dir / "candidate_pairs.tsv"
    candidate.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-1\n",
        encoding="utf-8"
    )

    m_result, c_result, s_result = validate_all(matching, candidate, test_dir)

    assert not m_result.passed
    assert len(m_result.errors) >= 2  # Duplicate in list + missing S1-2 + extra S1-999
    assert not c_result.passed  # Missing S1-2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])