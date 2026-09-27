"""Submission file validation for Phase 11 final output.

Validates matching_results.tsv and candidate_pairs.tsv against the challenge
requirements. Reusable, does not require test inference to have completed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DELIM = "\t"
MAX_EXAMPLES = 5
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    errors: list[str]
    warnings: list[str]
    stats: dict

    def __bool__(self) -> bool:
        return self.passed


def read_ids(path: Path) -> set[str]:
    """Return the set of first-column entity IDs from a source TSV."""
    with open(path, encoding="utf-8") as f:
        next(f, None)
        return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}


def examples(items: set[str]) -> str:
    """Return a short, human-readable sample of items for an error message."""
    items = sorted(items)
    shown = ", ".join(items[:MAX_EXAMPLES])
    if len(items) > MAX_EXAMPLES:
        return f"{len(items)} total, e.g. {shown}, ..."
    return shown


def load_match_targets(test_dir: Path, warnings: list[str]) -> Optional[set[str]]:
    """Return the set of valid S2/S3 match IDs, or None if unavailable."""
    targets = set()
    for name in ("test_source2.tsv", "test_source3.tsv"):
        path = test_dir / name
        if not path.is_file():
            warnings.append(
                f"{path} not found — skipping the (optional) check that matched "
                f"IDs exist in the test set. Every other rule is still checked. "
                f"This is the lighter-memory mode; provide test_source2/3.tsv to "
                f"enable the ID-existence check."
            )
            return None
        targets |= read_ids(path)
    return targets


def validate_id_list_file(
    path: Path,
    expected_header: list[str],
    col_label: str,
    required: set[str],
    valid_ids: Optional[set[str]],
    errors: list[str],
    warnings: list[str],
) -> Optional[dict[str, set[str]]]:
    """Validate one results-style TSV (matching or candidate).

    Returns a {source1_id: set(matched/candidate ids)} mapping, or None on a
    fatal problem (missing file, empty file, or a broken header) that stops parsing.
    """
    if not path.is_file():
        errors.append(f"File not found: {path}")
        return None

    name = path.name
    mapping = {}
    seen: set[str] = set()
    dup_rows: set[str] = set()
    intra_dupes: set[str] = set()
    self_matches: set[str] = set()
    wrong_prefix: set[str] = set()
    unknown: set[str] = set()
    n_rows = 0
    empties = 0

    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if not header:
            errors.append(f"{name} is empty.")
            return None
        if DELIM not in header and "," in header:
            errors.append(
                f"{name}: header has no TAB but contains commas — the file looks "
                "COMMA-separated. Submissions must be TAB-separated (.tsv); "
                "write it with df.to_csv(sep='\\t', index=False)."
            )
            return None
        cols = [c.strip().lower() for c in header.rstrip("\n").split(DELIM)]
        if cols != expected_header:
            errors.append(
                f"{name}: unexpected header {cols}. "
                f"Expected exactly {expected_header} (tab-separated)."
            )
            return None

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    errors.append(
                        f"{name}: malformed row (no tab) at line {line_num}: "
                        f"{line.rstrip()!r}"
                    )
                continue

            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)

            ids = rest.rstrip("\n").split(",") if rest.strip() else []
            if not ids:
                empties += 1
                mapping[s1] = set()
                continue
            if len(ids) != len(set(ids)):
                intra_dupes.add(s1)
            id_set = set(ids)
            mapping[s1] = id_set
            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)
                elif valid_ids is not None and mid not in valid_ids:
                    unknown.add(mid)

    findings = [
        (
            dup_rows,
            f"{name}: duplicate source1_entity_id row(s): {{ex}}. "
            "Each S1 entity may appear on only one row.",
        ),
        (
            intra_dupes,
            f"{name}: repeated ID inside a {col_label} list for: {{ex}}. "
            "No duplicate IDs are allowed within a list.",
        ),
        (
            self_matches,
            f"{name}: {col_label} contains Source-1 IDs (self-matches): {{ex}}. "
            "Only S2-/S3- IDs are allowed.",
        ),
        (
            wrong_prefix,
            f"{name}: {col_label} contains IDs without an S2-/S3- prefix: {{ex}}.",
        ),
        (
            unknown,
            f"{name}: {col_label} references IDs not in the test "
            "Source-2/3 files: {{ex}}.",
        ),
        (
            required - seen,
            f"{name}: required S1 entity(ies) missing: {{ex}}. "
            "Every entity in test_source1.tsv needs a row (empty = no match).",
        ),
        (
            seen - required,
            f"{name}: row(s) using an S1 ID that is not in the test set: {{ex}}.",
        ),
    ]
    for offenders, message in findings:
        if offenders:
            errors.append(message.format(ex=examples(offenders)))

    warnings.append(f"  {name}: {n_rows} rows ({empties} empty, {n_rows - empties} non-empty).")
    return mapping


def validate_matching_results(
    matching_path: Path,
    test_dir: Path,
    check_ids: bool = False,
) -> ValidationResult:
    """Validate matching_results.tsv against challenge requirements."""
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    source1 = test_dir / "test_source1.tsv"
    if not source1.is_file():
        errors.append(f"Test source1 file not found: {source1} (check test_dir).")
        return ValidationResult(False, errors, warnings, stats)

    required = read_ids(source1)
    stats["required_s1"] = len(required)

    valid_ids = None
    if check_ids:
        valid_ids = load_match_targets(test_dir, warnings)
        if valid_ids is not None:
            stats["valid_s2_s3_ids"] = len(valid_ids)
    else:
        warnings.append(
            "ID-existence check is OFF (the default) — not checking that matched "
            "IDs exist in the test set. Every other rule is still checked. "
            "Re-run with check_ids=True to enable it (needs test_source2/3.tsv; uses "
            "more memory). A nonexistent ID only lowers your score, never rejects "
            "your submission."
        )

    matched = validate_id_list_file(
        matching_path, MATCHING_HEADER, "matched_entity_ids", required, valid_ids,
        errors, warnings,
    )

    if matched is not None:
        stats["matching_rows"] = len(matched)
        stats["matching_non_empty"] = sum(1 for v in matched.values() if v)
        stats["matching_empty"] = sum(1 for v in matched.values() if not v)
        stats["total_matched_ids"] = sum(len(v) for v in matched.values())
        stats["unique_matched_ids"] = len(set().union(*matched.values())) if matched else 0

    passed = len(errors) == 0
    return ValidationResult(passed, errors, warnings, stats)


def validate_candidate_pairs(
    candidate_path: Path,
    test_dir: Path,
    check_ids: bool = False,
) -> ValidationResult:
    """Validate candidate_pairs.tsv against challenge requirements."""
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    source1 = test_dir / "test_source1.tsv"
    if not source1.is_file():
        errors.append(f"Test source1 file not found: {source1} (check test_dir).")
        return ValidationResult(False, errors, warnings, stats)

    required = read_ids(source1)
    stats["required_s1"] = len(required)

    valid_ids = None
    if check_ids:
        valid_ids = load_match_targets(test_dir, warnings)
        if valid_ids is not None:
            stats["valid_s2_s3_ids"] = len(valid_ids)
    else:
        warnings.append(
            "ID-existence check is OFF (the default) — not checking that candidate "
            "IDs exist in the test set. Every other rule is still checked."
        )

    candidate = validate_id_list_file(
        candidate_path, CANDIDATE_HEADER, "candidate_entity_ids", required, valid_ids,
        errors, warnings,
    )

    if candidate is not None:
        stats["candidate_rows"] = len(candidate)
        stats["candidate_non_empty"] = sum(1 for v in candidate.values() if v)
        stats["candidate_empty"] = sum(1 for v in candidate.values() if not v)
        stats["total_candidate_ids"] = sum(len(v) for v in candidate.values())
        stats["unique_candidate_ids"] = len(set().union(*candidate.values())) if candidate else 0

    passed = len(errors) == 0
    return ValidationResult(passed, errors, warnings, stats)


def validate_subset(
    matching_path: Path,
    candidate_path: Path,
) -> ValidationResult:
    """Validate that FINAL_MATCHES ⊆ CANDIDATE_PAIRS.

    Returns ValidationResult with the subset check results.
    If either file does not exist, returns PENDING status.
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    if not matching_path.is_file():
        warnings.append(f"matching_results.tsv not found: {matching_path} — subset check PENDING")
        return ValidationResult(False, errors, warnings, {"status": "PENDING", "reason": "matching file missing"})

    if not candidate_path.is_file():
        warnings.append(f"candidate_pairs.tsv not found: {candidate_path} — subset check PENDING")
        return ValidationResult(False, errors, warnings, {"status": "PENDING", "reason": "candidate file missing"})

    # Load both files
    matched_map: dict[str, set[str]] = {}
    candidate_map: dict[str, set[str]] = {}

    with open(matching_path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        if header != "source1_entity_id\tmatched_entity_ids":
            errors.append(f"matching_results.tsv: unexpected header {header}")
        else:
            for line in f:
                s1, _, rest = line.rstrip("\n").partition("\t")
                ids = rest.split(",") if rest.strip() else []
                matched_map[s1] = set(ids)

    with open(candidate_path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        if header != "source1_entity_id\tcandidate_entity_ids":
            errors.append(f"candidate_pairs.tsv: unexpected header {header}")
        else:
            for line in f:
                s1, _, rest = line.rstrip("\n").partition("\t")
                ids = rest.split(",") if rest.strip() else []
                candidate_map[s1] = set(ids)

    # Check subset
    not_subset = {
        s1 for s1, mids in matched_map.items()
        if mids - candidate_map.get(s1, set())
    }

    stats["s1_with_matches"] = sum(1 for v in matched_map.values() if v)
    stats["s1_with_candidates"] = sum(1 for v in candidate_map.values() if v)
    stats["s1_not_subset"] = len(not_subset)
    if not_subset:
        stats["example_not_subset"] = examples(not_subset)
        warnings.append(
            f"{len(not_subset)} S1 entity(ies) have matched IDs not present in "
            f"candidate_pairs.tsv, e.g. {examples(not_subset)}. Final matches "
            "normally come from your blocking candidates — double-check these."
        )
    else:
        stats["example_not_subset"] = ""

    passed = len(errors) == 0 and len(not_subset) == 0
    return ValidationResult(passed, errors, warnings, stats)


def validate_all(
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path,
    check_ids: bool = False,
) -> tuple[ValidationResult, ValidationResult, ValidationResult]:
    """Run all validation checks.

    Returns (matching_result, candidate_result, subset_result).
    """
    matching_result = validate_matching_results(matching_path, test_dir, check_ids)
    candidate_result = validate_candidate_pairs(candidate_path, test_dir, check_ids)
    subset_result = validate_subset(matching_path, candidate_path)
    return matching_result, candidate_result, subset_result


def check_deterministic_ordering(
    matching_path: Path,
    candidate_path: Path,
) -> ValidationResult:
    """Verify deterministic ordering: proba DESC, candidate id ASC.

    This check assumes the files were generated by the Phase 11c assembler
    which sorts by (-proba, candidate_id). Since we don't have probabilities
    here, we can only verify that within each S1, candidate IDs are sorted
    ascending (which is the tie-breaker when probabilities are equal).
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    for path, col_name in [(matching_path, "matched_entity_ids"), (candidate_path, "candidate_entity_ids")]:
        if not path.is_file():
            warnings.append(f"{path.name} not found — ordering check PENDING")
            continue

        with open(path, encoding="utf-8") as f:
            f.readline()  # header
            for line in f:
                s1, _, rest = line.rstrip("\n").partition("\t")
                ids = rest.split(",") if rest.strip() else []
                if len(ids) > 1:
                    # Check if IDs are sorted ascending (tie-breaker for equal proba)
                    if ids != sorted(ids):
                        errors.append(
                            f"{path.name}: S1 {s1} {col_name} not in deterministic order "
                            f"(expected ascending ID for tie-break): {ids}"
                        )
                        stats.setdefault("ordering_violations", []).append(s1)

    stats["files_checked"] = sum(1 for p in [matching_path, candidate_path] if p.is_file())
    passed = len(errors) == 0
    return ValidationResult(passed, errors, warnings, stats)