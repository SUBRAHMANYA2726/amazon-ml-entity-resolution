"""Phase 11e — Submission manifest generator.

Creates output/phase11/submission_manifest.json with actual values when available,
null/PENDING for values not yet available.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


MANIFEST_PATH = Path("output/phase11/submission_manifest.json")
PHASE9_THRESHOLD = Path("output/phase9/selected_threshold.json")
PHASE9_SUMMARY = Path("output/phase9/phase9_summary.json")
PHASE10_AUDIT = Path("output/phase10/phase10_audit.json")
MODEL_PATH = Path("output/phase 8/model_b_lightgbm.joblib")
MODEL_META = Path("output/phase 8/model_b_metadata.json")
MATCHING_PATH = Path("output/phase11/matching_results.tsv")
CANDIDATE_PATH = Path("output/phase11/candidate_pairs.tsv")
VALIDATION_RESULT = Path("output/phase11/validation_result.txt")


def get_git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def load_threshold() -> float:
    if PHASE9_THRESHOLD.is_file():
        with open(PHASE9_THRESHOLD, encoding="utf-8") as f:
            return float(json.load(f).get("selected_threshold", 0.922))
    return 0.922


def load_feature_count() -> int:
    if MODEL_META.is_file():
        with open(MODEL_META, encoding="utf-8") as f:
            meta = json.load(f)
            return len(meta.get("feature_column_order", []))
    return 84


def load_model_version() -> str:
    if MODEL_META.is_file():
        with open(MODEL_META, encoding="utf-8") as f:
            meta = json.load(f)
            return meta.get("model_version", "unknown")
    return "unknown"


def count_lines(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            # Skip header
            next(f, None)
            return sum(1 for _ in f)
    except Exception:
        return None


def get_validator_status() -> str:
    if not VALIDATION_RESULT.is_file():
        return "PENDING"
    try:
        content = VALIDATION_RESULT.read_text(encoding="utf-8")
        if "VALIDATOR RESULT: PASS" in content:
            return "PASS"
        elif "VALIDATOR RESULT: FAIL" in content:
            return "FAIL"
        elif "STATUS: PENDING" in content:
            return "PENDING"
        else:
            return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def get_test_s1_count() -> int | None:
    """Get the number of test S1 entities from the test file."""
    test_file = Path("dataset/student_resource/dataset/test/test_source1.tsv")
    if test_file.is_file():
        try:
            with open(test_file, encoding="utf-8") as f:
                next(f, None)
                return sum(1 for _ in f)
        except Exception:
            pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 11e submission manifest generator.")
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Collect data
    threshold = load_threshold()
    feature_count = load_feature_count()
    model_version = load_model_version()
    test_s1_count = get_test_s1_count()
    matching_rows = count_lines(MATCHING_PATH)
    candidate_rows = count_lines(CANDIDATE_PATH)
    validator_status = get_validator_status()
    git_branch = get_git_branch()
    git_commit = get_git_commit()

    manifest = {
        "project": "Amazon ML Challenge 2026 — Business Entity Resolution",
        "branch": git_branch,
        "commit": git_commit,
        "model_path": str(MODEL_PATH) if MODEL_PATH.is_file() else None,
        "model_version": model_version,
        "feature_count": feature_count,
        "final_threshold": threshold,
        "matching_results_path": str(MATCHING_PATH) if MATCHING_PATH.is_file() else None,
        "candidate_pairs_path": str(CANDIDATE_PATH) if CANDIDATE_PATH.is_file() else None,
        "validator_status": validator_status,
        "test_s1_count": test_s1_count,
        "matching_rows": matching_rows,
        "candidate_rows": candidate_rows,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Replace None with "PENDING" for JSON readability
    for key, value in manifest.items():
        if value is None:
            manifest[key] = "PENDING"

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {args.output}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())