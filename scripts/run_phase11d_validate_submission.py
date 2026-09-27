"""Phase 11d — Official validator wrapper.

Executes the official challenge validator, captures stdout/stderr, and saves
the result to output/phase11/validation_result.txt.

If submission files do not yet exist, reports PENDING — does not report PASS.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MATCHING = Path("output/phase11/matching_results.tsv")
DEFAULT_CANDIDATE = Path("output/phase11/candidate_pairs.tsv")
DEFAULT_TEST_DIR = Path("dataset/student_resource/dataset/test")
VALIDATOR_SCRIPT = Path("dataset/student_resource/utils/validate_submission.py")
OUTPUT_FILE = Path("output/phase11/validation_result.txt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 11d official validator wrapper.")
    p.add_argument("--matching", type=Path, default=DEFAULT_MATCHING)
    p.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    p.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument("--check-ids", action="store_true",
                   help="Also run validator with --check-ids (memory-heavy).")
    p.add_argument("--output", type=Path, default=OUTPUT_FILE)
    return p.parse_args()


def check_files_exist(matching: Path, candidate: Path) -> tuple[bool, str]:
    """Check if submission files exist. Returns (exists, message)."""
    missing = []
    if not matching.is_file():
        missing.append(f"matching_results.tsv: {matching}")
    if not candidate.is_file():
        missing.append(f"candidate_pairs.tsv: {candidate}")
    if missing:
        return False, "PENDING — submission files not generated yet:\n  " + "\n  ".join(missing)
    return True, ""


def run_validator(
    matching: Path,
    candidate: Path,
    test_dir: Path,
    check_ids: bool = False,
) -> tuple[int, str, str]:
    """Run the official validator. Returns (returncode, stdout, stderr)."""
    if not VALIDATOR_SCRIPT.is_file():
        return -1, "", f"Official validator not found at {VALIDATOR_SCRIPT}"

    cmd = [
        sys.executable, str(VALIDATOR_SCRIPT),
        "--matching", str(matching),
        "--candidate", str(candidate),
        "--test-dir", str(test_dir),
    ]
    if check_ids:
        cmd.append("--check-ids")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Validator timed out after 3600 seconds"
    except Exception as e:
        return -1, "", f"Failed to run validator: {e}"


def main() -> int:
    args = parse_args()
    start_time = time.time()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Check if submission files exist
    files_exist, msg = check_files_exist(args.matching, args.candidate)

    output_lines = [
        "Phase 11d — Official Validator Wrapper",
        "=" * 50,
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Matching file: {args.matching}",
        f"Candidate file: {args.candidate}",
        f"Test directory: {args.test_dir}",
        f"Check IDs: {args.check_ids}",
        "",
    ]

    if not files_exist:
        output_lines.extend([
            "STATUS: PENDING",
            msg,
            "",
            f"Elapsed: {time.time() - start_time:.2f}s",
        ])
        result_text = "\n".join(output_lines)
        print(result_text)
        args.output.write_text(result_text, encoding="utf-8")
        print(f"Result written to {args.output}")
        return 0  # Not an error, just pending

    output_lines.append("STATUS: RUNNING")
    output_lines.append("")

    # Run validator
    print("Running official validator...")
    returncode, stdout, stderr = run_validator(
        args.matching, args.candidate, args.test_dir, args.check_ids
    )

    elapsed = time.time() - start_time

    output_lines.extend([
        f"Return code: {returncode}",
        f"Elapsed: {elapsed:.2f}s",
        "",
        "STDOUT:",
        stdout or "(empty)",
        "",
        "STDERR:",
        stderr or "(empty)",
        "",
    ])

    if returncode == 0:
        output_lines.append("VALIDATOR RESULT: PASS")
    else:
        output_lines.append("VALIDATOR RESULT: FAIL")

    result_text = "\n".join(output_lines)
    print(result_text)

    args.output.write_text(result_text, encoding="utf-8")
    print(f"Result written to {args.output}")

    return 0 if returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())