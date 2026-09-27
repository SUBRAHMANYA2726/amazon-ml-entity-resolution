"""Phase 11c — Assemble final submission artifacts + validate.

Merges per-country pair checkpoints and probability checkpoints, applies the
Phase 9 operating threshold (0.922) plus Phase 10 retained rules
(intra-S1 dedup + deterministic ordering + fan-in safety), writes
output/matching_results.tsv and output/candidate_pairs.tsv, runs the full
audit battery and the official challenge validator.

Usage:
    python scripts/run_phase11c_assemble_submission.py
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from business_entity_resolution.utils.logging import configure_logging, get_logger
from business_entity_resolution.utils.seed import set_random_seed

LOGGER = get_logger("phase11c_runner")

TEST_DIR = Path("dataset/student_resource/dataset/test")
WORK_DIR = Path("output/phase11/work")
OUTPUT_DIR = Path("output")
COUNTRIES = ("india", "us", "france")
READ_KWARGS = {"sep": "\t", "dtype": str, "na_filter": False}
CACHE_READ_KWARGS = {"sep": ",", "dtype": str, "na_filter": False}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 11c assemble + validate.")
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--skip-validator", action="store_true")
    p.add_argument("--check-ids", action="store_true",
                   help="Also run validator with --check-ids (memory-heavy).")
    return p.parse_args()


def load_threshold(override: float | None) -> float:
    if override is not None:
        return float(override)
    with open("output/phase9/selected_threshold.json", encoding="utf-8") as f:
        return float(json.load(f)["selected_threshold"])


def assemble_country(country: str, threshold: float):
    """Yield (s1_id, candidate_ids_sorted, matched_ids_sorted) per anchor."""
    roster = pd.read_csv(WORK_DIR / f"anchors_{country}.csv.gz", **CACHE_READ_KWARGS)["entity_id"].astype(str).tolist()
    s1_ids = np.load(WORK_DIR / f"store_s1_ids_{country}.npy", allow_pickle=True).astype(str)
    cand_ids = np.load(WORK_DIR / f"store_cand_ids_{country}.npy", allow_pickle=True).astype(str)
    s1_index = {s: i for i, s in enumerate(s1_ids)}

    pairs_dir = WORK_DIR / f"pairs_{country}"
    proba_dir = WORK_DIR / f"proba_{country}"
    chunk_files = sorted(pairs_dir.glob("chunk_*.csv.gz"))

    # Accumulate per-anchor candidate/proba lists in roster order.
    cand_lists: dict[str, list] = {}
    for ci, cf in enumerate(chunk_files):
        pairs = pd.read_csv(cf, **CACHE_READ_KWARGS)
        npz = np.load(proba_dir / f"chunk_{ci:04d}.npz")
        assert len(pairs) == len(npz["proba"]), f"pairs/proba mismatch {country} chunk {ci}"
        s_list = pairs["source1_entity_id"].astype(str).tolist()
        c_list = pairs["candidate_entity_id"].astype(str).tolist()
        p_list = npz["proba"].astype(np.float64)
        # Cross-check integer keys agree with string ids.
        assert (s1_ids[npz["s1_idx"]] == np.array(s_list)).all()
        assert (cand_ids[npz["cand_idx"]] == np.array(c_list)).all()
        for s, c, p in zip(s_list, c_list, p_list):
            cand_lists.setdefault(s, []).append((c, float(p)))
        del pairs, npz
        gc.collect()

    n_match_pairs = 0
    n_zero_match = 0
    for s in roster:
        items = cand_lists.get(s, [])
        # Deterministic ordering: proba DESC, candidate id ASC.
        items.sort(key=lambda t: (-t[1], t[0]))
        seen = set()
        cands_out, match_out = [], []
        for c, p in items:
            if c in seen:
                continue
            seen.add(c)
            cands_out.append(c)
            if p >= threshold:
                match_out.append(c)
        n_match_pairs += len(match_out)
        if not match_out:
            n_zero_match += 1
        yield s, cands_out, match_out
    LOGGER.info("[%s] assembled: anchors=%d match_pairs=%d zero_match=%d",
                country, len(roster), n_match_pairs, n_zero_match)


def main() -> int:
    from business_entity_resolution.config.settings import load_settings

    args = parse_args()
    settings = load_settings(args.config)
    configure_logging(settings.logging)
    set_random_seed(settings.random_seed)
    t_start = time.time()

    threshold = load_threshold(args.threshold)
    LOGGER.info("Assembling submission with threshold %.4f", threshold)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("output/phase11").mkdir(parents=True, exist_ok=True)
    matching_path = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

    # Test S1 roster in FILE order (validator needs every test S1 exactly once).
    test_s1_order: list[str] = []
    for chunk in pd.read_csv(TEST_DIR / "test_source1.tsv", usecols=["entity_id"],
                             chunksize=500_000, **READ_KWARGS):
        test_s1_order.extend(chunk["entity_id"].astype(str).tolist())
    test_s1_set = set(test_s1_order)
    assert len(test_s1_set) == len(test_s1_order), "duplicate S1 in test file"
    LOGGER.info("Test S1 entities: %d", len(test_s1_order))

    per_country_rows: dict[str, dict[str, tuple[list, list]]] = {}
    for c in COUNTRIES:
        d = {}
        for s, cands_out, match_out in assemble_country(c, threshold):
            d[s] = (cands_out, match_out)
        per_country_rows[c] = d

    # Coverage check before writing.
    covered = set()
    for d in per_country_rows.values():
        covered.update(d.keys())
    missing = test_s1_set - covered
    extra = covered - test_s1_set
    LOGGER.info("covered=%d missing=%d extra=%d", len(covered), len(missing), len(extra))
    if missing or extra:
        LOGGER.critical("S1 coverage mismatch: missing=%d extra=%d", len(missing), len(extra))
        return 1

    stats = {"threshold": threshold, "test_s1": len(test_s1_order),
             "match_pairs": 0, "cand_pairs": 0, "zero_match": 0, "non_empty_match": 0,
             "per_country": {}}
    with open(matching_path, "w", encoding="utf-8") as fm, \
            open(candidate_path, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for s in test_s1_order:
            found = False
            for c in COUNTRIES:
                if s in per_country_rows[c]:
                    cands_out, match_out = per_country_rows[c][s]
                    found = True
                    break
            assert found, s
            fm.write(f"{s}\t{','.join(match_out)}\n")
            fc.write(f"{s}\t{','.join(cands_out)}\n")
            stats["match_pairs"] += len(match_out)
            stats["cand_pairs"] += len(cands_out)
            if match_out:
                stats["non_empty_match"] += 1
            else:
                stats["zero_match"] += 1
    del per_country_rows
    gc.collect()
    LOGGER.info("Wrote %s and %s: %s", matching_path, candidate_path, stats)

    # ---- Audit battery ----
    LOGGER.info("Running submission audit battery...")
    audit: dict = {"stats": stats, "checks": {}}

    # 1. Row-count / dup / format audit via chunked read.
    seen_s1: set[str] = set()
    dup_s1 = 0
    n_rows = 0
    bad_prefix = 0
    intra_dup = 0
    self_match = 0
    match_ids: set[str] = set()
    with open(matching_path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        assert header == "source1_entity_id\tmatched_entity_ids", header
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            n_rows += 1
            if s1 in seen_s1:
                dup_s1 += 1
            seen_s1.add(s1)
            ids = rest.split(",") if rest.strip() else []
            if len(ids) != len(set(ids)):
                intra_dup += 1
            for mid in ids:
                match_ids.add(mid)
                if mid.startswith("S1-"):
                    self_match += 1
                elif not (mid.startswith("S2-") or mid.startswith("S3-")):
                    bad_prefix += 1
    audit["checks"]["matching_rows"] = n_rows
    audit["checks"]["matching_dup_s1"] = dup_s1
    audit["checks"]["matching_intra_dup_lists"] = intra_dup
    audit["checks"]["matching_self_matches"] = self_match
    audit["checks"]["matching_bad_prefix"] = bad_prefix
    audit["checks"]["matching_missing_s1"] = len(test_s1_set - seen_s1)
    audit["checks"]["matching_extra_s1"] = len(seen_s1 - test_s1_set)
    audit["checks"]["unique_matched_ids"] = len(match_ids)

    n_crows = 0
    seen_c: set[str] = set()
    cand_ids_all: dict[str, set[str]] = {}
    with open(candidate_path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        assert header == "source1_entity_id\tcandidate_entity_ids", header
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            n_crows += 1
            seen_c.add(s1)
            ids = rest.split(",") if rest.strip() else []
            cand_ids_all[s1] = set(ids)
    audit["checks"]["candidate_rows"] = n_crows
    audit["checks"]["candidate_missing_s1"] = len(test_s1_set - seen_c)

    # 2. Subset check: matches ⊆ candidates (streaming, bounded memory).
    not_subset = 0
    with open(matching_path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            ids = rest.split(",") if rest.strip() else []
            if set(ids) - cand_ids_all.get(s1, set()):
                not_subset += 1
    audit["checks"]["matches_not_subset_of_candidates"] = not_subset
    del cand_ids_all
    gc.collect()

    # 3. Existence check: every matched ID occurs in test S2/S3 (chunked scan).
    remaining = set(match_ids)
    del match_ids
    for name in ("test_source2.tsv", "test_source3.tsv"):
        if not remaining:
            break
        for chunk in pd.read_csv(TEST_DIR / name, usecols=["entity_id"],
                                 chunksize=500_000, **READ_KWARGS):
            remaining -= set(chunk["entity_id"].astype(str).tolist())
            if not remaining:
                break
        gc.collect()
    audit["checks"]["matched_ids_not_in_test_s2_s3"] = len(remaining)
    audit["elapsed_seconds"] = round(time.time() - t_start, 2)
    with open("output/phase11/submission_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    LOGGER.info("Audit: %s", json.dumps(audit, indent=2))

    hard_fail = (
        audit["checks"]["matching_dup_s1"] != 0
        or audit["checks"]["matching_missing_s1"] != 0
        or audit["checks"]["matching_extra_s1"] != 0
        or audit["checks"]["matching_self_matches"] != 0
        or audit["checks"]["matching_bad_prefix"] != 0
        or audit["checks"]["matching_intra_dup_lists"] != 0
        or audit["checks"]["matches_not_subset_of_candidates"] != 0
        or audit["checks"]["matched_ids_not_in_test_s2_s3"] != 0
        or audit["checks"]["candidate_rows"] != len(test_s1_order)
        or n_rows != len(test_s1_order)
    )
    if hard_fail:
        LOGGER.critical("Submission audit FAILED")
        return 1

    # ---- Official validator ----
    if not args.skip_validator:
        cmd = [sys.executable, "dataset/student_resource/utils/validate_submission.py",
               "--matching", str(matching_path), "--candidate", str(candidate_path),
               "--test-dir", str(TEST_DIR)]
        LOGGER.info("Running official validator: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        with open("output/phase11/validator_result.txt", "w", encoding="utf-8") as f:
            f.write(f"CMD: {' '.join(cmd)}\nReturn code: {proc.returncode}\n\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n")
        if proc.returncode != 0:
            LOGGER.critical("Official validator FAILED")
            return 1
        if args.check_ids:
            cmd2 = cmd + ["--check-ids"]
            LOGGER.info("Running official validator with --check-ids ...")
            proc2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=7200)
            print(proc2.stdout)
            with open("output/phase11/validator_result_check_ids.txt", "w", encoding="utf-8") as f:
                f.write(f"CMD: {' '.join(cmd2)}\nReturn code: {proc2.returncode}\n\nSTDOUT:\n{proc2.stdout}\nSTDERR:\n{proc2.stderr}\n")
    LOGGER.info("Phase 11c complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
