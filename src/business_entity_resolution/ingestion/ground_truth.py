"""Ground-truth analysis restricted to the training split.

The ground truth is read but never modified, and never used to alter any raw
field. Every statistic below is counted from the training file itself.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from business_entity_resolution.config.settings import DatasetSettings
from business_entity_resolution.ingestion.dataset import LoadedFile, iter_chunks
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)

LIST_SEPARATOR = ","
WELL_FORMED_ID = re.compile(r"^S[123]-\d+$")


def analyse_ground_truth(
    loaded: LoadedFile,
    settings: DatasetSettings,
    source_column: str,
    matched_column: str,
) -> dict[str, Any]:
    """Return the full training ground-truth structure and match cardinality."""

    match_histogram: Counter[int] = Counter()
    contribution: Counter[str] = Counter()
    source_prefixes: Counter[str] = Counter()
    matched_prefixes: Counter[str] = Counter()
    malformed: list[str] = []
    total_rows = 0
    rows_with_matches = 0
    total_matched_ids = 0
    whitespace_padded_ids = 0
    unsorted_lists = 0
    separator_edge_rows = 0
    empty_string_match_rows = 0
    duplicate_id_rows = 0
    duplicate_id_occurrences = 0
    max_list_length = 0
    source_key_chunks: list[np.ndarray] = []
    matched_key_chunks: list[np.ndarray] = []
    matched_key_prefix_chunks: list[np.ndarray] = []
    composite_key_chunks: list[np.ndarray] = []

    for chunk in iter_chunks(loaded, settings):
        total_rows += len(chunk)
        left = chunk[source_column]
        right = chunk[matched_column]

        source_key_chunks.append(_id_keys(left))
        for value in left.fillna("").tolist():
            prefix = prefix_of(value)
            source_prefixes[prefix] += 1

        right_filled = right.fillna("")
        empty_string_match_rows += int((right_filled.eq("") & right.notna()).sum())
        for text in right_filled.tolist():
            if text == "":
                match_histogram[0] += 1
                continue
            if text != text.strip():
                whitespace_padded_ids += 1
            if text.startswith(LIST_SEPARATOR) or text.endswith(LIST_SEPARATOR):
                separator_edge_rows += 1
                if len(malformed) < 200:
                    malformed.append(text[:120])
            raw_items = text.split(LIST_SEPARATOR)
            items = [item.strip() for item in raw_items]
            if any(item != original for item, original in zip(items, raw_items)):
                whitespace_padded_ids += 1
            items = [item for item in items if item]
            total_matched_ids += len(items)
            match_histogram[len(items)] += 1
            max_list_length = max(max_list_length, len(items))
            if not items:
                continue
            rows_with_matches += 1
            if len(items) != len(set(items)):
                duplicate_id_rows += 1
                duplicate_id_occurrences += len(items) - len(set(items))
            if items != sorted(items):
                unsorted_lists += 1
            keys: list[int] = []
            for item in items:
                prefix = prefix_of(item)
                matched_prefixes[prefix] += 1
                contribution[prefix] += 1
                if not WELL_FORMED_ID.match(item) and len(malformed) < 200:
                    malformed.append(item)
                keys.append(_id_key(item))
            matched_key_chunks.append(np.asarray(keys, dtype=np.int64))
            matched_key_prefix_chunks.append(
                np.asarray([prefix_index(prefix_of(item)) for item in items], dtype=np.int64)
            )
            composite_key_chunks.append(
                np.asarray(
                    [prefix_index(prefix_of(item)) * 10**12 + _id_key(item) for item in items],
                    dtype=np.int64,
                )
            )

    rows_with_match_by_prefix = _rows_with_match_by_prefix(
        loaded, settings, source_column, matched_column
    )

    distribution = {str(key): int(value) for key, value in sorted(match_histogram.items())}
    distinct_sources = _distinct_int64(source_key_chunks)
    reuse = _matched_record_reuse(composite_key_chunks)

    return {
        "file": loaded.as_dict(),
        "source_column": source_column,
        "matched_column": matched_column,
        "list_separator": LIST_SEPARATOR,
        "total_rows": total_rows,
        "distinct_source1_ids": distinct_sources,
        "duplicate_source1_rows": total_rows - distinct_sources,
        "source_id_is_unique": total_rows == distinct_sources,
        "source_id_prefixes": dict(sorted(source_prefixes.items())),
        "source_id_prefix_rows": {
            prefix: {
                "rows": source_prefixes[prefix],
                "rows_with_at_least_one_match": rows_with_match_by_prefix.get(prefix, 0),
                "rows_with_no_match": source_prefixes[prefix]
                - rows_with_match_by_prefix.get(prefix, 0),
            }
            for prefix in sorted(source_prefixes)
        },
        "matched_id_prefixes": dict(sorted(matched_prefixes.items())),
        "s1_entities_with_zero_matches": distribution.get("0", 0),
        "s1_entities_with_one_match": distribution.get("1", 0),
        "s1_entities_with_multiple_matches": int(
            sum(value for key, value in match_histogram.items() if key > 1)
        ),
        "match_count_distribution": distribution,
        "total_matched_ids": total_matched_ids,
        "max_matches_per_s1": max_list_length,
        "rows_with_at_least_one_match": rows_with_matches,
        "average_matches_per_s1": round(total_matched_ids / total_rows, 6)
        if total_rows
        else 0.0,
        "median_matches_per_s1": _median_from_histogram(match_histogram, total_rows),
        "min_matches_per_s1": min(match_histogram) if match_histogram else 0,
        "match_contribution_by_prefix": dict(sorted(contribution.items())),
        "duplicate_id_rows": duplicate_id_rows,
        "duplicate_id_occurrences": duplicate_id_occurrences,
        "whitespace_padded_id_count": whitespace_padded_ids,
        "unsorted_id_list_rows": unsorted_lists,
        "separator_edge_case_rows": separator_edge_rows,
        "empty_string_match_rows": empty_string_match_rows,
        "malformed_entry_count": len(malformed),
        "malformed_entry_examples": malformed[:10],
        "matched_record_reuse": reuse,
        "_source_keys": np.concatenate(source_key_chunks) if source_key_chunks else np.empty(0, np.int64),
        "_matched_keys": np.concatenate(matched_key_chunks)
        if matched_key_chunks
        else np.empty(0, np.int64),
        "_matched_key_prefixes": np.concatenate(matched_key_prefix_chunks)
        if matched_key_prefix_chunks
        else np.empty(0, np.int64),
    }


def _matched_record_reuse(composite_key_chunks: list[np.ndarray]) -> dict[str, Any]:
    """Measure how often one S2/S3 record is claimed by more than one S1 entity.

    A record appearing in several match lists is the main entity-level leakage
    risk: a row-level split would place the same record in two folds.
    """

    if not composite_key_chunks:
        return {
            "distinct_matched_records": 0,
            "records_matched_by_multiple_s1_entities": 0,
            "records_matched_by_multiple_s1_entities_percentage": 0.0,
            "max_s1_entities_per_matched_record": 0,
        }
    stacked = np.unique(np.concatenate(composite_key_chunks))
    counts = np.unique(np.concatenate(composite_key_chunks), return_counts=True)[1]
    reused = int((counts > 1).sum())
    return {
        "distinct_matched_records": int(stacked.size),
        "records_matched_by_multiple_s1_entities": reused,
        "records_matched_by_multiple_s1_entities_percentage": round(
            100.0 * reused / stacked.size, 6
        ),
        "max_s1_entities_per_matched_record": int(counts.max()),
    }


def _rows_with_match_by_prefix(
    loaded: LoadedFile,
    settings: DatasetSettings,
    source_column: str,
    matched_column: str,
) -> dict[str, int]:
    """Return how many rows of each source-id prefix declare a match."""

    rows_with_match: Counter[str] = Counter()
    for chunk in iter_chunks(loaded, settings, columns=[source_column, matched_column]):
        matched = chunk[matched_column].fillna("").str.strip().ne("")
        for value in chunk[source_column][matched].fillna("").tolist():
            rows_with_match[prefix_of(value)] += 1
    return dict(rows_with_match)


def _median_from_histogram(histogram: Counter[int], total: int) -> float:
    """Return an exact median match count from a complete match-count histogram."""

    if not total:
        return 0.0
    cumulative = 0
    for size in sorted(histogram):
        cumulative += histogram[size]
        if cumulative * 2 >= total:
            return float(size)
    return float(max(histogram))


def prefix_of(value: str) -> str:
    """Return the identifier prefix an entity id declares through its own text."""

    text = str(value).strip()
    head = text.split("-", 1)[0]
    if not head or head == text:
        return "<unprefixed>"
    return head


def prefix_index(prefix: str) -> int:
    """Return a stable integer code for an observed identifier prefix."""

    order = ("S1", "S2", "S3")
    return order.index(prefix) if prefix in order else len(order)


def _id_key(value: object) -> int:
    """Return the exact integer body of an identifier, or -1 when absent.

    Only the segment after the final ``-`` is considered, so the digits inside a
    source prefix are never mixed into the numeric body.
    """

    text = str(value).strip()
    tail = text.rsplit("-", 1)[-1] if "-" in text else text
    digits = "".join(character for character in tail if character.isdigit())
    return int(digits) if digits else -1


def _id_keys(series: pd.Series) -> np.ndarray:
    """Return the integer body of every identifier in a column."""

    return (
        series.fillna("")
        .str.extract(r"(\d+)\s*$", expand=False)
        .fillna("")
        .map(_digits_only)
        .to_numpy(dtype=np.int64)
    )


def _digits_only(value: str) -> int:
    return int(value) if value else -1


def _distinct_int64(chunks: list[np.ndarray]) -> int:
    """Return the exact distinct count of a chunked int64 key stream."""

    if not chunks:
        return 0
    return int(np.unique(np.concatenate(chunks)).size)


def validate_ground_truth_ids(
    ground_truth: dict[str, Any],
    source_keys_by_prefix: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Check every declared ground-truth id against the training source keys.

    Comparison is exact on the integer body of each identifier. A declared id is
    reported as resolvable only when its prefix and its integer body both exist
    in the corresponding training source file.
    """

    matched_keys = ground_truth["_matched_keys"]
    matched_prefixes = ground_truth["_matched_key_prefixes"]
    order = ("S1", "S2", "S3")
    report: dict[str, Any] = {
        "total_declared_matched_ids": int(matched_keys.size),
        "per_prefix": {},
    }
    for code, prefix in enumerate(order):
        mask = matched_prefixes == code
        keys = matched_keys[mask]
        source = source_keys_by_prefix.get(prefix)
        entry: dict[str, Any] = {"declared_matches": int(keys.size)}
        if source is None:
            entry["source_key_available"] = False
            report["per_prefix"][prefix] = entry
            continue
        source_sorted = np.unique(source)
        entry["source_key_available"] = True
        entry["source_ids"] = int(source_sorted.size)
        position = np.searchsorted(source_sorted, keys)
        position_clipped = np.clip(position, 0, source_sorted.size - 1)
        found = source_sorted[position_clipped] == keys
        entry["resolvable_ids"] = int(found.sum())
        entry["unresolvable_ids"] = int((~found).sum())
        entry["unresolvable_examples"] = [
            f"{prefix}-{int(value)}" for value in keys[~found][:5]
        ]
        report["per_prefix"][prefix] = entry
    unresolvable_total = sum(
        entry.get("unresolvable_ids", 0) for entry in report["per_prefix"].values()
    )
    report["unresolvable_total"] = int(unresolvable_total)
    report["unresolvable_percentage"] = (
        round(100.0 * unresolvable_total / matched_keys.size, 6)
        if matched_keys.size
        else 0.0
    )
    return report


def coverage_against_sources(
    ground_truth_keys: np.ndarray,
    source_keys: np.ndarray,
    label: str,
    source_prefix: str = "S1",
) -> dict[str, Any]:
    """Return the exact overlap between ground-truth ids and a source file."""

    left = np.unique(ground_truth_keys)
    right = np.unique(source_keys)
    missing = np.setdiff1d(left, right, assume_unique=True)
    extra = np.setdiff1d(right, left, assume_unique=True)
    return {
        "label": label,
        "ground_truth_distinct_ids": int(left.size),
        "source_distinct_ids": int(right.size),
        "ground_truth_ids_missing_from_source": int(missing.size),
        "source_ids_absent_from_ground_truth": int(extra.size),
        "overlap": int(left.size - missing.size),
        "missing_examples": [f"{source_prefix}-{int(value)}" for value in missing[:5]],
        "absent_examples": [f"{source_prefix}-{int(value)}" for value in extra[:5]],
    }
