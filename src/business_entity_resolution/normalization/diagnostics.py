"""Normalization diagnostics computed from real data.

Every diagnostic answers a question a later phase needs answered before it
trusts a representation: how much did normalization change, what collapsed to
empty, where did distinct raw values collide, and what may have been normalized
too aggressively.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class FieldDiagnostics:
    """Diagnostics for one field of one source file."""

    file_key: str
    column: str
    representation: str
    rows: int = 0
    raw_non_empty: int = 0
    changed_count: int = 0
    changed_percentage: float = 0.0
    empty_after_normalization: int = 0
    empty_percentage: float = 0.0
    raw_distinct: int = 0
    normalized_distinct: int = 0
    collision_group_count: int = 0
    collision_value_count: int = 0
    collision_reduction_percentage: float = 0.0
    mean_token_count: float = 0.0
    median_token_count: float = 0.0
    max_token_count: int = 0
    mean_raw_length: float = 0.0
    mean_normalized_length: float = 0.0
    over_normalized_count: int = 0
    before_after: list[dict[str, str]] = field(default_factory=list)
    collisions: list[dict[str, Any]] = field(default_factory=list)
    over_normalized_examples: list[dict[str, str]] = field(default_factory=list)
    emptied_examples: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the diagnostics."""

        return {
            "file": self.file_key,
            "column": self.column,
            "representation": self.representation,
            "rows": self.rows,
            "raw_non_empty": self.raw_non_empty,
            "changed_count": self.changed_count,
            "changed_percentage": self.changed_percentage,
            "empty_after_normalization": self.empty_after_normalization,
            "empty_percentage": self.empty_percentage,
            "raw_distinct": self.raw_distinct,
            "normalized_distinct": self.normalized_distinct,
            "collision_group_count": self.collision_group_count,
            "collision_value_count": self.collision_value_count,
            "collision_reduction_percentage": self.collision_reduction_percentage,
            "mean_token_count": self.mean_token_count,
            "median_token_count": self.median_token_count,
            "max_token_count": self.max_token_count,
            "mean_raw_length": self.mean_raw_length,
            "mean_normalized_length": self.mean_normalized_length,
            "over_normalized_count": self.over_normalized_count,
            "before_after_examples": self.before_after,
            "collision_examples": self.collisions,
            "over_normalized_examples": self.over_normalized_examples,
            "emptied_examples": self.emptied_examples,
        }


def diagnose_column(
    raw: pd.Series,
    normalized: pd.Series,
    *,
    file_key: str,
    column: str,
    representation: str,
    token_column: pd.Series | None = None,
    over_normalized_ratio: float = 0.4,
    max_examples: int = 8,
) -> FieldDiagnostics:
    """Return the diagnostics of one representation of one column."""

    report = FieldDiagnostics(
        file_key=file_key, column=column, representation=representation, rows=int(len(raw))
    )
    raw_text = raw.fillna("").astype(str)
    normalized_text = normalized.fillna("").astype(str)
    non_empty_mask = raw_text.str.strip().ne("")
    report.raw_non_empty = int(non_empty_mask.sum())

    comparable = non_empty_mask
    changed_mask = comparable & (raw_text != normalized_text)
    report.changed_count = int(changed_mask.sum())
    report.changed_percentage = _percentage(changed_mask.sum(), report.raw_non_empty)

    empty_mask = comparable & normalized_text.str.strip().eq("")
    report.empty_after_normalization = int(empty_mask.sum())
    report.empty_percentage = _percentage(report.empty_after_normalization, report.raw_non_empty)

    if token_column is not None:
        counts = token_column.fillna("").astype(str).map(
            lambda value: len(value.split()) if value else 0
        )
        if len(counts):
            report.mean_token_count = round(float(counts.mean()), 4)
            report.median_token_count = float(counts.median())
            report.max_token_count = int(counts.max())

    raw_lengths = raw_text[comparable].str.len()
    normalized_lengths = normalized_text[comparable].str.len()
    if len(raw_lengths):
        report.mean_raw_length = round(float(raw_lengths.mean()), 4)
        report.mean_normalized_length = round(float(normalized_lengths.mean()), 4)

    ratio_mask = comparable & (raw_lengths > 0)
    short_mask = ratio_mask & (normalized_lengths < raw_lengths * over_normalized_ratio)
    report.over_normalized_count = int(short_mask.sum())

    raw_values = raw_text[comparable]
    normalized_values = normalized_text[comparable]
    report.raw_distinct = int(raw_values.nunique())
    report.normalized_distinct = int(normalized_values.nunique())

    groups = normalized_values.groupby(normalized_values).agg(list)
    sizes = groups.map(len)
    colliding = sizes[sizes > 1]
    report.collision_group_count = int(len(colliding))
    report.collision_value_count = int(colliding.sum()) if len(colliding) else 0
    if report.raw_distinct:
        report.collision_reduction_percentage = _percentage(
            report.raw_distinct - report.normalized_distinct, report.raw_distinct
        )

    report.before_after = _paired_examples(
        raw_text, normalized_text, changed_mask, max_examples
    )
    report.emptied_examples = _paired_examples(
        raw_text, normalized_text, empty_mask, max_examples
    )
    report.over_normalized_examples = _paired_examples(
        raw_text, normalized_text, short_mask, max_examples
    )
    report.collisions = _collision_examples(normalized_text, max_examples)
    return report


def _percentage(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 4) if whole else 0.0


def _paired_examples(
    raw: pd.Series,
    normalized: pd.Series,
    mask: pd.Series,
    limit: int,
) -> list[dict[str, str]]:
    """Return real before/after pairs for the values a mask selects."""

    if not mask.any():
        return []
    selected = pd.DataFrame({"raw": raw[mask], "normalized": normalized[mask]})
    selected = selected[selected["raw"] != selected["normalized"]]
    pairs = [
        {"raw": str(row["raw"]), "normalized": str(row["normalized"])}
        for _, row in selected.head(limit).iterrows()
    ]
    return pairs


def _collision_examples(normalized: pd.Series, limit: int) -> list[dict[str, Any]]:
    """Return real examples of distinct raw values sharing one representation."""

    if normalized.empty:
        return []
    frame = pd.DataFrame({"normalized": normalized})
    frame = frame[frame["normalized"].str.strip().ne("")]
    sizes = frame.groupby("normalized").size()
    targets = set(sizes[sizes > 1].sort_values(ascending=False).head(limit).index)
    if not targets:
        return []
    examples: list[dict[str, Any]] = []
    for value in frame[frame["normalized"].isin(targets)]["normalized"].unique():
        members = frame.loc[frame["normalized"] == value, "normalized"]
        raw_values = sorted({str(item) for item in members})
        examples.append(
            {
                "normalized": str(value),
                "distinct_raw_values": len(raw_values),
                "raw_examples": raw_values[:limit],
            }
        )
        if len(examples) >= limit:
            break
    return examples


def summarize(diagnostics: Sequence[FieldDiagnostics]) -> dict[str, Any]:
    """Return a compact roll-up across every measured field representation."""

    totals: Counter[str] = Counter()
    for item in diagnostics:
        totals["fields"] += 1
        totals["rows"] += item.rows
        totals["raw_non_empty"] += item.raw_non_empty
        totals["changed"] += item.changed_count
        totals["emptied"] += item.empty_after_normalization
        totals["collision_groups"] += item.collision_group_count
        totals["over_normalized"] += item.over_normalized_count
    return {
        "fields_measured": totals["fields"],
        "rows_measured": totals["rows"],
        "raw_non_empty_values": totals["raw_non_empty"],
        "values_changed": totals["changed"],
        "values_changed_percentage": _percentage(totals["changed"], totals["raw_non_empty"]),
        "values_emptied": totals["emptied"],
        "values_emptied_percentage": _percentage(totals["emptied"], totals["raw_non_empty"]),
        "collision_groups": totals["collision_groups"],
        "values_over_normalized": totals["over_normalized"],
    }
