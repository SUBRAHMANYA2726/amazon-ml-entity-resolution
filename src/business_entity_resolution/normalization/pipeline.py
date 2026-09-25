"""The normalization pipeline: raw values in, raw-plus-derived columns out.

The pipeline is a pure representation builder. It never mutates an input frame,
never replaces a raw column, and never emits a match decision. A frame coming
out of :func:`normalize_frame` still contains every original column with its
original bytes, plus new columns named ``<column>__<representation>``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from business_entity_resolution.config.settings import NormalizationSettings
from business_entity_resolution.normalization.fields import (
    STAGE_NAMES,
    FieldResult,
    NormalizerRegistry,
    classify_column,
)
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)

REPRESENTATION_SEPARATOR = "__"
RAW_SUFFIX = "raw"
TOKEN_REPRESENTATIONS = ("tokens", "alnum_tokens", "char_ngrams")
EXTRA_REPRESENTATIONS: Mapping[str, tuple[str, ...]] = {
    "name": ("name_core", "name_sorted", "name_tokens"),
    "address": (
        "address_sorted",
        "address_components",
        "address_marker_free",
        "address_numbers",
        "address_house_number",
        "address_postal",
        "address_unit",
    ),
    "country": ("country", "country_token"),
    "identifier": ("id_prefix", "id_body"),
    "generic": (),
}


@dataclass(frozen=True, slots=True)
class NormalizationPlan:
    """The exact representation columns a schema will produce."""

    columns: tuple[str, ...]
    representations_by_column: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the plan."""

        return {
            "columns": list(self.columns),
            "representations_by_column": {
                column: list(values) for column, values in self.representations_by_column.items()
            },
        }


class NormalizationPipeline:
    """Turn raw columns into side-by-side comparison representations."""

    def __init__(self, settings: NormalizationSettings) -> None:
        if settings is None:
            raise ValueError("Normalization settings are required")
        self.settings = settings
        self.registry = NormalizerRegistry(settings)

    def plan(self, columns: Iterable[str]) -> NormalizationPlan:
        """Return the representation columns a schema will produce."""

        produced: list[str] = []
        per_column: dict[str, tuple[str, ...]] = {}
        for column in columns:
            names = representations_for(column)
            per_column[column] = names
            produced.extend(
                f"{column}{REPRESENTATION_SEPARATOR}{name}" for name in names
            )
        return NormalizationPlan(columns=tuple(produced), representations_by_column=per_column)

    def representations_for(self, column: str) -> tuple[str, ...]:
        """Return every representation name produced for a column."""

        return representations_for(column)

    def normalize_value(self, column: str, value: object) -> str:
        """Return only the field-specific representation of one value."""

        return self.registry.normalize_value(column, value)

    def normalize_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Return one flat record of raw and derived values.

        The raw entries are copied through unchanged and are always present.
        """

        payload: dict[str, Any] = dict(record)
        for column, value in record.items():
            if column is None:
                continue
            result = self.registry.normalize(column, value)
            for name, produced in _representation_values(result).items():
                payload[f"{column}{REPRESENTATION_SEPARATOR}{name}"] = produced
        return payload

    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a new frame with raw columns preserved plus derived columns."""

        if frame.empty and not len(frame.columns):
            return frame.copy()
        normalized = frame.copy(deep=False)
        for column in frame.columns:
            for name, values in self._column_values(frame[column], column).items():
                normalized[f"{column}{REPRESENTATION_SEPARATOR}{name}"] = values
        return normalized

    def _column_values(self, series: pd.Series, column: str) -> dict[str, pd.Series]:
        """Return every representation of one column as a Series."""

        collected: dict[str, list[Any]] = {name: [] for name in representations_for(column)}
        for value in series.tolist():
            result = self.registry.normalize(column, value)
            values = _representation_values(result)
            for name in collected:
                collected[name].append(values.get(name))
        return {name: pd.Series(values, index=series.index, dtype="object")
                for name, values in collected.items()}


def representations_for(column: str) -> tuple[str, ...]:
    """Return the representation names produced for a column, in order."""

    kind = classify_column(column)
    names = [RAW_SUFFIX, *STAGE_NAMES, kind, *TOKEN_REPRESENTATIONS]
    names.extend(EXTRA_REPRESENTATIONS.get(kind, ()))
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def _representation_values(result: FieldResult | None) -> dict[str, Any]:
    """Flatten one field result into a name-to-value mapping."""

    if result is None:
        return {}
    return result.as_dict()


def normalize_records(
    pipeline: NormalizationPipeline,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize a sequence of records, preserving every raw value."""

    return [pipeline.normalize_record(record) for record in records]
