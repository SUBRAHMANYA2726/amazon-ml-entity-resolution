"""Deterministic, memory-bounded profiling of the discovered training data.

Every number this module reports is computed from the real bytes on disk. Two
measurement methods are used and always disclosed:

* ``exact`` - counted directly from every row (row counts, nulls, lengths,
  country labels, identifier structure, duplicate rows).
* ``uint64_content_hash`` - distinct-value counting over a 64-bit content hash.
  Two different values collide with probability below 1e-5 at this scale, and
  the method is recorded alongside every distinct count.
* ``sample`` - computed on a deterministic leading-row sample, with the sample
  size recorded. Used only for the exploratory pattern battery.

No value is ever invented, defaulted, or carried over from documentation.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from business_entity_resolution.config.settings import DatasetSettings
from business_entity_resolution.ingestion.dataset import (
    LoadedFile,
    iter_chunks,
    read_filtered,
    read_header,
    read_sample,
)
from business_entity_resolution.ingestion.discovery import DiscoveredDataset
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)

HASH_METHOD = "uint64_content_hash"
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
ZERO_WIDTH = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
WHITESPACE_RUN = re.compile(r"\s+")
NON_ASCII = re.compile(r"[^\x00-\x7f]")
PUNCTUATION_CHARS = ".,;:!?\"()[]{}/\\|@#$%^*+=<>~`_"
EMAIL_LIKE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_LIKE = re.compile(r"(?:\+\d[\d\s().-]{6,}\d)")
US_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
NUMERIC_ZIP = re.compile(r"\b\d{6}\b")
ALNUM_RUN = re.compile(r"[0-9a-z]+")

PATTERN_BATTERY: dict[str, str] = {
    "ampersand": r"&",
    "word_and": r"\band\b",
    "period": r"\.",
    "comma": r",",
    "hyphen": r"-",
    "slash": r"/",
    "apostrophe": r"'",
    "parenthesis": r"[()]",
    "hash": r"#",
    "ampersand_word": r"\bW\b",
    "dba": r"\bd\.?b\.?a\.?\b",
    "underscore": r"_",
    "percent": r"%",
    "at_sign": r"@",
    "double_space": r"  ",
    "leading_space": r"^ ",
    "trailing_space": r" $",
    "non_ascii": NON_ASCII.pattern,
    "control_character": CONTROL_CHARACTERS.pattern,
    "zero_width": ZERO_WIDTH.pattern,
    "has_digit": r"\d",
    "digit_run_6_plus": r"\d{6,}",
    "email_like": EMAIL_LIKE.pattern,
    "phone_like": PHONE_LIKE.pattern,
    "us_zip_like": US_ZIP.pattern,
    "numeric_zip_like": NUMERIC_ZIP.pattern,
    "newline": r"[\r\n]",
    "tab_inside_value": r"\t",
}

UNIT_MARKER_PATTERN = (
    r"\b(?:apt|apartment|suite|ste|unit|flat|rm|room|fl|floor|flr|bldg|building|"
    r"shop|office|opp|opposite|near|behind|beside|next)\b\.?"
)
STREET_DESIGNATOR_PATTERN = (
    r"\b(?:st|str|street|rd|road|ave|av|avenue|blvd|boulevard|dr|drive|ln|lane|"
    r"ct|court|cir|circle|pl|place|pkwy|parkway|hwy|highway|aly|alley|ter|terrace|"
    r"trl|trail|way|pkg|plz|plaza|square|sq|gt|ground|nr|number|sec|section)\b\.?"
)
DIRECTION_PATTERN = r"\b(?:north|south|east|west|ne|nw|se|sw|n|s|e|w)\b\.?"


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Full-dataset statistics for one real column of one real file."""

    name: str
    row_count: int
    non_null_count: int
    null_count: int
    null_percentage: float
    distinct_count: int
    distinct_method: str
    duplicate_value_count: int
    inferred_dtype: str
    min_length: int | None
    max_length: int | None
    mean_length: float | None
    median_length: float | None
    whitespace_only_count: int
    top_values: tuple[tuple[str, int], ...] = ()
    sample_values: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the column statistics."""

        return {
            "name": self.name,
            "row_count": self.row_count,
            "non_null_count": self.non_null_count,
            "null_count": self.null_count,
            "null_percentage": self.null_percentage,
            "distinct_count": self.distinct_count,
            "distinct_method": self.distinct_method,
            "duplicate_value_count": self.duplicate_value_count,
            "inferred_dtype": self.inferred_dtype,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "mean_length": self.mean_length,
            "median_length": self.median_length,
            "whitespace_only_count": self.whitespace_only_count,
            "top_values": [
                {"value": value, "count": count} for value, count in self.top_values
            ],
            "sample_values": list(self.sample_values),
        }


@dataclass(frozen=True, slots=True)
class FileProfile:
    """Schema and full-dataset statistics for one real dataset file."""

    schema: dict[str, Any]
    row_count: int
    duplicate_row_count: int
    duplicate_row_method: str
    columns: tuple[ColumnProfile, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the file profile."""

        payload = dict(self.schema)
        payload.update(
            {
                "row_count": self.row_count,
                "duplicate_row_count": self.duplicate_row_count,
                "duplicate_row_method": self.duplicate_row_method,
                "columns": [column.as_dict() for column in self.columns],
            }
        )
        return payload


@dataclass
class _ColumnAccumulator:
    """Streaming accumulator for one column of one file."""

    name: str
    chunks: list[np.ndarray] = field(default_factory=list)
    null_count: int = 0
    whitespace_only: int = 0
    length_counts: Counter[int] = field(default_factory=Counter)
    samples: list[str] = field(default_factory=list)
    max_samples: int = 8
    top_n: int = 25

    def update(self, values: pd.Series) -> None:
        """Fold one chunk of the column into the accumulator."""

        text = values
        null_mask = text.isna()
        self.null_count += int(null_mask.sum())
        present = text[~null_mask]
        if present.empty:
            self.chunks.append(np.empty(0, dtype=np.uint64))
            return
        stripped = present.str.strip()
        self.whitespace_only += int((stripped == "").sum())
        lengths = present.str.len()
        self.length_counts.update(int(value) for value in lengths.tolist())
        self.chunks.append(pd.util.hash_array(present.to_numpy(dtype=object), encoding="utf-8").astype(np.uint64))
        if len(self.samples) < self.max_samples:
            self.samples.extend(str(value) for value in present.head(self.max_samples).tolist())

    def finish(self, top_n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the unique hashes and their exact occurrence counts."""

        self.top_n = top_n
        if not self.chunks:
            return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)
        stacked = np.concatenate(self.chunks) if len(self.chunks) > 1 else self.chunks[0]
        self.chunks = [stacked]  # keep the array alive for the materialization pass
        unique, counts = np.unique(stacked, return_counts=True)
        return unique, counts

    def hashes(self) -> np.ndarray:
        """Return the accumulated per-row hash array."""

        if not self.chunks:
            return np.empty(0, dtype=np.uint64)
        return self.chunks[0]

    def length_stats(self) -> dict[str, Any]:
        """Return exact length statistics derived from the full length histogram."""

        if not self.length_counts:
            return {
                "min_length": None,
                "max_length": None,
                "mean_length": None,
                "median_length": None,
            }
        lengths = np.fromiter(
            (length for length, count in self.length_counts.items() for _ in range(count)),
            dtype=np.int64,
            count=sum(self.length_counts.values()),
        )
        return {
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "mean_length": round(float(lengths.mean()), 4),
            "median_length": float(np.median(lengths)),
        }

    def materialise(self, unique: np.ndarray, counts: np.ndarray, top_n: int) -> tuple[tuple[str, int], ...]:
        """Return the top values, resolved from the file in a second bounded pass."""

        if unique.size == 0:
            return ()
        order = np.argsort(-counts, kind="stable")[:top_n]
        wanted = {int(unique[index]): int(counts[index]) for index in order}
        return tuple(sorted(wanted.items(), key=lambda item: (-item[1], item[0])))


@dataclass(slots=True)
class SampleProfile:
    """Exploratory statistics measured on a deterministic leading-row sample."""

    method: str = "sample"
    rows: int = 0
    pattern_counts: dict[str, int] = field(default_factory=dict)
    pattern_examples: dict[str, list[str]] = field(default_factory=dict)
    case_profile: dict[str, int] = field(default_factory=dict)
    case_examples: dict[str, list[str]] = field(default_factory=dict)
    script_profile: dict[str, int] = field(default_factory=dict)
    component_counts: Counter = field(default_factory=Counter)
    address_examples: dict[str, list[str]] = field(default_factory=dict)
    token_frequencies: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    last_token_frequencies: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    position_token_frequencies: dict[str, list[tuple[str, int]]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the sample profile."""

        return {
            "method": self.method,
            "rows": self.rows,
            "pattern_counts": dict(self.pattern_counts),
            "pattern_examples": {key: list(value) for key, value in self.pattern_examples.items()},
            "case_profile": dict(self.case_profile),
            "case_examples": {key: list(value) for key, value in self.case_examples.items()},
            "script_profile": dict(self.script_profile),
            "address_component_count_distribution": {
                str(key): value for key, value in sorted(self.component_counts.items())
            },
            "address_examples": {key: list(value) for key, value in self.address_examples.items()},
            "top_tokens": {
                key: [{"token": token, "count": count} for token, count in value]
                for key, value in self.token_frequencies.items()
            },
            "top_trailing_tokens": {
                key: [{"token": token, "count": count} for token, count in value]
                for key, value in self.last_token_frequencies.items()
            },
            "top_leading_tokens": {
                key: [{"token": token, "count": count} for token, count in value]
                for key, value in self.position_token_frequencies.items()
            },
        }


def profile_file(loaded: LoadedFile, settings: DatasetSettings) -> FileProfile:
    """Compute full-dataset schema and column statistics for one file."""

    LOGGER.info("Profiling %s", loaded.key)
    accumulators = {column: _ColumnAccumulator(column) for column in loaded.columns}
    row_hashes: list[np.ndarray] = []
    row_count = 0

    for chunk in iter_chunks(loaded, settings):
        row_count += len(chunk)
        row_hashes.append(
            pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype=np.uint64)
        )
        for column in loaded.columns:
            accumulators[column].update(chunk[column])

    stacked_rows = np.concatenate(row_hashes) if row_hashes else np.empty(0, dtype=np.uint64)
    distinct_rows = int(np.unique(stacked_rows).size)
    del row_hashes
    del stacked_rows

    columns: list[ColumnProfile] = []
    for column in loaded.columns:
        accumulator = accumulators.pop(column)
        unique, counts = accumulator.finish(settings.top_values)
        lengths = accumulator.length_stats()
        non_null = row_count - accumulator.null_count
        top_pairs = accumulator.materialise(unique, counts, settings.top_values)
        top_values = _resolve_top_values(loaded, settings, column, top_pairs)
        columns.append(
            ColumnProfile(
                name=column,
                row_count=row_count,
                non_null_count=non_null,
                null_count=accumulator.null_count,
                null_percentage=round(100.0 * accumulator.null_count / row_count, 4)
                if row_count
                else 0.0,
                distinct_count=int(unique.size),
                distinct_method=HASH_METHOD,
                duplicate_value_count=non_null - int(unique.size),
                inferred_dtype=infer_dtype(column, unique.size, non_null, lengths),
                min_length=lengths["min_length"],
                max_length=lengths["max_length"],
                mean_length=lengths["mean_length"],
                median_length=lengths["median_length"],
                whitespace_only_count=accumulator.whitespace_only,
                top_values=top_values,
                sample_values=tuple(accumulator.samples[: settings.max_examples]),
            )
        )
        del accumulator

    return FileProfile(
        schema=loaded.as_dict(),
        row_count=row_count,
        duplicate_row_count=row_count - distinct_rows,
        duplicate_row_method=HASH_METHOD,
        columns=tuple(columns),
    )


def _resolve_top_values(
    loaded: LoadedFile,
    settings: DatasetSettings,
    column: str,
    top_pairs: Sequence[tuple[int, int]],
) -> tuple[tuple[str, int], ...]:
    """Materialize the most frequent values of a column in a second bounded pass.

    The lookup is vectorized: only the handful of rows whose content hash is
    one of the wanted keys is turned back into text.
    """

    if not top_pairs:
        return ()
    wanted = np.asarray(sorted({pair[0] for pair in top_pairs}), dtype=np.uint64)
    resolved: dict[int, str] = {}
    for chunk in iter_chunks(loaded, settings, columns=[column]):
        values = chunk[column]
        hashes = pd.util.hash_array(
            values.to_numpy(dtype=object), encoding="utf-8"
        ).astype(np.uint64)
        position = np.searchsorted(wanted, hashes)
        position_clipped = np.clip(position, 0, wanted.size - 1)
        hit = (wanted[position_clipped] == hashes) & np.isin(hashes, wanted)
        if not hit.any():
            continue
        keys = hashes[hit]
        texts = values[hit].tolist()
        for key, text in zip(keys.tolist(), texts):
            resolved.setdefault(int(key), text)
        if len(resolved) == wanted.size:
            break
    return tuple((resolved[pair[0]], pair[1]) for pair in top_pairs if pair[0] in resolved)


def infer_dtype(
    column: str,
    distinct: int,
    non_null: int,
    lengths: dict[str, Any],
) -> str:
    """Infer a column dtype from measured cardinality and length behavior."""

    if non_null == 0:
        return "empty"
    if column.endswith("_id") or column == "id":
        return "identifier_string"
    if distinct == non_null and non_null > 0:
        return "string_high_cardinality"
    if lengths.get("max_length") is not None and lengths["max_length"] <= 3:
        return "short_label_string"
    return "string"


def profile_sample(
    loaded: LoadedFile,
    settings: DatasetSettings,
    *,
    rows: int | None = None,
) -> dict[str, SampleProfile]:
    """Run the exploratory pattern battery over a deterministic leading sample."""

    sample = read_sample(loaded, settings, rows=rows)
    LOGGER.info("Sampled %d row(s) from %s for pattern discovery", len(sample), loaded.key)
    profiles: dict[str, SampleProfile] = {}

    for column in loaded.columns:
        if column.endswith("_id") or column == "id":
            continue
        profiles[column] = _profile_sample_column(column, sample[column], settings)

    if any(column.endswith("address") for column in loaded.columns):
        address_column = next(
            column for column in loaded.columns if column.endswith("address")
        )
        profiles[f"{address_column}_structure"] = _profile_address_structure(
            sample[address_column], settings
        )
    return profiles


def _profile_sample_column(
    column: str,
    values: pd.Series,
    settings: DatasetSettings,
) -> SampleProfile:
    """Measure pattern, case, script, and token statistics for one sample column."""

    text = values.dropna()
    profile = SampleProfile(rows=int(len(values)))
    if text.empty:
        return profile

    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for name, pattern in PATTERN_BATTERY.items():
        mask = text.str.contains(pattern, regex=True, na=False)
        counts[name] = int(mask.sum())
        found = text[mask].head(settings.max_examples).tolist()
        if found:
            examples[name] = [str(value) for value in found]

    for name, pattern in (
        ("unit_marker", UNIT_MARKER_PATTERN),
        ("street_designator", STREET_DESIGNATOR_PATTERN),
        ("direction", DIRECTION_PATTERN),
    ):
        mask = text.str.contains(pattern, regex=True, na=False)
        counts[name] = int(mask.sum())
        found = text[mask].head(settings.max_examples).tolist()
        if found:
            examples[name] = [str(value) for value in found]

    case_profile, case_examples = _case_profile(text, settings)
    profile.pattern_counts = counts
    profile.pattern_examples = examples
    profile.case_profile = case_profile
    profile.case_examples = case_examples
    profile.script_profile = _script_profile(text)
    profile.token_frequencies[column] = _top_tokens(
        (token for token_list in _tokens(text) for token in token_list), settings.top_values
    )
    profile.last_token_frequencies[column] = _top_tokens(
        (token_list[-1] for token_list in _tokens(text) if token_list), settings.top_values
    )
    profile.position_token_frequencies[column] = _top_tokens(
        (token_list[0] for token_list in _tokens(text) if token_list), settings.top_values
    )
    return profile


def _case_profile(
    text: pd.Series, settings: DatasetSettings
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Classify observed capitalization behaviour without assuming any locale."""

    upper = text.str.upper()
    lower = text.str.lower()
    counts = {
        "all_upper": int((text == upper).sum()),
        "all_lower": int((text == lower).sum()),
        "has_upper": int((text != lower).sum()),
        "has_lower": int((text != upper).sum()),
        "has_digit": int(text.str.contains(r"\d", regex=True).sum()),
        "is_blank_like": int(text.str.strip().eq("").sum()),
    }
    examples: dict[str, list[str]] = {}
    for label, mask in (
        ("all_upper", text == upper),
        ("all_lower", text == lower),
        ("mixed", (text != upper) & (text != lower)),
    ):
        found = text[mask].head(settings.max_examples).tolist()
        if found:
            examples[label] = [str(value) for value in found]
    return counts, examples


def _script_profile(text: pd.Series) -> dict[str, int]:
    """Count values by the dominant Unicode script actually observed."""

    counter: Counter[str] = Counter()
    for value in text.head(50_000).tolist():
        script = detect_script(str(value))
        counter[script] += 1
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def detect_script(value: str) -> str:
    """Return a coarse script label for one value without external data."""

    has_kana = any("\u3040" <= character <= "\u30ff" for character in value)
    has_hangul = any("\uac00" <= character <= "\ud7af" for character in value)
    has_cjk = any("\u4e00" <= character <= "\u9fff" for character in value)
    has_cyrillic = any("\u0400" <= character <= "\u04ff" for character in value)
    has_greek = any("\u0370" <= character <= "\u03ff" for character in value)
    has_arabic = any("\u0600" <= character <= "\u06ff" for character in value)
    has_devanagari = any("\u0900" <= character <= "\u097f" for character in value)
    has_latin_extended = any("\u00c0" <= character <= "\u024f" for character in value)
    labels = [
        ("kana", has_kana),
        ("hangul", has_hangul),
        ("cjk", has_cjk),
        ("cyrillic", has_cyrillic),
        ("greek", has_greek),
        ("arabic", has_arabic),
        ("devanagari", has_devanagari),
        ("latin_extended", has_latin_extended),
    ]
    active = [label for label, present in labels if present]
    if active:
        return "+".join(active)
    return "ascii_or_common"


def _profile_address_structure(values: pd.Series, settings: DatasetSettings) -> SampleProfile:
    """Profile the comma-separated structure of the observed address values."""

    text = values.fillna("")
    profile = SampleProfile(rows=int(len(values)))
    component_counts: Counter[int] = Counter()
    examples: dict[str, list[str]] = {}

    component_lengths = text.str.count(",") + 1
    for count in component_lengths.tolist():
        component_counts[int(count)] += 1
    profile.component_counts = component_counts

    for label, mask in (
        ("empty_or_blank", text.str.strip().eq("")),
        ("single_component", component_lengths <= 1),
        ("many_components", component_lengths >= 4),
        ("us_zip_like", text.str.contains(US_ZIP, regex=True, na=False)),
        ("numeric_zip_like", text.str.contains(NUMERIC_ZIP, regex=True, na=False)),
        ("phone_like", text.str.contains(PHONE_LIKE, regex=True, na=False)),
        ("email_like", text.str.contains(EMAIL_LIKE, regex=True, na=False)),
    ):
        found = text[mask].head(settings.max_examples).tolist()
        if found:
            examples[label] = [str(value) for value in found]
    profile.address_examples = examples
    return profile


def _tokens(text: pd.Series) -> Iterator[list[str]]:
    """Yield lowercase alphanumeric token lists for sampled values."""

    for value in text.tolist():
        yield ALNUM_RUN.findall(unicodedata.normalize("NFKC", str(value)).lower())


def _top_tokens(token_lists: Iterable[str], top_n: int) -> list[tuple[str, int]]:
    """Return the most frequent tokens with deterministic tie-breaking."""

    counter: Counter[str] = Counter(token for token in token_lists if token)
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:top_n]


def exact_pattern_counts(
    loaded: LoadedFile,
    settings: DatasetSettings,
    column: str,
    patterns: Sequence[str],
) -> dict[str, int]:
    """Count a focused set of patterns across every row of one column."""

    totals = {name: 0 for name in patterns}
    for chunk in iter_chunks(loaded, settings, columns=[column]):
        for name, pattern in patterns.items():
            totals[name] += int(
                chunk[column].str.contains(pattern, regex=True, na=False).sum()
            )
    return totals


def identifier_structure(
    loaded: LoadedFile,
    settings: DatasetSettings,
    column: str,
) -> dict[str, Any]:
    """Describe the identifier pattern of a column from every observed value.

    Identifiers are parsed into an optional alphabetic prefix plus a numeric
    body so uniqueness is verified on an exact 64-bit integer key rather than
    on a string.
    """

    prefixes: Counter[str] = Counter()
    non_numeric: list[str] = []
    numeric_chunks: list[np.ndarray] = []
    total = 0
    nulls = 0
    length_min: int | None = None
    length_max: int | None = None
    samples: list[str] = []

    for chunk in iter_chunks(loaded, settings, columns=[column]):
        series = chunk[column]
        total += len(series)
        nulls += int(series.isna().sum())
        text = series.fillna("")
        lengths = text.str.len()
        if len(lengths):
            chunk_min = int(lengths.min())
            chunk_max = int(lengths.max())
            length_min = chunk_min if length_min is None else min(length_min, chunk_min)
            length_max = chunk_max if length_max is None else max(length_max, chunk_max)
        prefix, numeric = _split_identifier(text)
        prefixes.update(prefix)
        bad = numeric.isna() & text.ne("")
        if int(bad.sum()):
            non_null_text = text[~bad]
            non_numeric.extend(str(value) for value in non_null_text.head(5).tolist())
        numeric_chunks.append(
            pd.to_numeric(numeric, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        )
        if len(samples) < settings.max_examples:
            samples.extend(str(value) for value in text.head(settings.max_examples).tolist())

    numeric_all = np.concatenate(numeric_chunks) if numeric_chunks else np.empty(0, np.int64)
    valid = numeric_all[numeric_all >= 0]
    unique_valid = int(np.unique(valid).size)
    prefix_patterns = sorted(prefixes.items(), key=lambda item: (-item[1], item[0]))

    return {
        "column": column,
        "row_count": total,
        "null_count": nulls,
        "distinct_count": int(np.unique(numeric_all).size),
        "distinct_method": "exact_int64_key" if not non_numeric else HASH_METHOD,
        "distinct_numeric_id_count": unique_valid,
        "duplicate_id_count": int(valid.size - unique_valid),
        "id_is_unique": unique_valid == int(valid.size) and nulls == 0,
        "length_min": length_min,
        "length_max": length_max,
        "prefix_counts": {prefix: count for prefix, count in prefix_patterns},
        "prefix_patterns": [
            {"prefix": prefix, "count": count, "pattern": _prefix_pattern(prefix)}
            for prefix, count in prefix_patterns
        ],
        "non_numeric_id_count": int(numeric_all.size - valid.size),
        "non_numeric_id_examples": non_numeric[: settings.max_examples],
        "sample_values": samples[: settings.max_examples],
        "_keys": numeric_all,
    }


def _split_identifier(text: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split identifier text into a source prefix and a numeric body.

    The prefix keeps the digits that belong to the label, so ``S1-925783039``
    yields the prefix ``S1`` and the body ``925783039``.
    """

    prefix = text.str.extract(r"^([^\d]*\d*)", expand=False).fillna("")
    numeric = text.str.extract(r"(\d+)\s*$", expand=False)
    return prefix, numeric


def _prefix_pattern(prefix: str) -> str:
    """Return a literal or wildcard description of an observed id prefix."""

    if not prefix:
        return "<no prefix>"
    if prefix.isalnum():
        return prefix
    return "<non-alphanumeric prefix>"


def build_pair_sample(
    loaded_sources: dict[str, LoadedFile],
    settings: DatasetSettings,
    ground_truth: LoadedFile,
    source_column: str,
    matched_column: str,
) -> dict[str, Any]:
    """Sample real ground-truth-linked pairs to surface observed variation.

    Only training data is touched. For every sampled S1 row one match per
    distinct matched-source prefix is taken, so the sample covers S2 and S3
    links separately, and the compared records are streamed out of the training
    source files.
    """

    pairs: list[tuple[str, str]] = []
    wanted_by_source: dict[str, set[str]] = {}
    for chunk in iter_chunks(ground_truth, settings, columns=[source_column, matched_column]):
        if len(pairs) >= settings.pair_sample_rows:
            break
        declared = chunk[chunk[matched_column].notna()]
        for source_id, matched in declared.itertuples(index=False, name=None):
            if len(pairs) >= settings.pair_sample_rows:
                break
            chosen: dict[str, str] = {}
            for candidate in str(matched).split(","):
                candidate = candidate.strip()
                if not candidate:
                    continue
                chosen.setdefault(_source_label(candidate), candidate)
            if not chosen:
                continue
            for label, candidate in sorted(chosen.items()):
                pairs.append((str(source_id), candidate))
                wanted_by_source.setdefault(label, set()).add(candidate)
            wanted_by_source.setdefault("S1", set()).add(str(source_id))

    LOGGER.info(
        "Sampling %d ground-truth-linked pair(s) from the training split", len(pairs)
    )
    records: dict[str, dict[str, dict[str, str | None]]] = {}
    for label, wanted in wanted_by_source.items():
        loaded = loaded_sources.get(label)
        if loaded is None:
            LOGGER.warning("No training file discovered for prefix %s; skipping", label)
            continue
        key_column = _identifier_column(loaded)
        records[label] = read_filtered(
            loaded, settings, key_column=key_column, wanted=wanted
        )
    return {
        "pairs": pairs,
        "records": records,
        "sample_rows": len(pairs),
    }


def _identifier_column(loaded: LoadedFile) -> str:
    """Return the identifier column of a source file."""

    for column in loaded.columns:
        if column == "id" or column.endswith("_id"):
            return column
    return loaded.columns[0]


def _source_label(entity_id: str) -> str:
    """Return the source label an identifier declares through its own prefix."""

    head = entity_id.split("-", 1)[0]
    return head if head else "S?"


def collect_country_distribution(
    loaded: LoadedFile, settings: DatasetSettings, column: str
) -> dict[str, int]:
    """Return the exact country-label distribution of one training file."""

    counter: Counter[str] = Counter()
    for chunk in iter_chunks(loaded, settings, columns=[column]):
        counter.update(str(value) for value in chunk[column].dropna().tolist())
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def collect_length_histogram(
    loaded: LoadedFile, settings: DatasetSettings, column: str
) -> dict[str, int]:
    """Return an exact histogram of value lengths for one column."""

    counter: Counter[int] = Counter()
    for chunk in iter_chunks(loaded, settings, columns=[column]):
        counter.update(
            int(value) for value in chunk[column].dropna().str.len().tolist()
        )
    return {str(key): value for key, value in sorted(counter.items())}


def load_headers(dataset: DiscoveredDataset, settings: DatasetSettings) -> dict[str, LoadedFile]:
    """Read the header of every discovered file without reading its rows."""

    headers: dict[str, LoadedFile] = {}
    for item in dataset.files:
        loaded = read_header(item, settings)
        headers[item.relative_path] = loaded
    return headers
