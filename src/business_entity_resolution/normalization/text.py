"""Conservative, staged text cleaning.

Each function is one deliberately small step. Nothing here decides that two
records match, and nothing here discards a raw value: the caller always keeps
the original string and stores the result as an additional representation.

The stages are ordered from least to most destructive so diagnostics can report
exactly how much each stage changed. The punctuation stage uses a cached
translation table so bulk runs stay practical without changing behaviour.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from business_entity_resolution.config.settings import NormalizationSettings

CONTROL_CHARACTERS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"
    r"\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]"
)
WHITESPACE_RUN = re.compile(r"\s+")

# Typographic and full-width variants that are always safe to fold to ASCII.
CHARACTER_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "­": "", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", "　": " ", " ": " ",
    "…": "...",
}

FOLD_SOURCE = re.compile("[" + "".join(CHARACTER_FOLD) + "]")

PUNCTUATION_CATEGORIES = frozenset(
    {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po", "Sm", "Sk", "So"}
)

#: Staged text representations stored beside every raw value, in order.
STAGE_NAMES = (
    "unicode_normalized",
    "case_normalized",
    "whitespace_normalized",
    "punctuation_normalized",
    "cleaned",
)


@dataclass(frozen=True, slots=True)
class TextRepresentation:
    """Every intermediate representation of one raw value.

    ``raw`` is always the untouched input. Each later attribute is a strictly
    derived view, so a consumer can choose how aggressive to be.
    """

    raw: str
    unicode_normalized: str
    case_normalized: str
    whitespace_normalized: str
    punctuation_normalized: str
    cleaned: str

    @property
    def stages(self) -> tuple[str, ...]:
        """Return the stage names in the order they are applied."""

        return ("raw", *STAGE_NAMES)

    def as_dict(self) -> dict[str, str]:
        """Return the representations as an ordered mapping."""

        return {name: getattr(self, name) for name in self.stages}

    def changed_at(self, stage: str) -> bool:
        """Return whether a stage changed the value produced by the stage before."""

        names = self.stages
        if stage not in names:
            raise KeyError(f"Unknown normalization stage: {stage}")
        index = names.index(stage)
        if index == 0:
            return False
        return getattr(self, stage) != getattr(self, names[index - 1])


class PunctuationTranslator:
    """Cached character-level cleaning policy.

    The table is keyed by the exact character set the data uses, so the same
    character always maps to the same replacement. ``observe`` extends the table
    with newly seen characters, and once a batch is registered the same batch
    can be translated again cheaply.
    """

    def __init__(self, settings: NormalizationSettings) -> None:
        self.settings = settings
        self.keep = frozenset(settings.keep_characters)
        self._table: dict[int, str | None] = {}
        self._unseen: re.Pattern[str] | None = None

    @property
    def table(self) -> dict[int, str | None]:
        """Return the cached ordinal-to-replacement table."""

        return self._table

    def observe(self, values: Iterable[str]) -> int:
        """Register every character in the given values; return how many are new.

        The already-registered check is a single C-level scan, so repeated
        observation of the same kind of data costs almost nothing.
        """

        joined = "".join(values)
        if self._unseen is not None and not self._unseen.search(joined):
            return 0
        seen = set(joined)
        added = 0
        for character in seen:
            code = ord(character)
            if code in self._table:
                continue
            self._table[code] = self._replacement(character)
            added += 1
        self._rebuild_unseen_pattern()
        return added

    def observe_series(self, values: pd.Series) -> int:
        """Register every character in a series without building it twice."""

        return self.observe(values.fillna("").astype(str).tolist())

    def _rebuild_unseen_pattern(self) -> None:
        if not self._table:
            self._unseen = None
            return
        known = "".join(sorted(chr(code) for code in self._table))
        self._unseen = re.compile(f"[^{re.escape(known)}]")

    def _replacement(self, character: str) -> str | None:
        folded = CHARACTER_FOLD.get(character, character)
        if not folded:
            return None
        if self.settings.strip_control_characters and CONTROL_CHARACTERS.match(folded):
            return None
        if folded in self.keep:
            return folded
        if folded.isspace():
            return " "
        category = unicodedata.category(folded)
        if category in PUNCTUATION_CATEGORIES or category in {"Sc", "Sk"}:
            return " " if self.settings.punctuation_to_space else None
        if category[0] in {"L", "N", "M"}:
            return folded
        if category == "Zs":
            return " "
        if self.settings.strip_symbols:
            return None
        return folded

    def __call__(self, value: str) -> str:
        """Return the cleaned value using the cached table.

        The table must already cover every character in ``value``; callers use
        :meth:`observe` on a whole batch (or on the single value) first.
        """

        if not value:
            return value
        return value.translate(self._table)


def is_missing(value: object) -> bool:
    """Return whether a raw value counts as missing without guessing semantics."""

    return value is None or (isinstance(value, float) and value != value) or value == ""


def to_unicode(value: str, form: str) -> str:
    """Return the value in the requested Unicode normalization form."""

    if not form or form.lower() == "none":
        return value
    return unicodedata.normalize(form, value)


def strip_control_characters(value: str) -> str:
    """Remove control and zero-width characters, keeping ordinary spaces.

    A value made only of control characters normalizes to an empty string; the
    diagnostics report that explicitly rather than hiding it.
    """

    return CONTROL_CHARACTERS.sub("", value)


def casefold(value: str, enabled: bool) -> str:
    """Return a case-folded value, or the input when folding is disabled."""

    return value.casefold() if enabled else value


def collapse_whitespace(value: str, enabled: bool) -> str:
    """Collapse every whitespace run to one space and trim the edges."""

    if not enabled:
        return value
    return WHITESPACE_RUN.sub(" ", value).strip()


def fold_punctuation(value: str) -> str:
    """Replace typographic punctuation variants with their ASCII equivalents."""

    return FOLD_SOURCE.sub(lambda match: CHARACTER_FOLD[match.group(0)], value)


def normalize_punctuation(value: str, settings: NormalizationSettings) -> str:
    """Normalize punctuation conservatively using the configured policy."""

    return PunctuationTranslator(settings)(fold_punctuation(value))


def apply_text_pipeline(
    value: object,
    settings: NormalizationSettings,
    translator: PunctuationTranslator | None = None,
) -> TextRepresentation | None:
    """Run the staged text pipeline over one raw value.

    Returns ``None`` for a missing value so callers can keep missingness
    distinguishable from an empty string.
    """

    if is_missing(value):
        return None
    raw = str(value)
    engine = translator or PunctuationTranslator(settings)
    unicode_stage = to_unicode(raw, settings.unicode_form)
    if settings.strip_control_characters:
        unicode_stage = strip_control_characters(unicode_stage)
    case_stage = casefold(unicode_stage, settings.casefold)
    whitespace_stage = collapse_whitespace(case_stage, settings.collapse_whitespace)
    punctuation_stage = engine(whitespace_stage)
    cleaned = collapse_whitespace(punctuation_stage, settings.collapse_whitespace)
    return TextRepresentation(
        raw=raw,
        unicode_normalized=unicode_stage,
        case_normalized=case_stage,
        whitespace_normalized=whitespace_stage,
        punctuation_normalized=punctuation_stage,
        cleaned=cleaned,
    )


def apply_text_pipeline_series(
    frame: pd.DataFrame,
    settings: NormalizationSettings,
    translator: PunctuationTranslator | None = None,
    *,
    stages: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run the staged text pipeline over a whole frame, column by column.

    Every stage uses pandas' compiled string routines, so bulk runs stay
    practical. Requesting only ``("cleaned",)`` fuses the stages into a single
    pass; the result is identical and ``tests/test_normalization_text.py`` asserts
    that equivalence.
    """

    engine = translator or PunctuationTranslator(settings)
    wanted = tuple(stages) if stages is not None else STAGE_NAMES
    unknown = [name for name in wanted if name not in STAGE_NAMES]
    if unknown:
        raise ValueError(f"Unknown normalization stages requested: {unknown}")
    fused = tuple(wanted) == ("cleaned",)

    output: dict[str, pd.Series] = {}
    for column in frame.columns:
        engine.observe_series(frame[column])
        text = frame[column].fillna("").astype(str)
        unicode_stage = _normalize_series(text, settings)
        if not fused and "unicode_normalized" in wanted:
            output[f"{column}__unicode_normalized"] = unicode_stage
        case_stage = unicode_stage.str.casefold() if settings.casefold else unicode_stage
        if not fused and "case_normalized" in wanted:
            output[f"{column}__case_normalized"] = case_stage
        if not fused and "whitespace_normalized" in wanted:
            output[f"{column}__whitespace_normalized"] = _collapse_series(
                case_stage, settings
            )
        working = case_stage if fused else _collapse_series(case_stage, settings)
        punctuation_stage = working.map(engine)
        if not fused and "punctuation_normalized" in wanted:
            output[f"{column}__punctuation_normalized"] = punctuation_stage
        output[f"{column}__cleaned"] = _collapse_series(punctuation_stage, settings)
    return pd.DataFrame(output, index=frame.index)


def _normalize_series(text: pd.Series, settings: NormalizationSettings) -> pd.Series:
    """Return the Unicode-normalized, control-cleaned series."""

    if settings.unicode_form and settings.unicode_form.lower() != "none":
        result = text.str.normalize(settings.unicode_form)
    else:
        result = text
    if settings.strip_control_characters:
        result = result.str.replace(CONTROL_CHARACTERS, "", regex=True)
    return result


def _collapse_series(text: pd.Series, settings: NormalizationSettings) -> pd.Series:
    """Return the whitespace-collapsed series."""

    if not settings.collapse_whitespace:
        return text
    return text.str.replace(WHITESPACE_RUN, " ", regex=True).str.strip()


def describe_settings(settings: NormalizationSettings) -> dict[str, Any]:
    """Return a JSON-serializable view of the active normalization policy."""

    return {
        "unicode_form": settings.unicode_form,
        "strip_control_characters": settings.strip_control_characters,
        "casefold": settings.casefold,
        "collapse_whitespace": settings.collapse_whitespace,
        "punctuation_to_space": settings.punctuation_to_space,
        "strip_symbols": settings.strip_symbols,
        "keep_characters": settings.keep_characters,
        "ampersand_to_and": settings.ampersand_to_and,
        "country_open_set": settings.country_open_set,
        "country_case": settings.country_case,
        "char_ngram_sizes": list(settings.char_ngram_sizes),
        "token_sort": settings.token_sort,
        "min_token_length": settings.min_token_length,
        "over_normalized_length_ratio": settings.over_normalized_length_ratio,
        "rule_table_sizes": {
            "general_legal_suffixes": len(settings.general_legal_suffixes),
            "name_abbreviations": len(settings.name_abbreviations),
            "street_designators": len(settings.street_designators),
            "unit_markers": len(settings.unit_markers),
            "direction_tokens": len(settings.direction_tokens),
            "address_tokens": len(settings.address_token_map),
        },
    }
