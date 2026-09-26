"""Field-specific normalization built only from observed training patterns.

Each normalizer returns *additional* representations of a value. None of them
removes the raw value, and none of them can decide that two records are the same
entity: they only make values easier to compare later.

Rule tables are supplied by configuration and are expected to be populated from
patterns that were actually observed in the training sources. With empty tables
the normalizers still behave correctly, they simply apply fewer rewrites.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

from business_entity_resolution.config.settings import NormalizationSettings
from business_entity_resolution.normalization import tokens as token_utils
from business_entity_resolution.normalization.text import (
    STAGE_NAMES,
    PunctuationTranslator,
    TextRepresentation,
    apply_text_pipeline,
    is_missing,
)

NUMBER = re.compile(r"\d+")
DIGIT_GROUPS = re.compile(r"\d+")
COMPONENT_SEPARATOR = ","

NAME_HINTS = ("name", "nom", "nombre", "title")
ADDRESS_HINTS = ("address", "addr", "street", "strasse", "via", "locality")
COUNTRY_HINTS = ("country", "nation", "pais", "pays", "land")
IDENTIFIER_HINTS = ("id", "identifier", "uuid", "key")

KIND_NAME = "name"
KIND_ADDRESS = "address"
KIND_COUNTRY = "country"
KIND_IDENTIFIER = "identifier"
KIND_GENERIC = "generic"

__all__ = [
    "STAGE_NAMES",
    "FieldNormalizer",
    "FieldResult",
    "NormalizerRegistry",
    "classify_column",
]


@dataclass(frozen=True, slots=True)
class FieldResult:
    """Every representation produced for one raw value of one field."""

    column: str
    kind: str
    raw: str
    stages: dict[str, str]
    tokens: tuple[str, ...] = ()
    alnum_tokens: tuple[str, ...] = ()
    char_ngrams: tuple[str, ...] = ()
    extras: dict[str, str] = field(default_factory=dict)

    def column_value(self, name: str) -> str | None:
        """Return one named representation, or ``None`` when it does not exist."""

        if name in self.stages:
            return self.stages[name]
        return self.extras.get(name)

    def as_dict(self) -> dict[str, Any]:
        """Return a flat mapping of every representation."""

        payload: dict[str, Any] = {
            "raw": self.raw,
            **self.stages,
            "tokens": token_utils.join_tokens(self.tokens),
            "alnum_tokens": token_utils.join_tokens(self.alnum_tokens),
            "char_ngrams": token_utils.join_tokens(self.char_ngrams),
        }
        payload.update(self.extras)
        return payload


@lru_cache(maxsize=512)
def classify_column(column: str) -> str:
    """Infer a field kind from a column name, without assuming a fixed schema."""

    lowered = column.casefold()
    if lowered in COUNTRY_HINTS or lowered.endswith("_country"):
        return KIND_COUNTRY
    if any(hint in lowered for hint in ADDRESS_HINTS):
        return KIND_ADDRESS
    if any(hint in lowered for hint in NAME_HINTS):
        return KIND_NAME
    if lowered in IDENTIFIER_HINTS or any(
        lowered.endswith(hint) for hint in IDENTIFIER_HINTS
    ):
        return KIND_IDENTIFIER
    return KIND_GENERIC


class FieldNormalizer:
    """Base normalizer that always preserves the raw value."""

    kind = KIND_GENERIC

    def __init__(
        self,
        settings: NormalizationSettings,
        translator: PunctuationTranslator | None = None,
    ) -> None:
        self.settings = settings
        self.translator = translator or PunctuationTranslator(settings)

    def normalize(self, column: str, value: object) -> FieldResult | None:
        """Return every representation of one raw value, or ``None`` if missing."""

        if is_missing(value):
            return None
        raw = str(value)
        self.translator.observe((raw,))
        stages = apply_text_pipeline(raw, self.settings, self.translator)
        assert stages is not None
        return self.build(column, raw, stages)

    def build(self, column: str, raw: str, stages: TextRepresentation) -> FieldResult:
        """Return the generic field result shared by every normalizer."""

        cleaned = stages.cleaned
        return FieldResult(
            column=column,
            kind=self.kind,
            raw=raw,
            stages={name: getattr(stages, name) for name in STAGE_NAMES},
            tokens=tuple(token_utils.whitespace_tokens(cleaned)),
            alnum_tokens=tuple(token_utils.alphanumeric_tokens(cleaned)),
            char_ngrams=tuple(
                token_utils.char_ngrams(cleaned, self.settings.char_ngram_sizes)
            ),
        )

    def shared_tail(self, result: FieldResult, normalized: str) -> FieldResult:
        """Return the result with the field-specific representation attached."""

        return FieldResult(
            column=result.column,
            kind=result.kind,
            raw=result.raw,
            stages=result.stages,
            tokens=result.tokens,
            alnum_tokens=result.alnum_tokens,
            char_ngrams=result.char_ngrams,
            extras={**result.extras, self.kind: normalized},
        )


class GenericNormalizer(FieldNormalizer):
    """A field with no domain rule: only the staged text representations."""

    kind = KIND_GENERIC


class IdentifierNormalizer(FieldNormalizer):
    """An identifier field: never rewritten, only cleaned and tokenized.

    Case folding is disabled here so an identifier representation always keeps
    the exact prefix the file declared.
    """

    kind = KIND_IDENTIFIER

    def __init__(
        self,
        settings: NormalizationSettings,
        translator: PunctuationTranslator | None = None,
    ) -> None:
        super().__init__(replace(settings, casefold=False), translator)

    def build(self, column: str, raw: str, stages: TextRepresentation) -> FieldResult:
        result = super().build(column, raw, stages)
        return FieldResult(
            column=result.column,
            kind=result.kind,
            raw=result.raw,
            stages=result.stages,
            tokens=result.tokens,
            alnum_tokens=result.alnum_tokens,
            char_ngrams=result.char_ngrams,
            extras={
                **result.extras,
                "id_body": _identifier_body(raw),
                "id_prefix": _identifier_prefix(raw),
            },
        )


class CountryNormalizer(FieldNormalizer):
    """Open-set country label cleaning.

    The label is cleaned and case-folded but never mapped to a fixed country
    universe, so a country absent from the training data is still usable.
    """

    kind = KIND_COUNTRY

    def build(self, column: str, raw: str, stages: TextRepresentation) -> FieldResult:
        result = super().build(column, raw, stages)
        label = result.stages["punctuation_normalized"]
        label = self._apply_case(label)
        if not self.settings.country_open_set:
            raise ValueError(
                "country.open_set must stay true; the pipeline is country-agnostic"
            )
        return FieldResult(
            column=result.column,
            kind=result.kind,
            raw=result.raw,
            stages=result.stages,
            tokens=result.tokens,
            alnum_tokens=result.alnum_tokens,
            char_ngrams=result.char_ngrams,
            extras={**result.extras, "country": label, "country_token": label.replace(" ", "_")},
        )

    def _apply_case(self, label: str) -> str:
        case = self.settings.country_case
        if case == "upper":
            return label.upper()
        if case == "lower":
            return label.lower()
        if case == "fold":
            return label.casefold()
        return label


class NameNormalizer(FieldNormalizer):
    """Business-name normalization from observed training patterns.

    Applied in order: ampersand to the word ``and``, configured whole-token
    abbreviation expansion, configured legal-form synonym canonicalization, then
    re-spacing. Corporate-form tokens are kept in the main representation and
    removed only in the separate ``name_core`` representation.
    """

    kind = KIND_NAME

    def build(self, column: str, raw: str, stages: TextRepresentation) -> FieldResult:
        result = super().build(column, raw, stages)
        working = result.stages["punctuation_normalized"]
        if self.settings.ampersand_to_and:
            working = re.sub(r"\s*&\s*", " and ", working)
        working = _apply_token_map(working, self.settings.name_abbreviations)
        working = _apply_token_map(working, self.settings.general_legal_suffixes)
        ordered_tokens = token_utils.normalized_tokens(
            working, min_length=self.settings.min_token_length
        )
        core_tokens = [
            token
            for token in ordered_tokens
            if token not in self.settings.general_legal_suffixes
        ]
        sorted_tokens = token_utils.normalized_tokens(
            working, min_length=self.settings.min_token_length, sort=True
        )
        normalized = token_utils.join_tokens(
            sorted_tokens if self.settings.token_sort else ordered_tokens
        )
        result = self.shared_tail(result, normalized)
        return FieldResult(
            column=result.column,
            kind=result.kind,
            raw=result.raw,
            stages=result.stages,
            tokens=result.tokens,
            alnum_tokens=result.alnum_tokens,
            char_ngrams=result.char_ngrams,
            extras={
                **result.extras,
                "name_core": token_utils.join_tokens(core_tokens),
                "name_sorted": token_utils.join_tokens(sorted_tokens),
                "name_tokens": token_utils.join_tokens(ordered_tokens),
            },
        )


class AddressNormalizer(FieldNormalizer):
    """Address normalization from observed training patterns.

    Comma-separated structure is preserved, road/street designator and unit
    markers are canonicalized from configuration, numeric components are
    exposed separately, and an order-insensitive view is produced because
    component reordering is one of the observed variation types.
    """

    kind = KIND_ADDRESS

    def build(self, column: str, raw: str, stages: TextRepresentation) -> FieldResult:
        result = super().build(column, raw, stages)
        working = result.stages["punctuation_normalized"]
        components = [
            token_utils.join_tokens(
                token_utils.normalized_tokens(
                    self._normalize_component(component), min_length=1
                )
            )
            for component in working.split(COMPONENT_SEPARATOR)
        ]
        components = [component for component in components if component]
        markers = self.settings.missing_markers
        without_markers = [
            component
            for component in components
            if component.casefold() not in markers
        ] if markers else list(components)
        numbers = _numeric_components(working)
        postal = _postal_component(working)
        result = self.shared_tail(result, token_utils.join_tokens(components))
        return FieldResult(
            column=result.column,
            kind=result.kind,
            raw=result.raw,
            stages=result.stages,
            tokens=result.tokens,
            alnum_tokens=result.alnum_tokens,
            char_ngrams=result.char_ngrams,
            extras={
                **result.extras,
                "address_sorted": token_utils.join_tokens(sorted(components)),
                "address_components": f"{COMPONENT_SEPARATOR} ".join(components),
                "address_marker_free": token_utils.join_tokens(without_markers),
                "address_numbers": token_utils.join_tokens(numbers),
                "address_house_number": _leading_number(working),
                "address_postal": postal,
                "address_unit": _unit_component(working, self.settings),
            },
        )

    def _normalize_component(self, component: str) -> str:
        """Canonicalize one comma-separated address component."""

        text = _apply_token_map(component, self.settings.street_designators)
        text = _apply_token_map(text, self.settings.unit_markers)
        text = _apply_token_map(text, self.settings.direction_tokens)
        return _apply_token_map(text, self.settings.address_token_map)


class NormalizerRegistry:
    """Resolve a column to its normalizer from the column's own name."""

    def __init__(self, settings: NormalizationSettings) -> None:
        self.settings = settings
        translator = PunctuationTranslator(settings)
        self._normalizers = {
            KIND_NAME: NameNormalizer(settings, translator),
            KIND_ADDRESS: AddressNormalizer(settings, translator),
            KIND_COUNTRY: CountryNormalizer(settings, translator),
            KIND_IDENTIFIER: IdentifierNormalizer(settings, translator),
            KIND_GENERIC: GenericNormalizer(settings, translator),
        }

    def kind_for(self, column: str) -> str:
        """Return the field kind inferred for a column name."""

        return classify_column(column)

    def for_column(self, column: str) -> FieldNormalizer:
        """Return the normalizer responsible for a column name."""

        return self._normalizers[self.kind_for(column)]

    def normalize(self, column: str, value: object) -> FieldResult | None:
        """Normalize one value of one column, preserving the raw value."""

        return self.for_column(column).normalize(column, value)

    def normalize_value(self, column: str, value: object) -> str:
        """Return only the field-specific representation of one value."""

        result = self.normalize(column, value)
        if result is None:
            return ""
        normalized = result.column_value(self.kind_for(column))
        return normalized if normalized is not None else result.stages["cleaned"]


def _apply_token_map(value: str, mapping: Mapping[str, str]) -> str:
    """Rewrite whole tokens through a configured rule table."""

    if not mapping:
        return value
    tokens = token_utils.whitespace_tokens(value)
    if not tokens:
        return value
    rewritten: list[str] = []
    for token in tokens:
        key = token.casefold()
        if key in mapping:
            replacement = mapping[key]
            if replacement:
                rewritten.extend(token_utils.whitespace_tokens(replacement))
            continue
        rewritten.append(token)
    return token_utils.join_tokens(rewritten)


def _numeric_components(value: str) -> list[str]:
    """Return the numeric components observed in an address value."""

    return [group for group in DIGIT_GROUPS.findall(value) if group]


def _leading_number(value: str) -> str:
    """Return the numeric components of the first address component holding digits.

    This is the house-number or municipal-number position in both observed
    address styles; nothing beyond the first component is claimed.
    """

    for component in value.split(COMPONENT_SEPARATOR):
        groups = DIGIT_GROUPS.findall(component)
        if groups:
            return token_utils.join_tokens(groups)
    return ""


def _postal_component(value: str) -> str:
    """Return a trailing postal-code-like component when the data contains one."""

    for component in reversed(value.split(COMPONENT_SEPARATOR)):
        candidate = component.strip()
        if NUMBER.fullmatch(candidate) and len(candidate) in {5, 6, 9}:
            return candidate
    return ""


def _unit_component(value: str, settings: NormalizationSettings) -> str:
    """Return the unit or apartment component of an address, if present."""

    markers = set(settings.unit_markers)
    if not markers:
        return ""
    for component in value.split(COMPONENT_SEPARATOR):
        tokens = token_utils.alphanumeric_tokens(component)
        if any(token in markers for token in tokens):
            return token_utils.join_tokens(tokens)
    return ""


def _identifier_prefix(value: str) -> str:
    """Return the literal prefix an identifier declares before its first digit."""

    head = value.strip().split("-", 1)[0]
    return head if head != value.strip() else ""


def _identifier_body(value: str) -> str:
    """Return the numeric body of an identifier without altering its prefix."""

    text = value.strip()
    tail = text.rsplit("-", 1)[-1] if "-" in text else text
    digits = "".join(character for character in tail if character.isdigit())
    return digits
