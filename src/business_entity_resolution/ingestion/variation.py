"""Data-driven discovery of the variation patterns the real training data shows.

The examples come from actual training ground-truth links: a Source 1 record and
a record the ground truth says matches it. Nothing here is invented or taken
from documentation; every example is a pair that exists in the training split.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

ALNUM = re.compile(r"[0-9a-z]+")
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class VariationExample:
    """One observed difference between two records the ground truth links."""

    category: str
    left_id: str
    left_value: str
    right_id: str
    right_value: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serializable view of the example."""

        return {
            "category": self.category,
            "left_id": self.left_id,
            "left_value": self.left_value,
            "right_id": self.right_id,
            "right_value": self.right_value,
            "detail": self.detail,
        }


@dataclass
class VariationReport:
    """Category counts plus real examples for each observed variation type."""

    sample_pairs: int = 0
    compared_fields: list[str] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    examples: dict[str, list[VariationExample]] = field(default_factory=dict)
    equal_after_conservative_normalization: Counter = field(default_factory=Counter)

    def record(self, category: str, example: VariationExample, limit: int) -> None:
        """Count one observed variation and keep a bounded set of examples."""

        self.counts[category] += 1
        bucket = self.examples.setdefault(category, [])
        if len(bucket) < limit:
            bucket.append(example)

    def as_dict(self, limit: int) -> dict[str, Any]:
        """Return a JSON-serializable view of the variation report."""

        return {
            "method": "training_ground_truth_linked_pairs",
            "sample_pairs": self.sample_pairs,
            "compared_fields": list(self.compared_fields),
            "variation_counts": dict(sorted(self.counts.items())),
            "pairs_equal_after_conservative_normalization": dict(
                sorted(self.equal_after_conservative_normalization.items())
            ),
            "examples": {
                category: [example.as_dict() for example in values[:limit]]
                for category, values in sorted(self.examples.items())
            },
        }


def compare_pairs(
    pairs: list[tuple[str, str]],
    records: dict[str, dict[str, dict[str, str | None]]],
    fields: list[str],
    *,
    limit: int = 8,
    normalizer=None,
) -> VariationReport:
    """Classify the observed differences between linked record pairs."""

    report = VariationReport(sample_pairs=len(pairs), compared_fields=list(fields))
    for left_id, right_id in pairs:
        left_label = left_id.split("-", 1)[0]
        right_label = right_id.split("-", 1)[0]
        left_record = records.get(left_label, {}).get(left_id)
        right_record = records.get(right_label, {}).get(right_id)
        if not left_record or not right_record:
            continue
        for column in fields:
            left_value = _text(left_record.get(column))
            right_value = _text(right_record.get(column))
            if not left_value and not right_value:
                continue
            if not left_value or not right_value:
                report.record(
                    f"{column}__missing_on_one_side",
                    VariationExample(
                        f"{column}__missing_on_one_side",
                        left_id,
                        left_value,
                        right_id,
                        right_value,
                        "one side has no value",
                    ),
                    limit,
                )
                continue
            if left_value == right_value:
                report.record(
                    f"{column}__identical",
                    VariationExample(
                        f"{column}__identical",
                        left_id,
                        left_value,
                        right_id,
                        right_value,
                        "byte-identical values",
                    ),
                    limit,
                )
                continue
            for category, detail in _classify(left_value, right_value):
                report.record(
                    f"{column}__{category}",
                    VariationExample(
                        f"{column}__{category}",
                        left_id,
                        left_value,
                        right_id,
                        right_value,
                        detail,
                    ),
                    limit,
                )
            if normalizer is not None:
                normalized_left = normalizer(column, left_value)
                normalized_right = normalizer(column, right_value)
                if normalized_left == normalized_right:
                    report.equal_after_conservative_normalization[column] += 1
    return report


def _text(value: str | None) -> str:
    return "" if value is None or value != value else str(value).strip()


def _classify(left: str, right: str) -> list[tuple[str, str]]:
    """Return the variation categories two differing values actually exhibit."""

    categories: list[tuple[str, str]] = []
    left_fold, right_fold = _compatibility(left), _compatibility(right)
    if left_fold == right_fold and left != right:
        categories.append(
            (
                "unicode_form",
                "equal under NFKC but not byte-identical: character-width or composed/decomposed form",
            )
        )

    left_case, right_case = left_fold.casefold(), right_fold.casefold()
    if left_case != right_case:
        categories.append(("case", "case-only difference"))

    left_ws, right_ws = _collapse(left_case), _collapse(right_case)
    if left_ws != right_ws:
        if " ".join(left_ws.split()) == " ".join(right_ws.split()):
            categories.append(("whitespace", "whitespace-only difference"))
        if re.search(r"\s{2,}", left) or re.search(r"\s{2,}", right):
            categories.append(("whitespace_run", "repeated whitespace present"))
        if left != left.strip() or right != right.strip():
            categories.append(("whitespace_edge", "leading or trailing whitespace"))

    left_punct, right_punct = _punctuation(left_ws), _punctuation(right_ws)
    if left_punct != right_punct:
        categories.append(("punctuation", "different punctuation characters"))

    left_amp, right_amp = _ampersand(left_ws), _ampersand(right_ws)
    if left_amp != right_amp:
        categories.append(("ampersand_vs_and", "'&' versus the word 'and'"))

    left_tokens, right_tokens = _tokens(left_ws), _tokens(right_ws)
    if left_tokens == right_tokens:
        categories.append(("token_reordering", "same tokens in a different order"))
    elif sorted(left_tokens) == sorted(right_tokens):
        categories.append(("token_order", "same token multiset in a different order"))

    left_compact, right_compact = "".join(left_tokens), "".join(right_tokens)
    if left_compact != right_compact and _similarity(left_compact, right_compact) >= 0.72:
        categories.append(
            ("spelling_variation", "high character similarity, different spelling")
        )
    if _numeric_sets(left) != _numeric_sets(right):
        categories.append(("numeric_component", "different numeric components"))

    left_designators = _matches(left, DESIGNATORS)
    right_designators = _matches(right, DESIGNATORS)
    if left_designators != right_designators:
        categories.append(
            ("address_abbreviation", "different road/street designator forms")
        )

    left_units = _matches(left, UNIT_MARKERS)
    right_units = _matches(right, UNIT_MARKERS)
    if left_units != right_units:
        categories.append(("unit_marker", "different apartment/unit marker forms"))

    return categories


DESIGNATORS = {
    "st", "str", "street", "rd", "road", "ave", "av", "avenue", "blvd", "boulevard",
    "dr", "drive", "ln", "lane", "ct", "court", "cir", "circle", "pl", "place",
    "pkwy", "parkway", "hwy", "highway", "aly", "alley", "ter", "terrace", "trl",
    "trail", "way", "sq", "square", "plz", "plaza", "sec", "section",
}
UNIT_MARKERS = {
    "apt", "apartment", "suite", "ste", "unit", "flat", "rm", "room", "fl",
    "floor", "flr", "bldg", "building", "shop", "no", "nr", "number",
}


def _compatibility(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def _collapse(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _punctuation(value: str) -> str:
    return "".join(sorted(set(PUNCT.findall(value))))


def _ampersand(value: str) -> str:
    if "&" in value:
        return "ampersand"
    if re.search(r"\band\b", value):
        return "word"
    return "none"


def _tokens(value: str) -> list[str]:
    return ALNUM.findall(unicodedata.normalize("NFKC", value).casefold())


def _matches(value: str, vocabulary: set[str]) -> tuple[str, ...]:
    return tuple(sorted({token for token in _tokens(value) if token in vocabulary}))


def _numeric_sets(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+", value))


def _similarity(left: str, right: str) -> float:
    """Return a normalized character-bigram similarity in [0, 1]."""

    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    left_grams = _bigrams(left)
    right_grams = _bigrams(right)
    if not left_grams or not right_grams:
        return 0.0
    shared = len(left_grams & right_grams)
    return round(2 * shared / (len(left_grams) + len(right_grams)), 4)


def _bigrams(value: str) -> set[str]:
    compact = "".join(value.split())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[index : index + 2] for index in range(len(compact) - 1)}
