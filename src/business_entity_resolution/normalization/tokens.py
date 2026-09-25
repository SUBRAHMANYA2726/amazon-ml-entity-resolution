"""Deterministic, reusable tokenization utilities.

Four token views are supported so a later phase can choose its own granularity
without re-normalizing the raw text:

* ``whitespace`` tokens of any cleaned value,
* ``alphanumeric`` runs only,
* ``normalized`` alphanumeric tokens after dropping configured noise tokens,
* optional character n-grams of the compacted alphanumeric form.

Every function is pure and order-preserving, so the same input always yields the
same tokens.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

ALNUM_RUN = re.compile(r"[^\W_]+", re.UNICODE)
DIGIT_RUN = re.compile(r"\d+")
WHITESPACE = re.compile(r"\s+")
TOKEN_SEPARATOR = " "


def whitespace_tokens(value: str) -> list[str]:
    """Return whitespace-separated tokens of a value, empty tokens removed."""

    if not value:
        return []
    return [token for token in WHITESPACE.split(value.strip()) if token]


def alphanumeric_tokens(value: str) -> list[str]:
    """Return maximal alphanumeric runs of a value, order preserved."""

    if not value:
        return []
    return ALNUM_RUN.findall(value)


def normalized_tokens(
    value: str,
    *,
    drop: Iterable[str] = (),
    min_length: int = 1,
    sort: bool = False,
) -> list[str]:
    """Return filtered alphanumeric tokens, optionally in sorted order.

    ``drop`` holds whole tokens that carry no identifying signal (for example
    a corporate-form word). Tokens shorter than ``min_length`` are removed. The
    result is a list, never a set, so position information is not lost unless
    ``sort`` is requested explicitly.
    """

    excluded = {token for token in drop}
    tokens = [
        token
        for token in alphanumeric_tokens(value)
        if len(token) >= min_length and token not in excluded
    ]
    return sorted(tokens) if sort else tokens


def compact(value: str) -> str:
    """Return the value with every non-alphanumeric character removed."""

    return "".join(alphanumeric_tokens(value))


def char_ngrams(
    value: str,
    sizes: Sequence[int] = (3,),
    *,
    min_count: int = 1,
) -> list[str]:
    """Return character n-grams of the compacted value, deterministically ordered.

    Shorter values are emitted whole rather than dropped, so a short but valid
    value still produces a representation.
    """

    text = compact(value)
    if not text:
        return []
    grams: list[str] = []
    for size in sorted({int(item) for item in sizes}):
        if size < 1:
            continue
        if len(text) <= size:
            grams.append(text)
            continue
        grams.extend([text[index : index + size] for index in range(len(text) - size + 1)])
    if min_count > 1:
        unique: list[str] = []
        for gram in grams:
            if unique.count(gram) < min_count:
                unique.append(gram)
        grams = unique
    return grams


def join_tokens(tokens: Sequence[str]) -> str:
    """Join tokens into a single deterministic representation string."""

    return TOKEN_SEPARATOR.join(tokens)


def token_counts(values: Iterable[str]) -> Mapping[str, int]:
    """Return token frequencies for a sequence of token lists."""

    counter: dict[str, int] = {}
    for tokens in values:
        for token in tokens:
            counter[token] = counter.get(token, 0) + 1
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def longest_common_prefix(left: Sequence[str], right: Sequence[str]) -> int:
    """Return the length of the shared token prefix of two token lists."""

    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index
