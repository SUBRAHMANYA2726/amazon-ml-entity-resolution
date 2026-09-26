"""Deterministic similarity functions for Business Entity Resolution.

Provides string, token, character n-gram, edit-distance, and structured-field
similarity functions. All functions handle None, empty strings, and missing
values safely without letting missing fields become false exact matches.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

try:
    from rapidfuzz.distance import JaroWinkler, Levenshtein
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

WORD_PATTERN = re.compile(r"\w+")


def is_blank(value: object) -> bool:
    """Return True if value is None, NaN, empty, or whitespace-only."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    s = str(value).strip()
    return not s or s.lower() in ("none", "nan", "null")


def exact_equality(s1: object, s2: object) -> float:
    """Return 1.0 if both strings are non-blank and match case-insensitively, else 0.0.
    
    Missing values (None, empty, NaN) NEVER match, returning 0.0.
    """
    if is_blank(s1) or is_blank(s2):
        return 0.0
    return 1.0 if str(s1).strip().casefold() == str(s2).strip().casefold() else 0.0


def extract_tokens(text: object) -> tuple[str, ...]:
    """Extract lowercased alphanumeric tokens from text."""
    if is_blank(text):
        return ()
    return tuple(WORD_PATTERN.findall(str(text).casefold()))


def token_jaccard(
    tokens1: Sequence[str] | set[str] | object,
    tokens2: Sequence[str] | set[str] | object,
) -> float:
    """Compute Jaccard similarity between two token sets.
    
    Returns |A ∩ B| / |A ∪ B| in [0.0, 1.0]. Returns 0.0 if either is empty.
    """
    s1 = set(tokens1) if isinstance(tokens1, (Sequence, set)) else set(extract_tokens(tokens1))
    s2 = set(tokens2) if isinstance(tokens2, (Sequence, set)) else set(extract_tokens(tokens2))
    if not s1 or not s2:
        return 0.0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return float(intersection / union) if union > 0 else 0.0


def token_overlap(
    tokens1: Sequence[str] | set[str] | object,
    tokens2: Sequence[str] | set[str] | object,
) -> float:
    """Compute token overlap (containment) coefficient.
    
    Returns |A ∩ B| / min(|A|, |B|) in [0.0, 1.0]. Returns 0.0 if either is empty.
    """
    s1 = set(tokens1) if isinstance(tokens1, (Sequence, set)) else set(extract_tokens(tokens1))
    s2 = set(tokens2) if isinstance(tokens2, (Sequence, set)) else set(extract_tokens(tokens2))
    if not s1 or not s2:
        return 0.0
    intersection = len(s1.intersection(s2))
    min_size = min(len(s1), len(s2))
    return float(intersection / min_size) if min_size > 0 else 0.0


def extract_char_ngrams(text: object, n: int = 3) -> set[str]:
    """Extract boundary-padded character n-grams from text."""
    if is_blank(text):
        return set()
    s = f"_{str(text).strip().casefold()}_"
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def char_ngram_jaccard(s1: object, s2: object, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity between two strings."""
    ng1 = extract_char_ngrams(s1, n=n)
    ng2 = extract_char_ngrams(s2, n=n)
    if not ng1 or not ng2:
        return 0.0
    intersection = len(ng1.intersection(ng2))
    union = len(ng1.union(ng2))
    return float(intersection / union) if union > 0 else 0.0


def levenshtein_similarity(s1: object, s2: object) -> float:
    """Compute normalized Levenshtein similarity: 1.0 - (edit_distance / max_len).
    
    Returns value in [0.0, 1.0]. Returns 0.0 if either string is blank.
    """
    if is_blank(s1) or is_blank(s2):
        return 0.0
    str1 = str(s1).strip().casefold()
    str2 = str(s2).strip().casefold()
    if str1 == str2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return float(Levenshtein.normalized_similarity(str1, str2))

    # Pure Python fallback
    m, n = len(str1), len(str2)
    if m > n:
        str1, str2, m, n = str2, str1, n, m
    prev = list(range(m + 1))
    for j, c2 in enumerate(str2, 1):
        curr = [j] * (m + 1)
        for i, c1 in enumerate(str1, 1):
            cost = 0 if c1 == c2 else 1
            curr[i] = min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost)
        prev = curr
    dist = prev[m]
    return float(1.0 - dist / max(m, n))


def jaro_winkler_similarity(s1: object, s2: object, prefix_weight: float = 0.1) -> float:
    """Compute Jaro-Winkler similarity between two strings in [0.0, 1.0]."""
    if is_blank(s1) or is_blank(s2):
        return 0.0
    str1 = str(s1).strip().casefold()
    str2 = str(s2).strip().casefold()
    if str1 == str2:
        return 1.0
    if HAS_RAPIDFUZZ:
        return float(JaroWinkler.similarity(str1, str2, prefix_weight=prefix_weight))

    # Pure Python fallback for Jaro-Winkler
    len1, len2 = len(str1), len(str2)
    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if not s2_matches[j] and str1[i] == str2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    k = 0
    transpositions = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if str1[i] != str2[k]:
                transpositions += 1
            k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2.0) / matches) / 3.0

    prefix = 0
    for i in range(min(4, len1, len2)):
        if str1[i] == str2[i]:
            prefix += 1
        else:
            break

    return float(jaro + prefix * prefix_weight * (1.0 - jaro))


def common_prefix_ratio(s1: object, s2: object) -> float:
    """Length of longest common prefix divided by max(len1, len2)."""
    if is_blank(s1) or is_blank(s2):
        return 0.0
    str1 = str(s1).strip().casefold()
    str2 = str(s2).strip().casefold()
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 0.0
    prefix_len = 0
    for c1, c2 in zip(str1, str2):
        if c1 == c2:
            prefix_len += 1
        else:
            break
    return float(prefix_len / max_len)


def common_suffix_ratio(s1: object, s2: object) -> float:
    """Length of longest common suffix divided by max(len1, len2)."""
    if is_blank(s1) or is_blank(s2):
        return 0.0
    str1 = str(s1).strip().casefold()
    str2 = str(s2).strip().casefold()
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 0.0
    suffix_len = 0
    for c1, c2 in zip(reversed(str1), reversed(str2)):
        if c1 == c2:
            suffix_len += 1
        else:
            break
    return float(suffix_len / max_len)


@dataclass(frozen=True, slots=True)
class FieldComparisonResult:
    """Four-way indicators for comparison of an optional structured field."""

    exact_match: float  # 1.0 if both present and equal, else 0.0
    both_present: float  # 1.0 if neither is blank, else 0.0
    one_missing: float  # 1.0 if exactly one is blank, else 0.0
    both_missing: float  # 1.0 if both are blank, else 0.0


def compare_structured_field(v1: object, v2: object) -> FieldComparisonResult:
    """Deterministic 4-way comparison for any structured field.
    
    CRITICAL: Both missing NEVER counts as exact_match=1.0.
    """
    b1 = is_blank(v1)
    b2 = is_blank(v2)
    if b1 and b2:
        return FieldComparisonResult(exact_match=0.0, both_present=0.0, one_missing=0.0, both_missing=1.0)
    if b1 or b2:
        return FieldComparisonResult(exact_match=0.0, both_present=0.0, one_missing=1.0, both_missing=0.0)
    # Both are present
    match = 1.0 if str(v1).strip().casefold() == str(v2).strip().casefold() else 0.0
    return FieldComparisonResult(exact_match=match, both_present=1.0, one_missing=0.0, both_missing=0.0)
