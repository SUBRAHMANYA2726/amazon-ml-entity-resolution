"""Strategy B & C: Name Token and Address Token Blocking.

Strategy B (Name Token Blocking):
- Tokenizes normalized business names into clean alphanumeric tokens.
- Filters out universal stop words and corporate legal forms (LLC, Inc, Pvt Ltd).
- Enforces min_token_length and max_token_frequency to eliminate candidate explosion.
- Builds an inverted index (token -> candidate_ids).

Strategy C (Address Token Blocking):
- Tokenizes normalized addresses into postal codes, city prefixes, street designators, numbers.
- Builds composite keys (postal + city, postal + token, street + number).
- Protects against degenerate empty/null blocks.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import pandas as pd

from business_entity_resolution.blocking.exact import (
    STOP_TOKENS,
    WORD_TOKEN_RE,
    extract_house_number,
    extract_postal_code,
)
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)

# Common street designators to omit as solitary keys
STREET_STOP_WORDS = frozenset({
    "st", "street", "rd", "road", "ave", "avenue", "dr", "drive",
    "ln", "lane", "ct", "court", "blvd", "boulevard", "way", "pl",
    "place", "pkwy", "parkway", "cir", "circle", "fl", "floor",
    "ste", "suite", "apt", "unit", "bldg", "building", "box", "po",
})


def tokenize_name(text: str | None, min_len: int = 3) -> list[str]:
    """Tokenize name into meaningful, non-stopword tokens."""
    if not text or not isinstance(text, str):
        return []
    tokens = WORD_TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) >= min_len and t not in STOP_TOKENS]


def tokenize_address(text: str | None, min_len: int = 3) -> list[str]:
    """Tokenize address into meaningful spatial tokens."""
    if not text or not isinstance(text, str):
        return []
    tokens = WORD_TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) >= min_len and t not in STREET_STOP_WORDS and t not in STOP_TOKENS]


class NameTokenBlocker:
    """Strategy B: Inverted index on selective business name tokens."""

    def __init__(
        self,
        *,
        min_token_length: int = 3,
        max_token_frequency: int = 10000,
        max_candidates_per_token: int = 500,
        max_candidates_per_query: int = 50,
        require_two_tokens: bool = False,
    ) -> None:
        self.min_token_length = min_token_length
        self.max_token_frequency = max_token_frequency
        self.max_candidates_per_token = max_candidates_per_token
        self.max_candidates_per_query = max_candidates_per_query
        self.require_two_tokens = require_two_tokens

        # Inverted index: token -> list of candidate entity IDs
        self.index: dict[str, list[str]] = defaultdict(list)
        self.token_frequencies: Counter[str] = Counter()
        self._pruned_tokens: set[str] = set()

    def fit(self, candidates_df: pd.DataFrame, id_col: str = "entity_id") -> NameTokenBlocker:
        """Build name token inverted index from candidate DataFrame."""
        name_col = None
        for col in ("business_name__name_core", "business_name__name", "business_name"):
            if col in candidates_df.columns:
                name_col = col
                break

        names = candidates_df[name_col].fillna("").astype(str).tolist() if name_col else [""] * len(candidates_df)
        ids = candidates_df[id_col].astype(str).tolist()

        raw_index = defaultdict(list)
        for i in range(len(ids)):
            cand_id = ids[i]
            tokens = set(tokenize_name(names[i], min_len=self.min_token_length))
            for tok in tokens:
                raw_index[tok].append(cand_id)
                self.token_frequencies[tok] += 1

        # Prune tokens that exceed max_token_frequency (e.g. ubiquitous words)
        for tok, postings in raw_index.items():
            if len(postings) > self.max_token_frequency:
                self._pruned_tokens.add(tok)
            else:
                self.index[tok] = postings

        LOGGER.info(
            "NameTokenBlocker indexed %d tokens (%d pruned as too frequent)",
            len(self.index),
            len(self._pruned_tokens),
        )
        return self

    def block_query(self, anchor_id: str, name: str | None) -> dict[str, set[str]]:
        """Retrieve candidate IDs for an anchor using name tokens."""
        tokens = tokenize_name(name, min_len=self.min_token_length)
        if not tokens:
            return {}

        token_candidates: Counter[str] = Counter()
        for tok in set(tokens):
            if tok in self._pruned_tokens:
                continue
            postings = self.index.get(tok, [])
            if 0 < len(postings) <= self.max_candidates_per_token:
                for cand_id in postings:
                    if cand_id != anchor_id:
                        token_candidates[cand_id] += 1

        results: dict[str, set[str]] = {}
        # Sort candidates by number of shared tokens descending
        sorted_cands = token_candidates.most_common(self.max_candidates_per_query)
        min_overlap = 2 if self.require_two_tokens and len(tokens) >= 2 else 1

        for cand_id, count in sorted_cands:
            if count >= min_overlap:
                results[cand_id] = {"name_token"}

        return results

    def generate_pairs(
        self,
        anchors_df: pd.DataFrame,
        id_col: str = "entity_id",
    ) -> dict[str, dict[str, set[str]]]:
        """Generate candidates for all anchors in DataFrame."""
        name_col = None
        for col in ("business_name__name_core", "business_name__name", "business_name"):
            if col in anchors_df.columns:
                name_col = col
                break

        names = anchors_df[name_col].fillna("").astype(str).tolist() if name_col else [""] * len(anchors_df)
        ids = anchors_df[id_col].astype(str).tolist()

        all_cands: dict[str, dict[str, set[str]]] = {}
        for i in range(len(ids)):
            aid = ids[i]
            all_cands[aid] = self.block_query(aid, names[i])
        return all_cands


class AddressTokenBlocker:
    """Strategy C: Inverted index on composite address tokens."""

    def __init__(
        self,
        *,
        max_block_size: int = 5000,
        max_candidates_per_query: int = 50,
    ) -> None:
        self.max_block_size = max_block_size
        self.max_candidates_per_query = max_candidates_per_query

        # Inverted index: key -> list of candidate entity IDs
        self.index: dict[str, list[str]] = defaultdict(list)
        self._key_counts: Counter[str] = Counter()

    def fit(self, candidates_df: pd.DataFrame, id_col: str = "entity_id") -> AddressTokenBlocker:
        """Build composite address inverted index."""
        addr_col = None
        for col in ("business_address__address", "business_address"):
            if col in candidates_df.columns:
                addr_col = col
                break

        country_col = None
        for col in ("country__country", "country"):
            if col in candidates_df.columns:
                country_col = col
                break

        addrs = candidates_df[addr_col].fillna("").astype(str).tolist() if addr_col else [""] * len(candidates_df)
        countries = candidates_df[country_col].fillna("").astype(str).str.lower().tolist() if country_col else [""] * len(candidates_df)
        ids = candidates_df[id_col].astype(str).tolist()

        for i in range(len(ids)):
            cand_id = ids[i]
            addr = addrs[i]
            country = countries[i]
            if not addr:
                continue

            postal = extract_postal_code(addr)
            house = extract_house_number(addr)
            tokens = tokenize_address(addr, min_len=4)

            # Key 1: country + postal + house
            if country and postal and house:
                k1 = f"cph_{country}_{postal}_{house}"
                self.index[k1].append(cand_id)
                self._key_counts[k1] += 1

            # Key 2: country + house + street token
            if country and house and tokens:
                for tok in tokens[:2]:
                    k2 = f"chst_{country}_{house}_{tok}"
                    self.index[k2].append(cand_id)
                    self._key_counts[k2] += 1

            # Key 3: postal + street token
            if postal and tokens:
                for tok in tokens[:2]:
                    k3 = f"pst_{postal}_{tok}"
                    self.index[k3].append(cand_id)
                    self._key_counts[k3] += 1

        LOGGER.info(
            "AddressTokenBlocker indexed %d address keys",
            len(self.index),
        )
        return self

    def block_query(
        self,
        anchor_id: str,
        address: str | None,
        country: str | None,
    ) -> dict[str, set[str]]:
        """Retrieve candidate IDs for an anchor using address tokens."""
        if not address or not isinstance(address, str):
            return {}

        country_norm = country.lower().strip() if country else ""
        postal = extract_postal_code(address)
        house = extract_house_number(address)
        tokens = tokenize_address(address, min_len=4)

        keys_to_query = []
        if country_norm and postal and house:
            keys_to_query.append(f"cph_{country_norm}_{postal}_{house}")
        if country_norm and house and tokens:
            for tok in tokens[:2]:
                keys_to_query.append(f"chst_{country_norm}_{house}_{tok}")
        if postal and tokens:
            for tok in tokens[:2]:
                keys_to_query.append(f"pst_{postal}_{tok}")

        cand_counts: Counter[str] = Counter()
        for k in keys_to_query:
            postings = self.index.get(k, [])
            if 0 < len(postings) <= self.max_block_size:
                for cand_id in postings:
                    if cand_id != anchor_id:
                        cand_counts[cand_id] += 1

        results: dict[str, set[str]] = {}
        for cand_id, _ in cand_counts.most_common(self.max_candidates_per_query):
            results[cand_id] = {"address_token"}

        return results

    def generate_pairs(
        self,
        anchors_df: pd.DataFrame,
        id_col: str = "entity_id",
    ) -> dict[str, dict[str, set[str]]]:
        """Generate candidates for all anchors in DataFrame."""
        addr_col = None
        for col in ("business_address__address", "business_address"):
            if col in anchors_df.columns:
                addr_col = col
                break

        country_col = None
        for col in ("country__country", "country"):
            if col in anchors_df.columns:
                country_col = col
                break

        addrs = anchors_df[addr_col].fillna("").astype(str).tolist() if addr_col else [""] * len(anchors_df)
        countries = anchors_df[country_col].fillna("").astype(str).tolist() if country_col else [""] * len(anchors_df)
        ids = anchors_df[id_col].astype(str).tolist()

        all_cands: dict[str, dict[str, set[str]]] = {}
        for i in range(len(ids)):
            aid = ids[i]
            all_cands[aid] = self.block_query(aid, addrs[i], countries[i])
        return all_cands
