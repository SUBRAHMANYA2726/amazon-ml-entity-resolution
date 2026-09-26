"""Strategy A: Exact Blocking Keys.

Implements exact blocking using strong normalized fields:
- normalized business name
- alphabetically sorted business name (word-order invariant)
- country + normalized business name
- country + postal code + house number
- country + postal code + name
- country + house number + first significant name token

Missing and empty values are guarded strictly to avoid degenerate massive blocks.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from business_entity_resolution.contracts import CandidatePair
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)

# Regular expressions for structured field extraction
POSTAL_CODE_RE = re.compile(r"\b(\d{5}(?:-\d{4})?|\d{6})\b")
HOUSE_NUMBER_RE = re.compile(r"\b(\d{1,6}[A-Za-z]?)\b")
WORD_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOP_TOKENS = frozenset({
    "the", "and", "of", "in", "for", "at", "by", "on", "with", "to",
    "llc", "inc", "corp", "corporation", "ltd", "limited", "pvt",
    "co", "company", "pllc", "lp", "llp", "pc", "services", "group",
})


def extract_postal_code(text: str | None) -> str | None:
    """Extract standard 5-digit US ZIP or 6-digit Indian PIN code."""
    if not text or not isinstance(text, str):
        return None
    match = POSTAL_CODE_RE.search(text)
    if match:
        code = match.group(1).replace("-", "").strip()
        if len(code) in (5, 6, 9):
            return code[:5] if len(code) == 9 else code
    return None


def extract_house_number(text: str | None) -> str | None:
    """Extract leading or prominent street house number."""
    if not text or not isinstance(text, str):
        return None
    match = HOUSE_NUMBER_RE.search(text)
    return match.group(1).lower().strip() if match else None


def extract_first_significant_token(text: str | None) -> str | None:
    """Extract first significant name token (excluding stop words and legal forms)."""
    if not text or not isinstance(text, str):
        return None
    tokens = WORD_TOKEN_RE.findall(text.lower())
    for token in tokens:
        if len(token) >= 3 and token not in STOP_TOKENS:
            return token
    return tokens[0] if tokens else None


def clean_name_string(text: str | None) -> str | None:
    """Return lowercase alphanumeric normalized name string."""
    if not text or not isinstance(text, str):
        return None
    cleaned = " ".join(WORD_TOKEN_RE.findall(text.lower()))
    return cleaned if cleaned else None


def sorted_name_string(text: str | None) -> str | None:
    """Return alphabetically sorted name tokens (word-order invariant)."""
    if not text or not isinstance(text, str):
        return None
    tokens = sorted(t for t in WORD_TOKEN_RE.findall(text.lower()) if t not in STOP_TOKENS)
    return " ".join(tokens) if tokens else clean_name_string(text)


class ExactBlocker:
    """Generate candidate pairs using exact matches on strong normalized fields."""

    def __init__(
        self,
        *,
        max_block_size: int = 5000,
        enable_name_exact: bool = True,
        enable_name_sorted: bool = True,
        enable_country_name: bool = True,
        enable_country_postal_name: bool = True,
        enable_country_postal_house: bool = True,
        enable_country_house_first_token: bool = True,
    ) -> None:
        self.max_block_size = max_block_size
        self.enable_name_exact = enable_name_exact
        self.enable_name_sorted = enable_name_sorted
        self.enable_country_name = enable_country_name
        self.enable_country_postal_name = enable_country_postal_name
        self.enable_country_postal_house = enable_country_postal_house
        self.enable_country_house_first_token = enable_country_house_first_token

        # Inverted index: strategy_name -> key -> list of candidate entity IDs
        self._indexes: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self._candidate_sources: dict[str, str] = {}
        self._indexed_count = 0

    def fit(self, candidates_df: pd.DataFrame, id_col: str = "entity_id") -> ExactBlocker:
        """Build exact key inverted indexes over candidate records."""
        if candidates_df.empty:
            return self

        name_col = self._find_column(candidates_df, ("business_name__name_core", "business_name__name", "business_name"))
        addr_col = self._find_column(candidates_df, ("business_address__address", "business_address"))
        country_col = self._find_column(candidates_df, ("country__country", "country"))

        names = candidates_df[name_col].fillna("").astype(str).tolist() if name_col else [""] * len(candidates_df)
        addrs = candidates_df[addr_col].fillna("").astype(str).tolist() if addr_col else [""] * len(candidates_df)
        countries = candidates_df[country_col].fillna("").astype(str).str.lower().tolist() if country_col else [""] * len(candidates_df)
        ids = candidates_df[id_col].astype(str).tolist()

        for i in range(len(ids)):
            cand_id = ids[i]
            name = names[i]
            addr = addrs[i]
            country = countries[i]

            # Cache candidate source (e.g. S2 vs S3)
            prefix = cand_id.split("-")[0] if "-" in cand_id else "unknown"
            self._candidate_sources[cand_id] = prefix

            norm_name = clean_name_string(name)
            name_sorted = sorted_name_string(name)
            postal = extract_postal_code(addr)
            house = extract_house_number(addr)
            first_tok = extract_first_significant_token(name)

            if self.enable_name_exact and norm_name:
                self._indexes["exact_name"][norm_name].append(cand_id)

            if self.enable_name_sorted and name_sorted and name_sorted != norm_name:
                self._indexes["exact_name_sorted"][name_sorted].append(cand_id)

            if self.enable_country_name and country and norm_name:
                self._indexes["exact_country_name"][f"{country}_{norm_name}"].append(cand_id)

            if self.enable_country_postal_name and country and postal and norm_name:
                self._indexes["exact_country_postal_name"][f"{country}_{postal}_{norm_name}"].append(cand_id)

            if self.enable_country_postal_house and country and postal and house:
                self._indexes["exact_country_postal_house"][f"{country}_{postal}_{house}"].append(cand_id)

            if self.enable_country_house_first_token and country and house and first_tok:
                self._indexes["exact_country_house_token"][f"{country}_{house}_{first_tok}"].append(cand_id)

        self._indexed_count += len(candidates_df)
        LOGGER.info(
            "ExactBlocker indexed %d candidates across %d key strategies",
            len(candidates_df),
            len(self._indexes),
        )
        return self

    def block_query(
        self,
        anchor_id: str,
        name: str | None,
        address: str | None,
        country: str | None,
    ) -> dict[str, set[str]]:
        """Retrieve candidate IDs and their strategy provenance for a single anchor.

        Returns:
            Mapping of candidate_id -> set of strategy names that retrieved it.
        """
        candidates: dict[str, set[str]] = defaultdict(set)
        norm_name = clean_name_string(name)
        name_sorted = sorted_name_string(name)
        country_norm = country.lower().strip() if country else ""
        postal = extract_postal_code(address)
        house = extract_house_number(address)
        first_tok = extract_first_significant_token(name)

        keys_to_check = []
        if self.enable_name_exact and norm_name:
            keys_to_check.append(("exact_name", norm_name))

        if self.enable_name_sorted and name_sorted and name_sorted != norm_name:
            keys_to_check.append(("exact_name_sorted", name_sorted))

        if self.enable_country_name and country_norm and norm_name:
            keys_to_check.append(("exact_country_name", f"{country_norm}_{norm_name}"))

        if self.enable_country_postal_name and country_norm and postal and norm_name:
            keys_to_check.append(("exact_country_postal_name", f"{country_norm}_{postal}_{norm_name}"))

        if self.enable_country_postal_house and country_norm and postal and house:
            keys_to_check.append(("exact_country_postal_house", f"{country_norm}_{postal}_{house}"))

        if self.enable_country_house_first_token and country_norm and house and first_tok:
            keys_to_check.append(("exact_country_house_token", f"{country_norm}_{house}_{first_tok}"))

        for strat, key in keys_to_check:
            postings = self._indexes.get(strat, {}).get(key, [])
            if 0 < len(postings) <= self.max_block_size:
                for cand_id in postings:
                    if cand_id != anchor_id:
                        candidates[cand_id].add(strat)

        return dict(candidates)

    def generate_pairs(
        self,
        anchors_df: pd.DataFrame,
        id_col: str = "entity_id",
    ) -> dict[str, dict[str, set[str]]]:
        """Generate candidate pairs for all anchors in a DataFrame.

        Returns:
            Mapping of anchor_id -> {candidate_id -> set of strategies}
        """
        name_col = self._find_column(anchors_df, ("business_name__name_core", "business_name__name", "business_name"))
        addr_col = self._find_column(anchors_df, ("business_address__address", "business_address"))
        country_col = self._find_column(anchors_df, ("country__country", "country"))

        names = anchors_df[name_col].fillna("").astype(str).tolist() if name_col else [""] * len(anchors_df)
        addrs = anchors_df[addr_col].fillna("").astype(str).tolist() if addr_col else [""] * len(anchors_df)
        countries = anchors_df[country_col].fillna("").astype(str).tolist() if country_col else [""] * len(anchors_df)
        ids = anchors_df[id_col].astype(str).tolist()

        all_candidates: dict[str, dict[str, set[str]]] = {}
        for i in range(len(ids)):
            aid = ids[i]
            cands = self.block_query(aid, names[i], addrs[i], countries[i])
            all_candidates[aid] = cands

        return all_candidates

    @staticmethod
    def _find_column(df: pd.DataFrame, candidate_names: Sequence[str]) -> str | None:
        """Find first matching column from candidate names."""
        for name in candidate_names:
            if name in df.columns:
                return name
        return None
