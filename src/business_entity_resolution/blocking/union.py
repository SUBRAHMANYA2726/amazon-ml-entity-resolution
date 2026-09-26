"""Multi-Strategy Candidate Union and Deduplication Engine.

Merges candidates generated across all independent blocking strategies:
- Removes duplicate candidate pairs per anchor.
- Preserves candidate provenance (frozenset of strategy names that retrieved each candidate).
- Validates candidate IDs (disallows self-matches, ensures S2/S3 validity, disallows S1 candidates).
- Preserves source identity (distinguishing S2 from S3).
- Emits candidate pairs in standard contracts and submission TSV format.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from business_entity_resolution.contracts import CandidatePair
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)

VALID_CANDIDATE_PREFIXES = frozenset({"S2", "S3"})
VALID_TARGET_ID_RE = re.compile(r"^S[23]-\d+$")


@dataclass(frozen=True, slots=True)
class ProvenancedCandidate:
    """A candidate entity with its provenance and source designation."""

    candidate_id: str
    source: str  # "S2" or "S3"
    strategies: frozenset[str]


@dataclass(frozen=True, slots=True)
class AnchorCandidates:
    """All candidates retrieved for a single anchor entity."""

    anchor_id: str
    candidates: tuple[ProvenancedCandidate, ...] = ()

    @property
    def total_candidates(self) -> int:
        return len(self.candidates)

    @property
    def s2_candidates(self) -> tuple[ProvenancedCandidate, ...]:
        return tuple(c for c in self.candidates if c.source == "S2")

    @property
    def s3_candidates(self) -> tuple[ProvenancedCandidate, ...]:
        return tuple(c for c in self.candidates if c.source == "S3")

    def candidate_ids(self) -> list[str]:
        return [c.candidate_id for c in self.candidates]


class CandidateUnion:
    """Merge, deduplicate, validate, and track provenance across multiple strategy results."""

    def __init__(
        self,
        *,
        max_candidates_per_anchor: int | None = None,
        valid_candidate_ids: set[str] | None = None,
    ) -> None:
        self.max_candidates_per_anchor = max_candidates_per_anchor
        self.valid_candidate_ids = valid_candidate_ids

    def merge_strategy_results(
        self,
        strategy_candidate_maps: Sequence[tuple[str, Mapping[str, Mapping[str, Iterable[str]]]]],
    ) -> dict[str, AnchorCandidates]:
        """Merge candidates from multiple strategies.

        Args:
            strategy_candidate_maps: Sequence of (strategy_name, mapping of anchor_id -> {cand_id -> set_of_substrategies}).

        Returns:
            Mapping of anchor_id -> AnchorCandidates object.
        """
        # anchor_id -> candidate_id -> set of strategies
        aggregated: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

        for strat_name, cand_map in strategy_candidate_maps:
            for aid, cands in cand_map.items():
                for cid, substrats in cands.items():
                    # Validate candidate ID
                    if not self._is_valid_candidate(aid, cid):
                        continue
                    # Add strategy provenance
                    if substrats:
                        aggregated[aid][cid].update(substrats)
                    else:
                        aggregated[aid][cid].add(strat_name)

        # Convert to immutable AnchorCandidates objects
        results: dict[str, AnchorCandidates] = {}
        for aid, cands in aggregated.items():
            provenanced: list[ProvenancedCandidate] = []
            for cid, strats in cands.items():
                src = cid.split("-")[0] if "-" in cid else "unknown"
                provenanced.append(
                    ProvenancedCandidate(
                        candidate_id=cid,
                        source=src,
                        strategies=frozenset(strats),
                    )
                )

            # Optional cap on candidates per anchor (prioritizing candidates with highest strategy agreement)
            if self.max_candidates_per_anchor and len(provenanced) > self.max_candidates_per_anchor:
                # Sort by number of confirming strategies descending, then by candidate_id
                provenanced.sort(key=lambda x: (-len(x.strategies), x.candidate_id))
                provenanced = provenanced[: self.max_candidates_per_anchor]

            results[aid] = AnchorCandidates(anchor_id=aid, candidates=tuple(provenanced))

        return results

    def _is_valid_candidate(self, anchor_id: str, candidate_id: str) -> bool:
        """Validate candidate ID adheres to entity resolution constraints."""
        if not candidate_id or not isinstance(candidate_id, str):
            return False

        # No self-matches
        if candidate_id == anchor_id:
            return False

        # Anchor is S1; candidate must NOT be S1
        if candidate_id.startswith("S1-"):
            return False

        # Candidate must start with S2 or S3
        prefix = candidate_id.split("-")[0] if "-" in candidate_id else ""
        if prefix not in VALID_CANDIDATE_PREFIXES:
            return False

        # If a strict candidate whitelist is provided, check membership
        if self.valid_candidate_ids is not None and candidate_id not in self.valid_candidate_ids:
            return False

        return True

    @staticmethod
    def to_candidate_pairs(merged_candidates: Mapping[str, AnchorCandidates]) -> list[CandidatePair]:
        """Convert merged candidates to the pipeline's CandidatePair contracts."""
        pairs: list[CandidatePair] = []
        for aid, acands in merged_candidates.items():
            for c in acands.candidates:
                pairs.append(
                    CandidatePair(
                        anchor_ref=aid,
                        candidate_ref=c.candidate_id,
                        provenance=c.strategies,
                    )
                )
        return pairs

    @staticmethod
    def to_dataframe(merged_candidates: Mapping[str, AnchorCandidates]) -> pd.DataFrame:
        """Convert merged candidates to a flat evaluation/feature DataFrame."""
        rows = []
        for aid, acands in merged_candidates.items():
            for c in acands.candidates:
                rows.append({
                    "source1_entity_id": aid,
                    "candidate_entity_id": c.candidate_id,
                    "candidate_source": c.source,
                    "strategy_count": len(c.strategies),
                    "strategies": ";".join(sorted(c.strategies)),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def export_candidate_pairs_tsv(
        merged_candidates: Mapping[str, AnchorCandidates],
        output_path: str | Path,
    ) -> Path:
        """Export merged candidates to candidate_pairs.tsv format.

        Format:
        source1_entity_id\tcandidate_entity_ids
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for aid, acands in merged_candidates.items():
                cids = [c.candidate_id for c in acands.candidates]
                cand_str = ",".join(cids)
                f.write(f"{aid}\t{cand_str}\n")

        LOGGER.info("Exported candidate pairs to %s (%d anchors)", path, len(merged_candidates))
        return path
