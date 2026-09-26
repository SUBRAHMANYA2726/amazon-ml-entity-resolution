"""Entity-aware train/validation splitting module for Phase 6.

Enforces zero-leakage entity grouping: all candidate pairs belonging to the same
source1 entity must stay entirely within either training or validation.
Never splits individual pairs randomly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EntitySplitStatistics:
    """Detailed metadata and diagnostic statistics for an entity-aware split."""

    random_seed: int
    val_fraction: float
    split_method: str
    n_train_entities: int
    n_val_entities: int
    n_total_entities: int
    n_train_pairs: int
    n_val_pairs: int
    n_total_pairs: int
    train_positive_pairs: int
    train_negative_pairs: int
    train_positive_rate: float
    val_positive_pairs: int
    val_negative_pairs: int
    val_positive_rate: float
    train_scale_pos_weight: float

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EntitySplitResult:
    """Result of partitioning a pairwise dataset by entity anchor."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    train_entity_ids: frozenset[str]
    val_entity_ids: frozenset[str]
    statistics: EntitySplitStatistics


def entity_aware_train_val_split(
    df: pd.DataFrame,
    *,
    entity_col: str = "source1_entity_id",
    label_col: str = "ground_truth_label",
    val_fraction: float = 0.20,
    random_seed: int = 2026,
) -> EntitySplitResult:
    """Split candidate pairs deterministically by anchor entity ID.

    Parameters
    ----------
    df : pd.DataFrame
        Pairwise feature matrix or candidate dataset.
    entity_col : str
        Column name identifying the anchor entity (e.g. 'source1_entity_id').
    label_col : str
        Column name for ground-truth match labels (0 or 1).
    val_fraction : float
        Fraction of unique entities assigned to validation (default 0.20).
    random_seed : int
        Seed for deterministic random permutation.

    Returns
    -------
    EntitySplitResult
        Dataclass containing train_df, val_df, entity ID sets, and split statistics.

    Raises
    ------
    ValueError
        If inputs are invalid or if any anchor entity appears in both splits.
    """
    if entity_col not in df.columns:
        raise ValueError(f"Entity column '{entity_col}' not found in dataframe columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe columns: {list(df.columns)}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    unique_entities = np.sort(df[entity_col].dropna().unique())
    n_total_entities = len(unique_entities)

    if n_total_entities < 2:
        raise ValueError(f"Cannot perform entity split with fewer than 2 unique entities; got {n_total_entities}")

    # Deterministic permutation using a dedicated NumPy RandomState
    rng = np.random.RandomState(random_seed)
    shuffled_entities = unique_entities.copy()
    rng.shuffle(shuffled_entities)

    n_val = int(round(n_total_entities * val_fraction))
    n_val = max(1, min(n_val, n_total_entities - 1))
    n_train = n_total_entities - n_val

    val_entity_set = set(shuffled_entities[:n_val])
    train_entity_set = set(shuffled_entities[n_val:])

    # Strict Leakage Rule: Explicitly verify zero overlap between train and validation S1 IDs
    overlap = train_entity_set.intersection(val_entity_set)
    if overlap:
        raise ValueError(
            f"FATAL: Entity-aware split violated! Overlap detected between train and val: {len(overlap)} entities: {overlap}"
        )

    # Partition rows
    is_val = df[entity_col].isin(val_entity_set)
    train_df = df[~is_val].copy()
    val_df = df[is_val].copy()

    # Re-verify at row level
    train_row_entities = set(train_df[entity_col].unique())
    val_row_entities = set(val_df[entity_col].unique())
    row_overlap = train_row_entities.intersection(val_row_entities)
    if row_overlap:
        raise ValueError(
            f"FATAL: Row-level entity overlap detected in split! Overlapping IDs: {row_overlap}"
        )

    # Compute label statistics
    train_pos = int((train_df[label_col] == 1).sum())
    train_neg = int((train_df[label_col] == 0).sum())
    val_pos = int((val_df[label_col] == 1).sum())
    val_neg = int((val_df[label_col] == 0).sum())

    train_pos_rate = float(train_pos / len(train_df)) if len(train_df) > 0 else 0.0
    val_pos_rate = float(val_pos / len(val_df)) if len(val_df) > 0 else 0.0

    # Calculate exact scale_pos_weight (negative / positive)
    scale_pos_weight = float(train_neg / train_pos) if train_pos > 0 else 1.0

    stats = EntitySplitStatistics(
        random_seed=random_seed,
        val_fraction=val_fraction,
        split_method="entity_group_split_by_" + entity_col,
        n_train_entities=n_train,
        n_val_entities=n_val,
        n_total_entities=n_total_entities,
        n_train_pairs=len(train_df),
        n_val_pairs=len(val_df),
        n_total_pairs=len(df),
        train_positive_pairs=train_pos,
        train_negative_pairs=train_neg,
        train_positive_rate=train_pos_rate,
        val_positive_pairs=val_pos,
        val_negative_pairs=val_neg,
        val_positive_rate=val_pos_rate,
        train_scale_pos_weight=scale_pos_weight,
    )

    LOGGER.info(
        "Entity-aware split completed: %d train S1 (%d pairs, %d pos / %d neg), %d val S1 (%d pairs, %d pos / %d neg), scale_pos_weight=%.2f",
        n_train,
        len(train_df),
        train_pos,
        train_neg,
        n_val,
        len(val_df),
        val_pos,
        val_neg,
        scale_pos_weight,
    )

    return EntitySplitResult(
        train_df=train_df,
        val_df=val_df,
        train_entity_ids=frozenset(train_entity_set),
        val_entity_ids=frozenset(val_entity_set),
        statistics=stats,
    )
