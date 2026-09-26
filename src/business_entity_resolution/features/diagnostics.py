"""Feature quality validation, redundancy diagnostics, and statistical separation analysis.

Strictly non-ML diagnostics for Phase 5:
1. Feature quality validation: ranges, dtypes, nulls, NaNs, infinities, constant features.
2. Redundancy analysis: collinear pairs (|r| > 0.95), duplicate features.
3. Label-aware positive vs. negative distribution comparison.
4. Statistical feature separation diagnostic (Cohen's d standardized difference).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from business_entity_resolution.features.metadata import FEATURE_REGISTRY, FeatureSpec
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FeatureColumnStat:
    """Individual statistics and validity indicators for a single feature column."""

    name: str
    group: str
    dtype: str
    missing_pct: float
    min_val: float
    max_val: float
    mean_val: float
    median_val: float
    std_val: float
    unique_count: int
    is_constant: bool
    is_near_constant: bool
    has_nan: bool
    has_inf: bool
    range_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "dtype": self.dtype,
            "missing_pct": round(self.missing_pct, 4),
            "min_val": round(self.min_val, 4),
            "max_val": round(self.max_val, 4),
            "mean_val": round(self.mean_val, 4),
            "median_val": round(self.median_val, 4),
            "std_val": round(self.std_val, 4),
            "unique_count": self.unique_count,
            "is_constant": self.is_constant,
            "is_near_constant": self.is_near_constant,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "range_valid": self.range_valid,
        }


@dataclass(frozen=True, slots=True)
class FeatureQualityReport:
    """Complete feature quality and validation report."""

    total_rows: int
    total_features: int
    column_stats: tuple[FeatureColumnStat, ...]
    constant_features: tuple[str, ...]
    near_constant_features: tuple[str, ...]
    invalid_range_features: tuple[str, ...]
    nan_features: tuple[str, ...]
    inf_features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_features": self.total_features,
            "constant_features": list(self.constant_features),
            "near_constant_features": list(self.near_constant_features),
            "invalid_range_features": list(self.invalid_range_features),
            "nan_features": list(self.nan_features),
            "inf_features": list(self.inf_features),
            "column_stats": [cs.to_dict() for cs in self.column_stats],
        }


def validate_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> FeatureQualityReport:
    """Validate data quality across all feature columns."""
    LOGGER.info("Validating feature matrix quality for %d features across %d rows...", len(feature_cols), len(df))
    stats_list: list[FeatureColumnStat] = []
    constant_feats: list[str] = []
    near_constant_feats: list[str] = []
    invalid_range_feats: list[str] = []
    nan_feats: list[str] = []
    inf_feats: list[str] = []

    n_rows = len(df)
    for col in feature_cols:
        series = df[col]
        vals = series.to_numpy(dtype=np.float64)

        has_nan = bool(np.isnan(vals).any())
        has_inf = bool(np.isinf(vals).any())
        if has_nan:
            nan_feats.append(col)
        if has_inf:
            inf_feats.append(col)

        clean_vals = vals[np.isfinite(vals)] if (has_nan or has_inf) else vals
        if len(clean_vals) > 0:
            min_v = float(np.min(clean_vals))
            max_v = float(np.max(clean_vals))
            mean_v = float(np.mean(clean_vals))
            median_v = float(np.median(clean_vals))
            std_v = float(np.std(clean_vals))
        else:
            min_v = max_v = mean_v = median_v = std_v = 0.0

        n_unique = int(series.nunique(dropna=True))
        is_const = n_unique <= 1
        is_near = is_const or (std_v < 1e-4) or (float((vals == vals[0]).sum()) / n_rows > 0.999)

        if is_const:
            constant_feats.append(col)
        elif is_near:
            near_constant_feats.append(col)

        # Expected range check
        spec = FEATURE_REGISTRY.get(col)
        range_valid = True
        if spec and spec.expected_range:
            low, high = spec.expected_range
            if low is not None and min_v < (low - 1e-5):
                range_valid = False
            if high is not None and max_v > (high + 1e-5):
                range_valid = False

        if not range_valid:
            invalid_range_feats.append(col)

        missing_pct = float(series.isna().sum() / n_rows) if n_rows > 0 else 0.0

        stats_list.append(
            FeatureColumnStat(
                name=col,
                group=spec.group if spec else "unknown",
                dtype=str(series.dtype),
                missing_pct=missing_pct,
                min_val=min_v,
                max_val=max_v,
                mean_val=mean_v,
                median_val=median_v,
                std_val=std_v,
                unique_count=n_unique,
                is_constant=is_const,
                is_near_constant=is_near,
                has_nan=has_nan,
                has_inf=has_inf,
                range_valid=range_valid,
            )
        )

    return FeatureQualityReport(
        total_rows=n_rows,
        total_features=len(feature_cols),
        column_stats=tuple(stats_list),
        constant_features=tuple(constant_feats),
        near_constant_features=tuple(near_constant_feats),
        invalid_range_features=tuple(invalid_range_feats),
        nan_features=tuple(nan_feats),
        inf_features=tuple(inf_feats),
    )


def compute_redundancy_diagnostics(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    corr_threshold: float = 0.95,
) -> dict[str, Any]:
    """Compute feature correlations and identify highly collinear feature pairs."""
    LOGGER.info("Computing correlation matrix for redundancy diagnostics (|r| >= %.2f)...", corr_threshold)
    # Exclude constant features from correlation matrix
    valid_cols = [c for c in feature_cols if df[c].nunique() > 1]
    sub_df = df[valid_cols]

    corr_mat = sub_df.corr(method="pearson")
    collinear_pairs: list[dict[str, Any]] = []

    cols = list(corr_mat.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1 = cols[i]
            c2 = cols[j]
            val = corr_mat.iloc[i, j]
            if not np.isnan(val) and abs(val) >= corr_threshold:
                collinear_pairs.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "pearson_correlation": round(float(val), 4),
                    "is_exact_duplicate": bool(abs(val) >= 0.9999),
                })

    collinear_pairs.sort(key=lambda x: abs(x["pearson_correlation"]), reverse=True)
    return {
        "correlation_threshold": corr_threshold,
        "total_collinear_pairs": len(collinear_pairs),
        "exact_duplicates": [p for p in collinear_pairs if p["is_exact_duplicate"]],
        "collinear_pairs": collinear_pairs,
    }


def compute_separation_diagnostics(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "ground_truth_label",
) -> list[dict[str, Any]]:
    """Compute statistical separation diagnostic (Cohen's d) for distinguishing positive from negative pairs.
    
    This is purely a descriptive statistical diagnostic, NOT feature importance or ML model fitting.
    """
    if label_col not in df.columns:
        LOGGER.warning("Label column '%s' absent; skipping separation diagnostic.", label_col)
        return []

    pos_df = df[df[label_col] == 1]
    neg_df = df[df[label_col] == 0]

    diagnostics: list[dict[str, Any]] = []
    for col in feature_cols:
        pos_vals = pos_df[col].to_numpy(dtype=np.float64)
        neg_vals = neg_df[col].to_numpy(dtype=np.float64)

        pos_mean = float(np.mean(pos_vals)) if len(pos_vals) > 0 else 0.0
        neg_mean = float(np.mean(neg_vals)) if len(neg_vals) > 0 else 0.0
        pos_med = float(np.median(pos_vals)) if len(pos_vals) > 0 else 0.0
        neg_med = float(np.median(neg_vals)) if len(neg_vals) > 0 else 0.0

        pos_std = float(np.std(pos_vals)) if len(pos_vals) > 0 else 0.0
        neg_std = float(np.std(neg_vals)) if len(neg_vals) > 0 else 0.0

        pooled_std = math.sqrt((pos_std**2 + neg_std**2) / 2.0)
        cohen_d = (pos_mean - neg_mean) / pooled_std if pooled_std > 1e-6 else 0.0

        spec = FEATURE_REGISTRY.get(col)
        diagnostics.append({
            "feature_name": col,
            "feature_group": spec.group if spec else "unknown",
            "cohens_d_separation": round(float(cohen_d), 4),
            "abs_separation": round(abs(float(cohen_d)), 4),
            "pos_mean": round(pos_mean, 4),
            "neg_mean": round(neg_mean, 4),
            "mean_difference": round(pos_mean - neg_mean, 4),
            "pos_median": round(pos_med, 4),
            "neg_median": round(neg_med, 4),
        })

    diagnostics.sort(key=lambda x: x["abs_separation"], reverse=True)
    return diagnostics


def summarize_class_distribution(
    df: pd.DataFrame,
    label_col: str = "ground_truth_label",
) -> dict[str, Any]:
    """Compute exact candidate pairs class distribution overall and per source."""
    total = len(df)
    if label_col not in df.columns:
        return {"total_candidate_pairs": total}

    labels = df[label_col]
    pos = int(labels.sum())
    neg = total - pos

    s2_mask = df["candidate_source"] == "S2"
    s3_mask = df["candidate_source"] == "S3"

    s2_pos = int(df.loc[s2_mask, label_col].sum())
    s2_neg = int(s2_mask.sum() - s2_pos)
    s3_pos = int(df.loc[s3_mask, label_col].sum())
    s3_neg = int(s3_mask.sum() - s3_pos)

    # Candidates per anchor
    cands_per_anchor = df.groupby("source1_entity_id").size()

    return {
        "total_candidate_pairs": total,
        "positive_pairs": pos,
        "negative_pairs": neg,
        "positive_percentage": round(100.0 * pos / total, 2) if total > 0 else 0.0,
        "negative_percentage": round(100.0 * neg / total, 2) if total > 0 else 0.0,
        "s2_distribution": {
            "total_pairs": int(s2_mask.sum()),
            "positives": s2_pos,
            "negatives": s2_neg,
            "positive_rate": round(100.0 * s2_pos / s2_mask.sum(), 2) if s2_mask.sum() > 0 else 0.0,
        },
        "s3_distribution": {
            "total_pairs": int(s3_mask.sum()),
            "positives": s3_pos,
            "negatives": s3_neg,
            "positive_rate": round(100.0 * s3_pos / s3_mask.sum(), 2) if s3_mask.sum() > 0 else 0.0,
        },
        "candidates_per_anchor": {
            "mean": round(float(cands_per_anchor.mean()), 2),
            "median": round(float(cands_per_anchor.median()), 1),
            "p90": round(float(cands_per_anchor.quantile(0.90)), 1),
            "p95": round(float(cands_per_anchor.quantile(0.95)), 1),
            "p99": round(float(cands_per_anchor.quantile(0.99)), 1),
            "min": int(cands_per_anchor.min()),
            "max": int(cands_per_anchor.max()),
        },
    }
