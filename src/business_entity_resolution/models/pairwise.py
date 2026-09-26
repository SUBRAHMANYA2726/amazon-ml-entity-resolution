"""Supervised Pairwise ML Entity Matchers for Phase 6.

Implements concrete Matcher implementations using LightGBM and CatBoost.
Guarantees:
- Strict feature leakage prevention (blocks identifiers and labels).
- Exact feature column ordering preservation.
- Handling of class imbalance via training-derived scale_pos_weight.
- Probability scoring and validation (0 <= p <= 1, no NaN, no inf).
- Feature importance extraction labeled as MODEL FEATURE IMPORTANCE.
- Deterministic model serialization and reloadability.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from business_entity_resolution.models.base import Matcher
from business_entity_resolution.utils.logging import get_logger

LOGGER = get_logger(__name__)

# Strict Leakage Prevention: Columns that must NEVER be passed as model features
FORBIDDEN_FEATURE_COLUMNS = frozenset({
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "ground_truth_label",
    "label",
    "target",
    "direct_match_id",
})


def validate_and_extract_features(
    features: Any,
    expected_feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Validate input features, reject data leakage, and ensure column order.

    Parameters
    ----------
    features : pd.DataFrame or np.ndarray or Sequence
        Input feature matrix.
    expected_feature_names : Sequence[str] | None
        If provided, enforces presence and exact ordering of these features.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Clean 2D numpy array and feature name list.

    Raises
    ------
    ValueError
        If leakage columns are present, required columns are missing,
        or inputs contain invalid values.
    """
    if isinstance(features, pd.DataFrame):
        # 1. Strict Leakage Check
        found_forbidden = FORBIDDEN_FEATURE_COLUMNS.intersection(features.columns)
        if found_forbidden:
            raise ValueError(
                f"FATAL DATA LEAKAGE: Forbidden columns detected in model feature matrix: "
                f"{sorted(found_forbidden)}"
            )

        if expected_feature_names is not None:
            missing_cols = set(expected_feature_names) - set(features.columns)
            if missing_cols:
                raise ValueError(
                    f"Feature matrix is missing {len(missing_cols)} expected features: "
                    f"{sorted(missing_cols)[:5]}..."
                )
            # Order columns exactly as expected
            aligned_df = features[list(expected_feature_names)]
            feature_names = list(expected_feature_names)
        else:
            aligned_df = features
            feature_names = list(features.columns)

        X = aligned_df.to_numpy(dtype=np.float32)
    elif isinstance(features, np.ndarray):
        if expected_feature_names is None:
            raise ValueError("Expected feature names must be specified when passing raw NumPy arrays")
        if features.ndim != 2:
            raise ValueError(f"Feature array must be 2D, got shape {features.shape}")
        if features.shape[1] != len(expected_feature_names):
            raise ValueError(
                f"Feature array column count {features.shape[1]} does not match "
                f"expected count {len(expected_feature_names)}"
            )
        X = np.asarray(features, dtype=np.float32)
        feature_names = list(expected_feature_names)
    else:
        raise TypeError(f"Features must be a DataFrame or 2D array, got {type(features)}")

    # Verify no NaN or Inf in feature array
    if np.isnan(X).any():
        nan_cols = [feature_names[i] for i in range(X.shape[1]) if np.isnan(X[:, i]).any()]
        raise ValueError(f"Feature matrix contains NaN values in columns: {nan_cols[:5]}")
    if np.isinf(X).any():
        inf_cols = [feature_names[i] for i in range(X.shape[1]) if np.isinf(X[:, i]).any()]
        raise ValueError(f"Feature matrix contains infinite values in columns: {inf_cols[:5]}")

    return X, feature_names


def validate_probabilities(probs: np.ndarray) -> np.ndarray:
    """Validate that predicted probabilities are well-formed and in [0, 1].

    Raises
    ------
    ValueError
        If probabilities contain NaN, Inf, or values outside [0, 1].
    """
    probs_arr = np.asarray(probs, dtype=np.float64)
    if probs_arr.ndim != 1:
        probs_arr = probs_arr.ravel()

    if np.isnan(probs_arr).any():
        raise ValueError("Model prediction failed: Probabilities contain NaN values")
    if np.isinf(probs_arr).any():
        raise ValueError("Model prediction failed: Probabilities contain Infinite values")
    if (probs_arr < 0.0).any() or (probs_arr > 1.0).any():
        min_v = float(np.min(probs_arr))
        max_v = float(np.max(probs_arr))
        raise ValueError(
            f"Model prediction failed: Probabilities outside valid range [0, 1]: min={min_v}, max={max_v}"
        )
    return probs_arr


class BasePairwiseMatcher(Matcher):
    """Abstract base class for Phase 6 pairwise gradient boosted matchers."""

    def __init__(
        self,
        *,
        scale_pos_weight: float = 19.0,
        random_seed: int = 2026,
        feature_names: Sequence[str] | None = None,
        model_type: str = "base",
    ) -> None:
        self.scale_pos_weight = float(scale_pos_weight)
        self.random_seed = int(random_seed)
        self.feature_names = tuple(feature_names) if feature_names is not None else ()
        self.model_type = model_type
        self.model: Any = None
        self.is_fitted: bool = False
        self._training_metadata: dict[str, Any] = {}

    @abstractmethod
    def _create_underlying_model(self) -> Any:
        """Instantiate the estimator with deterministic configuration."""

    def fit(self, features: Any, labels: Any) -> None:
        """Fit the matcher using leakage-safe pairwise features and binary labels.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray
            84 Phase 5 features (no identifiers or labels allowed).
        labels : pd.Series or np.ndarray
            Binary ground-truth labels (0 or 1).
        """
        expected_names = self.feature_names if self.feature_names else None
        X, resolved_feature_names = validate_and_extract_features(features, expected_names)
        self.feature_names = tuple(resolved_feature_names)

        y = np.asarray(labels, dtype=np.int32).ravel()
        if len(y) != len(X):
            raise ValueError(f"Features length ({len(X)}) does not match labels length ({len(y)})")

        unique_labels = set(np.unique(y))
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Labels must be binary {0, 1}, got {unique_labels}")

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos == 0:
            raise ValueError("Training labels contain zero positive examples")

        LOGGER.info(
            "Fitting %s model on %d pairs (%d pos, %d neg, scale_pos_weight=%.2f)...",
            self.model_type,
            len(y),
            n_pos,
            n_neg,
            self.scale_pos_weight,
        )

        self.model = self._create_underlying_model()
        self.model.fit(X, y)
        self.is_fitted = True

        self._training_metadata = {
            "n_samples": len(y),
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "positive_rate": float(n_pos / len(y)),
            "scale_pos_weight": self.scale_pos_weight,
            "random_seed": self.random_seed,
            "feature_count": len(self.feature_names),
        }
        LOGGER.info("%s model fitting completed successfully", self.model_type)

    def score(self, features: Any) -> np.ndarray:
        """Predict match probabilities P(candidate is true match | pair features).

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray
            Pairwise features matching feature_names.

        Returns
        -------
        np.ndarray
            1D array of validated probabilities in [0, 1].
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"Cannot score: {self.model_type} matcher has not been fitted yet")

        X, _ = validate_and_extract_features(features, self.feature_names)
        raw_probs = self.model.predict_proba(X)[:, 1]
        return validate_probabilities(raw_probs)

    @abstractmethod
    def get_feature_importances(self) -> pd.DataFrame:
        """Return MODEL FEATURE IMPORTANCE table sorted by importance descending."""

    def save(self, model_file_path: str | Path, metadata_file_path: str | Path | None = None) -> None:
        """Serialize model to reloadable format and write companion metadata."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"Cannot save unfitted {self.model_type} model")

        model_path = Path(model_file_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_version": "1.0.0",
            "model_type": self.model_type,
            "scale_pos_weight": self.scale_pos_weight,
            "random_seed": self.random_seed,
            "feature_names": list(self.feature_names),
            "training_metadata": self._training_metadata,
            "model": self.model,
        }
        joblib.dump(payload, model_path, compress=3)
        LOGGER.info("Saved %s model artifact to %s", self.model_type, model_path)

        if metadata_file_path is not None:
            meta_path = Path(metadata_file_path)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_dict = {
                "model_version": "1.0.0",
                "model_type": self.model_type,
                "scale_pos_weight": self.scale_pos_weight,
                "random_seed": self.random_seed,
                "feature_column_order": list(self.feature_names),
                "feature_count": len(self.feature_names),
                "model_configuration": getattr(self, "hyperparameters", {}),
                "training_metadata": self._training_metadata,
                "preprocessing_metadata": {
                    "missing_value_policy": "native_tree_split",
                    "scaling": "none_tree_invariant",
                    "leakage_protection": "forbidden_identifiers_stripped",
                },
                "class_imbalance_strategy": {
                    "method": "scale_pos_weight",
                    "value": self.scale_pos_weight,
                },
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2)
            LOGGER.info("Saved model metadata JSON to %s", meta_path)


class LightGBMMatcher(BasePairwiseMatcher):
    """Pairwise entity matcher powered by LightGBM Classifier."""

    def __init__(
        self,
        *,
        scale_pos_weight: float = 19.0,
        random_seed: int = 2026,
        feature_names: Sequence[str] | None = None,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        **extra_kwargs: Any,
    ) -> None:
        super().__init__(
            scale_pos_weight=scale_pos_weight,
            random_seed=random_seed,
            feature_names=feature_names,
            model_type="lightgbm",
        )
        self.hyperparameters: dict[str, Any] = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_seed,
            "deterministic": True,
            "n_jobs": -1,
            "verbosity": -1,
            **extra_kwargs,
        }

    def _create_underlying_model(self) -> Any:
        import lightgbm as lgb
        return lgb.LGBMClassifier(**self.hyperparameters)

    def get_feature_importances(self) -> pd.DataFrame:
        """Return MODEL FEATURE IMPORTANCE table (split and gain metrics)."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot compute feature importance on unfitted model")

        split_importance = self.model.booster_.feature_importance(importance_type="split")
        gain_importance = self.model.booster_.feature_importance(importance_type="gain")
        total_gain = float(np.sum(gain_importance)) if np.sum(gain_importance) > 0 else 1.0

        df = pd.DataFrame({
            "feature_name": list(self.feature_names),
            "split_importance": split_importance,
            "gain_importance": gain_importance,
            "normalized_gain_importance": [g / total_gain for g in gain_importance],
            "importance_label": "MODEL FEATURE IMPORTANCE",
        })
        return df.sort_values(by="gain_importance", ascending=False).reset_index(drop=True)

    @classmethod
    def load(cls, model_file_path: str | Path) -> LightGBMMatcher:
        """Reload serialized LightGBM matcher."""
        payload = joblib.load(model_file_path)
        if payload.get("model_type") != "lightgbm":
            raise ValueError(f"Expected lightgbm model type, got {payload.get('model_type')}")

        matcher = cls(
            scale_pos_weight=payload["scale_pos_weight"],
            random_seed=payload["random_seed"],
            feature_names=payload["feature_names"],
        )
        matcher.model = payload["model"]
        matcher.is_fitted = True
        matcher._training_metadata = payload.get("training_metadata", {})
        LOGGER.info("Reloaded LightGBMMatcher from %s (%d features)", model_file_path, len(matcher.feature_names))
        return matcher


class CatBoostMatcher(BasePairwiseMatcher):
    """Pairwise entity matcher powered by CatBoost Classifier."""

    def __init__(
        self,
        *,
        scale_pos_weight: float = 19.0,
        random_seed: int = 2026,
        feature_names: Sequence[str] | None = None,
        iterations: int = 350,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        **extra_kwargs: Any,
    ) -> None:
        super().__init__(
            scale_pos_weight=scale_pos_weight,
            random_seed=random_seed,
            feature_names=feature_names,
            model_type="catboost",
        )
        self.hyperparameters: dict[str, Any] = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "scale_pos_weight": self.scale_pos_weight,
            "random_seed": self.random_seed,
            "thread_count": -1,
            "verbose": False,
            **extra_kwargs,
        }

    def _create_underlying_model(self) -> Any:
        import catboost as cb
        return cb.CatBoostClassifier(**self.hyperparameters)

    def get_feature_importances(self) -> pd.DataFrame:
        """Return MODEL FEATURE IMPORTANCE table."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot compute feature importance on unfitted model")

        importances = self.model.get_feature_importance()
        total_imp = float(np.sum(importances)) if np.sum(importances) > 0 else 1.0

        df = pd.DataFrame({
            "feature_name": list(self.feature_names),
            "importance": importances,
            "normalized_importance": [imp / total_imp for imp in importances],
            "importance_label": "MODEL FEATURE IMPORTANCE",
        })
        return df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    @classmethod
    def load(cls, model_file_path: str | Path) -> CatBoostMatcher:
        """Reload serialized CatBoost matcher."""
        payload = joblib.load(model_file_path)
        if payload.get("model_type") != "catboost":
            raise ValueError(f"Expected catboost model type, got {payload.get('model_type')}")

        matcher = cls(
            scale_pos_weight=payload["scale_pos_weight"],
            random_seed=payload["random_seed"],
            feature_names=payload["feature_names"],
        )
        matcher.model = payload["model"]
        matcher.is_fitted = True
        matcher._training_metadata = payload.get("training_metadata", {})
        LOGGER.info("Reloaded CatBoostMatcher from %s (%d features)", model_file_path, len(matcher.feature_names))
        return matcher
