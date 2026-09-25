"""Configuration model and loader for the schema-agnostic Phase 0 project."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

import yaml

from business_entity_resolution.exceptions import ConfigurationError

DEFAULT_CONFIG_PATH = Path("config.yaml")
_VALID_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})


@dataclass(frozen=True, slots=True)
class ProjectSettings:
    """Project metadata that is independent of the dataset schema."""

    name: str
    phase: int


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Relative project paths; no dataset filename is assumed."""

    data_root: Path
    raw_dir: Path
    processed_dir: Path
    train_dir: Path
    validation_dir: Path
    test_dir: Path
    experiments_dir: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    """Console logging settings."""

    level: str
    format: str
    date_format: str


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    """Generic experiment-recording settings."""

    retain_history: bool
    metadata_format: str


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Model placeholders that remain unselected during Phase 0."""

    primary: str | None
    benchmark: str | None
    options: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BlockingSettings:
    """Blocking configuration without field or strategy assumptions."""

    strategies: tuple[str, ...]
    options: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Metric and validation settings with no selected score threshold."""

    metric: str
    beta: float
    threshold_grid: tuple[float, ...]
    validation_unit: str


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated, immutable project configuration."""

    project: ProjectSettings
    paths: DataPaths
    random_seed: int
    logging: LoggingSettings
    experiments: ExperimentSettings
    models: ModelSettings
    blocking: BlockingSettings
    evaluation: EvaluationSettings
    source_path: Path


def load_settings(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load and validate a Phase 0 configuration file.

    Only generic project settings are interpreted. Dataset names, schemas,
    columns, and source-specific mappings are intentionally not accepted here.
    """

    path = Path(config_path)
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigurationError(f"Unable to read configuration file: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in configuration file: {path}") from error

    root = _mapping(raw, "root")
    project = _mapping_value(root, "project")
    paths = _mapping_value(root, "paths")
    logging = _mapping_value(root, "logging")
    experiments = _mapping_value(root, "experiments")
    models = _mapping_value(root, "models")
    blocking = _mapping_value(root, "blocking")
    evaluation = _mapping_value(root, "evaluation")

    project_settings = ProjectSettings(
        name=_text(project, "name"),
        phase=_non_negative_int(project, "phase"),
    )
    path_settings = DataPaths(
        data_root=_relative_path(paths, "data_root"),
        raw_dir=_relative_path(paths, "raw_dir"),
        processed_dir=_relative_path(paths, "processed_dir"),
        train_dir=_relative_path(paths, "train_dir"),
        validation_dir=_relative_path(paths, "validation_dir"),
        test_dir=_relative_path(paths, "test_dir"),
        experiments_dir=_relative_path(paths, "experiments_dir"),
        output_dir=_relative_path(paths, "output_dir"),
    )
    logging_settings = LoggingSettings(
        level=_log_level(logging, "level"),
        format=_text(logging, "format"),
        date_format=_text(logging, "date_format"),
    )
    experiment_settings = ExperimentSettings(
        retain_history=_bool(experiments, "retain_history"),
        metadata_format=_text(experiments, "metadata_format"),
    )
    model_settings = ModelSettings(
        primary=_optional_text(models, "primary"),
        benchmark=_optional_text(models, "benchmark"),
        options=_frozen_mapping(_mapping_value(models, "options")),
    )
    blocking_settings = BlockingSettings(
        strategies=_text_tuple(blocking, "strategies"),
        options=_frozen_mapping(_mapping_value(blocking, "options")),
    )
    evaluation_settings = EvaluationSettings(
        metric=_text(evaluation, "metric"),
        beta=_positive_number(evaluation, "beta"),
        threshold_grid=_number_tuple(evaluation, "threshold_grid"),
        validation_unit=_text(evaluation, "validation_unit"),
    )

    return Settings(
        project=project_settings,
        paths=path_settings,
        random_seed=_non_negative_int(root, "random_seed"),
        logging=logging_settings,
        experiments=experiment_settings,
        models=model_settings,
        blocking=blocking_settings,
        evaluation=evaluation_settings,
        source_path=path,
    )


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Expected a mapping at {location}")
    return value


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in mapping:
        raise ConfigurationError(f"Missing required configuration section or key: {key}")
    return _mapping(mapping[key], key)


def _frozen_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively freeze generic option values for deterministic settings."""

    return MappingProxyType({key: _freeze(value) for key, value in mapping.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _frozen_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Configuration value '{key}' must be a non-empty string")
    return value.strip()


def _optional_text(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Configuration value '{key}' must be a string or null")
    return value.strip()


def _bool(mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ConfigurationError(f"Configuration value '{key}' must be a boolean")
    return value


def _non_negative_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(f"Configuration value '{key}' must be a non-negative integer")
    return value


def _positive_number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"Configuration value '{key}' must be a positive number")
    return float(value)


def _text_tuple(mapping: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError(f"Configuration value '{key}' must be a list of strings")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ConfigurationError(f"Configuration value '{key}' must contain non-empty strings")
    return tuple(item.strip() for item in value)


def _number_tuple(mapping: Mapping[str, Any], key: str) -> tuple[float, ...]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError(f"Configuration value '{key}' must be a list of numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ConfigurationError(f"Configuration value '{key}' must contain numbers")
    return tuple(float(item) for item in value)


def _log_level(mapping: Mapping[str, Any], key: str) -> str:
    value = _text(mapping, key).upper()
    if value not in _VALID_LOG_LEVELS:
        allowed = ", ".join(sorted(_VALID_LOG_LEVELS))
        raise ConfigurationError(f"Configuration value '{key}' must be one of: {allowed}")
    return value


def _relative_path(mapping: Mapping[str, Any], key: str) -> Path:
    raw_path = _text(mapping, key)
    path = Path(raw_path)
    windows_path = PureWindowsPath(raw_path)
    posix_path = PurePosixPath(raw_path)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or posix_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in path.parts
        or ".." in posix_path.parts
    ):
        raise ConfigurationError(
            f"Configuration path '{key}' must be project-relative, not absolute or parent-relative"
        )
    return path
