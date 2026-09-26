"""Configuration model and loader for the project.

Phase 1 and Phase 2 extend the Phase 0 foundation with two optional sections:

* ``dataset`` locates the supplied challenge files by pattern instead of by a
  hard-coded filename, and records which split may be read.
* ``normalization`` holds configurable, train-derived comparison rules.

Both sections are optional so a bare Phase 0 configuration still loads.
"""

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
_VALID_UNICODE_FORMS = frozenset({"NFC", "NFD", "NFKC", "NFKD", "none"})
_VALID_SPLITS = frozenset({"train", "test"})


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
class DatasetSettings:
    """Location and read-scope rules for the supplied challenge files.

    Filenames are never encoded here. Files are discovered under ``root`` and
    classified from their own path and stem, so the configuration stays valid if
    the upstream archive is renamed or re-extracted.
    """

    root: Path
    read_split: str
    recursive: bool
    extensions: tuple[str, ...]
    ignore_globs: tuple[str, ...]
    split_keywords: Mapping[str, str]
    source_keywords: tuple[str, ...]
    ground_truth_keywords: tuple[str, ...]
    delimiter: str
    encoding: str
    header_rows: int
    chunksize: int
    sample_rows: int
    pair_sample_rows: int
    top_values: int
    max_examples: int


@dataclass(frozen=True, slots=True)
class NormalizationSettings:
    """Configurable, conservative comparison rules derived from training data.

    Every mapping here is a rule table, not a decision: no setting can declare
    two records to be the same entity.
    """

    unicode_form: str
    strip_control_characters: bool
    casefold: bool
    collapse_whitespace: bool
    punctuation_to_space: bool
    strip_symbols: bool
    keep_characters: str
    ampersand_to_and: bool
    general_legal_suffixes: Mapping[str, str]
    name_abbreviations: Mapping[str, str]
    street_designators: Mapping[str, str]
    unit_markers: Mapping[str, str]
    direction_tokens: Mapping[str, str]
    address_token_map: Mapping[str, str]
    missing_markers: tuple[str, ...]
    country_open_set: bool
    country_case: str
    char_ngram_sizes: tuple[int, ...]
    token_sort: bool
    min_token_length: int
    over_normalized_length_ratio: float


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
    dataset: DatasetSettings | None
    normalization: NormalizationSettings | None
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
    dataset_settings = _dataset_settings(root.get("dataset"))
    normalization_settings = _normalization_settings(root.get("normalization"))

    return Settings(
        project=project_settings,
        paths=path_settings,
        random_seed=_non_negative_int(root, "random_seed"),
        logging=logging_settings,
        experiments=experiment_settings,
        models=model_settings,
        blocking=blocking_settings,
        evaluation=evaluation_settings,
        dataset=dataset_settings,
        normalization=normalization_settings,
        source_path=path,
    )


def _dataset_settings(value: Any) -> DatasetSettings | None:
    """Parse the optional dataset-discovery section."""

    if value is None:
        return None
    mapping = _mapping(value, "dataset")
    split_keywords = _mapping_value(mapping, "split_keywords")
    return DatasetSettings(
        root=_relative_path(mapping, "root"),
        read_split=_choice(mapping, "read_split", _VALID_SPLITS),
        recursive=_bool(mapping, "recursive"),
        extensions=_lowercase_extensions(_text_tuple(mapping, "extensions")),
        ignore_globs=_text_tuple(mapping, "ignore_globs"),
        split_keywords=_frozen_mapping(
            {str(key).lower(): str(item).lower() for key, item in split_keywords.items()}
        ),
        source_keywords=tuple(
            keyword.lower() for keyword in _text_tuple(mapping, "source_keywords")
        ),
        ground_truth_keywords=tuple(
            keyword.lower() for keyword in _text_tuple(mapping, "ground_truth_keywords")
        ),
        delimiter=_single_character(mapping, "delimiter", description="delimiter"),
        encoding=_text(mapping, "encoding"),
        header_rows=_positive_int(mapping, "header_rows"),
        chunksize=_positive_int(mapping, "chunksize"),
        sample_rows=_positive_int(mapping, "sample_rows"),
        pair_sample_rows=_positive_int(mapping, "pair_sample_rows"),
        top_values=_positive_int(mapping, "top_values"),
        max_examples=_positive_int(mapping, "max_examples"),
    )


def _normalization_settings(value: Any) -> NormalizationSettings | None:
    """Parse the optional normalization rule section."""

    if value is None:
        return None
    mapping = _mapping(value, "normalization")
    text = _mapping_value(mapping, "text")
    name = _mapping_value(mapping, "name")
    address = _mapping_value(mapping, "address")
    country = _mapping_value(mapping, "country")
    tokens = _mapping_value(mapping, "tokens")
    diagnostics = _mapping_value(mapping, "diagnostics")
    return NormalizationSettings(
        unicode_form=_choice(text, "unicode_form", _VALID_UNICODE_FORMS, lower=False),
        strip_control_characters=_bool(text, "strip_control_characters"),
        casefold=_bool(text, "casefold"),
        collapse_whitespace=_bool(text, "collapse_whitespace"),
        punctuation_to_space=_bool(text, "punctuation_to_space"),
        strip_symbols=_bool(text, "strip_symbols"),
        keep_characters=_text(text, "keep_characters"),
        ampersand_to_and=_bool(text, "ampersand_to_and"),
        general_legal_suffixes=_rule_table(name, "general_legal_suffixes"),
        name_abbreviations=_rule_table(name, "abbreviations"),
        street_designators=_rule_table(address, "street_designators"),
        unit_markers=_rule_table(address, "unit_markers"),
        direction_tokens=_rule_table(address, "direction_tokens"),
        address_token_map=_rule_table(address, "address_tokens"),
        missing_markers=(
            tuple(item.strip().casefold() for item in _text_tuple(address, "missing_markers"))
            if "missing_markers" in address
            else ()
        ),
        country_open_set=_bool(country, "open_set"),
        country_case=_choice(country, "case", frozenset({"lower", "upper", "fold", "none"})),
        char_ngram_sizes=_ngram_sizes(_number_tuple(tokens, "char_ngram_sizes")),
        token_sort=_bool(tokens, "token_sort"),
        min_token_length=_non_negative_int(tokens, "min_token_length"),
        over_normalized_length_ratio=_ratio(diagnostics, "over_normalized_length_ratio"),
    )


def _rule_table(mapping: Mapping[str, Any], key: str) -> Mapping[str, str]:
    """Parse a conservative token-to-token rule table."""

    if key not in mapping:
        raise ConfigurationError(f"Missing required configuration section or key: {key}")
    table = _mapping(mapping[key], f"{key}")
    for token, replacement in table.items():
        if not isinstance(token, str) or not token.strip():
            raise ConfigurationError(f"Configuration key '{token!r}' must be a non-empty string")
        if not isinstance(replacement, str):
            raise ConfigurationError(
                f"Configuration value for '{token}' must be a string (use an empty "
                "string to drop the token)"
            )
    return _frozen_mapping({token.strip().lower(): replacement for token, replacement in table.items()})


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


def _choice(
    mapping: Mapping[str, Any],
    key: str,
    allowed: frozenset[str],
    *,
    lower: bool = True,
) -> str:
    raw = _text(mapping, key)
    value = raw.lower() if lower else raw.upper()
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ConfigurationError(f"Configuration value '{key}' must be one of: {options}")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"Configuration value '{key}' must be a positive integer")
    return value


def _lowercase_extensions(values: Sequence[str]) -> tuple[str, ...]:
    extensions: list[str] = []
    for value in values:
        suffix = value.strip().lower()
        if not suffix.startswith("."):
            raise ConfigurationError(
                f"Configuration extension {value!r} must start with a dot, e.g. '.tsv'"
            )
        extensions.append(suffix)
    if not extensions:
        raise ConfigurationError("At least one dataset extension must be configured")
    return tuple(extensions)


def _single_character(mapping: Mapping[str, Any], key: str, description: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"Configuration value '{key}' must be a non-empty string")
    if len(value) != 1:
        raise ConfigurationError(f"Configuration value '{key}' must be a single character")
    if value in {'"', "'"}:
        raise ConfigurationError(
            f"Configuration {description} {value!r} would break TSV parsing; "
            "use an escape-free literal such as '\\t'"
        )
    return value


def _ngram_sizes(values: Sequence[float]) -> tuple[int, ...]:
    sizes: list[int] = []
    for value in values:
        size = int(value)
        if size != value or size < 1 or size > 8:
            raise ConfigurationError(
                "Configuration value 'char_ngram_sizes' must contain integers between 1 and 8"
            )
        sizes.append(size)
    if not sizes:
        raise ConfigurationError("Configuration value 'char_ngram_sizes' must not be empty")
    return tuple(sorted(set(sizes)))


def _ratio(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise ConfigurationError(f"Configuration value '{key}' must be a ratio in (0, 1]")
    return float(value)


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
