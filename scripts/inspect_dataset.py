"""Phase 1 - dataset inspection and profiling for the training split.

The script discovers the real challenge files under the configured dataset root,
profiles every training file, profiles the training ground truth, and writes
reproducible artifacts to the configured output directory.

It never reads the test split. Test files are discovered and their structure is
recorded so availability is proven, but no test row is loaded, counted, or used
to influence any rule, statistic, or diagnostic.

Usage::

    python scripts/inspect_dataset.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from business_entity_resolution.config.settings import Settings, load_settings
from business_entity_resolution.exceptions import ConfigurationError
from business_entity_resolution.ingestion import reporting
from business_entity_resolution.ingestion.dataset import (
    LoadedFile,
    iter_chunks,
    read_header,
)
from business_entity_resolution.ingestion.discovery import (
    GROUND_TRUTH_ROLE,
    SOURCE_ROLE,
    DiscoveredDataset,
    discover_dataset,
)
from business_entity_resolution.ingestion.ground_truth import (
    analyse_ground_truth,
    coverage_against_sources,
    prefix_of,
    validate_ground_truth_ids,
)
from business_entity_resolution.ingestion.profiling import (
    HASH_METHOD,
    build_pair_sample,
    collect_country_distribution,
    exact_pattern_counts,
    identifier_structure,
    profile_file,
    profile_sample,
)
from business_entity_resolution.ingestion.variation import compare_pairs
from business_entity_resolution.normalization import NormalizationPipeline
from business_entity_resolution.normalization.text import (
    PunctuationTranslator,
    apply_text_pipeline_series,
)
from business_entity_resolution.utils.logging import configure_logging, get_logger
from business_entity_resolution.utils.seed import set_random_seed

LOGGER = get_logger(__name__)

FOCUSED_PATTERNS: dict[str, str] = {
    "ampersand": r"&",
    "word_and": r"\band\b",
    "non_ascii": r"[^\x00-\x7f]",
    "comma": r",",
    "has_digit": r"\d",
    "hash": r"#",
    "repeated_whitespace": r"\s{2,}",
    "us_zip_like": r"\b\d{5}(?:-\d{4})?\b",
    "numeric_zip_like": r"\b\d{6}\b",
    "email_like": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "phone_like": r"(?:\+\d[\d\s().-]{6,}\d)",
}
TEXT_COLUMN_HINTS = ("name", "address")


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 1 inspection command parser."""

    parser = argparse.ArgumentParser(
        description="Profile the real training data of the Amazon ML Challenge 2026."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory (default: the configured paths.output_dir).",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Skip the HTML review report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Phase 1 and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
    except ConfigurationError as error:
        build_parser().error(str(error))
        return 2
    if settings.dataset is None:
        build_parser().error("config.yaml must define a 'dataset' section for Phase 1")
        return 2

    logger = configure_logging(settings.logging)
    set_random_seed(settings.random_seed)
    project_root = args.config.resolve().parent
    output_dir = (project_root / (args.output_dir or settings.paths.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = discover_dataset(settings.dataset, project_root)
    headers = {item.relative_path: read_header(item, settings.dataset) for item in dataset.files}
    train_files = _train_files(dataset, settings)
    if not train_files:
        build_parser().error(
            f"No training source files were discovered under {dataset.root}"
        )
        return 2

    profile: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": settings.project.name,
        "phase": settings.project.phase,
        "dataset_root": str(dataset.root),
        "read_split": settings.dataset.read_split,
        "config_source": str(settings.source_path),
        "files": [],
        "ignored_paths": list(dataset.ignored),
        "sample_rows": settings.dataset.sample_rows,
    }

    profile["files"] = _file_inventory(dataset, headers, settings)
    profile["scope"] = _scope(dataset, settings)

    source_profiles: dict[str, Any] = {}
    identifiers: list[dict[str, Any]] = []
    sample_profiles: dict[str, Any] = {}
    focused: dict[str, Any] = {}
    source_keys: dict[str, np.ndarray] = {}

    for index in (1, 2, 3):
        loaded = _source_file(dataset, settings, index)
        if loaded is None:
            logger.warning("Training source %d was not discovered; skipping it.", index)
            continue
        file_profile = profile_file(loaded, settings.dataset)
        source_profiles[loaded.key] = file_profile.as_dict()
        _merge_file_result(profile, file_profile.as_dict())

        id_column = _id_column(loaded)
        structure = identifier_structure(loaded, settings.dataset, id_column)
        prefixes = list(structure["prefix_counts"])
        structure["prefix"] = prefixes[0] if len(prefixes) == 1 else "MIXED"
        source_keys[structure["prefix"]] = np.asarray(structure.pop("_keys"), dtype=np.int64)
        structure["file"] = loaded.key
        identifiers.append(structure)

        country_column = _country_column(loaded)
        if country_column:
            distribution = collect_country_distribution(loaded, settings.dataset, country_column)
            _set_column_payload(
                profile, loaded.key, country_column, "country_distribution", distribution
            )

        text_columns = [
            column
            for column in loaded.columns
            if any(hint in column.casefold() for hint in TEXT_COLUMN_HINTS)
        ]
        focused[loaded.key] = {
            column: exact_pattern_counts(loaded, settings.dataset, column, FOCUSED_PATTERNS)
            for column in text_columns
        }
        for column in text_columns:
            _set_column_payload(
                profile,
                loaded.key,
                column,
                "full_data_pattern_counts",
                focused[loaded.key][column],
            )
        sample_profiles[loaded.key] = {
            column: payload.as_dict()
            for column, payload in profile_sample(loaded, settings.dataset).items()
        }

    profile["identifiers"] = identifiers
    profile["focused_pattern_counts"] = focused

    ground_truth_loaded = _ground_truth_file(dataset, settings)
    if ground_truth_loaded is not None:
        source_column, matched_column = _ground_truth_columns(ground_truth_loaded)
        truth_file_profile = profile_file(ground_truth_loaded, settings.dataset)
        _merge_file_result(profile, truth_file_profile.as_dict())
        truth = analyse_ground_truth(
            ground_truth_loaded, settings.dataset, source_column, matched_column
        )
        truth_keys = np.asarray(truth.pop("_source_keys"), dtype=np.int64)
        validation = validate_ground_truth_ids(truth, source_keys)
        truth.pop("_matched_keys")
        truth.pop("_matched_key_prefixes")
        truth["id_resolution"] = validation
        profile["ground_truth"] = truth
        profile["ground_truth_validation"] = validation
        profile["coverage"] = _coverage(truth_keys, source_keys)
        profile["variations"] = _variations(
            dataset, settings, ground_truth_loaded, source_column, matched_column, headers
        )

    profile["entity_overlap"] = _entity_overlap(dataset, settings)
    profile["sample_profiles"] = sample_profiles
    profile["totals"] = _totals(profile["files"])
    profile["leakage"] = _leakage(profile, source_profiles)
    profile["validation_strategy"] = _validation_strategy(profile)
    profile["methodology"] = _methodology(settings)
    profile["normalization_inputs"] = _normalization_inputs(dataset, settings)

    json_path = output_dir / "dataset_profile.json"
    reporting.write_json(json_path, profile)
    markdown = reporting.render_markdown(profile)
    (output_dir / "dataset_profile.md").write_text(markdown, encoding="utf-8")
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", output_dir / "dataset_profile.md")
    if not args.skip_html:
        html_path = output_dir / "initial_data_report.html"
        html_path.write_text(reporting.render_html(profile, markdown), encoding="utf-8")
        logger.info("Wrote %s", html_path)
    return 0


def _train_files(dataset: DiscoveredDataset, settings: Settings) -> list[LoadedFile]:
    return [
        read_header(item, settings.dataset)
        for item in dataset.files
        if item.split == settings.dataset.read_split and item.role == SOURCE_ROLE
    ]


def _source_file(
    dataset: DiscoveredDataset, settings: Settings, index: int
) -> LoadedFile | None:
    for item in dataset.files:
        if (
            item.split == settings.dataset.read_split
            and item.role == SOURCE_ROLE
            and item.source_index == index
        ):
            return read_header(item, settings.dataset)
    return None


def _ground_truth_file(dataset: DiscoveredDataset, settings: Settings) -> LoadedFile | None:
    for item in dataset.files:
        if item.split == settings.dataset.read_split and item.role == GROUND_TRUTH_ROLE:
            return read_header(item, settings.dataset)
    return None


def _id_column(loaded: LoadedFile) -> str:
    for column in loaded.columns:
        if column == "id" or column.endswith("_id"):
            return column
    return loaded.columns[0]


def _country_column(loaded: LoadedFile) -> str | None:
    for column in loaded.columns:
        if "country" in column.casefold():
            return column
    return None


def _ground_truth_columns(loaded: LoadedFile) -> tuple[str, str]:
    source_column = next(
        (column for column in loaded.columns if column != "matched_entity_ids" and "id" in column.casefold()),
        loaded.columns[0],
    )
    matched_column = next(
        (column for column in loaded.columns if column != source_column), loaded.columns[-1]
    )
    return source_column, matched_column


def _file_inventory(
    dataset: DiscoveredDataset, headers: dict[str, LoadedFile], settings: Settings
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for item in dataset.files:
        payload = item.as_dict()
        header = headers[item.relative_path]
        payload["delimiter"] = repr(settings.dataset.delimiter)
        payload["encoding"] = settings.dataset.encoding
        payload["columns"] = list(header.columns)
        payload["column_count"] = len(header.columns)
        payload["read_in_scope"] = item.split == settings.dataset.read_split
        inventory.append(payload)
    return inventory


def _scope(dataset: DiscoveredDataset, settings: Settings) -> dict[str, Any]:
    read = [item.relative_path for item in dataset.files if item.split == settings.dataset.read_split]
    not_read = [item.relative_path for item in dataset.files if item.split != settings.dataset.read_split]
    return {
        "read_split": settings.dataset.read_split,
        "files_read": len(read),
        "files_not_read": len(not_read),
        "read_files": read,
        "not_read_files": not_read,
        "test_split_policy": (
            "discovered for availability and structure only; no test row is read, "
            "counted, or used to fit any rule or statistic"
        ),
    }


def _totals(files: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "files_discovered": len(files),
        "files_profiled": sum(1 for entry in files if entry.get("row_count") is not None),
        "rows_profiled": sum(
            entry.get("row_count", 0) or 0
            for entry in files
            if entry.get("read_in_scope")
        ),
    }


def _merge_file_result(profile: dict[str, Any], payload: dict[str, Any]) -> None:
    for entry in profile["files"]:
        if entry["relative_path"] == payload["relative_path"]:
            entry.update(
                {
                    "row_count": payload["row_count"],
                    "duplicate_row_count": payload["duplicate_row_count"],
                    "duplicate_row_method": payload["duplicate_row_method"],
                    "columns": payload["columns"],
                }
            )
            return
    profile["files"].append(payload)


def _set_column_payload(
    profile: dict[str, Any], file_key: str, column: str, name: str, value: Any
) -> None:
    for entry in profile["files"]:
        if entry["relative_path"] != file_key:
            continue
        for column_payload in entry.get("columns", []):
            if isinstance(column_payload, dict) and column_payload.get("name") == column:
                column_payload[name] = value
                return


def _coverage(truth_keys: np.ndarray, source_keys: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compare the ground-truth left column only with the source it belongs to.

    The ground truth's left column holds S1 identifiers, so comparing it with the
    S2 or S3 key sets would be meaningless.
    """

    reference = next(
        (prefix for prefix in ("S1",) if prefix in source_keys),
        next(iter(source_keys), None),
    )
    if reference is None:
        return {}
    return {reference: coverage_against_sources(truth_keys, source_keys[reference], reference, reference)}


def _variations(
    dataset: DiscoveredDataset,
    settings: Settings,
    ground_truth: LoadedFile,
    source_column: str,
    matched_column: str,
    headers: dict[str, LoadedFile],
) -> dict[str, Any]:
    loaded_sources = {
        f"S{item.source_index}": headers[item.relative_path]
        for item in dataset.files
        if item.split == settings.dataset.read_split
        and item.role == SOURCE_ROLE
        and item.source_index is not None
    }
    sample = build_pair_sample(
        loaded_sources, settings.dataset, ground_truth, source_column, matched_column
    )
    if not sample["records"]:
        return {}
    text_columns = sorted(
        {
            column
            for records in sample["records"].values()
            for record in list(records.values())[:1]
            for column in record
            if column != source_column
            and not (column == "id" or column.endswith("_id") or column.endswith("_ids"))
        }
    )
    pipeline = NormalizationPipeline(settings.normalization)
    report = compare_pairs(
        sample["pairs"],
        sample["records"],
        text_columns,
        limit=settings.dataset.max_examples,
        normalizer=lambda column, value: pipeline.normalize_value(column, value),
    )
    return report.as_dict(settings.dataset.max_examples)


def _entity_overlap(
    dataset: DiscoveredDataset,
    settings: Settings,
) -> dict[str, Any]:
    """Measure exact overlap of conservative-normalized name+address+country keys.

    Keys come from the vectorized Phase 2 text pipeline and are compared as
    64-bit content hashes, so the whole training set is measured exactly while
    the resident set stays proportional to the file rather than the row count.
    A bounded set of real example rows is retained in the same pass.
    """

    hashes: dict[str, np.ndarray] = {}
    examples_by_label: dict[str, dict[int, str]] = {}
    for index in (1, 2, 3):
        loaded = _source_file(dataset, settings, index)
        if loaded is None:
            continue
        label = f"S{index}"
        hashes[label], examples_by_label[label] = _entity_key_hashes(loaded, settings)
        LOGGER.info(
            "Entity key set for %s: %d rows, %d distinct keys",
            label,
            hashes[label].size,
            np.unique(hashes[label]).size,
        )

    labels = sorted(hashes)
    pairs: dict[str, Any] = {}
    for position, left in enumerate(labels):
        for right in labels[position + 1 :]:
            shared = np.intersect1d(hashes[left], hashes[right])
            left_unique = int(np.unique(hashes[left]).size)
            right_unique = int(np.unique(hashes[right]).size)
            smaller = min(left_unique, right_unique) or 1
            wanted = [int(value) for value in shared[: settings.dataset.max_examples]]
            examples = [
                examples_by_label[label][key]
                for label in (left, right)
                for key in wanted
                if key in examples_by_label[label]
            ]
            pairs[f"{left} vs {right}"] = {
                "shared": int(shared.size),
                "left_keys": left_unique,
                "right_keys": right_unique,
                "share_of_smaller_percent": round(100.0 * shared.size / smaller, 4),
                "examples": examples[: settings.dataset.max_examples],
            }
            del shared
    return {
        "key_definition": (
            "conservative Phase 2 text normalization of "
            "business_name | business_address | country, compared as 64-bit content hashes"
        ),
        "distinct_keys": {
            label: int(np.unique(values).size) for label, values in hashes.items()
        },
        "pairs": pairs,
    }


EXAMPLE_ROW_LIMIT = 4000


def _entity_key_hashes(
    loaded: LoadedFile, settings: Settings
) -> tuple[np.ndarray, dict[int, str]]:
    """Return one 64-bit entity key per row plus a bounded set of real rows."""

    columns = [
        column
        for column in loaded.columns
        if not (column == "id" or column.endswith("_id"))
    ]
    translator = PunctuationTranslator(settings.normalization)
    keys: list[pd.Series] = []
    examples: dict[int, str] = {}
    for chunk in iter_chunks(loaded, settings.dataset, columns=columns):
        cleaned = apply_text_pipeline_series(
            chunk, settings.normalization, translator, stages=("cleaned",)
        )
        key = cleaned[f"{columns[0]}__cleaned"]
        for column in columns[1:]:
            key = key + "|" + cleaned[f"{column}__cleaned"]
        hashes = pd.util.hash_array(key.to_numpy(dtype=object), encoding="utf-8").astype(
            np.uint64
        )
        keys.append(pd.Series(hashes, index=chunk.index))
        if len(examples) < EXAMPLE_ROW_LIMIT:
            _collect_examples(chunk, columns, hashes, examples)
    stacked = (
        pd.concat(keys).to_numpy(dtype=np.uint64)
        if keys
        else np.empty(0, dtype=np.uint64)
    )
    return stacked, examples


def _collect_examples(
    chunk: pd.DataFrame,
    columns: Sequence[str],
    hashes: np.ndarray,
    examples: dict[int, str],
) -> None:
    """Keep the first real row seen for each entity key, bounded by a fixed limit."""

    frame = chunk[list(columns)].fillna("").astype(str)
    joined = frame[columns[0]]
    for column in columns[1:]:
        joined = joined + " | " + frame[column]
    for key, text in zip(hashes.tolist(), joined.tolist()):
        if int(key) not in examples:
            examples[int(key)] = str(text)
            if len(examples) >= EXAMPLE_ROW_LIMIT:
                return


def _leakage(
    profile: dict[str, Any], source_profiles: dict[str, Any]
) -> dict[str, Any]:
    """Assemble train-only leakage and duplication risks from measured values."""

    risks: list[dict[str, str]] = []
    for key, payload in source_profiles.items():
        risks.append(
            {
                "risk": f"{key}: duplicate rows inside one source file",
                "value": payload["duplicate_row_count"],
                "assessment": "informational; exact duplicates cannot be split across folds safely",
            }
        )
    truth = profile.get("ground_truth", {})
    if truth:
        risks.append(
            {
                "risk": "ground truth exposes every S1 label, so a random row split leaks entity identity",
                "value": truth["total_rows"],
                "assessment": "validation must split by S1 entity, never by candidate row",
            }
        )
        reuse = truth.get("matched_record_reuse", {})
        risks.append(
            {
                "risk": "S2/S3 records claimed by more than one S1 entity",
                "value": reuse.get("records_matched_by_multiple_s1_entities", 0),
                "assessment": (
                    "a row-level split would duplicate the same record across folds; "
                    f"worst case is {reuse.get('max_s1_entities_per_matched_record', 0)} S1 entities per record"
                ),
            }
        )
        risks.append(
            {
                "risk": "ground-truth rows whose match list is malformed",
                "value": truth["malformed_entry_count"],
                "assessment": "malformed lists must be parsed defensively, never trusted blindly",
            }
        )
    coverage = profile.get("coverage", {})
    for prefix, entry in sorted(coverage.items()):
        risks.append(
            {
                "risk": f"{prefix}: source ids absent from the ground truth left column",
                "value": entry["source_ids_absent_from_ground_truth"],
                "assessment": "these records can never be labelled, so they must not drive validation scoring",
            }
        )
    risks.append(
        {
            "risk": "exact duplicate entities across sources are indistinguishable from true matches",
            "value": _shared_overlap_total(profile),
            "assessment": "an exact-key match is a valid positive but must never be the only positive class",
        }
    )
    risks.append(
        {
            "risk": "test-split distribution and unseen countries are unknown by design",
            "value": "not measured",
            "assessment": "test data was deliberately not read; open-set country handling stays in place",
        }
    )
    return {
        "risks": risks,
        "notes": [
            "- The train/test boundary was not crossed: no test row was read, so no train-fitted "
            "quantity can have been influenced by the test split.",
            "- Because train and test are separate record sets, entity overlap between them is "
            "unknown and must be re-measured at inference time, not assumed.",
        ],
    }


def _shared_overlap_total(profile: dict[str, Any]) -> int:
    overlap = profile.get("entity_overlap", {}).get("pairs", {})
    return sum(entry["shared"] for entry in overlap.values())


def _validation_strategy(profile: Mapping[str, Any]) -> dict[str, Any]:
    truth = profile.get("ground_truth", {})
    total = truth.get("total_rows", 0) or 1
    singletons = truth.get("s1_entities_with_zero_matches", 0)
    multi = truth.get("s1_entities_with_multiple_matches", 0)
    return {
        "recommendations": [
            "Split at the S1 entity level, never at the candidate-row level. Every S1 entity and "
            "all of its matched S2/S3 records must stay in the same fold.",
            f"Stratify the split on match cardinality: {round(100.0 * singletons / total, 2)}% of S1 "
            f"entities are singletons and {round(100.0 * multi / total, 2)}% have multiple matches, "
            "so an unstratified split will swing the macro F0.5 average.",
            "Group near-duplicate S1 entities together using the conservative name+address key so a "
            "franchise or a chain location cannot appear in two folds.",
            "Score with macro F0.5 over S1 entities exactly as the official metric does, counting an "
            "empty prediction on a singleton as a perfect score and any false merge as zero.",
            "Report per-source recall separately (S2 and S3 contribute different volumes), because a "
            "global average hides a source-specific regression.",
            "Re-measure candidate recall per fold before any threshold is selected; a precision gain "
            "on a blocked candidate set is meaningless if recall is already lost in blocking.",
        ]
    }


def _methodology(settings: Settings) -> dict[str, Any]:
    return {
        "distinct_value_method": HASH_METHOD,
        "exact_counters": "counted over every row of the training file",
        "pattern_battery": (
            f"deterministic leading-row sample of {settings.dataset.sample_rows} rows per training "
            "source file, in file order"
        ),
        "test_split_policy": "discovered and structurally described only; no test row was read",
        "ground_truth_policy": "training ground truth only; read-only, never modified",
        "chunksize": settings.dataset.chunksize,
        "raw_data_modified": False,
    }


def _normalization_inputs(dataset: DiscoveredDataset, settings: Settings) -> dict[str, Any]:
    """Record the train files Phase 2 normalization is allowed to use."""

    return {
        "allowed_files": [
            item.relative_path
            for item in dataset.files
            if item.split == settings.dataset.read_split
        ],
        "excluded_from_fitting": [
            item.relative_path
            for item in dataset.files
            if item.split != settings.dataset.read_split
        ],
    }


if __name__ == "__main__":
    sys.exit(main())
