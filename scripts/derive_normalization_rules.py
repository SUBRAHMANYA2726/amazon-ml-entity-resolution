"""Derive normalization rule tables from the observed training data.

Every table this script proposes comes from tokens that actually occur in the
training sources. Nothing is invented, nothing is imported from documentation,
and no rule is fitted on the test split.

The script is read-only: it prints the proposed tables and writes them to
``output/derived_normalization_rules.json`` for review. Copying an approved
table into ``config.yaml`` is a deliberate, human-reviewed step.

Usage::

    python scripts/derive_normalization_rules.py --config config.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from business_entity_resolution.config.settings import Settings, load_settings
from business_entity_resolution.exceptions import ConfigurationError
from business_entity_resolution.ingestion import reporting
from business_entity_resolution.ingestion.dataset import LoadedFile, iter_chunks, read_header
from business_entity_resolution.ingestion.discovery import SOURCE_ROLE, discover_dataset
from business_entity_resolution.utils.logging import configure_logging, get_logger
from business_entity_resolution.utils.seed import set_random_seed

LOGGER = get_logger(__name__)

TOKEN = re.compile(r"[0-9a-z]+")
DIGITS = re.compile(r"\d")

#: A rule is only proposed when both forms are actually seen this often.
#: The threshold keeps one-off coincidences out of the configuration.
MIN_SUPPORT = 100

#: Designator candidates are the tokens a house-number component can end with.
DESIGNATOR_CANDIDATES = (
    "street", "st", "str", "road", "rd", "avenue", "ave", "av", "boulevard", "blvd",
    "drive", "dr", "lane", "ln", "court", "ct", "circle", "cir", "place", "pl",
    "plaza", "plz", "parkway", "pkwy", "highway", "hwy", "alley", "aly",
    "terrace", "ter", "trail", "trl", "way", "square", "sq", "walk", "loop",
    "path", "run", "row", "pike", "point", "ridge", "ridge", "crossing",
    "xing", "turnpike", "tpke", "bypass", "junction", "jct", "causeway", "cwy",
    "expressway", "expy", "freeway", "fwy", "gateway", "gtwy", "square",
    "corners", "xing", "estate", "gardens", "heights", "hills", "park", "village",
)
UNIT_CANDIDATES = (
    "apt", "apartment", "apartments", "suite", "ste", "unit", "flat", "rm", "room",
    "fl", "floor", "flr", "fl.", "bldg", "building", "shop", "office", "opp",
    "opposite", "near", "behind", "beside", "next", "lot", "no", "nr", "number",
    "ph", "phone", "fax", "sec", "section", "block", "blk", "door", "gate",
    "po", "box", "p.o", "pin", "zip", "landmark",
)
DIRECTION_CANDIDATES = (
    "north", "south", "east", "west", "northeast", "northwest", "southeast",
    "southwest", "ne", "nw", "se", "sw", "n", "s", "e", "w",
)
MISSING_MARKERS = (
    "null", "n/a", "na", "none", "nil", "unknown", "undefined", "not", "available",
    "nan", "-", "--", "---", "?", "no", "blank", "empty", "missing",
)
LEGAL_FORM_CANDIDATES = (
    "limited", "ltd", "llc", "llp", "lp", "inc", "incorporated", "corp",
    "corporation", "co", "company", "pvt", "private", "pllc", "pc", "pa",
    "plc", "gmbh", "sarl", "sa", "nv", "bv", "ab", "as", "oy", "aps", "spa",
    "pte", "pty", "kk", "kft", "doo", "sro", "trust", "holdings", "group",
    "enterprises", "partners", "ventures", "associates", "international",
    "services", "service", "solutions", "systems", "technologies", "technology",
)
DESIGNATOR_CANONICAL = {
    "st": "street", "str": "street",
    "rd": "road",
    "ave": "avenue", "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "pl": "place",
    "plz": "plaza",
    "pkwy": "parkway",
    "hwy": "highway",
    "aly": "alley",
    "ter": "terrace",
    "trl": "trail",
    "sq": "square",
    "tpke": "turnpike",
    "cwy": "causeway",
    "expy": "expressway",
    "fwy": "freeway",
    "gtwy": "gateway",
    "xing": "crossing",
    "jct": "junction",
    "byp": "bypass",
}
UNIT_CANONICAL = {
    "fl": "floor",
    "flr": "floor",
    "apt": "apartment",
    "ste": "suite",
    "bldg": "building",
    "blk": "block",
    "rm": "room",
    "no": "number",
    "nr": "number",
    "opp": "opposite",
}
DIRECTION_CANONICAL = {
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}
LEGAL_FORM_CANONICAL = {
    "ltd": "limited",
    "pvt": "private",
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "pllc": "pllc",
    "llc": "llc",
    "llp": "llp",
    "lp": "lp",
    "pc": "pc",
}
MISSING_TOKEN_VALUES = {"null", "na", "n/a", "none", "nan", "nil", "undefined", "-"}


def build_parser() -> argparse.ArgumentParser:
    """Build the rule-derivation command parser."""

    parser = argparse.ArgumentParser(
        description="Derive normalization rule tables from the training sources."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Derive the rule tables and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
    except ConfigurationError as error:
        build_parser().error(str(error))
        return 2
    if settings.dataset is None or settings.normalization is None:
        build_parser().error("config.yaml must define 'dataset' and 'normalization' sections")
        return 2

    configure_logging(settings.logging)
    set_random_seed(settings.random_seed)
    project_root = args.config.resolve().parent
    output_dir = (project_root / (args.output_dir or settings.paths.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = discover_dataset(settings.dataset, project_root)
    name_counters: dict[str, Counter] = {}
    designator_counter: Counter = Counter()
    unit_counter: Counter = Counter()
    direction_counter: Counter = Counter()
    missing_counter: Counter = Counter()
    designator_examples: dict[str, list[str]] = {}
    unit_examples: dict[str, list[str]] = {}
    missing_examples: dict[str, list[str]] = {}

    for item in dataset.files:
        if item.split != settings.dataset.read_split or item.role != SOURCE_ROLE:
            continue
        loaded = read_header(item, settings.dataset)
        name_column = _column(loaded, "name")
        address_column = _column(loaded, "address")
        for chunk in iter_chunks(loaded, settings.dataset):
            if name_column:
                name_counters[item.relative_path] = _name_tokens(
                    chunk[name_column], name_counters.get(item.relative_path, Counter())
                )
            if address_column:
                (
                    designator_counter,
                    unit_counter,
                    direction_counter,
                    missing_counter,
                    designator_examples,
                    unit_examples,
                    missing_examples,
                ) = _address_tokens(
                    chunk[address_column],
                    designator_counter,
                    unit_counter,
                    direction_counter,
                    missing_counter,
                    designator_examples,
                    unit_examples,
                    missing_examples,
                    settings.dataset.max_examples,
                )

    merged_names: Counter = Counter()
    for counter in name_counters.values():
        merged_names.update(counter)

    def supported(observed: Counter, variant: str, canonical: str) -> bool:
        """Return whether both sides of a synonym pair are actually observed."""

        return observed.get(variant, 0) >= MIN_SUPPORT and observed.get(canonical, 0) >= MIN_SUPPORT

    designator_observed: Counter = designator_counter
    unit_observed: Counter = unit_counter
    direction_observed: Counter = direction_counter

    derived: dict[str, Any] = {
        "method": "tokens observed in the training sources only",
        "min_support": MIN_SUPPORT,
        "read_split": settings.dataset.read_split,
        "files_analysed": [item.relative_path for item in dataset.files if item.split == settings.dataset.read_split],
        "business_name": {
            "trailing_token_frequencies": _top(merged_names, 60),
            "observed_legal_form_tokens": {
                token: count
                for token, count in sorted(merged_names.items(), key=lambda item: -item[1])
                if token in LEGAL_FORM_CANDIDATES
            },
            "proposed_legal_form_synonyms": {
                variant: canonical
                for variant, canonical in sorted(LEGAL_FORM_CANONICAL.items())
                if variant != canonical
                and supported(merged_names, variant, canonical)
            },
            "excluded_legal_form_tokens": {
                variant: canonical
                for variant, canonical in sorted(LEGAL_FORM_CANONICAL.items())
                if variant != canonical
                and not supported(merged_names, variant, canonical)
            },
        },
        "business_address": {
            "observed_street_designators": {
                token: count for token, count in _top(designator_observed, 200)
            },
            "proposed_street_designators": {
                variant: DESIGNATOR_CANONICAL[variant]
                for variant, canonical in sorted(DESIGNATOR_CANONICAL.items())
                if supported(designator_observed, variant, canonical)
            },
            "excluded_street_designators": {
                variant: canonical
                for variant, canonical in sorted(DESIGNATOR_CANONICAL.items())
                if not supported(designator_observed, variant, canonical)
            },
            "observed_unit_markers": {token: count for token, count in _top(unit_observed, 60)},
            "proposed_unit_markers": {
                variant: UNIT_CANONICAL[variant]
                for variant, canonical in sorted(UNIT_CANONICAL.items())
                if supported(unit_observed, variant, canonical)
            },
            "excluded_unit_markers": {
                variant: canonical
                for variant, canonical in sorted(UNIT_CANONICAL.items())
                if not supported(unit_observed, variant, canonical)
            },
            "observed_direction_tokens": {
                token: count for token, count in _top(direction_observed, 30)
            },
            "proposed_direction_tokens": {
                variant: DIRECTION_CANONICAL[variant]
                for variant, canonical in sorted(DIRECTION_CANONICAL.items())
                if supported(direction_observed, variant, canonical)
            },
            "excluded_direction_tokens": {
                variant: canonical
                for variant, canonical in sorted(DIRECTION_CANONICAL.items())
                if not supported(direction_observed, variant, canonical)
            },
            "observed_missing_markers": {token: count for token, count in _top(missing_counter, 30)},
            "proposed_missing_markers": {
                token: count
                for token, count in _top(missing_counter, 60)
                if token in MISSING_TOKEN_VALUES and count >= MIN_SUPPORT
            },
            "excluded_missing_markers": {
                token: count
                for token, count in _top(missing_counter, 60)
                if token in MISSING_TOKEN_VALUES and count < MIN_SUPPORT
            },
            "designator_examples": designator_examples,
            "unit_examples": unit_examples,
            "missing_marker_examples": missing_examples,
        },
    }

    path = output_dir / "derived_normalization_rules.json"
    reporting.write_json(path, derived)
    LOGGER.info("Wrote %s", path)
    print(reporting.render_yaml_rules(derived))
    return 0


def _column(loaded: LoadedFile, hint: str) -> str | None:
    for column in loaded.columns:
        if hint in column.casefold():
            return column
    return None


def _name_tokens(values: pd.Series, counter: Counter) -> Counter:
    text = values.fillna("").astype(str)
    for value in text.tolist():
        tokens = TOKEN.findall(unicodedata.normalize("NFKC", value).casefold())
        if tokens:
            counter[tokens[-1]] += 1
    return counter


def _address_tokens(
    values: pd.Series,
    designator: Counter,
    unit: Counter,
    direction: Counter,
    missing: Counter,
    designator_examples: dict[str, list[str]],
    unit_examples: dict[str, list[str]],
    missing_examples: dict[str, list[str]],
    limit: int,
) -> tuple[Counter, Counter, Counter, Counter, dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    for value in values.fillna("").astype(str).tolist():
        tokens = TOKEN.findall(unicodedata.normalize("NFKC", value).casefold())
        if not tokens:
            continue
        for token in tokens:
            if token in MISSING_MARKERS:
                missing[token] += 1
                if len(missing_examples.setdefault(token, [])) < limit:
                    missing_examples[token].append(value[:160])
        for index, token in enumerate(tokens):
            if token in DIRECTION_CANDIDATES and len(token) <= 5:
                direction[token] += 1
            if token in UNIT_CANDIDATES and len(token) >= 2:
                unit[token] += 1
                if len(unit_examples.setdefault(token, [])) < limit:
                    unit_examples[token].append(value[:160])
            previous = tokens[index - 1] if index else ""
            if DIGITS.search(previous) and token in DESIGNATOR_CANDIDATES:
                designator[token] += 1
                if len(designator_examples.setdefault(token, [])) < limit:
                    designator_examples[token].append(value[:160])
    return designator, unit, direction, missing, designator_examples, unit_examples, missing_examples


def _top(counter: Counter, limit: int) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


if __name__ == "__main__":
    sys.exit(main())
