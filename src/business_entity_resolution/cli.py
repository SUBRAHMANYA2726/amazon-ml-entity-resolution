"""Safe Phase 0 command-line entry point.

The command validates configuration and prepares logging only. It deliberately
does not inspect data, train a model, generate candidates, or write outputs.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from business_entity_resolution.config.settings import DEFAULT_CONFIG_PATH, load_settings
from business_entity_resolution.exceptions import ConfigurationError
from business_entity_resolution.utils.logging import configure_logging, get_logger
from business_entity_resolution.utils.seed import set_random_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the non-data Phase 0 validation command parser."""

    parser = argparse.ArgumentParser(
        description="Validate Phase 0 configuration without processing challenge data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the generic project configuration file (default: config.yaml).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate configuration and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
    except ConfigurationError as error:
        build_parser().error(str(error))
        return 2

    logger = configure_logging(settings.logging)
    set_random_seed(settings.random_seed)
    logger.info(
        "Phase %s configuration is valid for %s; no dataset operations were run.",
        settings.project.phase,
        settings.project.name,
    )
    get_logger(__name__).debug("Configuration source: %s", settings.source_path)
    return 0
