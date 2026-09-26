"""Command-line entry point for the Business Entity Resolution pipeline.

Supports:
- --stage validate: Validates configuration and environment without processing data.
- --stage blocking: Runs Phase 3 multi-strategy blocking, candidate generation, and recall evaluation.
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
    """Build the command parser."""

    parser = argparse.ArgumentParser(
        description="Business Entity Resolution pipeline CLI."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["validate", "blocking"],
        default="validate",
        help="Pipeline execution stage (default: validate).",
    )
    parser.add_argument(
        "--num-anchors",
        type=int,
        default=2500,
        help="Number of anchors to evaluate during blocking (default: 2500).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/phase3"),
        help="Output directory for blocking results and reports.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate configuration or run pipeline stage."""

    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
    except ConfigurationError as error:
        build_parser().error(str(error))
        return 2

    logger = configure_logging(settings.logging)
    set_random_seed(settings.random_seed)

    if args.stage == "validate":
        logger.info(
            "Phase %s configuration is valid for %s; validation check passed.",
            settings.project.phase,
            settings.project.name,
        )
        get_logger(__name__).debug("Configuration source: %s", settings.source_path)
        return 0

    if args.stage == "blocking":
        logger.info("Executing Phase 3 Multi-Strategy Blocking stage...")
        from scripts.run_phase3_blocking import main as run_blocking_main
        # Re-invoke runner
        return run_blocking_main()

    return 0
