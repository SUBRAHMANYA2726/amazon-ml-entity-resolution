"""Tests for reusable project logging."""

from __future__ import annotations

import logging
from pathlib import Path

from business_entity_resolution.config.settings import load_settings
from business_entity_resolution.utils.logging import configure_logging, get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_configures_project_logger_once() -> None:
    """The project logger is configured without using the root logger."""

    settings = load_settings(PROJECT_ROOT / "config.yaml")
    logger = configure_logging(settings.logging)
    configure_logging(settings.logging)

    assert logger.name == "business_entity_resolution"
    assert logger.level == logging.INFO
    assert logger.propagate is False
    assert len(logger.handlers) == 1


def test_returns_module_logger_in_project_namespace() -> None:
    """Module loggers inherit the project namespace."""

    logger = get_logger("tests.foundation")

    assert logger.name == "business_entity_resolution.tests.foundation"
