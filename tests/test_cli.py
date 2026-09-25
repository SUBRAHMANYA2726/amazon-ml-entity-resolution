"""Tests for the safe, non-data Phase 0 command-line entry point."""

from __future__ import annotations

from pathlib import Path

from business_entity_resolution.cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase_zero_command_validates_config_without_data_processing() -> None:
    """The only current entry point completes from generic configuration alone."""

    assert main(["--config", str(PROJECT_ROOT / "config.yaml")]) == 0
