"""Tests for generic, dataset-independent configuration behavior."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from business_entity_resolution.config.settings import load_settings
from business_entity_resolution.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_loads_project_configuration() -> None:
    """The committed configuration remains schema-agnostic and relative."""

    settings = load_settings(PROJECT_ROOT / "config.yaml")

    assert settings.project.phase in (0, 1, 2, 3)
    assert settings.paths.raw_dir == Path("data/raw")
    assert settings.models.primary is None
    assert isinstance(settings.blocking.strategies, tuple)
    assert settings.evaluation.threshold_grid == ()
    assert all(
        not getattr(settings.paths, field.name).is_absolute()
        for field in fields(settings.paths)
    )


def test_settings_load_deterministically() -> None:
    """Repeated loads produce equal immutable configuration objects."""

    first = load_settings(PROJECT_ROOT / "config.yaml")
    second = load_settings(PROJECT_ROOT / "config.yaml")

    assert first == second


def test_option_mappings_are_read_only() -> None:
    """Configuration consumers cannot mutate loaded generic options."""

    settings = load_settings(PROJECT_ROOT / "config.yaml")

    with pytest.raises(TypeError):
        settings.models.options["future_option"] = "value"  # type: ignore[index]


def test_rejects_machine_specific_paths(tmp_path: Path) -> None:
    """Configuration paths must remain project-relative."""

    content = (PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8")
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        content.replace("data_root: data", "data_root: /machine-specific"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="project-relative"):
        load_settings(invalid_config)


def test_reports_missing_configuration_file(tmp_path: Path) -> None:
    """A missing config receives a clear project-specific error."""

    with pytest.raises(ConfigurationError, match="does not exist"):
        load_settings(tmp_path / "absent.yaml")
