"""Tests for deterministic standard-library seeding."""

from __future__ import annotations

import random

import pytest

from business_entity_resolution.utils.seed import set_random_seed


def test_seed_reproduces_standard_library_randomness() -> None:
    """The configured seed produces repeatable random sequences."""

    set_random_seed(2026)
    first = [random.random() for _ in range(3)]
    set_random_seed(2026)
    second = [random.random() for _ in range(3)]

    assert first == second


def test_seed_rejects_invalid_values() -> None:
    """Invalid seeds fail before affecting the runtime."""

    with pytest.raises(ValueError, match="non-negative integer"):
        set_random_seed(-1)
