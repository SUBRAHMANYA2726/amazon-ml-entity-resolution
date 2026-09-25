"""Deterministic standard-library runtime setup."""

from __future__ import annotations

import os
import random


def set_random_seed(seed: int) -> None:
    """Seed standard-library randomness without importing future ML libraries."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
