"""Shared pytest configuration."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "protected: an invariant that blocks a merge when it fails",
    )
    config.addinivalue_line("markers", "slow: takes more than a second")


@pytest.fixture(autouse=True)
def deterministic_seeds():
    """Every test starts from the same random state."""
    import random

    import numpy as np

    random.seed(42)
    np.random.seed(42)
    yield
