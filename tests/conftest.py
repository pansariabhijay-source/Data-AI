"""Test fixtures and shared configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture
def sample_classification_df() -> pd.DataFrame:
    """Generate a small classification dataset for testing."""
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "feat1": rng.normal(0, 1, n),
        "feat2": rng.normal(0, 1, n),
        "feat3": rng.choice(["a", "b", "c"], n),
        "target": rng.choice([0, 1], n),
    })
    return df


@pytest.fixture
def sample_regression_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "feat1": rng.normal(0, 1, n),
        "feat2": rng.normal(5, 2, n),
        "feat3": rng.choice(["x", "y"], n),
        "target": rng.normal(100, 20, n),
    })
    return df


@pytest.fixture
def tmp_csv(sample_classification_df: pd.DataFrame, tmp_path: Path) -> str:
    path = tmp_path / "test_data.csv"
    sample_classification_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def settings():
    from core.config import load_settings
    return load_settings(overrides={
        "pipeline": {"random_seed": 42, "artifact_dir": "test_artifacts", "report_dir": "test_reports"},
        "logging": {"level": "DEBUG"},
    })
