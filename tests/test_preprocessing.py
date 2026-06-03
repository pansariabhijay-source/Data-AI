"""Tests for preprocessing service (decoupled from CrewAI)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("crewai", MagicMock())
sys.modules.setdefault("crewai.tools", MagicMock())

import numpy as np
import pandas as pd
import pytest

from core.config import PreprocessingConfig
from agents.preprocessing.tools import PreprocessingService


@pytest.fixture
def service():
    return PreprocessingService(PreprocessingConfig())


def test_remove_duplicates(service):
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": [4, 4, 5, 6]})
    result, count = service.remove_duplicates(df)
    assert count == 1
    assert len(result) == 3


def test_handle_missing_numeric(service):
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [10, 20, 30, 40]})
    result, strategies = service.handle_missing_values(df)
    assert result["a"].isnull().sum() == 0
    assert "a" in strategies


def test_handle_missing_categorical(service):
    df = pd.DataFrame({"a": ["x", None, "y", "x"]})
    result, strategies = service.handle_missing_values(df)
    assert result["a"].isnull().sum() == 0


def test_drop_high_null_columns(service):
    n = 100
    df = pd.DataFrame({
        "good": range(n),
        "bad": [np.nan] * 80 + list(range(20)),
    })
    result, strategies = service.handle_missing_values(df)
    assert "bad" not in result.columns


def test_handle_outliers_iqr(service):
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 100]})
    result, handled = service.handle_outliers(df)
    assert "a" in handled
    assert result["a"].max() < 100


def test_fix_dtypes_numeric_string(service):
    df = pd.DataFrame({"a": ["1", "2", "3", "4"]})
    result, fixes = service.fix_dtypes(df)
    assert pd.api.types.is_numeric_dtype(result["a"])


def test_detect_high_cardinality(service):
    df = pd.DataFrame({"a": [f"val_{i}" for i in range(100)]})
    high = service.detect_high_cardinality(df)
    assert "a" in high
