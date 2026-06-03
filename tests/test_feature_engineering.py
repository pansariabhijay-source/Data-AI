"""Tests for feature engineering service (decoupled from CrewAI)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("crewai", MagicMock())
sys.modules.setdefault("crewai.tools", MagicMock())

import numpy as np
import pandas as pd
import pytest

from core.config import FeatureEngineeringConfig
from core.constants import ProblemType
from agents.feature_engineering.tools import FeatureEngineeringService


@pytest.fixture
def service():
    return FeatureEngineeringService(FeatureEngineeringConfig())


def test_encode_low_cardinality(service):
    df = pd.DataFrame({"color": ["red", "blue", "green", "red"], "val": [1, 2, 3, 4]})
    result, mapping = service.encode_categoricals(df)
    assert "color" not in result.columns
    assert "color" in mapping


def test_encode_high_cardinality(service):
    service._config.max_onehot_cardinality = 3
    df = pd.DataFrame({"cat": [f"v{i}" for i in range(10)], "val": range(10)})
    result, mapping = service.encode_categoricals(df)
    assert "cat" in result.columns
    assert pd.api.types.is_numeric_dtype(result["cat"])


def test_remove_low_variance(service):
    df = pd.DataFrame({"constant": [1] * 100, "varied": np.random.randn(100)})
    result, removed = service.remove_low_variance(df)
    assert "constant" in removed


def test_remove_correlated(service):
    x = np.random.randn(100)
    df = pd.DataFrame({"a": x, "b": x + np.random.randn(100) * 0.01, "c": np.random.randn(100)})
    result, removed = service.remove_correlated(df)
    assert len(removed) > 0


def test_scale_features(service):
    df = pd.DataFrame({"a": [100, 200, 300], "b": [1, 2, 3]})
    result, mapping = service.scale_features(df)
    assert abs(result["a"].mean()) < 1e-10
