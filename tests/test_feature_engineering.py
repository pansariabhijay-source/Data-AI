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
    # High cardinality but NOT id-like (10 categories spread over 100 rows) →
    # label-encoded into a numeric feature.
    service._config.max_onehot_cardinality = 3
    df = pd.DataFrame({"cat": [f"v{i % 10}" for i in range(100)], "val": range(100)})
    result, mapping = service.encode_categoricals(df)
    assert "cat" in result.columns
    assert pd.api.types.is_numeric_dtype(result["cat"])
    assert "label_encoded" in mapping["cat"]


def test_encode_drops_id_like(service):
    # Near-unique column (id-like) is dropped, not encoded into noise.
    service._config.max_onehot_cardinality = 3
    df = pd.DataFrame({"user_id": [f"u{i}" for i in range(100)], "val": range(100)})
    result, mapping = service.encode_categoricals(df)
    assert "user_id" not in result.columns
    assert "dropped" in mapping["user_id"]


def test_low_variance_scale_invariant(service):
    # A genuinely informative low-magnitude column (0/1 flag) must NOT be dropped
    # just because its raw variance is small next to a large-magnitude column.
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "income": rng.normal(50000, 15000, 200),   # huge raw variance
        "flag": rng.binomial(1, 0.4, 200),          # tiny raw variance, real signal
    })
    result, removed = service.remove_low_variance(df)
    assert "flag" not in removed
    assert "flag" in result.columns


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
