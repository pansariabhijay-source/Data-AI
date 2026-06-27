"""Tests for data collection service (decoupled from CrewAI for compatibility)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock crewai before importing service modules — Python 3.14 has pydantic v1 issues
sys.modules.setdefault("crewai", MagicMock())
sys.modules.setdefault("crewai.tools", MagicMock())

import numpy as np
import pandas as pd
import pytest

from core.config import DataCollectionConfig
from core.constants import ProblemType
from agents.data_collection.tools import DataCollectionService


@pytest.fixture
def service():
    return DataCollectionService(DataCollectionConfig())


def test_load_csv(service, tmp_csv):
    df = service.load_csv(tmp_csv)
    assert len(df) == 100
    assert "target" in df.columns


def test_load_csv_file_not_found(service):
    with pytest.raises(Exception, match="not found"):
        service.load_csv("/nonexistent/path.csv")


def test_detect_classification(service, sample_classification_df):
    pt = service.detect_problem_type(sample_classification_df, "target")
    assert pt == ProblemType.CLASSIFICATION


def test_detect_regression(service, sample_regression_df):
    pt = service.detect_problem_type(sample_regression_df, "target")
    assert pt == ProblemType.REGRESSION


def test_detect_clustering_no_target(service, sample_classification_df):
    pt = service.detect_problem_type(sample_classification_df, None)
    assert pt == ProblemType.CLUSTERING


def test_detect_regression_numeric_target_stored_as_string(service):
    """A continuous numeric target dirtied by a junk row (object dtype) must still
    be detected as regression, not a high-cardinality classification."""
    rng = np.random.RandomState(0)
    vals = rng.uniform(0, 1000, 500).round(2).astype(object)
    vals[0] = "JUNK"  # stray non-numeric value from a malformed header/export row
    df = pd.DataFrame({"feat": rng.normal(size=500), "target": vals})
    assert service.detect_problem_type(df, "target") == ProblemType.REGRESSION


def test_detect_classification_high_cardinality_capped(service):
    """A numeric target is only classification when it has few distinct values; a
    many-valued continuous target is regression even at a low unique/row ratio."""
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"target": rng.uniform(0, 1, 5000)})  # 5000 distinct floats
    assert service.detect_problem_type(df, "target") == ProblemType.REGRESSION


def test_detect_classification_low_cardinality_integer(service):
    df = pd.DataFrame({"target": np.array([0, 1, 2, 0, 1, 2, 1, 0] * 50)})
    assert service.detect_problem_type(df, "target") == ProblemType.CLASSIFICATION


def test_profile_dataset(service, sample_classification_df):
    meta = service.profile_dataset(sample_classification_df, "target")
    assert meta.n_rows == 100
    assert meta.n_columns == 4
    assert "target" in meta.column_names
