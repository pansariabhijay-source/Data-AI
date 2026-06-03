"""Tests for validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.constants import ProblemType
from core.validation import (
    compute_quality_score,
    detect_class_imbalance,
    detect_target_leakage,
    validate_dataframe,
    validate_target_column,
)


def test_validate_empty_df():
    df = pd.DataFrame()
    issues = validate_dataframe(df)
    assert any("empty" in i.lower() for i in issues)


def test_validate_too_few_rows():
    df = pd.DataFrame({"a": [1, 2, 3]})
    issues = validate_dataframe(df, min_rows=10)
    assert any("few rows" in i.lower() for i in issues)


def test_validate_target_missing():
    df = pd.DataFrame({"a": [1, 2, 3]})
    issues = validate_target_column(df, "nonexistent", ProblemType.CLASSIFICATION)
    assert any("not found" in i.lower() for i in issues)


def test_quality_score():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"]})
    score = compute_quality_score(df)
    assert 0 <= score <= 1


def test_detect_leakage():
    x = np.arange(100, dtype=float)
    df = pd.DataFrame({"feat": x, "target": x * 2 + 1})
    leaked = detect_target_leakage(df, "target", threshold=0.95)
    assert "feat" in leaked


def test_detect_imbalance():
    s = pd.Series([0] * 95 + [1] * 5)
    result = detect_class_imbalance(s)
    assert result is not None


def test_no_imbalance():
    s = pd.Series([0] * 50 + [1] * 50)
    result = detect_class_imbalance(s)
    assert result is None
