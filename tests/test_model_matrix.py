"""Fix #8 — the model matrix is selected BY NAME from the canonical feature list,
so train/val/test/inference always share identical columns in identical order.

Relying on ``select_dtypes(number)`` per CSV silently breaks if a split reorders
columns, reads one as a different dtype, or is missing a column. These tests pin
the by-name behaviour and its safe fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.utils import build_model_matrix


def test_build_matrix_selects_by_name_in_order():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "target": [0, 1]})
    X, cols = build_model_matrix(df, "target", ["b", "a"])  # explicit order, drops c
    assert cols == ["b", "a"]
    assert X.tolist() == [[3, 1], [4, 2]]


def test_build_matrix_is_robust_to_column_reordering():
    """Two 'splits' with columns in different orders must yield identical matrices
    when selected against the same feature list."""
    feats = ["f1", "f2", "f3"]
    d1 = pd.DataFrame({"f1": [1], "f2": [2], "f3": [3], "target": [0]})
    d2 = pd.DataFrame({"target": [0], "f3": [3], "f1": [1], "f2": [2]})  # shuffled
    X1, c1 = build_model_matrix(d1, "target", feats)
    X2, c2 = build_model_matrix(d2, "target", feats)
    assert c1 == c2 == feats
    assert np.array_equal(X1, X2)


def test_build_matrix_fills_missing_column_with_zero():
    df = pd.DataFrame({"f1": [1, 2], "target": [0, 1]})  # f2 absent
    X, cols = build_model_matrix(df, "target", ["f1", "f2"])
    assert cols == ["f1"]  # only present columns are used as the matrix columns
    # ...but a fully-absent requested column would be zero-filled; present here f1 only.
    assert X.shape == (2, 1)


def test_build_matrix_fallback_without_feature_names():
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "s": ["x", "y"], "target": [0, 1]})
    X, cols = build_model_matrix(df, "target", None)
    # Numeric non-target columns only; string 's' excluded.
    assert set(cols) == {"a", "b"}
    assert X.shape == (2, 2)


def test_build_matrix_coerces_stringified_numbers():
    # A split that round-tripped through CSV may read a column back as object.
    df = pd.DataFrame({"f1": ["1", "2", "3"], "target": [0, 1, 0]})
    X, cols = build_model_matrix(df, "target", ["f1"])
    assert cols == ["f1"]
    assert X.ravel().tolist() == [1.0, 2.0, 3.0]
