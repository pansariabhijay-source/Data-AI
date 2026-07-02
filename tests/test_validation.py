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


def test_detect_leakage_binary_string_target():
    """Thresholded leakage (corr well below 0.95 but perfect single-feature ranking)
    must be caught for a non-numeric classification target — the old corr-only check
    silently missed this (e.g. weather's RISK_MM -> RainTomorrow)."""
    rng = np.random.RandomState(0)
    amount = rng.uniform(0, 10, 400)
    target = np.where(amount > 1.0, "Yes", "No")  # target is a threshold of `amount`
    df = pd.DataFrame({
        "amount": amount,                       # perfect ranker -> leakage
        "noise": rng.normal(size=400),          # unrelated
        "target": target,
    })
    leaked = detect_target_leakage(df, "target")
    assert "amount" in leaked
    assert "noise" not in leaked


def test_detect_leakage_does_not_flag_moderate_predictor():
    """A genuinely strong-but-not-perfect feature must not be stripped as leakage."""
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 500)
    feat = y + rng.normal(0, 0.6, 500)  # informative but noisy (AUC well under 0.999)
    df = pd.DataFrame({"feat": feat, "target": y})
    assert "feat" not in detect_target_leakage(df, "target")


def test_detect_leakage_categorical_string_feature():
    """A pure-string feature that (almost) perfectly determines the target is
    leakage, but the numeric-AUC path coerced it to NaN and skipped it. The
    out-of-fold target-encoding path must now catch it."""
    rng = np.random.RandomState(0)
    n = 600
    y = rng.randint(0, 2, n)
    # Category encodes the answer: label 1 -> "fraud_*", label 0 -> "ok_*".
    status = np.where(y == 1, "flagged_fraud", "cleared_ok")
    df = pd.DataFrame({
        "status": status,                       # string, perfectly predictive -> leak
        "noise_cat": rng.choice(["a", "b", "c"], n),
        "target": y,
    })
    leaked = detect_target_leakage(df, "target")
    assert "status" in leaked
    assert "noise_cat" not in leaked


def test_detect_leakage_does_not_flag_highcard_string_id():
    """A near-unique string ID column perfectly 'predicts' the target in-sample but
    must NOT be flagged: out-of-fold encoding collapses unseen categories to the
    global mean, so it scores ~0.5. (IDs are handled by the FE ID guard instead.)"""
    rng = np.random.RandomState(2)
    n = 600
    y = rng.randint(0, 2, n)
    ids = [f"user_{i}" for i in range(n)]  # unique per row
    df = pd.DataFrame({"user_id": ids, "target": y})
    assert "user_id" not in detect_target_leakage(df, "target")


def test_detect_leakage_categorical_benign_not_flagged():
    """A low-cardinality categorical only weakly related to the target is not leakage."""
    rng = np.random.RandomState(3)
    n = 500
    cat = rng.choice(["red", "green", "blue"], n)
    # Mild association only.
    y = np.where(cat == "red", rng.binomial(1, 0.6, n), rng.binomial(1, 0.45, n))
    df = pd.DataFrame({"color": cat, "target": y})
    assert "color" not in detect_target_leakage(df, "target")


def test_detect_imbalance():
    s = pd.Series([0] * 95 + [1] * 5)
    result = detect_class_imbalance(s)
    assert result is not None


def test_no_imbalance():
    s = pd.Series([0] * 50 + [1] * 50)
    result = detect_class_imbalance(s)
    assert result is None
