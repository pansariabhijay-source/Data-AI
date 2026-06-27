"""Tests for SMOTE / imbalanced-target resampling guards."""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import ProblemType
from core.resampling import maybe_resample


@pytest.fixture
def imbalanced():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(1030, 5))
    y = np.array([0] * 1000 + [1] * 30)
    return X, y


def test_resample_applies_on_imbalanced_binary(imbalanced):
    X, y = imbalanced
    r = maybe_resample(X, y, ProblemType.CLASSIFICATION, enabled=True, sampling_strategy=0.25)
    assert r.applied
    assert r.method == "SMOTE"
    # minority oversampled toward 25% of the majority (1000 * 0.25 = 250)
    assert r.minority_after == 250
    assert len(r.y) == 1250


def test_resample_disabled_returns_original(imbalanced):
    X, y = imbalanced
    r = maybe_resample(X, y, ProblemType.CLASSIFICATION, enabled=False)
    assert not r.applied
    assert r.reason == "disabled"
    assert len(r.y) == len(y)


def test_resample_skips_already_balanced():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(1000, 4))
    y = np.array([0] * 500 + [1] * 500)
    r = maybe_resample(X, y, ProblemType.CLASSIFICATION, enabled=True)
    assert not r.applied
    assert "balanced" in r.reason


def test_resample_skips_multiclass():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(900, 4))
    y = np.array([0] * 600 + [1] * 200 + [2] * 100)
    r = maybe_resample(X, y, ProblemType.CLASSIFICATION, enabled=True)
    assert not r.applied
    assert "binary" in r.reason


def test_resample_skips_regression():
    rng = np.random.RandomState(3)
    X = rng.normal(size=(200, 3))
    y = rng.normal(size=200)
    r = maybe_resample(X, y, ProblemType.REGRESSION, enabled=True)
    assert not r.applied


def test_resample_respects_max_train_samples(imbalanced):
    X, y = imbalanced
    r = maybe_resample(X, y, ProblemType.CLASSIFICATION, enabled=True, max_train_samples=100)
    assert not r.applied
    assert "too large" in r.reason


def test_resample_fallback_when_too_few_minority_for_smote():
    """With fewer minority samples than k_neighbors, SMOTE can't build neighbours;
    it must fall back to RandomOverSampler rather than crash."""
    rng = np.random.RandomState(4)
    X = rng.normal(size=(403, 4))
    y = np.array([0] * 400 + [1] * 3)  # only 3 minority
    r = maybe_resample(X, y, ProblemType.CLASSIFICATION, enabled=True,
                       sampling_strategy=0.25, k_neighbors=5)
    assert r.applied  # k auto-shrinks to minority-1, or falls back
    assert r.minority_after > 3
