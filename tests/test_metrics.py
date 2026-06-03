"""Tests for metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import ProblemType
from core.metrics import (
    compute_metrics,
    get_primary_metric,
    is_metric_higher_better,
)


def test_classification_metrics():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    metrics = compute_metrics(y_true, y_pred, ProblemType.CLASSIFICATION)
    assert "f1" in metrics
    assert "accuracy" in metrics
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["accuracy"] <= 1


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
    metrics = compute_metrics(y_true, y_pred, ProblemType.REGRESSION)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] > 0
    assert metrics["r2"] > 0


def test_primary_metric():
    assert get_primary_metric(ProblemType.CLASSIFICATION) == "f1"
    assert get_primary_metric(ProblemType.REGRESSION) == "r2"
    assert get_primary_metric(ProblemType.CLUSTERING) == "silhouette_score"


def test_metric_direction():
    assert is_metric_higher_better("f1") is True
    assert is_metric_higher_better("rmse") is False
    assert is_metric_higher_better("r2") is True
