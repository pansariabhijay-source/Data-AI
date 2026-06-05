"""Tests for metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import ProblemType
from core.metrics import (
    compute_metrics,
    confusion_counts,
    find_optimal_threshold,
    get_primary_metric,
    is_metric_higher_better,
    positive_class_proba,
    predict_with_optimal_threshold,
    selection_score,
)


def test_classification_metrics():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    metrics = compute_metrics(y_true, y_pred, ProblemType.CLASSIFICATION)
    assert "f1" in metrics
    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["accuracy"] <= 1


def test_classification_probability_metrics_include_pr_auc():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])
    metrics = compute_metrics(y_true, y_pred, ProblemType.CLASSIFICATION, y_prob)
    assert "roc_auc" in metrics and "pr_auc" in metrics
    assert 0 <= metrics["pr_auc"] <= 1


def test_confusion_counts_binary():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    cm = confusion_counts(y_true, y_pred)
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_confusion_counts_non_binary_returns_none():
    assert confusion_counts(np.array([0, 1, 2]), np.array([0, 1, 2])) is None


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


def test_positive_class_proba_shapes():
    proba_2d = np.array([[0.9, 0.1], [0.3, 0.7]])
    assert np.allclose(positive_class_proba(proba_2d), [0.1, 0.7])
    assert np.allclose(positive_class_proba(np.array([0.1, 0.7])), [0.1, 0.7])


def test_find_optimal_threshold_beats_half_cutoff():
    # Imbalanced 90/10 where the model under-scores positives (max proba 0.45),
    # so the default 0.5 cutoff predicts NOTHING positive (F1=0). The tuned
    # threshold must recover a strictly better F1.
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(0)
    y = np.array([0] * 90 + [1] * 10)
    proba = np.concatenate([rng.uniform(0.0, 0.25, 90), rng.uniform(0.25, 0.45, 10)])
    thr = find_optimal_threshold(y, proba)
    assert thr < 0.5
    f1_tuned = f1_score(y, (proba >= thr).astype(int))
    f1_half = f1_score(y, (proba >= 0.5).astype(int))
    assert f1_tuned > f1_half


def test_find_optimal_threshold_single_class():
    # No positives → cannot derive a threshold; fall back to 0.5.
    assert find_optimal_threshold(np.zeros(10), np.linspace(0, 1, 10)) == 0.5


def test_predict_with_optimal_threshold_maps_labels():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.1, 0.9]])
    preds, thr = predict_with_optimal_threshold(y_true, proba, classes=np.array([0, 1]))
    assert set(np.unique(preds)).issubset({0, 1})
    assert 0.0 <= thr <= 1.0
    # The two genuine positives should be recovered at the tuned threshold.
    assert preds[2] == 1 and preds[3] == 1


def test_selection_score_binary_uses_ranking_metrics():
    """Binary classification selection averages PR-AUC and ROC-AUC, ignoring the
    threshold-dependent F1 — a model with weaker F1 but stronger ranking wins."""
    strong_rank = {"f1": 0.70, "pr_auc": 0.90, "roc_auc": 0.95}
    strong_f1 = {"f1": 0.85, "pr_auc": 0.60, "roc_auc": 0.65}
    s_rank = selection_score(strong_rank, ProblemType.CLASSIFICATION, n_classes=2)
    s_f1 = selection_score(strong_f1, ProblemType.CLASSIFICATION, n_classes=2)
    assert s_rank == pytest.approx(0.925)
    assert s_rank > s_f1


def test_selection_score_binary_falls_back_to_f1():
    """Without ranking metrics (e.g. a model with no probabilities), fall back to F1."""
    score = selection_score({"f1": 0.8}, ProblemType.CLASSIFICATION, n_classes=2)
    assert score == pytest.approx(0.8)


def test_selection_score_multiclass_uses_f1():
    score = selection_score(
        {"f1": 0.7, "roc_auc": 0.99}, ProblemType.CLASSIFICATION, n_classes=3
    )
    assert score == pytest.approx(0.7)


def test_selection_score_regression_uses_r2():
    score = selection_score({"r2": 0.42, "rmse": 3.1}, ProblemType.REGRESSION)
    assert score == pytest.approx(0.42)
