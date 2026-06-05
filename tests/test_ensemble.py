"""Tests for the probability-averaging ensemble."""

from __future__ import annotations

import numpy as np
import pytest

from core.ensemble import ProbabilityAveragingEnsemble


class _StubModel:
    def __init__(self, proba: np.ndarray, classes: np.ndarray) -> None:
        self._proba = proba
        self.classes_ = classes

    def predict_proba(self, X):  # noqa: ANN001
        return self._proba


def test_equal_weight_average():
    classes = np.array([0, 1])
    a = _StubModel(np.array([[0.8, 0.2], [0.4, 0.6]]), classes)
    b = _StubModel(np.array([[0.6, 0.4], [0.2, 0.8]]), classes)
    ens = ProbabilityAveragingEnsemble([a, b], classes)
    proba = ens.predict_proba(np.zeros((2, 1)))
    assert np.allclose(proba, [[0.7, 0.3], [0.3, 0.7]])
    assert list(ens.predict(np.zeros((2, 1)))) == [0, 1]


def test_weighted_average_favours_member():
    classes = np.array([0, 1])
    a = _StubModel(np.array([[0.9, 0.1]]), classes)
    b = _StubModel(np.array([[0.1, 0.9]]), classes)
    ens = ProbabilityAveragingEnsemble([a, b], classes, weights=[3.0, 1.0])
    proba = ens.predict_proba(np.zeros((1, 1)))
    # 3:1 weighting pulls toward member a (class 0).
    assert proba[0, 0] > 0.6


def test_requires_members():
    with pytest.raises(ValueError):
        ProbabilityAveragingEnsemble([], np.array([0, 1]))


def test_pickles_roundtrip():
    import joblib, io

    classes = np.array([0, 1])
    ens = ProbabilityAveragingEnsemble(
        [_StubModel(np.array([[0.5, 0.5]]), classes)], classes
    )
    buf = io.BytesIO()
    joblib.dump(ens, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    assert restored.classes_.tolist() == [0, 1]
