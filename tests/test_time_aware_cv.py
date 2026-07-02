"""Fix #2 — tuning and champion-selection cross-validation must be time-aware on
out-of-time runs.

A shuffled K-fold on data that was deliberately split chronologically leaks the
future into hyperparameter selection, undoing the whole point of the out-of-time
protocol. These tests pin the CV scheme to the run's evaluation strategy.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("crewai", MagicMock())
sys.modules.setdefault("crewai.tools", MagicMock())

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from core.config import load_settings
from core.constants import ProblemType
from agents.improvement.tools import ImprovementService
from agents.training.tools import _time_ordered_subsample


def test_time_ordered_subsample_preserves_order_and_caps():
    X = np.arange(100).reshape(-1, 1).astype(float)
    y = np.arange(100)
    Xc, yc = _time_ordered_subsample(X, y, cap=10)
    assert len(Xc) <= 10
    # Strictly increasing => original chronological order preserved (no shuffle).
    assert np.all(np.diff(yc) > 0)
    # Spans the full range (even stride), not just the head/tail.
    assert yc[0] == 0 and yc[-1] == 99


def test_time_ordered_subsample_noop_when_small():
    X = np.arange(5).reshape(-1, 1).astype(float)
    y = np.arange(5)
    Xc, yc = _time_ordered_subsample(X, y, cap=100)
    assert np.array_equal(yc, y)


def _capture_cv_for(time_ordered: bool):
    """Run one RandomizedSearch tuning round with the CV constructor intercepted,
    and return the ``cv`` object that was passed to RandomizedSearchCV."""
    import sklearn.model_selection as skms

    captured = {}
    real = skms.RandomizedSearchCV

    class _FakeSearch:
        """Minimal stand-in that just fits the base estimator (no real search),
        so the test stays fast and deterministic while capturing the cv arg."""
        def __init__(self, estimator, *args, **kwargs):
            captured["cv"] = kwargs.get("cv")
            self._est = estimator

        def fit(self, X, y):
            self.best_estimator_ = self._est.fit(X, y)
            self.best_params_ = {}
            self.best_score_ = 0.0
            return self

    skms.RandomizedSearchCV = _FakeSearch  # read by the function's local import
    try:
        settings = load_settings()
        svc = ImprovementService(settings.improvement, seed=42)
        rng = np.random.RandomState(0)
        X = rng.normal(size=(80, 3))
        y = (X[:, 0] + rng.normal(size=80) > 0).astype(int)
        svc._tune_with_randomized_search(
            "LogisticRegression", X, y, ProblemType.CLASSIFICATION, seed=42,
            time_ordered=time_ordered,
        )
    finally:
        skms.RandomizedSearchCV = real
    return captured.get("cv")


def test_tuning_uses_timeseriessplit_when_out_of_time():
    cv = _capture_cv_for(time_ordered=True)
    assert isinstance(cv, TimeSeriesSplit)


def test_tuning_uses_stratified_kfold_otherwise():
    cv = _capture_cv_for(time_ordered=False)
    assert isinstance(cv, StratifiedKFold)
    assert cv.shuffle is True
