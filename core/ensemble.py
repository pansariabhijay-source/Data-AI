"""
Lightweight model ensembling for the training stage.

`ProbabilityAveragingEnsemble` is a soft-voting wrapper over already-fitted
estimators. Unlike sklearn's ``VotingClassifier`` it does NOT refit the base
models, so it preserves expensive per-model work (e.g. early-stopped boosters)
and simply averages their calibrated probabilities. Defined at module scope so
it pickles cleanly via joblib.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ProbabilityAveragingEnsemble:
    """Soft-voting ensemble that averages the ``predict_proba`` of fitted models.

    All members must share the same ``classes_`` ordering (guaranteed when they
    were trained on the same target). ``weights`` lets stronger members count for
    more; defaults to equal weighting.
    """

    def __init__(
        self,
        estimators: list[Any],
        classes: np.ndarray,
        member_names: list[str] | None = None,
        weights: list[float] | None = None,
    ) -> None:
        if not estimators:
            raise ValueError("ProbabilityAveragingEnsemble requires >= 1 estimator")
        self.estimators = estimators
        self.classes_ = np.asarray(classes)
        self.member_names = member_names or [type(e).__name__ for e in estimators]
        if weights is not None and len(weights) != len(estimators):
            raise ValueError("weights length must match estimators length")
        self.weights = np.asarray(weights, dtype=float) if weights is not None else None

    def predict_proba(self, X: Any) -> np.ndarray:
        probs = np.array([est.predict_proba(X) for est in self.estimators])
        if self.weights is not None:
            w = self.weights / self.weights.sum()
            return np.tensordot(w, probs, axes=([0], [0]))
        return probs.mean(axis=0)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
