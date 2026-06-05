"""Tests for model registry."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from core.constants import ProblemType
from core.model_registry import (
    apply_imbalance_handling,
    build_default_registry,
    fit_model,
    get_fitted_n_estimators,
    ModelRegistry,
    ModelSpec,
)


def test_build_default_registry():
    registry = build_default_registry()
    clf_models = registry.list_models(ProblemType.CLASSIFICATION)
    reg_models = registry.list_models(ProblemType.REGRESSION)
    clu_models = registry.list_models(ProblemType.CLUSTERING)
    assert len(clf_models) >= 3
    assert len(reg_models) >= 3
    assert len(clu_models) >= 3


def test_create_instance():
    registry = build_default_registry()
    model = registry.create_instance(ProblemType.CLASSIFICATION, "LogisticRegression")
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_get_nonexistent_model():
    registry = build_default_registry()
    with pytest.raises(Exception, match="not registered"):
        registry.get(ProblemType.CLASSIFICATION, "NonExistentModel")


def test_classifiers_handle_imbalance():
    """Every classifier must apply class weighting or scale_pos_weight."""
    registry = build_default_registry()
    for spec in registry.list_models(ProblemType.CLASSIFICATION):
        weighted = spec.default_params.get("class_weight") == "balanced"
        assert weighted or spec.imbalance_param, f"{spec.name} has no imbalance handling"


def test_apply_imbalance_handling_sets_scale_pos_weight():
    registry = build_default_registry()
    spec = registry.get(ProblemType.CLASSIFICATION, "XGBClassifier")
    model = registry.create_instance(ProblemType.CLASSIFICATION, "XGBClassifier")
    y = np.array([0] * 80 + [1] * 20)  # neg/pos = 4.0
    apply_imbalance_handling(model, spec, y)
    assert model.get_params()["scale_pos_weight"] == pytest.approx(4.0)


def test_apply_imbalance_handling_ignores_multiclass():
    registry = build_default_registry()
    spec = registry.get(ProblemType.CLASSIFICATION, "XGBClassifier")
    model = registry.create_instance(ProblemType.CLASSIFICATION, "XGBClassifier")
    before = model.get_params().get("scale_pos_weight")
    apply_imbalance_handling(model, spec, np.array([0, 1, 2, 0, 1, 2]))
    assert model.get_params().get("scale_pos_weight") == before


def test_fit_model_early_stops_booster():
    pytest.importorskip("xgboost")
    registry = build_default_registry()
    spec = registry.get(ProblemType.CLASSIFICATION, "XGBClassifier")
    assert spec.supports_early_stopping
    model = registry.create_instance(ProblemType.CLASSIFICATION, "XGBClassifier")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5)); y = (X[:, 0] > 0).astype(int)
    Xv = rng.normal(size=(100, 5)); yv = (Xv[:, 0] > 0).astype(int)
    fit_model(model, spec, X, y, eval_X=Xv, eval_y=yv)
    trees = get_fitted_n_estimators(model)
    # Trivially separable → early stopping should use far fewer than the 2000 cap.
    assert trees is not None and trees < 2000


def test_fit_model_plain_for_non_booster():
    spec = ModelSpec(name="rf", problem_type=ProblemType.CLASSIFICATION,
                     factory=RandomForestClassifier)
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    X = np.random.RandomState(0).normal(size=(50, 3)); y = (X[:, 0] > 0).astype(int)
    out = fit_model(model, spec, X, y, eval_X=X, eval_y=y)
    assert out.n_estimators == 10  # untouched, no early stopping


def test_svc_skipped_on_large_data():
    registry = build_default_registry()
    svc = registry.get(ProblemType.CLASSIFICATION, "SVC")
    assert svc.max_train_samples is not None and svc.max_train_samples < 100000
