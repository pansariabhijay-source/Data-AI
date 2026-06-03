"""Tests for model registry."""

from __future__ import annotations

import pytest
from core.constants import ProblemType
from core.model_registry import build_default_registry, ModelRegistry


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
