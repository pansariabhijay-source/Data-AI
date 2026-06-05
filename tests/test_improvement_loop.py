"""Tests for the iterative hyperparameter-tuning loop control flow.

The per-round tuning is stubbed so we test the loop's stopping logic
(target reached / patience plateau / keep global best) deterministically,
without running real RandomizedSearch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("crewai", MagicMock())
sys.modules.setdefault("crewai.tools", MagicMock())

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from core.config import load_settings
from core.state import ModelResult, PipelineState
from agents.improvement.tools import ImprovementService


@pytest.fixture
def tiny_run(tmp_path):
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    df["target"] = (df["a"] + rng.normal(size=60) > 0).astype(int)
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    df.iloc[:40].to_csv(train, index=False)
    df.iloc[40:].to_csv(val, index=False)
    state = PipelineState(
        run_id="loop_test", target_column="target",
        raw_data_path=str(train), train_path=str(train), val_path=str(val),
    )
    state.problem_type = "classification"
    state.best_model_name = "LogisticRegression"
    state.best_metric_value = 0.5
    state.model_results = [ModelResult(
        model_name="LogisticRegression", model_type="classification",
        metrics={"f1": 0.5, "pr_auc": 0.5, "roc_auc": 0.5}, is_best=True, status="trained",
    )]
    return state


def _fitted_model():
    m = LogisticRegression(max_iter=200)
    m.fit(np.random.RandomState(1).normal(size=(20, 3)), [0, 1] * 10)
    return m


def _scripted_service(settings, scores):
    """Return an ImprovementService whose tuning rounds yield the given scores."""
    svc = ImprovementService(settings.improvement, seed=42)
    seq = iter(scores)

    def fake_round(*args, **kwargs):
        try:
            score = next(seq)
        except StopIteration:
            return None
        return {
            "model": _fitted_model(), "params": {"C": 1.0},
            "metrics": {"f1": score, "pr_auc": score, "roc_auc": score},
            "threshold": 0.5, "score": score,
        }

    svc._run_one_tuning_round = fake_round  # type: ignore[method-assign]
    return svc


def test_loop_stops_on_target(tiny_run):
    s = load_settings()
    s.improvement.tuning_target_metric = 0.8
    s.improvement.tuning_max_iterations = 10
    s.improvement.tuning_patience = 10
    svc = _scripted_service(s, [0.6, 0.85, 0.9])  # hits target at round 2
    out = svc.run(tiny_run, s)
    msgs = out.experiment_history[-1].improvements_applied
    assert any("Reached target" in m for m in msgs)
    # stopped at round 2 — the third score is never consumed
    assert any("round 2" in m.lower() for m in msgs)
    assert out.best_metric_value == pytest.approx(0.85)


def test_loop_stops_on_patience(tiny_run):
    s = load_settings()
    s.improvement.tuning_target_metric = 0.0  # no target
    s.improvement.tuning_max_iterations = 10
    s.improvement.tuning_patience = 2
    # improve once, then two non-improving rounds -> early stop
    svc = _scripted_service(s, [0.7, 0.65, 0.66, 0.9])
    out = svc.run(tiny_run, s)
    msgs = out.experiment_history[-1].improvements_applied
    assert any("early stop" in m.lower() for m in msgs)
    assert out.best_metric_value == pytest.approx(0.7)  # 0.9 never reached


def test_loop_respects_max_iterations(tiny_run):
    s = load_settings()
    s.improvement.tuning_target_metric = 0.99  # never reached
    s.improvement.tuning_max_iterations = 3
    s.improvement.tuning_patience = 10
    svc = _scripted_service(s, [0.6, 0.7, 0.8, 0.85, 0.9])  # would keep climbing
    out = svc.run(tiny_run, s)
    # only 3 rounds run -> best is the 3rd score
    assert out.best_metric_value == pytest.approx(0.8)
