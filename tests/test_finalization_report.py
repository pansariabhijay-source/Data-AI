"""Smoke test for markdown report generation including new test-performance section."""

from __future__ import annotations

from pathlib import Path

from agents.finalization.tools import FinalizationService
from core.state import ModelResult, PipelineState


def _state() -> PipelineState:
    return PipelineState(
        run_id="t1",
        problem_type="classification",
        target_column="is_fraud",
        best_model_name="XGBClassifier",
        best_metric_name="f1",
        best_metric_value=0.86,
        best_threshold=0.73,
        test_metrics={"f1": 0.85, "precision": 0.9, "recall": 0.8, "roc_auc": 0.98, "pr_auc": 0.84},
        test_confusion={"tn": 5998, "fp": 2, "fn": 13, "tp": 61},
        model_results=[
            ModelResult(model_name="XGBClassifier", status="trained", is_best=True,
                        metrics={"f1": 0.86, "precision": 0.91, "recall": 0.82, "roc_auc": 0.98, "pr_auc": 0.85},
                        decision_threshold=0.73, training_time_seconds=1.2),
            ModelResult(model_name="VotingEnsemble", status="trained",
                        metrics={"f1": 0.85, "precision": 0.9, "recall": 0.8, "roc_auc": 0.98, "pr_auc": 0.84},
                        training_time_seconds=0.3),
        ],
    )


def test_report_renders_test_section(tmp_path: Path):
    svc = FinalizationService()
    out = tmp_path / "report.md"
    svc._generate_markdown_report(_state(), out)
    text = out.read_text(encoding="utf-8")
    assert "Held-Out Test Performance" in text
    assert "Confusion Matrix" in text
    assert "PR_AUC" in text  # PR-AUC surfaced in the metrics table
    assert "VotingEnsemble" in text
    assert "0.73" in text  # tuned threshold mentioned
