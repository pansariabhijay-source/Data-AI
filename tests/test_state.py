"""Tests for core.state module."""

from __future__ import annotations

import json

from core.state import (
    DatasetMetadata,
    ErrorReport,
    ExperimentRecord,
    ModelResult,
    PipelineState,
    PreprocessingSummary,
)


def test_pipeline_state_creation():
    state = PipelineState(run_id="test_run_001")
    assert state.run_id == "test_run_001"
    assert state.problem_type is None
    assert state.retry_count == 0
    assert state.failed is False
    assert state.completed_stages == []


def test_state_serialization_roundtrip():
    state = PipelineState(
        run_id="test_123",
        problem_type="classification",
        target_column="species",
        best_model_name="RandomForest",
        best_metric_value=0.95,
    )
    state.model_results.append(ModelResult(
        model_name="RandomForest", model_type="classification",
        metrics={"f1": 0.95, "accuracy": 0.96}, status="trained",
    ))
    state.add_error(ErrorReport(
        severity="medium", stage="training",
        error_type="overfitting", root_cause="Train-val gap > 0.15",
    ))

    json_str = state.to_checkpoint()
    restored = PipelineState.from_checkpoint(json_str)

    assert restored.run_id == "test_123"
    assert restored.best_model_name == "RandomForest"
    assert len(restored.model_results) == 1
    assert restored.model_results[0].metrics["f1"] == 0.95
    assert len(restored.error_reports) == 1


def test_mark_stage_lifecycle():
    state = PipelineState(run_id="test")
    state.mark_stage_start("preprocessing")
    assert state.current_stage == "preprocessing"
    assert "preprocessing" in state.stage_timestamps
    assert "start" in state.stage_timestamps["preprocessing"]

    state.mark_stage_end("preprocessing")
    assert "preprocessing" in state.completed_stages
    assert "end" in state.stage_timestamps["preprocessing"]


def test_get_retryable_errors():
    state = PipelineState(run_id="test")
    state.add_error(ErrorReport(severity="high", error_type="low_perf", retryable=True))
    state.add_error(ErrorReport(severity="low", error_type="info", retryable=False))
    assert len(state.get_retryable_errors()) == 1


def test_nested_models_serialize():
    meta = DatasetMetadata(n_rows=1000, n_columns=10, column_names=["a", "b"])
    summary = PreprocessingSummary(rows_before=1000, rows_after=950, quality_score=0.85)
    state = PipelineState(
        run_id="test", dataset_metadata=meta, preprocessing_summary=summary,
    )
    data = json.loads(state.to_checkpoint())
    assert data["dataset_metadata"]["n_rows"] == 1000
    assert data["preprocessing_summary"]["quality_score"] == 0.85
