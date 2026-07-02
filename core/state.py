"""
Pipeline state — single source of truth for the entire pipeline run.

Design:
- Pydantic BaseModel for validation + serialization safety
- Nested schemas for structured sub-objects
- Survives retries, supports checkpointing, recovery, and debugging
- Never mutated by LLM — only by tool functions
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    file_path: str = ""
    n_rows: int = 0
    n_columns: int = 0
    column_names: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    memory_usage_mb: float = 0.0
    null_counts: dict[str, int] = Field(default_factory=dict)
    null_percentages: dict[str, float] = Field(default_factory=dict)
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    datetime_columns: list[str] = Field(default_factory=list)
    boolean_columns: list[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    unique_counts: dict[str, int] = Field(default_factory=dict)


class PreprocessingSummary(BaseModel):
    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0
    duplicates_removed: int = 0
    nulls_filled: dict[str, str] = Field(default_factory=dict)
    outliers_handled: dict[str, int] = Field(default_factory=dict)
    dtypes_fixed: dict[str, str] = Field(default_factory=dict)
    high_cardinality_columns: list[str] = Field(default_factory=list)
    columns_dropped: list[str] = Field(default_factory=list)
    quality_score: float = 0.0


class FeatureEngineeringSummary(BaseModel):
    features_created: list[str] = Field(default_factory=list)
    features_removed: list[str] = Field(default_factory=list)
    # Human-readable reason per removed feature (col -> why), for the report's
    # explanation section. Covers leakage / ID / (near-)constant / correlated drops.
    removal_reasons: dict[str, str] = Field(default_factory=dict)
    encoding_applied: dict[str, str] = Field(default_factory=dict)
    scaling_applied: dict[str, str] = Field(default_factory=dict)
    feature_importances: dict[str, float] = Field(default_factory=dict)
    selected_features: list[str] = Field(default_factory=list)
    n_features_before: int = 0
    n_features_after: int = 0


class ModelResult(BaseModel):
    model_name: str = ""
    model_type: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    train_metrics: dict[str, float] = Field(default_factory=dict)
    training_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    model_path: Optional[str] = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    is_best: bool = False
    status: str = "pending"
    # F1-optimal decision threshold on positive-class probability (binary only).
    # None means the default 0.5 / argmax decision rule was used.
    decision_threshold: Optional[float] = None


class ErrorReport(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    severity: str = "medium"
    stage: str = ""
    error_type: str = ""
    root_cause: str = ""
    traceback_str: Optional[str] = None
    recommended_fix: str = ""
    retryable: bool = False


class ExperimentRecord(BaseModel):
    iteration: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    best_model: str = ""
    best_metric_name: str = ""
    best_metric_value: float = 0.0
    improvements_applied: list[str] = Field(default_factory=list)
    model_results: list[ModelResult] = Field(default_factory=list)


class PipelineState(BaseModel):
    """Root state object tracking the entire pipeline lifecycle."""

    # Identity
    run_id: str = ""
    problem_type: Optional[str] = None
    target_column: Optional[str] = None

    # Dataset
    dataset_metadata: Optional[DatasetMetadata] = None

    # Paths
    raw_data_path: Optional[str] = None
    cleaned_data_path: Optional[str] = None
    featured_data_path: Optional[str] = None
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    split_ratios: dict[str, float] = Field(default_factory=dict)

    # Features
    selected_features: list[str] = Field(default_factory=list)
    feature_importances: dict[str, float] = Field(default_factory=dict)

    # Models
    model_results: list[ModelResult] = Field(default_factory=list)
    best_model_name: Optional[str] = None
    best_model_path: Optional[str] = None
    best_metric_name: Optional[str] = None
    best_metric_value: Optional[float] = None
    best_threshold: Optional[float] = None
    # Honest held-out test-set evaluation of the champion model (computed once,
    # at the chosen decision threshold). Empty until the test split is scored.
    test_metrics: dict[str, float] = Field(default_factory=dict)
    # 95% bootstrap confidence interval [low, high] per test metric — conveys how
    # much the single-slice test score could vary, so a point estimate isn't read
    # as more precise than it is.
    test_metric_cis: dict[str, list[float]] = Field(default_factory=dict)
    test_confusion: dict[str, int] = Field(default_factory=dict)
    # Precision/recall at fixed alert rates on the test set (binary classification),
    # for choosing an operating point given review capacity (fraud use case).
    test_operating_points: list[dict[str, Any]] = Field(default_factory=list)

    # Summaries
    preprocessing_summary: Optional[PreprocessingSummary] = None
    feature_engineering_summary: Optional[FeatureEngineeringSummary] = None

    # Errors
    error_reports: list[ErrorReport] = Field(default_factory=list)

    # Retry
    retry_count: int = 0
    max_retries: int = 3
    experiment_history: list[ExperimentRecord] = Field(default_factory=list)

    # Quality
    data_quality_flags: dict[str, Any] = Field(default_factory=dict)

    # Artifacts
    artifacts: dict[str, str] = Field(default_factory=dict)
    report_paths: dict[str, str] = Field(default_factory=dict)

    # Timestamps
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    stage_timestamps: dict[str, dict[str, str]] = Field(default_factory=dict)

    # Status
    current_stage: Optional[str] = None
    completed_stages: list[str] = Field(default_factory=list)
    failed: bool = False
    failure_reason: Optional[str] = None

    def mark_stage_start(self, stage: str) -> None:
        self.current_stage = stage
        self.stage_timestamps.setdefault(stage, {})["start"] = datetime.now(timezone.utc).isoformat()

    def mark_stage_end(self, stage: str) -> None:
        self.stage_timestamps.setdefault(stage, {})["end"] = datetime.now(timezone.utc).isoformat()
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)

    def add_error(self, error: ErrorReport) -> None:
        self.error_reports.append(error)

    def get_retryable_errors(self) -> list[ErrorReport]:
        return [e for e in self.error_reports if e.retryable]

    def to_checkpoint(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_checkpoint(cls, json_str: str) -> PipelineState:
        return cls.model_validate_json(json_str)
