"""
Custom exception hierarchy for the autonomous data science pipeline.

Design principles:
- Every stage has a dedicated exception so callers can handle per-stage failures.
- `RetryableError` is a mixin; combine with stage errors for retry-eligible failures.
- All exceptions carry structured context (stage, detail, cause) for observability.
"""

from __future__ import annotations

from typing import Any, Optional

from core.constants import PipelineStage


class PipelineError(Exception):
    """Root exception for all pipeline failures.

    Attributes:
        stage: The pipeline stage where the error occurred.
        detail: Human-readable explanation of what went wrong.
        context: Arbitrary structured data for debugging.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: Optional[PipelineStage] = None,
        detail: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.detail = detail or message
        self.context = context or {}

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.stage:
            parts.append(f"stage={self.stage.value}")
        if self.context:
            parts.append(f"context={self.context}")
        return " | ".join(parts)


class RetryableError(PipelineError):
    """Mixin-style exception indicating the failure is transient and retryable."""

    pass


# ── Stage-specific exceptions ───────────────────────────────────────────────


class ConfigurationError(PipelineError):
    """Invalid or missing configuration."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage=None, **kwargs)


class DataCollectionError(PipelineError):
    """Failure during data ingestion or validation."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage=PipelineStage.DATA_COLLECTION, **kwargs)


class PreprocessingError(PipelineError):
    """Failure during data cleaning or preprocessing."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage=PipelineStage.PREPROCESSING, **kwargs)


class FeatureEngineeringError(PipelineError):
    """Failure during feature engineering or selection."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage=PipelineStage.FEATURE_ENGINEERING, **kwargs)


class SplittingError(PipelineError):
    """Failure during data splitting."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage=PipelineStage.DATA_SPLITTING, **kwargs)


class TrainingError(PipelineError):
    """Failure during model training."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage=PipelineStage.MODEL_TRAINING, **kwargs)


class ErrorDetectionError(PipelineError):
    """Failure during error detection / auditing."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage=PipelineStage.ERROR_DETECTION, **kwargs)


class ImprovementError(PipelineError):
    """Failure during improvement / retry loop."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage=PipelineStage.IMPROVEMENT, **kwargs)


class FinalizationError(PipelineError):
    """Failure during finalization / artifact saving."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, stage=PipelineStage.FINALIZATION, **kwargs)


class DataValidationError(PipelineError):
    """Data fails schema or quality validation checks."""

    pass


class ModelRegistryError(PipelineError):
    """Failure in model registry operations (lookup, registration)."""

    pass
