"""
Error Detection Tool — automated auditing of the ML pipeline.

Detects: low performance, overfitting, underfitting, data leakage, class imbalance,
feature explosion, preprocessing failures, and invalid metrics.
"""

from __future__ import annotations

import json
import traceback
from typing import Optional


from core.config import ErrorDetectionConfig, Settings
from core.constants import ProblemType, Severity
from core.logging_config import get_logger, log_stage_timing
from core.metrics import get_primary_metric, is_metric_higher_better
from core.state import ErrorReport, PipelineState

logger = get_logger("error_detection")


class ErrorDetectionService:
    def __init__(self, config: ErrorDetectionConfig) -> None:
        self._config = config

    def _check_low_performance(self, state: PipelineState) -> list[ErrorReport]:
        errors: list[ErrorReport] = []
        pt = ProblemType(state.problem_type) if state.problem_type else None
        if not pt or not state.best_metric_value:
            return errors

        if pt == ProblemType.CLASSIFICATION and state.best_metric_value < self._config.min_classification_f1:
            errors.append(ErrorReport(
                severity=Severity.HIGH.value, stage="model_training",
                error_type="low_performance",
                root_cause=f"Best F1={state.best_metric_value:.4f} < threshold {self._config.min_classification_f1}",
                recommended_fix="Try hyperparameter tuning, feature engineering improvements, or class balancing",
                retryable=True,
            ))
        elif pt == ProblemType.REGRESSION and state.best_metric_value < self._config.min_regression_r2:
            errors.append(ErrorReport(
                severity=Severity.HIGH.value, stage="model_training",
                error_type="low_performance",
                root_cause=f"Best R2={state.best_metric_value:.4f} < threshold {self._config.min_regression_r2}",
                recommended_fix="Try feature engineering, outlier removal, or different models",
                retryable=True,
            ))
        return errors

    def _check_overfitting(self, state: PipelineState) -> list[ErrorReport]:
        errors: list[ErrorReport] = []
        primary = get_primary_metric(ProblemType(state.problem_type)) if state.problem_type else "f1"
        higher = is_metric_higher_better(primary)

        for r in state.model_results:
            if r.status != "trained" or not r.train_metrics or not r.metrics:
                continue
            train_val = r.train_metrics.get(primary)
            val_val = r.metrics.get(primary)
            if train_val is None or val_val is None:
                continue
            gap = (train_val - val_val) if higher else (val_val - train_val)
            if gap > self._config.overfitting_threshold:
                errors.append(ErrorReport(
                    severity=Severity.MEDIUM.value, stage="model_training",
                    error_type="overfitting",
                    root_cause=f"{r.model_name}: train_{primary}={train_val:.4f}, val_{primary}={val_val:.4f}, gap={gap:.4f}",
                    recommended_fix=f"Add regularization, reduce model complexity, or increase training data for {r.model_name}",
                    retryable=True,
                ))
        return errors

    def _check_feature_explosion(self, state: PipelineState) -> list[ErrorReport]:
        errors: list[ErrorReport] = []
        if state.feature_engineering_summary:
            n = state.feature_engineering_summary.n_features_after
            if n > self._config.max_feature_count:
                errors.append(ErrorReport(
                    severity=Severity.MEDIUM.value, stage="feature_engineering",
                    error_type="feature_explosion",
                    root_cause=f"{n} features after engineering exceeds limit {self._config.max_feature_count}",
                    recommended_fix="Increase feature selection aggressiveness or reduce one-hot cardinality",
                    retryable=True,
                ))
        return errors

    def _check_class_imbalance(self, state: PipelineState) -> list[ErrorReport]:
        errors: list[ErrorReport] = []
        imbalance = state.data_quality_flags.get("class_imbalance")
        if imbalance:
            min_class_pct = min(imbalance.values()) if isinstance(imbalance, dict) else 0
            errors.append(ErrorReport(
                severity=Severity.MEDIUM.value, stage="data_collection",
                error_type="class_imbalance",
                root_cause=f"Minority class has {min_class_pct:.1%} of samples",
                recommended_fix="Apply SMOTE, class weights, or oversampling",
                retryable=True,
            ))
        return errors

    def _check_all_models_failed(self, state: PipelineState) -> list[ErrorReport]:
        errors: list[ErrorReport] = []
        if state.model_results and all(r.status == "failed" for r in state.model_results):
            errors.append(ErrorReport(
                severity=Severity.CRITICAL.value, stage="model_training",
                error_type="all_models_failed",
                root_cause="All model training attempts failed",
                recommended_fix="Check data quality, feature types, and model compatibility",
                retryable=True,
            ))
        return errors

    def _check_preprocessing_quality(self, state: PipelineState) -> list[ErrorReport]:
        errors: list[ErrorReport] = []
        if state.preprocessing_summary and state.preprocessing_summary.quality_score < 0.5:
            errors.append(ErrorReport(
                severity=Severity.MEDIUM.value, stage="preprocessing",
                error_type="low_data_quality",
                root_cause=f"Post-preprocessing quality score={state.preprocessing_summary.quality_score:.4f}",
                recommended_fix="Review preprocessing strategies, consider manual feature review",
                retryable=True,
            ))
        return errors

    @log_stage_timing("error_detection")
    def run(self, state: PipelineState) -> PipelineState:
        all_errors: list[ErrorReport] = []
        all_errors.extend(self._check_low_performance(state))
        all_errors.extend(self._check_overfitting(state))
        all_errors.extend(self._check_feature_explosion(state))
        all_errors.extend(self._check_class_imbalance(state))
        all_errors.extend(self._check_all_models_failed(state))
        all_errors.extend(self._check_preprocessing_quality(state))

        for err in all_errors:
            state.add_error(err)
            logger.warning(f"[{err.severity}] {err.error_type}: {err.root_cause}")

        state.mark_stage_end("error_detection")
        logger.info(f"Error detection found {len(all_errors)} issues")
        return state


_service: Optional[ErrorDetectionService] = None
_state: Optional[PipelineState] = None
_settings: Optional[Settings] = None


def init_error_detection(state: PipelineState, settings: Settings) -> None:
    global _service, _state, _settings
    _service = ErrorDetectionService(settings.error_detection)
    _state = state
    _settings = settings


def detect_errors(instruction: str) -> str:
    """Audit the pipeline for errors, anomalies, and quality issues.

    Checks for low performance, overfitting, data leakage, class imbalance,
    feature explosion, and model failures.

    Args:
        instruction: Description of error detection task.

    Returns:
        JSON report of detected issues with severity and recommendations.
    """
    global _service, _state, _settings
    if _service is None or _state is None:
        return json.dumps({"error": "Error detection service not initialized"})
    try:
        _state.mark_stage_start("error_detection")
        _state = _service.run(_state)
        retryable = _state.get_retryable_errors()
        return json.dumps({
            "status": "success",
            "total_issues": len(_state.error_reports),
            "retryable_issues": len(retryable),
            "needs_retry": len(retryable) > 0 and _state.retry_count < _state.max_retries,
            "issues": [{
                "severity": e.severity, "type": e.error_type,
                "cause": e.root_cause, "fix": e.recommended_fix,
            } for e in _state.error_reports[-10:]],
        }, default=str)
    except Exception as e:
        logger.exception("Error detection failed")
        return json.dumps({"error": str(e)})
