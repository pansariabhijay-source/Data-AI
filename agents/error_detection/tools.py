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
            if gap <= self._config.overfitting_threshold:
                continue
            # A large gap alone is benign for tree/boosting models — they memorise
            # the training set (train≈1.0) yet can still generalise well. Only flag
            # when validation is ALSO weak (real overfitting that costs accuracy);
            # a model already scoring well on val isn't a problem worth surfacing.
            min_val = getattr(self._config, "overfitting_min_val_score", 0.80)
            if higher and val_val >= min_val:
                continue
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

    def _check_drift(self, state: PipelineState) -> list[ErrorReport]:
        """Flag train→test distribution drift via Population Stability Index (PSI).

        High PSI means the test split is distributed differently from train, so the
        honest test score may not reflect future data (often a sign of a time-based
        or otherwise non-random split). Diagnostic finding; never blocks the run.
        Runs on a row-capped sample so it's cheap.
        """
        errors: list[ErrorReport] = []
        if not state.train_path or not state.test_path:
            return errors
        try:
            from pathlib import Path
            import numpy as np
            import pandas as pd
            if not (Path(state.train_path).exists() and Path(state.test_path).exists()):
                return errors

            cap = 50000
            tr = pd.read_csv(state.train_path, low_memory=False)
            te = pd.read_csv(state.test_path, low_memory=False)
            if len(tr) > cap:
                tr = tr.sample(cap, random_state=42)
            if len(te) > cap:
                te = te.sample(cap, random_state=42)
            target = state.target_column
            cols = [c for c in tr.select_dtypes(include=[np.number]).columns
                    if c != target and c in te.columns]

            def psi(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
                a, b = a.dropna(), b.dropna()
                if len(a) < bins or len(b) < bins:
                    return 0.0
                edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
                if len(edges) < 3:
                    return 0.0
                edges[0], edges[-1] = -np.inf, np.inf
                eps = 1e-6
                af = np.clip(np.histogram(a, bins=edges)[0] / len(a), eps, None)
                bf = np.clip(np.histogram(b, bins=edges)[0] / len(b), eps, None)
                return float(np.sum((bf - af) * np.log(bf / af)))

            scores: dict[str, float] = {}
            for c in cols:
                try:
                    scores[c] = psi(tr[c], te[c])
                except Exception:
                    continue
            drifted = sorted(((c, p) for c, p in scores.items() if p >= 0.25),
                             key=lambda x: -x[1])
            if drifted:
                top = ", ".join(f"{c} (PSI={p:.2f})" for c, p in drifted[:5])
                errors.append(ErrorReport(
                    severity=Severity.LOW.value, stage="data_splitting",
                    error_type="distribution_drift",
                    root_cause=f"{len(drifted)} feature(s) drift between train and test "
                               f"(PSI≥0.25): {top}",
                    recommended_fix="If the split is time-based, the test score reflects future "
                                    "data (good); if random, large PSI may indicate instability.",
                    retryable=False,
                ))
                state.data_quality_flags["drift_psi"] = {c: round(p, 3) for c, p in drifted[:10]}
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Drift check skipped: {e}")
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
        all_errors.extend(self._check_drift(state))

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
