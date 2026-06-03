"""
Model Training Tool — train, evaluate, and rank models using the model registry.

Supports parallel training, automatic metric selection, and per-model timing.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import psutil

from core.config import Settings, TrainingConfig
from core.constants import ProblemType
from core.exceptions import TrainingError
from core.logging_config import get_logger, log_stage_timing
from core.metrics import compute_metrics, get_primary_metric, is_metric_higher_better
from core.model_registry import ModelRegistry, ModelSpec, build_default_registry
from core.state import ErrorReport, ExperimentRecord, ModelResult, PipelineState
from core.utils import ensure_directory, get_timestamp

logger = get_logger("training")


class ModelTrainingService:
    def __init__(self, config: TrainingConfig, registry: ModelRegistry, seed: int = 42) -> None:
        self._config = config
        self._registry = registry
        self._seed = seed

    def _train_single_model(
        self, spec: ModelSpec, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, problem_type: ProblemType,
        artifact_dir: Path,
    ) -> ModelResult:
        """Train a single model and return its results."""
        start = time.perf_counter()
        mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

        try:
            model = self._registry.create_instance(problem_type, spec.name)
            if problem_type == ProblemType.CLUSTERING:
                model.fit(X_train)
                preds = model.predict(X_train) if hasattr(model, "predict") else model.labels_
                metrics = compute_metrics(X_train, preds, problem_type)
                train_metrics = metrics.copy()
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                y_prob = None
                if spec.supports_probabilities and hasattr(model, "predict_proba"):
                    try:
                        y_prob = model.predict_proba(X_val)
                    except Exception:
                        pass
                metrics = compute_metrics(y_val, preds, problem_type, y_prob)
                train_preds = model.predict(X_train)
                train_metrics = compute_metrics(y_train, train_preds, problem_type)

            elapsed = time.perf_counter() - start
            mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

            # Save model
            model_path = artifact_dir / f"{spec.name}.joblib"
            joblib.dump(model, model_path)

            result = ModelResult(
                model_name=spec.name, model_type=spec.problem_type.value,
                metrics=metrics, train_metrics=train_metrics,
                training_time_seconds=round(elapsed, 3),
                memory_usage_mb=round(mem_after - mem_before, 2),
                model_path=str(model_path),
                hyperparameters=spec.default_params,
                status="trained",
            )
            logger.info(f"{spec.name}: {metrics} ({elapsed:.2f}s)")
            return result

        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"{spec.name} training failed: {e}")
            return ModelResult(
                model_name=spec.name, model_type=spec.problem_type.value,
                training_time_seconds=round(elapsed, 3),
                status="failed",
                hyperparameters=spec.default_params,
            )

    @log_stage_timing("model_training")
    def run(self, state: PipelineState, settings: Settings) -> PipelineState:
        problem_type = ProblemType(state.problem_type) if state.problem_type else ProblemType.UNKNOWN
        if problem_type == ProblemType.UNKNOWN:
            raise TrainingError(
                "Problem type is unknown — cannot select models to train. "
                "Either include 'data_collection' before 'training' in your workflow, "
                "or set problem_type to 'classification', 'regression', or 'clustering' in your workflow config."
            )
        specs = self._registry.list_models(problem_type)
        if not specs:
            raise TrainingError(f"No models registered for {problem_type.value}")

        # Load data
        train_df = pd.read_csv(state.train_path, low_memory=False)
        target = state.target_column

        if problem_type == ProblemType.CLUSTERING:
            X_train = train_df.select_dtypes(include=[np.number]).values
            y_train = np.zeros(len(X_train))  # placeholder
            X_val, y_val = X_train, y_train
        else:
            if not target or target not in train_df.columns:
                raise TrainingError(f"Target '{target}' not in training data")
            X_train = train_df.drop(columns=[target]).select_dtypes(include=[np.number]).values
            y_train = train_df[target].values

            if state.val_path:
                val_df = pd.read_csv(state.val_path, low_memory=False)
                X_val = val_df.drop(columns=[target]).select_dtypes(include=[np.number]).values
                y_val = val_df[target].values
            else:
                X_val, y_val = X_train, y_train

        artifact_dir = ensure_directory(Path(settings.pipeline.artifact_dir) / state.run_id / "models")

        # Train models (parallel for CPU-bound, sequential fallback)
        results: list[ModelResult] = []
        n_workers = min(len(specs), max(1, os.cpu_count() or 1))

        if n_workers > 1 and len(specs) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        self._train_single_model, spec, X_train, y_train, X_val, y_val, problem_type, artifact_dir
                    ): spec.name
                    for spec in specs
                }
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            for spec in specs:
                results.append(self._train_single_model(spec, X_train, y_train, X_val, y_val, problem_type, artifact_dir))

        # Select best model
        primary_metric = get_primary_metric(problem_type)
        higher_better = is_metric_higher_better(primary_metric)
        trained = [r for r in results if r.status == "trained" and primary_metric in r.metrics]

        if trained:
            best = max(trained, key=lambda r: r.metrics[primary_metric]) if higher_better else min(trained, key=lambda r: r.metrics[primary_metric])
            best.is_best = True
            state.best_model_name = best.model_name
            state.best_model_path = best.model_path
            state.best_metric_name = primary_metric
            state.best_metric_value = best.metrics[primary_metric]
            logger.info(f"Best model: {best.model_name} ({primary_metric}={best.metrics[primary_metric]:.4f})")

        state.model_results = results
        state.experiment_history.append(ExperimentRecord(
            iteration=state.retry_count,
            best_model=state.best_model_name or "",
            best_metric_name=primary_metric,
            best_metric_value=state.best_metric_value or 0.0,
            model_results=results,
        ))
        state.mark_stage_end("model_training")
        return state


_service: Optional[ModelTrainingService] = None
_state: Optional[PipelineState] = None
_settings: Optional[Settings] = None


def init_training(state: PipelineState, settings: Settings) -> None:
    global _service, _state, _settings
    registry = build_default_registry(settings.pipeline.random_seed)
    _service = ModelTrainingService(settings.training, registry, settings.pipeline.random_seed)
    _state = state
    _settings = settings


def train_models(instruction: str) -> str:
    """Train all registered models and select the best one.

    Trains classification, regression, or clustering models based on detected
    problem type. Evaluates on validation set and ranks by primary metric.

    Args:
        instruction: Description of training task.

    Returns:
        JSON summary of training results with best model info.
    """
    global _service, _state, _settings
    if _service is None or _state is None or _settings is None:
        return json.dumps({"error": "Training service not initialized"})
    try:
        _state.mark_stage_start("model_training")
        _state = _service.run(_state, _settings)
        trained = [r for r in _state.model_results if r.status == "trained"]
        return json.dumps({
            "status": "success",
            "models_trained": len(trained),
            "models_failed": len(_state.model_results) - len(trained),
            "best_model": _state.best_model_name,
            "best_metric": _state.best_metric_name,
            "best_value": _state.best_metric_value,
            "all_results": [{
                "name": r.model_name, "status": r.status,
                "metrics": r.metrics, "time_s": r.training_time_seconds,
            } for r in _state.model_results],
        }, default=str)
    except Exception as e:
        logger.exception("Model training failed")
        _state.add_error(ErrorReport(
            severity="critical", stage="model_training",
            error_type=type(e).__name__, root_cause=str(e),
            traceback_str=traceback.format_exc(),
            recommended_fix="Check data dimensions and model compatibility",
            retryable=True,
        ))
        return json.dumps({"error": str(e)})
