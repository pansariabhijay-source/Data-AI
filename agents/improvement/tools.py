"""
Improvement Tool — hyperparameter tuning, retry orchestration, and model improvement.

Implements RandomizedSearchCV and optional Optuna for hyperparameter optimization.
Manages retry caps, early stopping, and cross-iteration comparison.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from core.config import ImprovementConfig, Settings
from core.constants import ProblemType
from core.exceptions import ImprovementError
from core.logging_config import get_logger, log_stage_timing
from core.metrics import compute_metrics, get_primary_metric, is_metric_higher_better
from core.model_registry import build_default_registry
from core.state import ErrorReport, ExperimentRecord, ModelResult, PipelineState
from core.utils import ensure_directory

logger = get_logger("improvement")


class ImprovementService:
    def __init__(self, config: ImprovementConfig, seed: int = 42) -> None:
        self._config = config
        self._seed = seed

    def _tune_with_randomized_search(
        self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
        problem_type: ProblemType, seed: int,
    ) -> tuple[Any, dict[str, Any]]:
        """Tune a model using RandomizedSearchCV."""
        from sklearn.model_selection import RandomizedSearchCV
        registry = build_default_registry(seed)
        spec = registry.get(problem_type, model_name)
        model = registry.create_instance(problem_type, model_name)

        if not spec.search_space:
            logger.info(f"No search space for {model_name}, skipping tuning")
            model.fit(X_train, y_train)
            return model, spec.default_params

        primary = get_primary_metric(problem_type)
        scoring_map = {"f1": "f1_weighted", "r2": "r2", "rmse": "neg_root_mean_squared_error",
                        "accuracy": "accuracy", "silhouette_score": None}
        scoring = scoring_map.get(primary, "f1_weighted")

        if scoring is None:
            model.fit(X_train, y_train)
            return model, spec.default_params

        n_iter = min(self._config.tuning_iterations, _count_combinations(spec.search_space))
        search = RandomizedSearchCV(
            model, spec.search_space, n_iter=n_iter, scoring=scoring,
            cv=3, random_state=seed, n_jobs=-1, error_score="raise",
        )
        search.fit(X_train, y_train)
        logger.info(f"{model_name} best params: {search.best_params_} (score={search.best_score_:.4f})")
        return search.best_estimator_, search.best_params_

    def _try_optuna(
        self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, problem_type: ProblemType,
    ) -> tuple[Optional[Any], dict[str, Any]]:
        """Optuna hyperparameter optimization (if enabled and installed)."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.info("Optuna not installed, skipping")
            return None, {}

        registry = build_default_registry(self._seed)
        spec = registry.get(problem_type, model_name)
        if not spec.search_space:
            return None, {}

        primary = get_primary_metric(problem_type)
        higher = is_metric_higher_better(primary)

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for key, values in spec.search_space.items():
                if all(isinstance(v, (int, float)) for v in values):
                    if all(isinstance(v, int) for v in values):
                        params[key] = trial.suggest_int(key, min(values), max(values))
                    else:
                        params[key] = trial.suggest_float(key, min(values), max(values), log=True)
                else:
                    params[key] = trial.suggest_categorical(key, values)

            merged = {**spec.default_params, **params}
            model = spec.factory(**merged)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            metrics = compute_metrics(y_val, preds, problem_type)
            return metrics.get(primary, 0.0)

        direction = "maximize" if higher else "minimize"
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=self._config.tuning_iterations, timeout=300)

        best_params = {**spec.default_params, **study.best_params}
        best_model = spec.factory(**best_params)
        best_model.fit(X_train, y_train)
        logger.info(f"Optuna best for {model_name}: {study.best_value:.4f}")
        return best_model, best_params

    @log_stage_timing("improvement")
    def run(self, state: PipelineState, settings: Settings) -> PipelineState:
        if state.retry_count >= state.max_retries:
            logger.warning(f"Max retries ({state.max_retries}) reached, skipping improvement")
            state.mark_stage_end("improvement")
            return state

        problem_type = ProblemType(state.problem_type) if state.problem_type else ProblemType.UNKNOWN
        target = state.target_column

        if problem_type == ProblemType.CLUSTERING:
            state.mark_stage_end("improvement")
            return state

        # Load data
        train_df = pd.read_csv(state.train_path, low_memory=False)
        X_train = train_df.drop(columns=[target]).select_dtypes(include=[np.number]).values
        y_train = train_df[target].values

        X_val, y_val = X_train, y_train
        if state.val_path:
            val_df = pd.read_csv(state.val_path, low_memory=False)
            X_val = val_df.drop(columns=[target]).select_dtypes(include=[np.number]).values
            y_val = val_df[target].values

        # Find best model to tune
        best = state.best_model_name
        if not best:
            state.mark_stage_end("improvement")
            return state

        artifact_dir = ensure_directory(Path(settings.pipeline.artifact_dir) / state.run_id / "models")
        primary = get_primary_metric(problem_type)
        prev_best = state.best_metric_value or 0.0

        improvements: list[str] = []

        # Try tuning
        if self._config.use_optuna:
            tuned_model, best_params = self._try_optuna(best, X_train, y_train, X_val, y_val, problem_type)
        else:
            tuned_model, best_params = self._tune_with_randomized_search(best, X_train, y_train, problem_type, self._seed)

        if tuned_model is not None:
            preds = tuned_model.predict(X_val)
            y_prob = None
            if hasattr(tuned_model, "predict_proba"):
                try:
                    y_prob = tuned_model.predict_proba(X_val)
                except Exception:
                    pass
            metrics = compute_metrics(y_val, preds, problem_type, y_prob)
            new_val = metrics.get(primary, 0.0)

            higher = is_metric_higher_better(primary)
            improved = (new_val > prev_best) if higher else (new_val < prev_best)

            if improved:
                model_path = artifact_dir / f"{best}_tuned_iter{state.retry_count}.joblib"
                joblib.dump(tuned_model, model_path)
                state.best_model_path = str(model_path)
                state.best_metric_value = new_val
                improvements.append(f"Tuned {best}: {primary} {prev_best:.4f}->{new_val:.4f}")
                logger.info(f"Improvement found: {primary} {prev_best:.4f}->{new_val:.4f}")

                # Update model results
                state.model_results.append(ModelResult(
                    model_name=f"{best}_tuned",
                    model_type=problem_type.value,
                    metrics=metrics, hyperparameters=best_params,
                    model_path=str(model_path), is_best=True, status="trained",
                ))
            else:
                logger.info(f"No improvement: {primary} {new_val:.4f} vs {prev_best:.4f}")
                improvements.append(f"Tuning attempted but no improvement for {best}")

        state.retry_count += 1
        state.experiment_history.append(ExperimentRecord(
            iteration=state.retry_count,
            best_model=state.best_model_name or "",
            best_metric_name=primary,
            best_metric_value=state.best_metric_value or 0.0,
            improvements_applied=improvements,
        ))
        state.mark_stage_end("improvement")
        return state


def _count_combinations(space: dict[str, list]) -> int:
    n = 1
    for v in space.values():
        n *= len(v)
    return n


_service: Optional[ImprovementService] = None
_state: Optional[PipelineState] = None
_settings: Optional[Settings] = None


def init_improvement(state: PipelineState, settings: Settings) -> None:
    global _service, _state, _settings
    _service = ImprovementService(settings.improvement, settings.pipeline.random_seed)
    _state = state
    _settings = settings


def improve_pipeline(instruction: str) -> str:
    """Improve pipeline performance through hyperparameter tuning.

    Uses RandomizedSearchCV or Optuna to tune the best model.
    Respects retry caps and tracks improvement history.

    Args:
        instruction: Description of improvement task.

    Returns:
        JSON summary of improvement results.
    """
    global _service, _state, _settings
    if _service is None or _state is None or _settings is None:
        return json.dumps({"error": "Improvement service not initialized"})
    try:
        _state.mark_stage_start("improvement")
        _state = _service.run(_state, _settings)
        return json.dumps({
            "status": "success",
            "retry_count": _state.retry_count,
            "max_retries": _state.max_retries,
            "best_model": _state.best_model_name,
            "best_metric_value": _state.best_metric_value,
            "history": [{
                "iteration": e.iteration,
                "best_value": e.best_metric_value,
                "improvements": e.improvements_applied,
            } for e in _state.experiment_history[-5:]],
        }, default=str)
    except Exception as e:
        logger.exception("Improvement failed")
        _state.add_error(ErrorReport(
            severity="high", stage="improvement",
            error_type=type(e).__name__, root_cause=str(e),
            traceback_str=traceback.format_exc(),
            recommended_fix="Check model compatibility and search space",
            retryable=False,
        ))
        return json.dumps({"error": str(e)})
