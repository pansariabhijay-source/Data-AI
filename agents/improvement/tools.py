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
from core.metrics import (
    compute_metrics,
    get_primary_metric,
    is_metric_higher_better,
    predict_with_optimal_threshold,
    selection_score,
)
from core.model_registry import apply_imbalance_handling, build_default_registry
from core.state import ErrorReport, ExperimentRecord, ModelResult, PipelineState
from core.utils import ensure_directory

logger = get_logger("improvement")


def _is_tunable_base_model(model_name: str, problem_type: ProblemType, seed: int) -> bool:
    """True only if ``model_name`` is a base model present in the registry.

    Ensembles (VotingEnsemble/StackingEnsemble) and ``*_tuned`` champions are not
    registered, so they cannot be hyperparameter-searched.
    """
    if not model_name:
        return False
    try:
        build_default_registry(seed).get(problem_type, model_name)
        return True
    except Exception:
        return False


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
        # Inject imbalance params (e.g. scale_pos_weight) — it's a fixed param, so
        # cloned estimators inside the search inherit it.
        model = apply_imbalance_handling(model, spec, y_train)

        if not spec.search_space:
            logger.info(f"No search space for {model_name}, skipping tuning")
            model.fit(X_train, y_train)
            return model, spec.default_params

        primary = get_primary_metric(problem_type)
        # For binary targets, optimise the minority-class F1 ("f1", pos_label=1)
        # rather than "f1_weighted", which is dominated by the majority class.
        binary = problem_type == ProblemType.CLASSIFICATION and len(np.unique(y_train)) == 2
        f1_scoring = "f1" if binary else "f1_weighted"
        scoring_map = {"f1": f1_scoring, "r2": "r2", "rmse": "neg_root_mean_squared_error",
                        "accuracy": "accuracy", "silhouette_score": None}
        scoring = scoring_map.get(primary, f1_scoring)

        if scoring is None:
            model.fit(X_train, y_train)
            return model, spec.default_params

        n_iter = min(self._config.tuning_iterations, _count_combinations(spec.search_space))
        # Stratified folds keep the minority class represented in every split —
        # essential for stable tuning on imbalanced targets.
        if binary or (problem_type == ProblemType.CLASSIFICATION):
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        else:
            cv = 5
        search = RandomizedSearchCV(
            model, spec.search_space, n_iter=n_iter, scoring=scoring,
            cv=cv, random_state=seed, n_jobs=-1, error_score="raise",
        )
        search.fit(X_train, y_train)
        logger.info(f"{model_name} best params: {search.best_params_} (score={search.best_score_:.4f})")
        return search.best_estimator_, search.best_params_

    def _try_optuna(
        self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, problem_type: ProblemType,
        seed: Optional[int] = None,
    ) -> tuple[Optional[Any], dict[str, Any]]:
        """Optuna hyperparameter optimization (if enabled and installed)."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.info("Optuna not installed, skipping")
            return None, {}
        if seed is None:
            seed = self._seed

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
            model = apply_imbalance_handling(spec.factory(**merged), spec, y_train)
            model.fit(X_train, y_train)
            if (
                problem_type == ProblemType.CLASSIFICATION
                and hasattr(model, "predict_proba")
                and len(np.unique(y_train)) == 2
            ):
                preds, _ = predict_with_optimal_threshold(
                    y_val, model.predict_proba(X_val), model.classes_
                )
            else:
                preds = model.predict(X_val)
            metrics = compute_metrics(y_val, preds, problem_type)
            return metrics.get(primary, 0.0)

        direction = "maximize" if higher else "minimize"
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=self._config.tuning_iterations, timeout=300)

        best_params = {**spec.default_params, **study.best_params}
        best_model = apply_imbalance_handling(spec.factory(**best_params), spec, y_train)
        best_model.fit(X_train, y_train)
        logger.info(f"Optuna best for {model_name}: {study.best_value:.4f}")
        return best_model, best_params

    def _run_one_tuning_round(
        self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, problem_type: ProblemType,
        n_classes: int, seed: int,
    ) -> Optional[dict[str, Any]]:
        """Tune ``model_name`` once and score it on the validation split.

        Returns a dict with the fitted model, its params, validation metrics, the
        F1-optimal threshold (binary only) and the selection score used to compare
        rounds — or None if tuning could not run.
        """
        # The champion may be an ensemble (VotingEnsemble/StackingEnsemble) or an
        # already-tuned model — none of which exist in the base model registry, so
        # there are no hyperparameters to grid-search. Detect that up front and
        # skip gracefully instead of letting registry.get() raise and fail the
        # whole improvement stage.
        if not _is_tunable_base_model(model_name, problem_type, self._seed):
            logger.info(
                f"'{model_name}' is not a tunable base model (e.g. an ensemble); "
                f"skipping hyperparameter search."
            )
            return None

        if self._config.use_optuna:
            tuned, params = self._try_optuna(model_name, X_train, y_train, X_val, y_val, problem_type, seed)
        else:
            tuned, params = self._tune_with_randomized_search(model_name, X_train, y_train, problem_type, seed)
        if tuned is None:
            return None

        y_prob = None
        if hasattr(tuned, "predict_proba"):
            try:
                y_prob = tuned.predict_proba(X_val)
            except Exception:
                y_prob = None

        threshold: Optional[float] = None
        if problem_type == ProblemType.CLASSIFICATION and y_prob is not None and n_classes == 2:
            preds, threshold = predict_with_optimal_threshold(y_val, y_prob, tuned.classes_)
        else:
            preds = tuned.predict(X_val)

        metrics = compute_metrics(y_val, preds, problem_type, y_prob)
        return {
            "model": tuned, "params": params, "metrics": metrics,
            "threshold": threshold, "score": selection_score(metrics, problem_type, n_classes),
        }

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
        n_classes = int(len(np.unique(y_train)))

        improvements: list[str] = []

        # The model we attempt to tune is the champion. If it's an ensemble
        # (VotingEnsemble/StackingEnsemble) it isn't a registry base model, so the
        # tuning round below detects that and skips gracefully — the ensemble is
        # already a blend of strong models, so there's no single set of
        # hyperparameters to search. This is correct, not a failure.
        tune_model = best

        # Baseline = the current champion's validation selection score (same metric
        # used to crown it during training, so the comparison is consistent).
        champ = next((r for r in state.model_results if r.is_best and r.metrics), None)
        baseline_score = (
            selection_score(champ.metrics, problem_type, n_classes)
            if champ else (state.best_metric_value or 0.0)
        )

        best_cand: Optional[dict[str, Any]] = None
        best_score = baseline_score
        target = self._config.tuning_target_metric
        max_iter = max(1, self._config.tuning_max_iterations)
        patience = max(1, self._config.tuning_patience)

        if target > 0 and baseline_score >= target:
            improvements.append(f"Champion already at target ({baseline_score:.4f} >= {target}); skipping tuning")
            logger.info(improvements[-1])
        else:
            # Iterative tuning loop: keep searching (fresh seed each round) until the
            # target score is reached, the round budget is spent, or we plateau.
            no_improve = 0
            for i in range(max_iter):
                cand = self._run_one_tuning_round(
                    tune_model, X_train, y_train, X_val, y_val, problem_type, n_classes,
                    seed=self._seed + 1 + i,
                )
                if cand is None:
                    improvements.append(
                        f"Champion '{tune_model}' is an ensemble of already-strong models; "
                        f"no single-model hyperparameters to tune — keeping it as-is."
                    )
                    break
                if cand["score"] > best_score + 1e-9:
                    best_score, best_cand, no_improve = cand["score"], cand, 0
                    improvements.append(f"Round {i + 1}: {tune_model} sel_score -> {best_score:.4f}")
                    logger.info(improvements[-1])
                else:
                    no_improve += 1
                    logger.info(f"Round {i + 1}: no improvement ({cand['score']:.4f} <= {best_score:.4f})")
                if target > 0 and best_score >= target:
                    improvements.append(f"Reached target {target} at round {i + 1}")
                    break
                if no_improve >= patience:
                    improvements.append(f"No improvement for {patience} round(s); early stop at round {i + 1}")
                    break

        if best_cand is not None:
            metrics = best_cand["metrics"]
            new_primary = metrics.get(primary, 0.0)
            model_path = artifact_dir / f"{tune_model}_tuned_iter{state.retry_count}.joblib"
            joblib.dump(best_cand["model"], model_path)
            state.best_model_path = str(model_path)
            state.best_metric_value = new_primary
            state.best_threshold = best_cand["threshold"]
            logger.info(f"Improvement kept: {tune_model} sel_score {baseline_score:.4f}->{best_score:.4f} "
                        f"({primary}={new_primary:.4f})")
            # Promote the tuned model to champion: it beat the previous best, so it
            # becomes the single is_best entry and the reported best_model_name.
            for r in state.model_results:
                r.is_best = False
            state.best_model_name = f"{tune_model}_tuned"
            state.model_results.append(ModelResult(
                model_name=f"{tune_model}_tuned",
                model_type=problem_type.value,
                metrics=metrics, hyperparameters=best_cand["params"],
                model_path=str(model_path), is_best=True, status="trained",
                decision_threshold=best_cand["threshold"],
            ))
        else:
            improvements.append(f"No improvement over champion ({best}) after tuning")
            logger.info(improvements[-1])

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
