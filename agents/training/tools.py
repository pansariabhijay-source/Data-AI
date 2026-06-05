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
from typing import Any, Callable, Optional

import joblib
import numpy as np
import pandas as pd
import psutil

from core.config import Settings, TrainingConfig
from core.constants import ProblemType
from core.exceptions import TrainingError
from core.logging_config import get_logger, log_stage_timing
from core.metrics import (
    compute_metrics,
    confusion_counts,
    get_primary_metric,
    positive_class_proba,
    predict_with_optimal_threshold,
    selection_score,
)
from core.model_registry import (
    ModelRegistry,
    ModelSpec,
    apply_imbalance_handling,
    build_default_registry,
    fit_model,
    get_fitted_n_estimators,
)
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
        artifact_dir: Path, has_validation: bool = False,
    ) -> ModelResult:
        """Train a single model and return its results."""
        start = time.perf_counter()
        mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

        threshold: Optional[float] = None
        try:
            model = self._registry.create_instance(problem_type, spec.name)
            if problem_type == ProblemType.CLUSTERING:
                model.fit(X_train)
                preds = model.predict(X_train) if hasattr(model, "predict") else model.labels_
                metrics = compute_metrics(X_train, preds, problem_type)
                train_metrics = metrics.copy()
            else:
                # Inject data-dependent imbalance params (e.g. scale_pos_weight) before fit.
                model = apply_imbalance_handling(model, spec, y_train)
                # Use early stopping against the validation set for boosters when we
                # have a genuine (non-degenerate) validation split.
                eval_X = X_val if has_validation else None
                eval_y = y_val if has_validation else None
                model = fit_model(model, spec, X_train, y_train, eval_X=eval_X, eval_y=eval_y)

                y_prob = None
                if spec.supports_probabilities and hasattr(model, "predict_proba"):
                    try:
                        y_prob = model.predict_proba(X_val)
                    except Exception:
                        y_prob = None

                # For binary classification, pick the F1-optimal threshold instead of
                # the naive 0.5 cutoff — critical for imbalanced targets like fraud.
                if (
                    problem_type == ProblemType.CLASSIFICATION
                    and y_prob is not None
                    and len(np.unique(y_train)) == 2
                ):
                    preds, threshold = predict_with_optimal_threshold(y_val, y_prob, model.classes_)
                    train_pos = positive_class_proba(model.predict_proba(X_train))
                    train_preds = np.where(
                        train_pos >= threshold, model.classes_[1], model.classes_[0]
                    )
                else:
                    preds = model.predict(X_val)
                    train_preds = model.predict(X_train)

                metrics = compute_metrics(y_val, preds, problem_type, y_prob)
                train_metrics = compute_metrics(y_train, train_preds, problem_type)

            elapsed = time.perf_counter() - start
            mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

            # Save model
            model_path = artifact_dir / f"{spec.name}.joblib"
            joblib.dump(model, model_path)

            # Record the params actually used, including the early-stopped tree count.
            used_params = dict(spec.default_params)
            fitted_trees = get_fitted_n_estimators(model)
            if fitted_trees is not None:
                used_params["n_estimators"] = fitted_trees

            result = ModelResult(
                model_name=spec.name, model_type=spec.problem_type.value,
                metrics=metrics, train_metrics=train_metrics,
                training_time_seconds=round(elapsed, 3),
                memory_usage_mb=round(mem_after - mem_before, 2),
                model_path=str(model_path),
                hyperparameters=used_params,
                status="trained",
                decision_threshold=threshold,
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

    def _build_ensemble(
        self, results: list[ModelResult], X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, problem_type: ProblemType,
        artifact_dir: Path, score_of: Callable[[ModelResult], float],
        n_classes: int, top_k: int = 3,
    ) -> Optional[ModelResult]:
        """Average the probabilities of the top-K classifiers into a soft-voting ensemble.

        Members are ranked (and weighted) by the same selection score used to crown
        the champion, so the ensemble is assembled from the strongest generalizers.
        """
        if problem_type != ProblemType.CLASSIFICATION:
            return None

        # Rank trained, probability-capable members by the selection score.
        candidates = [r for r in results if r.status == "trained" and r.model_path]
        candidates.sort(key=score_of, reverse=True)

        members, names, weights = [], [], []
        ref_classes = None
        for r in candidates:
            if len(members) >= top_k:
                break
            try:
                est = joblib.load(r.model_path)
                if not hasattr(est, "predict_proba"):
                    continue
                classes = np.asarray(getattr(est, "classes_", []))
                if ref_classes is None:
                    ref_classes = classes
                elif not np.array_equal(classes, ref_classes):
                    continue  # incompatible label ordering — skip
                members.append(est)
                names.append(r.model_name)
                weights.append(max(score_of(r), 1e-3))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Ensemble: could not load {r.model_name}: {e}")

        if len(members) < 2 or ref_classes is None:
            return None  # nothing to gain from a 1-member ensemble

        start = time.perf_counter()
        try:
            from core.ensemble import ProbabilityAveragingEnsemble

            ens = ProbabilityAveragingEnsemble(members, ref_classes, names, weights)
            y_prob = ens.predict_proba(X_val)
            threshold: Optional[float] = None
            if len(ref_classes) == 2:
                preds, threshold = predict_with_optimal_threshold(y_val, y_prob, ref_classes)
                train_pos = positive_class_proba(ens.predict_proba(X_train))
                train_preds = np.where(train_pos >= threshold, ref_classes[1], ref_classes[0])
            else:
                preds = ens.predict(X_val)
                train_preds = ens.predict(X_train)

            metrics = compute_metrics(y_val, preds, problem_type, y_prob)
            train_metrics = compute_metrics(y_train, ens.predict(X_train) if len(ref_classes) != 2 else train_preds, problem_type)

            model_path = artifact_dir / "VotingEnsemble.joblib"
            joblib.dump(ens, model_path)
            elapsed = round(time.perf_counter() - start, 3)
            logger.info(f"VotingEnsemble({'+'.join(names)}): f1={metrics.get('f1', 0):.4f} "
                        f"sel_score={selection_score(metrics, problem_type, n_classes):.4f}")
            return ModelResult(
                model_name="VotingEnsemble", model_type=problem_type.value,
                metrics=metrics, train_metrics=train_metrics,
                training_time_seconds=elapsed, model_path=str(model_path),
                hyperparameters={"members": names, "voting": "soft", "weights": [round(w, 4) for w in weights]},
                status="trained", decision_threshold=threshold,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Ensemble construction failed: {e}")
            return None

    def _evaluate_on_test(
        self, state: PipelineState, settings: Settings,
        problem_type: ProblemType, target: Optional[str],
    ) -> None:
        """Score the selected champion on the untouched test split (honest estimate)."""
        if not state.test_path or not state.best_model_path or problem_type == ProblemType.CLUSTERING:
            return
        try:
            test_df = pd.read_csv(state.test_path, low_memory=False)
            if not target or target not in test_df.columns:
                return
            X_test = test_df.drop(columns=[target]).select_dtypes(include=[np.number]).values
            y_test = test_df[target].values
            model = joblib.load(state.best_model_path)

            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                except Exception:
                    y_prob = None

            if (
                problem_type == ProblemType.CLASSIFICATION
                and y_prob is not None
                and state.best_threshold is not None
                and len(np.unique(y_test)) == 2
            ):
                classes = np.asarray(getattr(model, "classes_", [0, 1]))
                pos = positive_class_proba(y_prob)
                preds = np.where(pos >= state.best_threshold, classes[1], classes[0])
            else:
                preds = model.predict(X_test)

            state.test_metrics = compute_metrics(y_test, preds, problem_type, y_prob)
            cm = confusion_counts(y_test, preds)
            if cm:
                state.test_confusion = cm
            logger.info(f"Held-out TEST metrics: {state.test_metrics}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Test-set evaluation failed (non-critical): {e}")

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

        has_validation = False
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
                has_validation = len(X_val) > 0
            else:
                X_val, y_val = X_train, y_train
                has_validation = False

        # Drop models that don't scale to this many rows (e.g. kernel SVC is
        # O(n^2)+ and would hang on large data) rather than letting them stall the run.
        n_train = len(X_train)
        kept_specs = [s for s in specs if s.max_train_samples is None or n_train <= s.max_train_samples]
        skipped = [s.name for s in specs if s not in kept_specs]
        if skipped:
            logger.info(f"Skipping {skipped} — train set has {n_train} rows (exceeds their scalability limit)")
        specs = kept_specs or specs  # never end up with zero models

        artifact_dir = ensure_directory(Path(settings.pipeline.artifact_dir) / state.run_id / "models")

        # Train models (parallel for CPU-bound, sequential fallback)
        results: list[ModelResult] = []
        n_workers = min(len(specs), max(1, os.cpu_count() or 1))

        if n_workers > 1 and len(specs) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        self._train_single_model, spec, X_train, y_train, X_val, y_val,
                        problem_type, artifact_dir, has_validation,
                    ): spec.name
                    for spec in specs
                }
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            for spec in specs:
                results.append(self._train_single_model(
                    spec, X_train, y_train, X_val, y_val, problem_type, artifact_dir, has_validation,
                ))

        primary_metric = get_primary_metric(problem_type)

        # The champion is selected on a more stable, threshold-independent score than
        # the reported F1 (see core.metrics.selection_score) — for binary
        # classification F1@single-split-threshold over-fits the validation split and
        # can crown a worse-generalizing model.
        n_classes = int(len(np.unique(y_train))) if problem_type != ProblemType.CLUSTERING else 0
        score_of = lambda r: selection_score(r.metrics, problem_type, n_classes)

        # Build a soft-voting ensemble from the strongest classifiers — frequently
        # beats any single model by averaging out their individual errors.
        ensemble = self._build_ensemble(
            results, X_train, y_train, X_val, y_val, problem_type, artifact_dir,
            score_of, n_classes,
        )
        if ensemble is not None:
            results.append(ensemble)

        # Select best model
        trained = [r for r in results if r.status == "trained" and primary_metric in r.metrics]

        if trained:
            best = max(trained, key=score_of)
            best.is_best = True
            state.best_model_name = best.model_name
            state.best_model_path = best.model_path
            state.best_metric_name = primary_metric
            state.best_metric_value = best.metrics[primary_metric]
            state.best_threshold = best.decision_threshold
            thr_note = f", threshold={best.decision_threshold:.4f}" if best.decision_threshold is not None else ""
            logger.info(f"Best model: {best.model_name} ({primary_metric}={best.metrics[primary_metric]:.4f}{thr_note}, sel_score={score_of(best):.4f})")

            # Honest held-out evaluation of the champion on the untouched test set.
            self._evaluate_on_test(state, settings, problem_type, target)

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
