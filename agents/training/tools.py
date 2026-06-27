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
    operating_points,
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
    maybe_wrap_scaler,
)
from core.resampling import maybe_resample
from core.state import ErrorReport, ExperimentRecord, ModelResult, PipelineState
from core.utils import ensure_directory, get_timestamp

logger = get_logger("training")

# Cap on rows used to FIT candidate models. For tabular data a stratified sample
# trains near-identical models in a fraction of the time (model_training was the
# pipeline's dominant cost on large data). The champion is still scored on the
# full validation split and tested on the full untouched test set, so this only
# trades a touch of fitting data for a large speed-up. Override via
# TRAIN_SAMPLE_ROWS (0 disables subsampling entirely).
try:
    _TRAIN_SAMPLE_ROWS = int(os.environ.get("TRAIN_SAMPLE_ROWS", "100000"))
except ValueError:
    _TRAIN_SAMPLE_ROWS = 100000

# Keep at least this many rows of every class when subsampling, so rare classes
# (e.g. fraud positives) never get sampled away.
_MIN_ROWS_PER_CLASS = 2000

# Champion selection via k-fold cross-validation instead of a single validation
# split. Rationale: a single split is noisy — a ~0.0002 difference (luck of the
# draw) can crown a worse-generalising model. CV averages over k partitions so
# the pick is stable run-to-run.
#
# OFF BY DEFAULT (CV_SELECTION_FOLDS=0). The stability experiment
# (scripts/cv_selection_experiment.py) showed CV only changes the champion when
# selection is genuinely ambiguous (noisy/low-signal data, e.g. clustered
# boosting models) — there it's more stable AND slightly better on test; on
# separable problems it's a no-op. Since it adds real per-run fitting cost,
# enable it (CV_SELECTION_FOLDS=4) only when you value selection stability over
# speed. CV runs on a capped sample to bound the extra fitting.
try:
    _CV_SELECTION_FOLDS = int(os.environ.get("CV_SELECTION_FOLDS", "0"))
except ValueError:
    _CV_SELECTION_FOLDS = 0
try:
    _CV_SELECTION_MAX_ROWS = int(os.environ.get("CV_SELECTION_MAX_ROWS", "40000"))
except ValueError:
    _CV_SELECTION_MAX_ROWS = 40000
# Prefer models with a higher mean but also lower fold-to-fold variance.
_CV_STD_PENALTY = 0.25

# Calibrate the champion's probabilities (isotonic/sigmoid) so the F1-optimal
# threshold and any probability outputs are trustworthy — tree ensembles in
# particular are badly miscalibrated. Calibration is monotonic, so it never
# changes the AUC-based champion selection; we adopt it only if it improves the
# Brier score. Set CALIBRATE_CHAMPION=0 to disable.
try:
    _CALIBRATE_CHAMPION = int(os.environ.get("CALIBRATE_CHAMPION", "1"))
except (TypeError, ValueError):
    _CALIBRATE_CHAMPION = 1


def _subsample_for_training(
    X: np.ndarray, y: np.ndarray, problem_type: ProblemType, seed: int,
    cap: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a (stratified, for classification) subsample capped at ``cap`` rows.

    Class proportions are preserved, but every class keeps at least
    ``_MIN_ROWS_PER_CLASS`` rows so minority signal survives. No-op when the data
    is already within the cap or subsampling is disabled.
    """
    cap = _TRAIN_SAMPLE_ROWS if cap is None else cap
    n = len(X)
    if cap <= 0 or n <= cap:
        return X, y

    rng = np.random.default_rng(seed)
    if problem_type == ProblemType.CLASSIFICATION:
        classes, counts = np.unique(y, return_counts=True)
        keep: list[np.ndarray] = []
        for cls, cnt in zip(classes, counts):
            cls_idx = np.where(y == cls)[0]
            alloc = int(round(cap * cnt / n))
            alloc = max(alloc, min(int(cnt), _MIN_ROWS_PER_CLASS))
            alloc = min(alloc, int(cnt))
            sel = cls_idx if alloc >= cnt else rng.choice(cls_idx, size=alloc, replace=False)
            keep.append(sel)
        idx = np.concatenate(keep)
    else:
        idx = rng.choice(n, size=cap, replace=False)

    rng.shuffle(idx)
    logger.info(
        f"Subsampling training data {n} -> {len(idx)} rows for candidate fitting "
        f"(cap={cap}); champion is still scored on the full val/test splits."
    )
    return X[idx], y[idx]


class ModelTrainingService:
    def __init__(self, config: TrainingConfig, registry: ModelRegistry, seed: int = 42) -> None:
        self._config = config
        self._registry = registry
        self._seed = seed

    def _train_single_model(
        self, spec: ModelSpec, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, problem_type: ProblemType,
        artifact_dir: Path, has_validation: bool = False, resampled: bool = False,
    ) -> ModelResult:
        """Train a single model and return its results."""
        start = time.perf_counter()
        mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

        threshold: Optional[float] = None
        try:
            # When the training set was already (partially) balanced by SMOTE, turn
            # off the model's own class-weighting so the imbalance isn't corrected
            # twice (which over-shifts the decision boundary and hurts precision).
            # This must cover ALL weighting mechanisms:
            #   • sklearn models  → class_weight="balanced"
            #   • LightGBM        → is_unbalance=True  (NOT class_weight!)
            #   • XGBoost         → scale_pos_weight   (handled by apply_imbalance_handling guard below)
            overrides = {}
            if resampled:
                if "class_weight" in spec.default_params:
                    overrides["class_weight"] = None
                if spec.default_params.get("is_unbalance"):
                    overrides["is_unbalance"] = False
            model = self._registry.create_instance(problem_type, spec.name, overrides or None)
            if problem_type == ProblemType.CLUSTERING:
                # Distance-based clusterers need scaled features. Scale locally here
                # (rather than in the shared matrix) so the transform is self-contained;
                # clustering has no held-out split or downstream serving to leak into.
                X_fit = X_train
                if spec.requires_scaling:
                    from sklearn.preprocessing import StandardScaler
                    X_fit = StandardScaler().fit_transform(X_train)
                model.fit(X_fit)
                preds = model.predict(X_fit) if hasattr(model, "predict") else model.labels_
                metrics = compute_metrics(X_fit, preds, problem_type)
                train_metrics = metrics.copy()
            else:
                # Inject data-dependent imbalance params (e.g. scale_pos_weight) before
                # fit — unless SMOTE already balanced the data (see above).
                if not resampled:
                    model = apply_imbalance_handling(model, spec, y_train)
                # Scale-sensitive models (linear/SVC) get a per-fit StandardScaler so
                # they see standardized features WITHOUT leaking test stats; trees are
                # left on raw features. No-op for tree/boosting models.
                model = maybe_wrap_scaler(model, spec)
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

    def _cv_selection_scores(
        self, specs: list[ModelSpec], X: np.ndarray, y: np.ndarray,
        problem_type: ProblemType, n_classes: int, seed: int, resampled: bool,
    ) -> dict[str, tuple[float, float]]:
        """Cross-validate each base model and return ``{name: (mean, std)}``.

        The selection score is threshold-independent (PR-AUC + ROC-AUC based), so
        no per-fold thresholding is needed. Runs on a capped sample for speed;
        returns ``{}`` (caller falls back to single-split) when CV is disabled or
        the data can't support k stratified folds.
        """
        folds = _CV_SELECTION_FOLDS
        if folds < 2 or problem_type == ProblemType.CLUSTERING or len(X) < folds * 10:
            return {}

        from sklearn.model_selection import KFold, StratifiedKFold

        Xc, yc = _subsample_for_training(X, y, problem_type, seed, cap=_CV_SELECTION_MAX_ROWS)

        if problem_type == ProblemType.CLASSIFICATION:
            _, counts = np.unique(yc, return_counts=True)
            if counts.min() < folds:
                return {}  # too few minority samples for stratified CV
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        else:
            splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)

        scores: dict[str, tuple[float, float]] = {}
        for spec in specs:
            fold_scores: list[float] = []
            for tr_idx, va_idx in splitter.split(Xc, yc):
                try:
                    overrides = {}
                    if resampled:
                        if "class_weight" in spec.default_params:
                            overrides["class_weight"] = None
                        if spec.default_params.get("is_unbalance"):
                            overrides["is_unbalance"] = False
                    model = self._registry.create_instance(problem_type, spec.name, overrides or None)
                    if not resampled:
                        model = apply_imbalance_handling(model, spec, yc[tr_idx])
                    model = maybe_wrap_scaler(model, spec)
                    model = fit_model(model, spec, Xc[tr_idx], yc[tr_idx])
                    y_prob = None
                    if spec.supports_probabilities and hasattr(model, "predict_proba"):
                        try:
                            y_prob = model.predict_proba(Xc[va_idx])
                        except Exception:
                            y_prob = None
                    preds = model.predict(Xc[va_idx])
                    m = compute_metrics(yc[va_idx], preds, problem_type, y_prob)
                    fold_scores.append(selection_score(m, problem_type, n_classes))
                except Exception as e:
                    logger.debug(f"CV fold failed for {spec.name}: {e}")
                    continue
            if fold_scores:
                scores[spec.name] = (float(np.mean(fold_scores)), float(np.std(fold_scores)))

        if scores:
            ranked = ", ".join(
                f"{k}={m:.4f}±{s:.3f}"
                for k, (m, s) in sorted(scores.items(), key=lambda kv: -kv[1][0])
            )
            logger.info(f"CV selection ({folds}-fold) scores: {ranked}")
        return scores

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

    def _maybe_calibrate_champion(
        self, state: PipelineState, best: ModelResult,
        X_val: np.ndarray, y_val: np.ndarray, problem_type: ProblemType,
        artifact_dir: Path,
    ) -> None:
        """Calibrate the binary champion's probabilities on the val split.

        Uses CalibratedClassifierCV(cv="prefit") so the base model is not refit
        (cheap). Calibration is monotonic, so AUC/selection is unchanged; we adopt
        it only if it improves the Brier score, then re-derive the F1-optimal
        threshold on the calibrated probabilities. The honest test evaluation that
        follows then reflects the calibrated model.
        """
        if not _CALIBRATE_CHAMPION or problem_type != ProblemType.CLASSIFICATION:
            return
        if X_val is None or len(X_val) < 50 or len(np.unique(y_val)) != 2:
            return
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.frozen import FrozenEstimator
            from sklearn.metrics import brier_score_loss

            model = joblib.load(best.model_path)
            if not hasattr(model, "predict_proba"):
                return
            classes = np.asarray(getattr(model, "classes_", [0, 1]))
            pos = classes[-1]
            y_bin = (np.asarray(y_val) == pos).astype(int)

            brier_before = brier_score_loss(y_bin, positive_class_proba(model.predict_proba(X_val)))
            method = "isotonic" if len(X_val) >= 1000 else "sigmoid"
            # FrozenEstimator keeps the champion (trained on the full train split)
            # fixed and fits only the calibrator on val — the modern replacement
            # for the removed cv="prefit".
            calib = CalibratedClassifierCV(FrozenEstimator(model), method=method)
            calib.fit(X_val, y_val)
            brier_after = brier_score_loss(y_bin, positive_class_proba(calib.predict_proba(X_val)))

            if brier_after >= brier_before:
                logger.info(
                    f"Calibration ({method}) did not improve Brier "
                    f"({brier_before:.4f}->{brier_after:.4f}); keeping uncalibrated."
                )
                return

            cal_proba = calib.predict_proba(X_val)
            preds, threshold = predict_with_optimal_threshold(y_val, cal_proba, calib.classes_)
            metrics = compute_metrics(y_val, preds, problem_type, cal_proba)

            path = artifact_dir / f"{best.model_name}_calibrated.joblib"
            joblib.dump(calib, path)
            best.model_path = str(path)
            best.decision_threshold = threshold
            best.metrics = metrics
            state.best_model_path = str(path)
            state.best_threshold = threshold
            if state.best_metric_name in metrics:
                state.best_metric_value = metrics[state.best_metric_name]
            state.data_quality_flags["calibration"] = {
                "method": method,
                "brier_before": round(float(brier_before), 4),
                "brier_after": round(float(brier_after), 4),
            }
            logger.info(
                f"Calibrated champion ({method}): Brier {brier_before:.4f}->{brier_after:.4f}, "
                f"threshold->{threshold:.4f}"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Champion calibration skipped (non-critical): {e}")

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
            # Operating points (precision/recall at fixed alert rates) — lets a fraud
            # team pick a threshold for their review capacity.
            if (
                problem_type == ProblemType.CLASSIFICATION
                and y_prob is not None and len(np.unique(y_test)) == 2
            ):
                try:
                    classes = np.asarray(getattr(model, "classes_", [0, 1]))
                    pos_label = classes[1] if len(classes) == 2 else 1
                    state.test_operating_points = operating_points(
                        y_test, positive_class_proba(y_prob), pos_label=pos_label,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Operating-point computation skipped: {e}")
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
        resampled = False
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

            # Synthesise minority-class examples in the TRAIN split only (never
            # val/test) for imbalanced binary targets. When applied, each model's
            # class-weighting is neutralised downstream to avoid double-correcting.
            tcfg = settings.training
            rr = maybe_resample(
                X_train, y_train, problem_type,
                enabled=tcfg.use_smote,
                sampling_strategy=tcfg.smote_sampling_strategy,
                min_minority_fraction=tcfg.smote_min_minority_fraction,
                max_train_samples=tcfg.smote_max_train_samples,
                k_neighbors=tcfg.smote_k_neighbors,
                seed=settings.pipeline.random_seed,
            )
            if rr.applied:
                X_train, y_train = rr.X, rr.y
                resampled = True
                state.data_quality_flags["resampling"] = {
                    "method": rr.method, "minority_before": rr.minority_before,
                    "minority_after": rr.minority_after,
                    "target_ratio": tcfg.smote_sampling_strategy,
                }

        # Cap rows used to fit candidate models (stratified). The dominant cost on
        # large data; champion is still scored on the full val/test splits.
        if problem_type != ProblemType.CLUSTERING:
            X_train, y_train = _subsample_for_training(
                X_train, y_train, problem_type, settings.pipeline.random_seed,
            )

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
                        problem_type, artifact_dir, has_validation, resampled,
                    ): spec.name
                    for spec in specs
                }
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            for spec in specs:
                results.append(self._train_single_model(
                    spec, X_train, y_train, X_val, y_val, problem_type, artifact_dir,
                    has_validation, resampled,
                ))

        primary_metric = get_primary_metric(problem_type)

        # The champion is selected on a more stable, threshold-independent score than
        # the reported F1 (see core.metrics.selection_score) — for binary
        # classification F1@single-split-threshold over-fits the validation split and
        # can crown a worse-generalizing model.
        n_classes = int(len(np.unique(y_train))) if problem_type != ProblemType.CLUSTERING else 0

        # Prefer a k-fold CV score (mean − std penalty) over the single-split score
        # so the champion is stable run-to-run. Models without a CV score (e.g. the
        # ensemble, built below) fall back to their single-split selection score.
        cv_scores = self._cv_selection_scores(
            specs, X_train, y_train, problem_type, n_classes, self._seed, resampled,
        )

        def score_of(r: ModelResult) -> float:
            cs = cv_scores.get(r.model_name)
            if cs is not None:
                return cs[0] - _CV_STD_PENALTY * cs[1]
            return selection_score(r.metrics, problem_type, n_classes)

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

            # Detect near-random performance: if the best model's ROC-AUC is below
            # 0.55 (barely above coin-flip), the dataset likely has insufficient
            # signal for reliable ML. Surface this as a clear warning rather than
            # letting the user interpret cryptic numbers.
            if problem_type == ProblemType.CLASSIFICATION and n_classes == 2:
                best_roc = best.metrics.get("roc_auc", 1.0)
                if best_roc < 0.55:
                    warning_msg = (
                        f"LOW SIGNAL WARNING: best model ROC-AUC={best_roc:.3f} (near random 0.5). "
                        "This typically means the dataset's features have very low predictive "
                        "power for the target. Check that (1) the correct target column is "
                        "selected, (2) informative features are present and not dropped, and "
                        "(3) the target labels are not randomly assigned."
                    )
                    logger.warning(warning_msg)
                    state.data_quality_flags["low_signal"] = {
                        "best_roc_auc": best_roc,
                        "warning": warning_msg,
                    }

            # Calibrate the champion's probabilities (monotonic — selection above
            # is unchanged) so its threshold and probability outputs are trustworthy.
            self._maybe_calibrate_champion(state, best, X_val, y_val, problem_type, artifact_dir)

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
