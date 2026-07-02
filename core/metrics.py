"""
Unified metric computation utilities.

Provides a single entry point to compute all relevant metrics for any problem type.
Avoids metric computation duplication across agents.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional
from core.constants import ProblemType
from core.logging_config import get_logger

logger = get_logger("metrics")


def target_inverse_transform(data_quality_flags: Optional[dict]) -> Optional[Any]:
    """Return the callable mapping model-space regression predictions back to the
    target's ORIGINAL units, or ``None`` if the target was not transformed.

    Feature engineering log1p-transforms a heavily skewed regression target and
    records it under ``data_quality_flags["log_transformed_target"]``. The model
    then trains and predicts in log-space; ``np.expm1`` is the exact inverse used
    to report metrics (and, later, predictions) in the units the user cares about.
    """
    info = (data_quality_flags or {}).get("log_transformed_target")
    if info:
        return np.expm1
    return None


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    problem_type: ProblemType,
    y_prob: Optional[np.ndarray] = None,
    inverse_transform: Optional[Any] = None,
) -> dict[str, float]:
    """Compute all relevant metrics for the given problem type.

    Args:
        y_true: Ground truth labels/values.
        y_pred: Predicted labels/values.
        problem_type: The ML problem type.
        y_prob: Predicted probabilities (classification only).
        inverse_transform: Optional callable applied to BOTH ``y_true`` and
            ``y_pred`` before scoring a regression problem. When the pipeline
            log-transforms a skewed target, the model trains/predicts in
            log-space; passing ``np.expm1`` here makes the reported RMSE/MAE/R²
            reflect the target's ORIGINAL units (dollars, counts, …) instead of
            log-space, which is otherwise meaningless to a user. Ignored for
            classification/clustering.

    Returns:
        Dict mapping metric name to value.
    """
    if problem_type == ProblemType.CLASSIFICATION:
        return _classification_metrics(y_true, y_pred, y_prob)
    elif problem_type == ProblemType.REGRESSION:
        return _regression_metrics(y_true, y_pred, inverse_transform)
    elif problem_type == ProblemType.CLUSTERING:
        return _clustering_metrics(y_true, y_pred)
    else:
        logger.warning(f"Unknown problem type: {problem_type}, returning empty metrics")
        return {}


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        f1_score, precision_score, recall_score, roc_auc_score,
    )
    n_classes = len(np.unique(y_true))
    avg = "binary" if n_classes == 2 else "weighted"
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    if y_prob is not None:
        try:
            if n_classes == 2:
                prob = positive_class_proba(y_prob)
                metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
                # PR-AUC (average precision) — the most informative ranking metric
                # for imbalanced targets, where ROC-AUC can look deceptively high.
                metrics["pr_auc"] = float(average_precision_score(y_true, prob))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except (ValueError, IndexError) as e:
            logger.warning(f"Probability metric computation failed: {e}")
    return metrics


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[dict[str, int]]:
    """Return binary confusion-matrix counts (tn, fp, fn, tp), or None if not binary."""
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true)
    labels = np.unique(y_true)
    if len(labels) != 2:
        return None
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    except (ValueError, IndexError):
        return None
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, inverse_transform: Optional[Any] = None
) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Report in the target's original units when it was transformed for training
    # (e.g. log1p on a skewed target). RMSE/MAE are only meaningful in real units,
    # and R² differs between log- and original-space, so scoring the inverse-mapped
    # values is the honest number to show the user. Guard against overflow/NaN from
    # expm1 on extreme predictions by falling back to the raw (log-space) values.
    if inverse_transform is not None:
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                yt = inverse_transform(y_true)
                yp = inverse_transform(y_pred)
            if np.all(np.isfinite(yt)) and np.all(np.isfinite(yp)):
                y_true, y_pred = yt, yp
            else:
                logger.warning("inverse_transform produced non-finite values; "
                               "reporting regression metrics in transformed space")
        except Exception as e:
            logger.warning(f"inverse_transform failed ({e}); metrics in transformed space")
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _clustering_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """For clustering, `labels` is the data matrix and `predictions` are cluster labels."""
    from sklearn.metrics import silhouette_score
    n_unique = len(np.unique(predictions))
    if n_unique < 2:
        logger.warning("Silhouette score requires >= 2 clusters, got %d", n_unique)
        return {"silhouette_score": -1.0}
    return {"silhouette_score": float(silhouette_score(labels, predictions))}


def positive_class_proba(y_prob: np.ndarray) -> np.ndarray:
    """Extract the positive-class probability vector from a predict_proba output.

    Accepts either a 2-D ``(n_samples, n_classes)`` array (returns the second
    column) or an already-1-D probability vector.
    """
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
        return y_prob[:, 1]
    return y_prob.ravel()


def find_optimal_threshold(
    y_true: np.ndarray, pos_proba: np.ndarray, pos_label: Any = 1
) -> float:
    """Find the decision threshold on positive-class probability that maximises F1.

    On imbalanced problems the default 0.5 cutoff is almost always wrong: a model
    can rank well (high ROC-AUC) yet predict almost no positives at 0.5, crushing
    F1/recall. Sweeping the precision-recall curve recovers the threshold that
    actually balances precision and recall for the minority class.

    Returns 0.5 if a meaningful threshold cannot be derived.
    """
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true)
    pos_proba = np.asarray(pos_proba)
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        precision, recall, thresholds = precision_recall_curve(
            y_true, pos_proba, pos_label=pos_label
        )
    except (ValueError, IndexError) as e:
        logger.warning(f"Threshold optimisation failed: {e}")
        return 0.5
    if len(thresholds) == 0:
        return 0.5
    # precision/recall have one more element than thresholds; align by dropping it.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)

    # Guard against degenerate optimisation on near-random models: when all
    # candidate thresholds produce precision below 2× the base rate (i.e. the
    # model can't beat a naive prior by a meaningful margin), fall back to 0.5
    # rather than selecting a near-zero threshold that floods predictions with
    # false positives.
    base_rate = float(np.mean(np.asarray(y_true) == pos_label))
    min_precision = max(2.0 * base_rate, 0.05)
    valid_mask = precision[:-1] >= min_precision
    if valid_mask.any():
        # Among thresholds with acceptable precision, pick the one with best F1.
        masked_f1 = np.where(valid_mask, f1, -1.0)
        best_idx = int(np.argmax(masked_f1))
    else:
        # No threshold meets minimum precision — the model is near-random.
        # Fall back to 0.5 to avoid flooding predictions with false positives
        # (a near-zero F1-optimal threshold on random scores predicts almost
        # everything as positive, which makes accuracy/precision look terrible).
        logger.warning(
            f"Threshold optimisation: no threshold achieves precision >= {min_precision:.3f} "
            f"(base rate={base_rate:.3f}). Model may have near-random discriminative power. "
            "Falling back to threshold=0.5."
        )
        return 0.5
    return float(thresholds[best_idx])


def predict_with_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, classes: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return F1-optimal binary predictions and the threshold used.

    ``classes`` is the estimator's ``classes_`` array. Predictions are mapped back
    to the original class labels so this works for any binary label encoding.
    """
    classes = np.asarray(classes)
    pos = positive_class_proba(y_prob)
    pos_label = classes[1] if len(classes) == 2 else 1
    thr = find_optimal_threshold(y_true, pos, pos_label=pos_label)
    if len(classes) == 2:
        preds = np.where(pos >= thr, classes[1], classes[0])
    else:
        preds = (pos >= thr).astype(int)
    return preds, thr


def operating_points(
    y_true: np.ndarray,
    pos_proba: np.ndarray,
    pos_label: Any = 1,
    alert_rates: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05),
) -> list[dict]:
    """Precision/recall at fixed *alert rates* — the way a fraud team actually
    operates (review capacity, not an abstract threshold).

    For each target fraction of transactions flagged, returns the score threshold
    that flags that fraction and the resulting precision (of flagged, how many are
    fraud) and recall (of all fraud, how many we caught). This answers "if we can
    investigate X% of transactions, what do we catch?".
    """
    y_true = np.asarray(y_true)
    pos = np.asarray(pos_proba, dtype=float)
    n = len(y_true)
    total_pos = int(np.sum(y_true == pos_label))
    if n == 0 or total_pos == 0:
        return []
    order = np.argsort(-pos)  # highest risk first
    y_sorted = (y_true[order] == pos_label).astype(int)
    out: list[dict] = []
    seen: set[int] = set()
    for rate in alert_rates:
        k = max(1, int(round(rate * n)))
        if k in seen:
            continue
        seen.add(k)
        flagged = y_sorted[:k]
        caught = int(flagged.sum())
        out.append({
            "alert_rate": round(k / n, 4),
            "threshold": round(float(pos[order[k - 1]]), 4),
            "flagged": int(k),
            "precision": round(caught / k, 4),
            "recall": round(caught / total_pos, 4),
            "caught": caught,
            "total_fraud": total_pos,
        })
    return out


def bootstrap_metric_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    problem_type: ProblemType,
    y_prob: Optional[np.ndarray] = None,
    inverse_transform: Optional[Any] = None,
    metrics: tuple[str, ...] = (
        "f1", "pr_auc", "roc_auc", "precision", "recall", "r2", "rmse", "mae",
    ),
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    max_rows: int = 50000,
) -> dict[str, list[float]]:
    """Percentile bootstrap confidence intervals for held-out test metrics.

    A single test slice gives a point estimate with no sense of its uncertainty —
    "PR-AUC 0.78" reads as precise when it may be 0.78 ± 0.05. Resampling the test
    rows with replacement and recomputing each metric yields an honest
    ``[low, high]`` interval at confidence ``1 - alpha`` (default 95%).

    Returns ``{metric: [low, high]}`` for the metrics that are computable; empty
    when the test set is too small. Cost is bounded by capping rows and resamples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n < 20 or n_boot <= 0:
        return {}
    yprob = np.asarray(y_prob) if y_prob is not None else None

    rng = np.random.default_rng(seed)
    if n > max_rows:
        base = rng.choice(n, size=max_rows, replace=False)
        y_true, y_pred = y_true[base], y_pred[base]
        if yprob is not None:
            yprob = yprob[base]
        n = max_rows

    samples: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yp = yprob[idx] if yprob is not None else None
        m = compute_metrics(
            y_true[idx], y_pred[idx], problem_type, yp, inverse_transform=inverse_transform
        )
        for k in metrics:
            v = m.get(k)
            if v is not None and np.isfinite(v):
                samples[k].append(float(v))

    lo_p, hi_p = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    out: dict[str, list[float]] = {}
    min_valid = max(30, n_boot // 10)
    for k, vals in samples.items():
        if len(vals) >= min_valid:
            out[k] = [
                round(float(np.percentile(vals, lo_p)), 4),
                round(float(np.percentile(vals, hi_p)), 4),
            ]
    return out


def get_primary_metric(problem_type: ProblemType) -> str:
    """Return the primary metric name used for reporting and comparison."""
    return {"classification": "f1", "regression": "r2", "clustering": "silhouette_score"}.get(
        problem_type.value, "f1"
    )


def selection_score(
    metrics: dict[str, float], problem_type: ProblemType, n_classes: Optional[int] = None
) -> float:
    """Higher-is-better scalar used to *select* the champion (not what's reported).

    For binary classification this is the mean of the available threshold-independent
    ranking metrics (PR-AUC and ROC-AUC) rather than F1 at a threshold tuned on a
    single validation split. Thresholded F1 over-fits that split and can crown a
    model that ranks worse on held-out data; averaging two complementary ranking
    metrics is minority-focused *and* lowers selection variance. The reported F1 and
    the F1-optimal decision threshold are unchanged — only which model wins. Falls
    back to F1 when ranking metrics are unavailable.
    """
    if problem_type == ProblemType.CLASSIFICATION:
        if n_classes == 2:
            ranking = [metrics[k] for k in ("pr_auc", "roc_auc") if k in metrics]
            if ranking:
                return float(np.mean(ranking))
        return float(metrics.get("f1", 0.0))
    if problem_type == ProblemType.REGRESSION:
        return float(metrics.get("r2", -1e18))
    if problem_type == ProblemType.CLUSTERING:
        return float(metrics.get("silhouette_score", -1e18))
    return float(metrics.get("f1", 0.0))


def is_metric_higher_better(metric_name: str) -> bool:
    """Return True if higher values are better for the given metric."""
    lower_is_better = {"rmse", "mae", "mse"}
    return metric_name.lower() not in lower_is_better
