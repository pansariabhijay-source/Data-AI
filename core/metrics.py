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


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    problem_type: ProblemType,
    y_prob: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute all relevant metrics for the given problem type.

    Args:
        y_true: Ground truth labels/values.
        y_pred: Predicted labels/values.
        problem_type: The ML problem type.
        y_prob: Predicted probabilities (classification only).

    Returns:
        Dict mapping metric name to value.
    """
    if problem_type == ProblemType.CLASSIFICATION:
        return _classification_metrics(y_true, y_pred, y_prob)
    elif problem_type == ProblemType.REGRESSION:
        return _regression_metrics(y_true, y_pred)
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


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    best_idx = int(np.argmax(f1))
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


def get_primary_metric(problem_type: ProblemType) -> str:
    """Return the primary metric name used for model comparison."""
    return {"classification": "f1", "regression": "r2", "clustering": "silhouette_score"}.get(
        problem_type.value, "f1"
    )


def is_metric_higher_better(metric_name: str) -> bool:
    """Return True if higher values are better for the given metric."""
    lower_is_better = {"rmse", "mae", "mse"}
    return metric_name.lower() not in lower_is_better
