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
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )
    n_classes = len(np.unique(y_true))
    avg = "binary" if n_classes == 2 else "weighted"
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    if y_prob is not None:
        try:
            if n_classes == 2:
                prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except (ValueError, IndexError) as e:
            logger.warning(f"ROC-AUC computation failed: {e}")
    return metrics


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


def get_primary_metric(problem_type: ProblemType) -> str:
    """Return the primary metric name used for model comparison."""
    return {"classification": "f1", "regression": "r2", "clustering": "silhouette_score"}.get(
        problem_type.value, "f1"
    )


def is_metric_higher_better(metric_name: str) -> bool:
    """Return True if higher values are better for the given metric."""
    lower_is_better = {"rmse", "mae", "mse"}
    return metric_name.lower() not in lower_is_better
