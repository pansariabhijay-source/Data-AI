"""
Imbalanced-target resampling (SMOTE) for the training stage.

``maybe_resample`` synthesises minority-class examples in the TRAINING set only —
never the validation or test split — so honest generalization estimates are
preserved. It is deliberately conservative:

* Binary classification only (multiclass SMOTE is far more fragile).
* Skips data that is already reasonably balanced (minority above a threshold).
* Skips very large training sets (synthetic blow-up / cost guard).
* PARTIAL oversampling. Empirically, moderate oversampling (minority/majority
  ratio ~0.1-0.5) beats class-weighting on extreme imbalance, but fully balancing
  (ratio 1.0) hurts gradient boosters — so the target ratio is capped, never forced
  to 1.0.
* Falls back to ``RandomOverSampler`` when SMOTE can't run (e.g. fewer minority
  samples than ``k_neighbors``), and returns the data untouched on any failure.

When resampling succeeds the training set is (partially) balanced, so the model's
existing class-weighting (``class_weight="balanced"`` / ``scale_pos_weight``) must
be neutralised by the caller to avoid double-correcting the imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from core.constants import ProblemType
from core.logging_config import get_logger

logger = get_logger("resampling")


@dataclass
class ResampleResult:
    X: np.ndarray
    y: np.ndarray
    applied: bool
    method: Optional[str] = None
    minority_before: Optional[int] = None
    minority_after: Optional[int] = None
    reason: Optional[str] = None  # why it was skipped, when applied is False


def maybe_resample(
    X: np.ndarray,
    y: np.ndarray,
    problem_type: ProblemType,
    *,
    enabled: bool = True,
    sampling_strategy: float = 0.25,
    min_minority_fraction: float = 0.20,
    max_train_samples: int = 500_000,
    k_neighbors: int = 5,
    seed: int = 42,
) -> ResampleResult:
    """Partially oversample the minority class of a binary training set with SMOTE.

    Returns a :class:`ResampleResult`; ``applied`` is False (with ``reason`` set)
    whenever a guard trips, in which case ``X``/``y`` are the originals unchanged.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    def skip(reason: str) -> ResampleResult:
        return ResampleResult(X=X, y=y, applied=False, reason=reason)

    if not enabled:
        return skip("disabled")
    if problem_type != ProblemType.CLASSIFICATION:
        return skip("not classification")

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return skip("not binary")
    n = int(counts.sum())
    minority = int(counts.min())
    majority = int(counts.max())
    minority_frac = minority / n if n else 0.0

    if minority_frac >= min_minority_fraction:
        return skip(f"already balanced (minority={minority_frac:.3f})")
    if n > max_train_samples:
        return skip(f"train too large ({n} > {max_train_samples})")
    if minority < 2:
        return skip(f"too few minority samples ({minority})")

    # Don't *reduce* the minority: only oversample toward the target ratio if the
    # current ratio is below it.
    current_ratio = minority / majority
    if current_ratio >= sampling_strategy:
        return skip(f"minority/majority {current_ratio:.3f} already >= target {sampling_strategy}")

    k = min(k_neighbors, minority - 1)
    try:
        from imblearn.over_sampling import SMOTE

        sampler: Any = SMOTE(
            sampling_strategy=sampling_strategy, k_neighbors=k, random_state=seed
        )
        method = "SMOTE"
        Xr, yr = sampler.fit_resample(X, y)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"SMOTE failed ({e}); falling back to RandomOverSampler")
        try:
            from imblearn.over_sampling import RandomOverSampler

            sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=seed)
            method = "RandomOverSampler"
            Xr, yr = sampler.fit_resample(X, y)
        except Exception as e2:  # noqa: BLE001
            logger.warning(f"Resampling unavailable ({e2}); training on original data")
            return skip(f"resampling error: {e2}")

    new_minority = int(np.unique(yr, return_counts=True)[1].min())
    logger.info(
        f"{method}: oversampled minority {minority} -> {new_minority} "
        f"(train {n} -> {len(yr)} rows, target ratio {sampling_strategy})"
    )
    return ResampleResult(
        X=Xr, y=yr, applied=True, method=method,
        minority_before=minority, minority_after=new_minority,
    )
