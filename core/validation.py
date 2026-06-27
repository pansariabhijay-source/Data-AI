"""
Data validation utilities for schema checks, quality scoring, and leakage detection.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from core.constants import ProblemType
from core.logging_config import get_logger

logger = get_logger("validation")


def validate_dataframe(df: pd.DataFrame, min_rows: int = 10, max_cols: int = 500) -> list[str]:
    """Run basic sanity checks on a DataFrame. Returns list of issues found."""
    issues: list[str] = []
    if df.empty:
        issues.append("DataFrame is empty")
        return issues
    if len(df) < min_rows:
        issues.append(f"Too few rows: {len(df)} < {min_rows}")
    if len(df.columns) > max_cols:
        issues.append(f"Too many columns: {len(df.columns)} > {max_cols}")
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        issues.append(f"Duplicate column names: {dup_cols}")
    return issues


def validate_target_column(df: pd.DataFrame, target: str, problem_type: ProblemType) -> list[str]:
    """Validate target column exists and is compatible with problem type."""
    issues: list[str] = []
    if target not in df.columns:
        issues.append(f"Target column '{target}' not found in DataFrame")
        return issues
    null_pct = df[target].isnull().mean()
    if null_pct > 0:
        issues.append(f"Target column has {null_pct:.1%} null values")
    if problem_type == ProblemType.CLASSIFICATION:
        n_unique = df[target].nunique()
        if n_unique < 2:
            issues.append(f"Classification target has only {n_unique} unique value(s)")
        if n_unique > 100:
            issues.append(f"Classification target has {n_unique} classes — may be regression")
    elif problem_type == ProblemType.REGRESSION:
        if not pd.api.types.is_numeric_dtype(df[target]):
            issues.append(f"Regression target '{target}' is not numeric")
    return issues


def compute_quality_score(df: pd.DataFrame) -> float:
    """Compute a data quality score in [0, 1] based on completeness, uniqueness, consistency."""
    if df.empty:
        return 0.0
    completeness = 1.0 - df.isnull().mean().mean()
    dup_ratio = df.duplicated().mean()
    uniqueness = 1.0 - dup_ratio
    # Consistency: ratio of columns with consistent dtypes (no mixed types)
    consistency_scores = []
    for col in df.columns:
        try:
            if df[col].dtype == object:
                inferred = pd.to_numeric(df[col], errors="coerce")
                pct_numeric = inferred.notna().mean()
                consistency_scores.append(1.0 if pct_numeric < 0.1 or pct_numeric > 0.9 else 0.5)
            else:
                consistency_scores.append(1.0)
        except Exception:
            consistency_scores.append(0.5)
    consistency = float(np.mean(consistency_scores)) if consistency_scores else 1.0
    score = 0.4 * completeness + 0.3 * uniqueness + 0.3 * consistency
    return round(min(max(score, 0.0), 1.0), 4)


def detect_target_leakage(
    df: pd.DataFrame, target: str, threshold: float = 0.95, auc_threshold: float = 0.999
) -> list[str]:
    """Detect features that almost perfectly determine the target (potential leakage).

    Two complementary, high-precision signals so the check is robust to the target's
    dtype (the old correlation-only check silently did nothing for a string target):

    * **Regression / numeric target** — absolute Pearson correlation ``>= threshold``.
    * **Binary classification target** — single-feature ROC-AUC ``>= auc_threshold``.
      Correlation misses thresholded leakage (e.g. ``RainTomorrow`` derived from a
      ``RISK_MM`` rainfall column has only ~0.69 corr but a perfect 1.0 AUC), whereas
      a feature that ranks the target perfectly on its own is leakage by definition.

    Thresholds are deliberately near-1 to avoid stripping legitimately strong
    predictors. ID-like columns are left to the feature-engineering ID guard.
    """
    leaked: list[str] = []
    if target not in df.columns:
        return leaked

    y = df[target]
    target_is_numeric = pd.api.types.is_numeric_dtype(y)
    n_unique_target = int(y.nunique(dropna=True))

    # Treat a numeric target with many distinct values as a regression target.
    if target_is_numeric and n_unique_target > 20:
        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            if col == target:
                continue
            try:
                corr = abs(numeric_df[col].corr(numeric_df[target]))
                if corr >= threshold:
                    leaked.append(col)
                    logger.warning(f"Potential leakage: '{col}' has {corr:.3f} correlation with target")
            except Exception:
                continue
        return leaked

    # Classification target. Only the binary case has a clean single-feature AUC test.
    if n_unique_target != 2:
        return leaked
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return leaked

    y_enc = pd.factorize(y)[0]  # 0/1 encoding of the two classes
    for col in df.columns:
        if col == target:
            continue
        feat = pd.to_numeric(df[col], errors="coerce")
        mask = feat.notna() & (y_enc >= 0)
        if mask.sum() < 10:
            continue
        yv = y_enc[mask.to_numpy()]
        if len(np.unique(yv)) < 2:
            continue
        try:
            auc = roc_auc_score(yv, feat[mask])
            auc = max(auc, 1.0 - auc)  # direction-agnostic ranking power
            if auc >= auc_threshold:
                leaked.append(col)
                logger.warning(f"Potential leakage: '{col}' has single-feature AUC {auc:.4f} vs target")
        except Exception:
            continue
    return leaked


def detect_class_imbalance(
    series: pd.Series, threshold: float = 0.1
) -> Optional[dict[str, float]]:
    """Detect class imbalance. Returns class distribution if minority < threshold."""
    counts = series.value_counts(normalize=True)
    if counts.min() < threshold:
        return counts.to_dict()
    return None
