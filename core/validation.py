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

    Three complementary, high-precision signals so the check is robust to the
    dtype of BOTH the target and the feature:

    * **Regression / numeric target** — absolute Pearson correlation ``>= threshold``.
    * **Binary target, numeric feature** — single-feature ROC-AUC ``>= auc_threshold``.
      Correlation misses thresholded leakage (e.g. ``RainTomorrow`` derived from a
      ``RISK_MM`` rainfall column has only ~0.69 corr but a perfect 1.0 AUC), whereas
      a feature that ranks the target perfectly on its own is leakage by definition.
    * **Binary target, categorical (string) feature** — out-of-fold target-encoded
      ROC-AUC ``>= auc_threshold``. The numeric-AUC path coerces each feature with
      ``to_numeric`` first, so a pure-string column that perfectly predicts the
      target (e.g. a ``status``/``disposition`` text field) became all-NaN and was
      silently skipped. Out-of-fold encoding catches it while self-guarding against
      false positives on high-cardinality ID columns: their categories are unseen
      across folds, so they collapse to the global mean and carry no AUC signal.

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
    n_rows = len(df)
    for col in df.columns:
        if col == target:
            continue
        feat = pd.to_numeric(df[col], errors="coerce")
        numeric_frac = float(feat.notna().mean())
        if numeric_frac >= 0.5:
            # Numeric (or mostly-numeric) feature — direct single-feature AUC.
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
        else:
            # Categorical / string feature — out-of-fold target-encoded AUC.
            nunique = int(df[col].nunique(dropna=True))
            # Guard: need >= 2 categories, and skip near-unique ID-like columns
            # (they carry no generalizable signal and are handled by the FE ID guard).
            if nunique < 2 or (n_rows and nunique / n_rows > 0.5):
                continue
            auc = _categorical_leakage_auc(df[col], y_enc)
            if auc is not None and auc >= auc_threshold:
                leaked.append(col)
                logger.warning(
                    f"Potential leakage: categorical '{col}' has out-of-fold "
                    f"target-encoded AUC {auc:.4f} vs target"
                )
    return leaked


def _categorical_leakage_auc(
    col: pd.Series, y_enc: np.ndarray, n_splits: int = 5, seed: int = 42,
    max_rows: int = 20000,
) -> Optional[float]:
    """Direction-agnostic out-of-fold target-encoded ROC-AUC of a categorical
    feature against a binary target, or ``None`` if it can't be computed.

    Out-of-fold (K-fold) target-mean encoding is what makes this a *leakage* test
    rather than a memorization test: within-sample target encoding trivially
    "predicts" the target for any high-cardinality column, but encoding each fold
    from the OTHER folds' means means an ID-like column (categories unseen across
    folds) collapses to the global mean and scores ~0.5, while a genuinely leaky
    low-cardinality column still separates the classes almost perfectly.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    cats = col.astype("string").fillna("__nan__").astype(str).to_numpy()
    y_enc = np.asarray(y_enc)
    n = len(cats)
    if n != len(y_enc) or n < 20:
        return None

    # Bound cost on large data — leakage is near-perfect, so a stratified sample
    # detects it just as well as the full frame.
    if n > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_rows, replace=False)
        cats, y_enc = cats[idx], y_enc[idx]
        n = max_rows

    if len(np.unique(y_enc)) < 2:
        return None
    _, counts = np.unique(y_enc, return_counts=True)
    folds = int(min(n_splits, counts.min()))
    if folds < 2:
        return None

    global_mean = float(np.mean(y_enc))
    oof = np.full(n, global_mean, dtype=float)
    try:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for tr, va in skf.split(np.zeros(n), y_enc):
            means = pd.Series(y_enc[tr]).groupby(cats[tr]).mean()
            mapped = pd.Series(cats[va]).map(means)
            oof[va] = mapped.fillna(global_mean).to_numpy()
        auc = roc_auc_score(y_enc, oof)
        return float(max(auc, 1.0 - auc))
    except Exception:
        return None


def detect_class_imbalance(
    series: pd.Series, threshold: float = 0.1
) -> Optional[dict[str, float]]:
    """Detect class imbalance. Returns class distribution if minority < threshold."""
    counts = series.value_counts(normalize=True)
    if counts.min() < threshold:
        return counts.to_dict()
    return None
