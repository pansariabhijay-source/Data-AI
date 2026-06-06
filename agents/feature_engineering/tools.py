"""
Feature Engineering Tool — encoding, scaling, feature generation, and selection.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.config import FeatureEngineeringConfig, Settings
from core.constants import ProblemType
from core.exceptions import FeatureEngineeringError
from core.logging_config import get_logger, log_stage_timing
from core.state import ErrorReport, FeatureEngineeringSummary, PipelineState
from core.utils import ensure_directory

logger = get_logger("feature_engineering")

# Mutual-information feature ranking is O(n log n) (kNN density estimation) and
# dominates feature_engineering time on large data. It's only used to RANK
# features, and a row sample preserves the ranking of the genuinely informative
# columns (only zero-MI noise columns reshuffle). Cap the rows fed to MI here;
# selection is still applied to the full frame. Override via MI_SAMPLE_ROWS
# (0 disables sampling).
try:
    _MI_SAMPLE_ROWS = int(os.environ.get("MI_SAMPLE_ROWS", "50000"))
except (TypeError, ValueError):
    _MI_SAMPLE_ROWS = 50000


class FeatureEngineeringService:
    """Deterministic feature engineering and selection logic."""

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        self._config = config

    def encode_categoricals(self, df: pd.DataFrame, target: Optional[str] = None) -> tuple[pd.DataFrame, dict[str, str]]:
        """One-hot or label encode categorical columns based on cardinality.

        ID-like columns (near-unique values, e.g. user_id) are dropped rather than
        label-encoded: encoding them injects a high-cardinality noise feature that
        models latch onto and overfit, while carrying no generalizable signal.
        """
        encoding_map: dict[str, str] = {}
        cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
        if target and target in cat_cols:
            cat_cols.remove(target)

        n_rows = max(len(df), 1)
        for col in cat_cols:
            nunique = df[col].nunique()
            if nunique <= self._config.max_onehot_cardinality:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                encoding_map[col] = f"one_hot ({nunique} categories)"
            elif nunique / n_rows > self._config.id_uniqueness_ratio:
                df = df.drop(columns=[col])
                encoding_map[col] = f"dropped (ID-like, {nunique} unique)"
                logger.info(f"Dropped ID-like column '{col}' ({nunique} unique values)")
            else:
                # Frequency-encode high-cardinality columns: map each category to
                # its relative frequency. Unlike label encoding (which injects an
                # arbitrary ordinal that models misread as magnitude), this is a
                # meaningful, leakage-safe signal — rare categories (low frequency)
                # are often the predictive ones (e.g. a rare device/merchant in
                # fraud). Selection downstream drops it if uninformative.
                s = df[col].astype(str)
                freq = s.map(s.value_counts(normalize=True))
                df[col] = freq.astype(float).fillna(0.0)
                encoding_map[col] = f"frequency_encoded ({nunique} categories)"

        return df, encoding_map

    def encode_target(self, df: pd.DataFrame, target: str, problem_type: ProblemType) -> pd.DataFrame:
        """Label encode the target column if it's categorical classification."""
        if problem_type != ProblemType.CLASSIFICATION:
            return df
        if target not in df.columns:
            return df
        if pd.api.types.is_numeric_dtype(df[target]):
            return df
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target].astype(str))
        logger.info(f"Label encoded target '{target}' with {len(le.classes_)} classes")
        return df

    def extract_datetime_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Extract useful features from datetime columns.

        Beyond the raw parts (year/month/dow/hour), add:
        * **cyclical** sin/cos for month, day-of-week and hour, so models (esp.
          linear ones) see that December is adjacent to January and 23:00 to 00:00
          rather than maximally far apart;
        * **is_weekend** (a common, strong signal);
        * **recency_days** — days before the most recent timestamp in the column
          (captures "how old / how recent", e.g. freshly-created scam postings).

        Downstream feature selection prunes any of these that aren't informative.
        """
        created: list[str] = []
        dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        for col in dt_cols:
            s = df[col]
            for attr, suffix in [("year", "_year"), ("month", "_month"), ("dayofweek", "_dow"), ("hour", "_hour")]:
                try:
                    df[col + suffix] = getattr(s.dt, attr)
                    created.append(col + suffix)
                except Exception:
                    pass
            # Cyclical encodings (period-aware): sin/cos so the wrap-around is smooth.
            for attr, period, suffix in [("month", 12, "_month"), ("dayofweek", 7, "_dow"), ("hour", 24, "_hour")]:
                try:
                    vals = getattr(s.dt, attr).astype(float)
                    df[col + suffix + "_sin"] = np.sin(2 * np.pi * vals / period)
                    df[col + suffix + "_cos"] = np.cos(2 * np.pi * vals / period)
                    created += [col + suffix + "_sin", col + suffix + "_cos"]
                except Exception:
                    pass
            try:
                df[col + "_is_weekend"] = (s.dt.dayofweek >= 5).astype(int)
                created.append(col + "_is_weekend")
            except Exception:
                pass
            try:
                df[col + "_recency_days"] = (s.max() - s).dt.days
                created.append(col + "_recency_days")
            except Exception:
                pass
            df = df.drop(columns=[col])
        return df, created

    def scale_features(self, df: pd.DataFrame, target: Optional[str] = None) -> tuple[pd.DataFrame, dict[str, str]]:
        """Scale numeric features."""
        scaling_map: dict[str, str] = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in numeric_cols:
            numeric_cols.remove(target)

        if not numeric_cols or self._config.scaling_method == "none":
            return df, scaling_map

        if self._config.scaling_method == "standard":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif self._config.scaling_method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif self._config.scaling_method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            return df, scaling_map

        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        for col in numeric_cols:
            scaling_map[col] = self._config.scaling_method

        return df, scaling_map

    def remove_low_variance(self, df: pd.DataFrame, target: Optional[str] = None) -> tuple[pd.DataFrame, list[str]]:
        """Remove near-constant features using a *scale-invariant* variance test.

        Raw variance is meaningless across columns of different units (an income
        column has variance in the millions, a 0-1 flag in the hundredths), so an
        absolute threshold would keep high-magnitude noise and drop low-magnitude
        signal. We min-max scale each column to [0, 1] first, making the threshold
        comparable across columns; a truly constant column has scaled variance 0.
        """
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.preprocessing import MinMaxScaler

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in numeric_cols:
            numeric_cols.remove(target)
        if not numeric_cols:
            return df, []

        try:
            scaled = MinMaxScaler().fit_transform(df[numeric_cols].fillna(0))
            selector = VarianceThreshold(threshold=self._config.variance_threshold)
            selector.fit(scaled)
            mask = selector.get_support()
            removed = [c for c, m in zip(numeric_cols, mask) if not m]
            if removed:
                df = df.drop(columns=removed)
                logger.info(f"Removed {len(removed)} low-variance features: {removed}")
            return df, removed
        except Exception as e:
            logger.warning(f"Variance threshold failed: {e}")
            return df, []

    def remove_correlated(self, df: pd.DataFrame, target: Optional[str] = None) -> tuple[pd.DataFrame, list[str]]:
        """Remove one of each pair of highly correlated features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in numeric_cols:
            numeric_cols.remove(target)
        if len(numeric_cols) < 2:
            return df, []

        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self._config.correlation_threshold)]
        if to_drop:
            df = df.drop(columns=to_drop)
            logger.info(f"Removed {len(to_drop)} correlated features: {to_drop}")
        return df, to_drop

    def select_k_best(self, df: pd.DataFrame, target: str, problem_type: ProblemType) -> tuple[pd.DataFrame, dict[str, float]]:
        """Select top K features using mutual information."""
        if target not in df.columns:
            return df, {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
        if not numeric_cols:
            return df, {}

        k = min(self._config.select_k_best, len(numeric_cols))
        X = df[numeric_cols].fillna(0)
        y = df[target]

        # Rank on a row sample to bound MI cost on large data (selection below is
        # still applied to the full frame). The sample preserves the ordering of
        # the informative features; only zero-MI noise columns reshuffle.
        if _MI_SAMPLE_ROWS > 0 and len(X) > _MI_SAMPLE_ROWS:
            samp = np.random.default_rng(42).choice(len(X), size=_MI_SAMPLE_ROWS, replace=False)
            X_mi, y_mi = X.iloc[samp], y.iloc[samp]
        else:
            X_mi, y_mi = X, y

        try:
            if problem_type == ProblemType.CLASSIFICATION:
                from sklearn.feature_selection import mutual_info_classif
                scores = mutual_info_classif(X_mi, y_mi, random_state=42)
            else:
                from sklearn.feature_selection import mutual_info_regression
                scores = mutual_info_regression(X_mi, y_mi, random_state=42)

            importance = dict(zip(numeric_cols, [float(s) for s in scores]))
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            selected = [f for f, _ in sorted_features[:k]]
            # Keep target + selected features + any non-numeric features
            non_numeric = [c for c in df.columns if c not in numeric_cols and c != target]
            keep = [target] + selected + non_numeric
            df = df[[c for c in keep if c in df.columns]]
            logger.info(f"Selected top {k} features by mutual information")
            return df, importance
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return df, {}

    @log_stage_timing("feature_engineering")
    def run(self, state: PipelineState, settings: Settings) -> PipelineState:
        data_path = state.cleaned_data_path
        if not data_path:
            raise FeatureEngineeringError("No cleaned data path available")

        df = pd.read_csv(data_path, low_memory=False)
        n_before = len(df.columns)
        target = state.target_column
        problem_type = ProblemType(state.problem_type) if state.problem_type else ProblemType.UNKNOWN

        # Act on leakage detected during data collection: a feature that is ~perfectly
        # correlated with the target leaks the answer, inflating training metrics while
        # the model fails to generalize. Detection without removal is a silent trap.
        leakage_dropped: list[str] = []
        if self._config.drop_leakage_columns:
            leaked = state.data_quality_flags.get("potential_leakage") or []
            leakage_dropped = [c for c in leaked if c in df.columns and c != target]
            if leakage_dropped:
                df = df.drop(columns=leakage_dropped)
                logger.warning(f"Dropped {len(leakage_dropped)} leakage columns: {leakage_dropped}")

        # Encode target first
        if target:
            df = self.encode_target(df, target, problem_type)

        # Datetime features
        df, dt_created = self.extract_datetime_features(df)

        # Encode categoricals
        df, encoding_map = self.encode_categoricals(df, target)

        # Remove low variance
        df, low_var_removed = self.remove_low_variance(df, target)

        # Remove correlated
        df, corr_removed = self.remove_correlated(df, target)

        # Feature selection
        importances: dict[str, float] = {}
        if target and target in df.columns and problem_type != ProblemType.CLUSTERING:
            df, importances = self.select_k_best(df, target, problem_type)

        # Scale features (after selection)
        df, scaling_map = self.scale_features(df, target)

        # Save
        artifact_dir = ensure_directory(Path(settings.pipeline.artifact_dir) / state.run_id)
        featured_path = artifact_dir / "featured_data.csv"
        df.to_csv(featured_path, index=False)

        all_removed = leakage_dropped + low_var_removed + corr_removed
        state.featured_data_path = str(featured_path)
        state.selected_features = [c for c in df.columns if c != target]
        state.feature_importances = importances
        state.feature_engineering_summary = FeatureEngineeringSummary(
            features_created=dt_created,
            features_removed=all_removed,
            encoding_applied=encoding_map,
            scaling_applied=scaling_map,
            feature_importances=importances,
            selected_features=state.selected_features,
            n_features_before=n_before,
            n_features_after=len(df.columns),
        )
        state.mark_stage_end("feature_engineering")
        logger.info(f"Feature engineering: {n_before}->{len(df.columns)} columns")
        return state


_service: Optional[FeatureEngineeringService] = None
_state: Optional[PipelineState] = None
_settings: Optional[Settings] = None


def init_feature_engineering(state: PipelineState, settings: Settings) -> None:
    global _service, _state, _settings
    _service = FeatureEngineeringService(settings.feature_engineering)
    _state = state
    _settings = settings


def engineer_features(instruction: str) -> str:
    """Engineer and select features for ML modeling.

    Encodes categoricals, extracts datetime features, removes low-variance
    and correlated features, selects top features, and scales numerics.

    Args:
        instruction: Natural language description of feature engineering task.

    Returns:
        JSON summary of feature engineering results.
    """
    global _service, _state, _settings
    if _service is None or _state is None or _settings is None:
        return json.dumps({"error": "Feature engineering service not initialized"})
    try:
        _state.mark_stage_start("feature_engineering")
        _state = _service.run(_state, _settings)
        s = _state.feature_engineering_summary
        return json.dumps({
            "status": "success",
            "features_before": s.n_features_before if s else 0,
            "features_after": s.n_features_after if s else 0,
            "features_created": len(s.features_created) if s else 0,
            "features_removed": len(s.features_removed) if s else 0,
        }, default=str)
    except Exception as e:
        logger.exception("Feature engineering failed")
        _state.add_error(ErrorReport(
            severity="critical", stage="feature_engineering",
            error_type=type(e).__name__, root_cause=str(e),
            traceback_str=traceback.format_exc(),
            recommended_fix="Check encoding and scaling settings",
            retryable=True,
        ))
        return json.dumps({"error": str(e)})
