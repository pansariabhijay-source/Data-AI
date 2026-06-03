"""
Feature Engineering Tool — encoding, scaling, feature generation, and selection.
"""

from __future__ import annotations

import json
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


class FeatureEngineeringService:
    """Deterministic feature engineering and selection logic."""

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        self._config = config

    def encode_categoricals(self, df: pd.DataFrame, target: Optional[str] = None) -> tuple[pd.DataFrame, dict[str, str]]:
        """One-hot or label encode categorical columns based on cardinality."""
        encoding_map: dict[str, str] = {}
        cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
        if target and target in cat_cols:
            cat_cols.remove(target)

        for col in cat_cols:
            nunique = df[col].nunique()
            if nunique <= self._config.max_onehot_cardinality:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                encoding_map[col] = f"one_hot ({nunique} categories)"
            else:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoding_map[col] = f"label_encoded ({nunique} categories)"

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
        """Extract useful features from datetime columns."""
        created: list[str] = []
        dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        for col in dt_cols:
            for attr, suffix in [("year", "_year"), ("month", "_month"), ("dayofweek", "_dow"), ("hour", "_hour")]:
                new_col = col + suffix
                try:
                    df[new_col] = getattr(df[col].dt, attr)
                    created.append(new_col)
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
        """Remove features with variance below threshold."""
        from sklearn.feature_selection import VarianceThreshold
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in numeric_cols:
            numeric_cols.remove(target)
        if not numeric_cols:
            return df, []

        selector = VarianceThreshold(threshold=self._config.variance_threshold)
        try:
            selector.fit(df[numeric_cols])
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

        try:
            if problem_type == ProblemType.CLASSIFICATION:
                from sklearn.feature_selection import mutual_info_classif
                scores = mutual_info_classif(X, y, random_state=42)
            else:
                from sklearn.feature_selection import mutual_info_regression
                scores = mutual_info_regression(X, y, random_state=42)

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

        all_removed = low_var_removed + corr_removed
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
