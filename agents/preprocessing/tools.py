"""
Preprocessing Tool — clean, impute, and prepare data for feature engineering.

Handles: missing values, duplicates, outliers, dtype fixing, high-cardinality detection.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.config import PreprocessingConfig, Settings
from core.constants import ProblemType
from core.exceptions import PreprocessingError
from core.logging_config import get_logger, log_stage_timing
from core.state import ErrorReport, PipelineState, PreprocessingSummary
from core.utils import ensure_directory, get_memory_usage_mb, optimize_dataframe_memory, read_csv_safe
from core.validation import compute_quality_score

logger = get_logger("preprocessing")

# Whether a value is MISSING is often predictive (especially for fraud), but
# imputation erases that signal. Add a binary `<col>__was_missing` feature before
# filling, for columns whose null fraction is at least this much. Downstream
# feature selection drops the indicator if it turns out to be uninformative.
# Disable with ADD_MISSING_INDICATORS=0.
try:
    _ADD_MISSING_INDICATORS = int(os.environ.get("ADD_MISSING_INDICATORS", "1"))
except ValueError:
    _ADD_MISSING_INDICATORS = 1
try:
    _MISSING_INDICATOR_MIN_FRAC = float(os.environ.get("MISSING_INDICATOR_MIN_FRAC", "0.01"))
except ValueError:
    _MISSING_INDICATOR_MIN_FRAC = 0.01


class PreprocessingService:
    """Deterministic data cleaning and preprocessing logic."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self._config = config

    def remove_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        before = len(df)
        df = df.drop_duplicates(keep=self._config.duplicate_keep)
        removed = before - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        return df, removed

    def handle_missing_values(self, df: pd.DataFrame, target: Optional[str] = None) -> tuple[pd.DataFrame, dict[str, str]]:
        """Fill or drop missing values based on column type and null ratio."""
        strategies: dict[str, str] = {}
        cols_to_drop: list[str] = []

        # Drop rows with a missing target first, so the row count is stable before
        # we add any missingness-indicator columns (which must align to df length).
        if target and target in df.columns:
            n_before = len(df)
            df = df.dropna(subset=[target])
            if len(df) < n_before:
                strategies[target] = f"dropped {n_before - len(df)} rows with null target"

        n_indicators = 0
        for col in df.columns:
            if col == target or col.endswith("__was_missing"):
                continue

            null_pct = df[col].isnull().mean()
            if null_pct == 0:
                continue

            if null_pct > self._config.max_null_threshold:
                cols_to_drop.append(col)
                strategies[col] = f"dropped (>{self._config.max_null_threshold:.0%} null)"
                continue

            # Preserve the missingness signal as a feature before imputing.
            if _ADD_MISSING_INDICATORS and null_pct >= _MISSING_INDICATOR_MIN_FRAC:
                df[f"{col}__was_missing"] = df[col].isnull().astype("int8")
                n_indicators += 1

            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                strategies[col] = f"filled with median ({median_val:.4g})"
            elif pd.api.types.is_bool_dtype(df[col]):
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else False
                df[col] = df[col].fillna(mode_val)
                strategies[col] = f"filled with mode ({mode_val})"
            else:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
                df[col] = df[col].fillna(mode_val)
                strategies[col] = f"filled with mode ({mode_val})"

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} high-null columns: {cols_to_drop}")
        if n_indicators:
            logger.info(f"Added {n_indicators} missing-value indicator feature(s)")

        return df, strategies

    def fix_dtypes(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        """Attempt to infer and fix column dtypes."""
        fixes: dict[str, str] = {}
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) in ("string", "str"):
                # Try numeric conversion
                numeric = pd.to_numeric(df[col], errors="coerce")
                pct_numeric = numeric.notna().mean()
                if pct_numeric > 0.8:
                    df[col] = numeric
                    fixes[col] = "converted to numeric"
                    continue

                # Try datetime BEFORE unit-extraction. (Pandas >=2 removed
                # ``infer_datetime_format`` — it now infers per-element by default, so
                # passing it raises TypeError and silently leaves date columns as
                # strings.) This ordering matters: a date like "2023-01-01 00:00:00"
                # would otherwise be clobbered by the unit-extractor below, which would
                # pull the leading number ("2023") and destroy the timestamp.
                try:
                    import warnings as _warnings
                    with _warnings.catch_warnings():
                        _warnings.simplefilter("ignore")
                        dt = pd.to_datetime(df[col], errors="coerce")
                    if dt.notna().mean() > 0.8:
                        df[col] = dt
                        fixes[col] = "converted to datetime"
                        continue
                except Exception:
                    pass

                # Try unit-laden numerics, e.g. "963 hp", "3990 cc", "2.5 sec",
                # "$1,100,000", "800 Nm", "340 km/h". Plain to_numeric fails on these
                # because of the currency/thousands separators and trailing units, so
                # the column would be wrongly treated as a high-cardinality category
                # and its real signal lost. Strip $ and commas, then pull the first
                # numeric token; convert only if most rows yield a number AND the
                # result isn't constant (avoids turning a code/label into a number).
                stripped = df[col].astype(str).str.replace(r"[,$]", "", regex=True)
                extracted = pd.to_numeric(
                    stripped.str.extract(r"(-?\d+\.?\d*)", expand=False), errors="coerce"
                )
                if extracted.notna().mean() > 0.8 and extracted.nunique() > 1:
                    df[col] = extracted
                    fixes[col] = "extracted numeric from units"
                    continue

                # Try boolean
                unique_lower = set(df[col].dropna().astype(str).str.lower().unique())
                if unique_lower.issubset({"true", "false", "yes", "no", "0", "1"}):
                    bool_map = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}
                    df[col] = df[col].astype(str).str.lower().map(bool_map)
                    fixes[col] = "converted to boolean"

        return df, fixes

    def handle_outliers(self, df: pd.DataFrame, target: Optional[str] = None) -> tuple[pd.DataFrame, dict[str, int]]:
        """Winsorize outliers using IQR method on *continuous* numeric columns.

        Guards against erasing real signal: discrete/low-cardinality columns
        (binary flags, encoded categoricals) are skipped, and a column is left
        untouched if winsorizing would clip more than ``max_outlier_fraction`` of
        its rows — at that point the "outliers" are the distribution itself
        (e.g. a minority class or rare-event cluster), not noise.
        """
        handled: dict[str, int] = {}
        skipped: list[str] = []
        if self._config.outlier_method != "iqr":
            return df, handled

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in numeric_cols:
            numeric_cols.remove(target)

        n_rows = len(df)
        max_clip = self._config.max_outlier_fraction * n_rows if n_rows else 0

        for col in numeric_cols:
            non_null = df[col].dropna()
            # Skip discrete columns — clipping them destroys category structure.
            if non_null.nunique() <= self._config.min_unique_for_outlier:
                skipped.append(col)
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                skipped.append(col)
                continue

            lower = q1 - self._config.iqr_multiplier * iqr
            upper = q3 + self._config.iqr_multiplier * iqr
            outlier_mask = (df[col] < lower) | (df[col] > upper)
            count = int(outlier_mask.sum())
            if count == 0:
                continue
            # If "outliers" dominate, this is the real distribution — don't clip.
            if count > max_clip:
                skipped.append(col)
                logger.info(
                    f"Skipped outlier clipping for '{col}': {count}/{n_rows} "
                    f"({count / n_rows:.1%}) flagged — treated as signal, not noise"
                )
                continue
            df[col] = df[col].clip(lower=lower, upper=upper)
            handled[col] = count

        if handled:
            total = sum(handled.values())
            logger.info(f"Winsorized {total} outliers across {len(handled)} columns")
        return df, handled

    def detect_high_cardinality(self, df: pd.DataFrame) -> list[str]:
        """Detect categorical columns with cardinality above threshold."""
        high_card: list[str] = []
        for col in df.select_dtypes(include=["object", "category", "str"]).columns:
            nunique = df[col].nunique()
            if nunique > self._config.max_cardinality:
                high_card.append(col)
        return high_card

    @log_stage_timing("preprocessing")
    def run(self, state: PipelineState, settings: Settings) -> PipelineState:
        """Execute the full preprocessing stage."""
        data_path = state.cleaned_data_path or state.raw_data_path
        if not data_path:
            raise PreprocessingError("No data path available")

        df = read_csv_safe(data_path, low_memory=False)
        rows_before = len(df)
        cols_before = len(df.columns)

        # Fix dtypes
        df, dtype_fixes = self.fix_dtypes(df)

        # Remove duplicates
        df, dups_removed = self.remove_duplicates(df)

        # Handle missing values
        df, null_strategies = self.handle_missing_values(df, state.target_column)

        # Handle outliers
        df, outliers = self.handle_outliers(df, state.target_column)

        # Detect high cardinality
        high_card = self.detect_high_cardinality(df)

        # Optimize memory
        df = optimize_dataframe_memory(df)

        # Quality score after cleaning
        quality = compute_quality_score(df)

        # Save cleaned data
        artifact_dir = ensure_directory(Path(settings.pipeline.artifact_dir) / state.run_id)
        cleaned_path = artifact_dir / "cleaned_data.csv"
        df.to_csv(cleaned_path, index=False)

        # Update state
        cols_dropped = [c for c, s in null_strategies.items() if "dropped" in s and "rows" not in s]
        state.cleaned_data_path = str(cleaned_path)
        state.preprocessing_summary = PreprocessingSummary(
            rows_before=rows_before, rows_after=len(df),
            columns_before=cols_before, columns_after=len(df.columns),
            duplicates_removed=dups_removed,
            nulls_filled=null_strategies, outliers_handled=outliers,
            dtypes_fixed=dtype_fixes,
            high_cardinality_columns=high_card,
            columns_dropped=cols_dropped, quality_score=quality,
        )
        state.mark_stage_end("preprocessing")
        logger.info(f"Preprocessing complete: {rows_before}->{len(df)} rows, quality={quality:.4f}")
        return state


_service: Optional[PreprocessingService] = None
_state: Optional[PipelineState] = None
_settings: Optional[Settings] = None


def init_preprocessing(state: PipelineState, settings: Settings) -> None:
    global _service, _state, _settings
    _service = PreprocessingService(settings.preprocessing)
    _state = state
    _settings = settings


def preprocess_data(instruction: str) -> str:
    """Clean and preprocess the dataset.

    Handles missing values, duplicates, outliers (IQR winsorization),
    dtype fixes, and high-cardinality detection.

    Args:
        instruction: Natural language description of preprocessing task.

    Returns:
        JSON summary of preprocessing results.
    """
    global _service, _state, _settings
    if _service is None or _state is None or _settings is None:
        return json.dumps({"error": "Preprocessing service not initialized"})
    try:
        _state.mark_stage_start("preprocessing")
        _state = _service.run(_state, _settings)
        s = _state.preprocessing_summary
        result = {
            "status": "success",
            "rows": f"{s.rows_before}->{s.rows_after}" if s else "N/A",
            "columns": f"{s.columns_before}->{s.columns_after}" if s else "N/A",
            "duplicates_removed": s.duplicates_removed if s else 0,
            "quality_score": s.quality_score if s else 0,
        }
        return json.dumps(result, default=str)
    except Exception as e:
        logger.exception("Preprocessing failed")
        _state.add_error(ErrorReport(
            severity="critical", stage="preprocessing",
            error_type=type(e).__name__, root_cause=str(e),
            traceback_str=traceback.format_exc(),
            recommended_fix="Check data format and column types",
            retryable=True,
        ))
        return json.dumps({"error": str(e)})
