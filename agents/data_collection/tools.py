"""
Data Collection Tool — ingest, profile, and validate datasets.

Supports CSV, database (SQLAlchemy), and URL ingestion.
Performs automatic schema detection, problem-type inference, and data profiling.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.config import DataCollectionConfig, Settings
from core.constants import ProblemType
from core.exceptions import DataCollectionError
from core.logging_config import get_logger, log_stage_timing
from core.state import DatasetMetadata, ErrorReport, PipelineState
from core.utils import get_memory_usage_mb, infer_column_types, optimize_dataframe_memory, read_csv_safe
from core.validation import (
    compute_quality_score,
    detect_class_imbalance,
    detect_target_leakage,
    validate_dataframe,
)

logger = get_logger("data_collection")


class DataCollectionService:
    """Deterministic data ingestion and profiling logic."""

    def __init__(self, config: DataCollectionConfig) -> None:
        self._config = config

    def load_csv(self, file_path: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """Load CSV with optional chunked reading for large files."""
        path = Path(file_path)
        if not path.exists():
            raise DataCollectionError(f"File not found: {file_path}")
        if not path.suffix.lower() in (".csv", ".tsv", ".txt"):
            raise DataCollectionError(f"Unsupported file type: {path.suffix}")

        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        file_size_mb = path.stat().st_size / (1024 * 1024)
        effective_chunk = chunk_size or self._config.chunk_size

        if file_size_mb > 100:
            logger.info(f"Large file ({file_size_mb:.0f}MB), using chunked loading")
            chunks = []
            for chunk in read_csv_safe(path, sep=sep, chunksize=effective_chunk, low_memory=False):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = read_csv_safe(path, sep=sep, low_memory=False)

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path.name}")
        return df

    def load_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """Load data from a SQL database."""
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
            logger.info(f"Loaded {len(df)} rows from database")
            return df
        except Exception as e:
            raise DataCollectionError(f"Database load failed: {e}") from e

    def profile_dataset(self, df: pd.DataFrame, target: Optional[str] = None) -> DatasetMetadata:
        """Generate comprehensive dataset profile."""
        col_types = infer_column_types(df)
        null_counts = df.isnull().sum().to_dict()
        null_pcts = df.isnull().mean().to_dict()

        metadata = DatasetMetadata(
            n_rows=len(df),
            n_columns=len(df.columns),
            column_names=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            memory_usage_mb=get_memory_usage_mb(df),
            null_counts={k: int(v) for k, v in null_counts.items()},
            null_percentages={k: round(float(v), 4) for k, v in null_pcts.items()},
            numeric_columns=col_types["numeric"],
            categorical_columns=col_types["categorical"],
            datetime_columns=col_types["datetime"],
            boolean_columns=col_types["boolean"],
            target_column=target,
            unique_counts={col: int(df[col].nunique()) for col in df.columns},
        )
        return metadata

    def detect_problem_type(self, df: pd.DataFrame, target: Optional[str] = None) -> ProblemType:
        """Heuristic-based problem type detection.

        Robust to numeric targets stored as strings (e.g. a price/volume column
        dirtied by a junk header row) by attempting numeric coercion first. A
        numeric target is only treated as classification when it has few distinct
        values — a high-cardinality continuous target (479-class "Volume") is
        regression, not a 479-way classification.
        """
        from core.constants import (
            DEFAULT_MAX_CLASSIFICATION_UNIQUE,
            DEFAULT_MAX_INT_CLASSIFICATION_UNIQUE,
            DEFAULT_NUMERIC_COERCE_FRACTION,
        )

        if target is None or target not in df.columns:
            logger.info("No target column — defaulting to clustering")
            return ProblemType.CLUSTERING

        col = df[target].dropna()
        if len(col) == 0:
            return ProblemType.CLASSIFICATION

        numeric = col
        if not pd.api.types.is_numeric_dtype(col):
            # A column dirtied by stray text (junk rows, "N/A") is still numeric
            # if the overwhelming majority of values parse as numbers.
            coerced = pd.to_numeric(col, errors="coerce")
            if coerced.notna().mean() < DEFAULT_NUMERIC_COERCE_FRACTION:
                # Numbers carrying units ("2.5 sec", "963 hp", "$1,100,000") don't
                # parse directly. Strip $/commas and pull the first numeric token —
                # preprocessing will do the same conversion, so the target type must
                # agree or every model is handed a continuous target as "classes".
                stripped = col.astype(str).str.replace(r"[,$]", "", regex=True)
                coerced = pd.to_numeric(
                    stripped.str.extract(r"(-?\d+\.?\d*)", expand=False), errors="coerce"
                )
            if coerced.notna().mean() >= DEFAULT_NUMERIC_COERCE_FRACTION:
                numeric = coerced.dropna()
            else:
                return ProblemType.CLASSIFICATION  # genuinely categorical / text

        n_unique = int(numeric.nunique())
        if n_unique <= 2:
            return ProblemType.CLASSIFICATION
        if n_unique <= DEFAULT_MAX_CLASSIFICATION_UNIQUE:
            return ProblemType.CLASSIFICATION

        # Integer-coded targets with a modest number of levels relative to the row
        # count are still discrete classes; everything else is regression.
        values = numeric.to_numpy(dtype=float)
        is_integer_like = bool(np.all(np.isfinite(values)) and np.all(np.equal(np.mod(values, 1), 0)))
        ratio = n_unique / max(len(numeric), 1)
        if is_integer_like and n_unique <= DEFAULT_MAX_INT_CLASSIFICATION_UNIQUE and ratio < 0.05:
            return ProblemType.CLASSIFICATION
        return ProblemType.REGRESSION

    @log_stage_timing("data_collection")
    def run(self, state: PipelineState, settings: Settings) -> PipelineState:
        """Execute the full data collection stage."""
        if not state.raw_data_path:
            raise DataCollectionError("No raw_data_path set in pipeline state")

        # Load data
        df = self.load_csv(state.raw_data_path)

        # Optimize memory
        df = optimize_dataframe_memory(df)

        # Validate
        issues = validate_dataframe(df, max_cols=self._config.max_columns)
        if issues:
            for issue in issues:
                logger.warning(f"Data validation issue: {issue}")
            state.data_quality_flags["validation_issues"] = issues

        # Detect problem type
        problem_type = self.detect_problem_type(df, state.target_column)
        state.problem_type = problem_type.value
        logger.info(f"Detected problem type: {problem_type.value}")

        # Profile
        metadata = self.profile_dataset(df, state.target_column)
        metadata.file_path = state.raw_data_path
        state.dataset_metadata = metadata

        # Quality score
        quality = compute_quality_score(df)
        state.data_quality_flags["quality_score"] = quality
        logger.info(f"Data quality score: {quality:.4f}")

        # Leakage check
        if state.target_column and state.target_column in df.columns:
            leaked = detect_target_leakage(df, state.target_column)
            if leaked:
                state.data_quality_flags["potential_leakage"] = leaked
                state.add_error(ErrorReport(
                    severity="high", stage="data_collection",
                    error_type="potential_leakage",
                    root_cause=f"Columns with >0.95 correlation to target: {leaked}",
                    recommended_fix="Review and potentially remove leaked features",
                    retryable=False,
                ))

            # Class imbalance check
            if problem_type == ProblemType.CLASSIFICATION:
                imbalance = detect_class_imbalance(df[state.target_column])
                if imbalance:
                    state.data_quality_flags["class_imbalance"] = imbalance
                    logger.warning(f"Class imbalance detected: {imbalance}")

            # Zero-signal warning: if ALL numeric features have near-zero correlation
            # with the target the dataset is likely synthetic noise. Flag it loudly so
            # the user understands why models will perform at chance level.
            try:
                numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_features = [c for c in numeric_features if c != state.target_column]
                if numeric_features and len(df) > 100:
                    target_vals = pd.to_numeric(df[state.target_column], errors="coerce")
                    corrs = df[numeric_features].apply(
                        lambda col: abs(pd.to_numeric(col, errors="coerce").corr(target_vals))
                    ).dropna()
                    max_corr = float(corrs.max()) if len(corrs) else 0.0
                    avg_corr = float(corrs.mean()) if len(corrs) else 0.0
                    state.data_quality_flags["feature_signal"] = {
                        "max_abs_corr": round(max_corr, 4),
                        "avg_abs_corr": round(avg_corr, 4),
                    }
                    if max_corr < 0.05:
                        state.data_quality_flags["low_signal_warning"] = (
                            f"All numeric features have very low correlation with the target "
                            f"(max |corr|={max_corr:.4f}, avg={avg_corr:.4f}). "
                            f"The dataset may be synthetic noise or the true signal may be "
                            f"entirely in categorical features. Model performance will likely "
                            f"be near-random on numeric features alone."
                        )
                        logger.warning(
                            f"LOW SIGNAL: max feature-target |correlation|={max_corr:.4f}. "
                            f"Models will likely perform at chance level."
                        )
                        state.add_error(ErrorReport(
                            severity="high",
                            stage="data_collection",
                            error_type="low_predictive_signal",
                            root_cause=(
                                f"All numeric features have near-zero correlation with the target "
                                f"(max |corr|={max_corr:.4f}). This strongly suggests the dataset "
                                f"is synthetic noise or the fraud labels were generated randomly "
                                f"relative to the features."
                            ),
                            recommended_fix=(
                                "Verify the dataset is correctly labelled. "
                                "Consider using a different dataset with real fraud signal, "
                                "or add features known to correlate with the outcome."
                            ),
                            retryable=False,
                        ))
            except Exception as _sig_err:
                logger.debug(f"Signal check failed (non-critical): {_sig_err}")

        # Save cleaned path (same as raw for now — preprocessing will create new)
        state.cleaned_data_path = state.raw_data_path
        state.mark_stage_end("data_collection")
        return state


# ── Singleton service ───────────────────────────────────────────────────────

_service: Optional[DataCollectionService] = None
_state: Optional[PipelineState] = None
_settings: Optional[Settings] = None


def init_data_collection(state: PipelineState, settings: Settings) -> None:
    global _service, _state, _settings
    _service = DataCollectionService(settings.data_collection)
    _state = state
    _settings = settings


def collect_data(instruction: str) -> str:
    """Collect and profile a dataset. The instruction should describe what data to load.

    This tool loads the dataset from the configured path, profiles it,
    detects the ML problem type, checks for data quality issues, and
    validates the target column.

    Args:
        instruction: Natural language description of data collection task.

    Returns:
        JSON summary of data collection results.
    """
    global _service, _state, _settings
    if _service is None or _state is None or _settings is None:
        return json.dumps({"error": "Data collection service not initialized"})
    try:
        _state.mark_stage_start("data_collection")
        _state = _service.run(_state, _settings)
        result = {
            "status": "success",
            "problem_type": _state.problem_type,
            "n_rows": _state.dataset_metadata.n_rows if _state.dataset_metadata else 0,
            "n_columns": _state.dataset_metadata.n_columns if _state.dataset_metadata else 0,
            "quality_score": _state.data_quality_flags.get("quality_score", 0),
            "issues": _state.data_quality_flags.get("validation_issues", []),
        }
        return json.dumps(result, default=str)
    except Exception as e:
        logger.exception("Data collection failed")
        _state.add_error(ErrorReport(
            severity="critical", stage="data_collection",
            error_type=type(e).__name__, root_cause=str(e),
            traceback_str=traceback.format_exc(),
            recommended_fix="Check data path and format",
            retryable=True,
        ))
        return json.dumps({"error": str(e)})
