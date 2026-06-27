"""
Data Splitting Tool — stratified/standard splits with leakage prevention.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

from core.config import Settings, SplittingConfig
from core.constants import ProblemType
from core.exceptions import SplittingError
from core.logging_config import get_logger, log_stage_timing
from core.state import ErrorReport, PipelineState
from core.utils import ensure_directory

logger = get_logger("splitting")


class DataSplittingService:
    def __init__(self, config: SplittingConfig, seed: int = 42) -> None:
        self._config = config
        self._seed = seed

    @log_stage_timing("data_splitting")
    def run(self, state: PipelineState, settings: Settings) -> PipelineState:
        data_path = state.featured_data_path or state.cleaned_data_path
        if not data_path:
            raise SplittingError("No data path available for splitting")

        df = pd.read_csv(data_path, low_memory=False)
        target = state.target_column
        problem_type = ProblemType(state.problem_type) if state.problem_type else ProblemType.UNKNOWN

        if problem_type == ProblemType.CLUSTERING:
            # No split needed for clustering — use full dataset
            artifact_dir = ensure_directory(Path(settings.pipeline.artifact_dir) / state.run_id)
            train_path = artifact_dir / "train.csv"
            df.to_csv(train_path, index=False)
            state.train_path = str(train_path)
            state.split_ratios = {"train": 1.0}
            state.mark_stage_end("data_splitting")
            logger.info("Clustering mode: skipping split, using full dataset")
            return state

        if not target or target not in df.columns:
            raise SplittingError(f"Target column '{target}' not found in dataset")

        train_ratio = self._config.train_ratio
        val_ratio = self._config.val_ratio
        test_ratio = self._config.test_ratio

        # Decide between out-of-time (chronological) and stratified-random splitting.
        time_col = self._detect_time_column(df, target) if self._config.time_aware_split != "off" else None
        if self._config.time_aware_split == "on" and time_col is None:
            logger.warning("time_aware_split='on' but no usable time column found — using stratified random")

        train = val = test = None
        strategy = "stratified-random"
        if time_col is not None:
            train, val, test = self._chronological_split(df, time_col, train_ratio, val_ratio, test_ratio)
            # Guard: a chronological slice can be degenerate on imbalanced data (e.g.
            # the newest period contains no fraud). If any split loses a class, the
            # out-of-time split is unusable — fall back to a stratified random split.
            if problem_type == ProblemType.CLASSIFICATION:
                n_classes = df[target].nunique()
                if any(part[target].nunique() < n_classes for part in (train, val, test)):
                    logger.warning(
                        "Out-of-time split produced a class-incomplete fold "
                        "(rare class concentrated in time) — falling back to stratified random."
                    )
                    train = None
            if train is not None:
                strategy = "out-of-time"
                logger.info(f"Out-of-time split on '{time_col}': train(oldest)={len(train)}, val={len(val)}, test(newest)={len(test)}")

        if train is None:
            train, val, test = self._random_split(df, target, problem_type, train_ratio, val_ratio, test_ratio)
            strategy = "stratified-random"
            logger.info(f"Stratified-random split: train={len(train)}, val={len(val)}, test={len(test)}")

        # CRITICAL: drop the hidden split-time key so it never becomes a model
        # feature (it would be a perfect proxy for the chronological label boundary).
        from core.constants import AXIOM_SPLIT_TIME_COL
        train = train.drop(columns=[AXIOM_SPLIT_TIME_COL], errors="ignore")
        val = val.drop(columns=[AXIOM_SPLIT_TIME_COL], errors="ignore")
        test = test.drop(columns=[AXIOM_SPLIT_TIME_COL], errors="ignore")

        # Save splits
        artifact_dir = ensure_directory(Path(settings.pipeline.artifact_dir) / state.run_id)
        train_path = artifact_dir / "train.csv"
        val_path = artifact_dir / "val.csv"
        test_path = artifact_dir / "test.csv"

        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)

        state.train_path = str(train_path)
        state.val_path = str(val_path)
        state.test_path = str(test_path)
        state.split_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
        state.data_quality_flags["split_strategy"] = strategy

        state.mark_stage_end("data_splitting")
        logger.info(f"Split complete ({strategy}): train={len(train)}, val={len(val)}, test={len(test)}")
        return state

    def _detect_time_column(self, df: pd.DataFrame, target: str) -> Optional[str]:
        """Find a usable time axis to sort by, or None.

        Prefers the hidden key threaded from feature engineering (the true primary
        timestamp). Otherwise looks for a raw time-like column that survived as a
        feature (e.g. creditcard's numeric ``Time`` in seconds) by name + shape:
        numeric/datetime, mostly non-null, and high-cardinality (a genuine axis, not
        a decomposed part like ``*_year`` or a cyclical ``*_sin``).
        """
        import re
        from core.constants import AXIOM_SPLIT_TIME_COL, TIME_COLUMN_NAME_PATTERN

        if AXIOM_SPLIT_TIME_COL in df.columns and df[AXIOM_SPLIT_TIME_COL].notna().mean() > 0.5:
            return AXIOM_SPLIT_TIME_COL

        pat = re.compile(TIME_COLUMN_NAME_PATTERN)
        decomposed = ("_year", "_month", "_dow", "_hour", "_sin", "_cos",
                      "_is_weekend", "_recency_days", "_was_missing")
        best, best_card = None, 0
        n = max(len(df), 1)
        for c in df.columns:
            if c == target or not pat.search(c) or c.endswith(decomposed):
                continue
            s = df[c]
            if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)):
                continue
            if s.notna().mean() < 0.9:
                continue
            card = s.nunique()
            # Require a real spread (not a handful of buckets) to call it a time axis.
            if card > max(20, 0.05 * n) and card > best_card:
                best, best_card = c, card
        return best

    def _chronological_split(self, df, time_col, train_ratio, val_ratio, test_ratio):
        """Sort by time and slice oldest→train, middle→val, newest→test (no shuffle)."""
        ordered = df.sort_values(time_col, kind="mergesort")  # stable
        n = len(ordered)
        i_train = int(round(n * train_ratio))
        i_val = int(round(n * (train_ratio + val_ratio)))
        return ordered.iloc[:i_train], ordered.iloc[i_train:i_val], ordered.iloc[i_val:]

    def _random_split(self, df, target, problem_type, train_ratio, val_ratio, test_ratio):
        from sklearn.model_selection import train_test_split

        stratify_col = df[target] if problem_type == ProblemType.CLASSIFICATION else None
        try:
            train_val, test = train_test_split(
                df, test_size=test_ratio, random_state=self._seed, stratify=stratify_col,
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed ({e}), falling back to random")
            train_val, test = train_test_split(df, test_size=test_ratio, random_state=self._seed)

        val_fraction = val_ratio / (train_ratio + val_ratio)
        stratify_tv = train_val[target] if problem_type == ProblemType.CLASSIFICATION else None
        try:
            train, val = train_test_split(
                train_val, test_size=val_fraction, random_state=self._seed, stratify=stratify_tv,
            )
        except ValueError:
            train, val = train_test_split(train_val, test_size=val_fraction, random_state=self._seed)
        return train, val, test


_service: Optional[DataSplittingService] = None
_state: Optional[PipelineState] = None
_settings: Optional[Settings] = None


def init_splitting(state: PipelineState, settings: Settings) -> None:
    global _service, _state, _settings
    _service = DataSplittingService(settings.splitting, settings.pipeline.random_seed)
    _state = state
    _settings = settings


def split_data(instruction: str) -> str:
    """Split the dataset into train, validation, and test sets.

    Uses stratified splitting for classification, standard for regression,
    and bypasses splitting for clustering.

    Args:
        instruction: Description of splitting task.

    Returns:
        JSON summary of split results.
    """
    global _service, _state, _settings
    if _service is None or _state is None or _settings is None:
        return json.dumps({"error": "Splitting service not initialized"})
    try:
        _state.mark_stage_start("data_splitting")
        _state = _service.run(_state, _settings)
        return json.dumps({
            "status": "success",
            "split_ratios": _state.split_ratios,
            "train_path": _state.train_path,
            "val_path": _state.val_path,
            "test_path": _state.test_path,
        }, default=str)
    except Exception as e:
        logger.exception("Data splitting failed")
        _state.add_error(ErrorReport(
            severity="critical", stage="data_splitting",
            error_type=type(e).__name__, root_cause=str(e),
            traceback_str=traceback.format_exc(),
            recommended_fix="Check target column and class distribution",
            retryable=True,
        ))
        return json.dumps({"error": str(e)})
