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

        from sklearn.model_selection import train_test_split

        train_ratio = self._config.train_ratio
        val_ratio = self._config.val_ratio
        test_ratio = self._config.test_ratio

        stratify_col = df[target] if problem_type == ProblemType.CLASSIFICATION else None

        # First split: train+val vs test
        try:
            train_val, test = train_test_split(
                df, test_size=test_ratio, random_state=self._seed,
                stratify=stratify_col,
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed ({e}), falling back to random")
            train_val, test = train_test_split(df, test_size=test_ratio, random_state=self._seed)

        # Second split: train vs val
        val_fraction = val_ratio / (train_ratio + val_ratio)
        stratify_tv = train_val[target] if problem_type == ProblemType.CLASSIFICATION else None
        try:
            train, val = train_test_split(
                train_val, test_size=val_fraction, random_state=self._seed,
                stratify=stratify_tv,
            )
        except ValueError:
            train, val = train_test_split(train_val, test_size=val_fraction, random_state=self._seed)

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

        state.mark_stage_end("data_splitting")
        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return state


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
