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

        # Decide the split strategy. Precedence (each guarded by class-completeness):
        #   1. grouped out-of-time  — entity + time: whole entities, oldest→newest
        #   2. out-of-time          — time only: chronological
        #   3. group-aware          — entity only: whole entities, stratified
        #   4. stratified-random    — fallback
        from core.constants import AXIOM_SPLIT_GROUP_COL
        time_col = self._detect_time_column(df, target) if self._config.time_aware_split != "off" else None
        if self._config.time_aware_split == "on" and time_col is None:
            logger.warning("time_aware_split='on' but no usable time column found — using stratified random")
        group_col = (
            AXIOM_SPLIT_GROUP_COL
            if (self._config.group_aware_split != "off" and AXIOM_SPLIT_GROUP_COL in df.columns)
            else None
        )
        entity_name = state.data_quality_flags.get("entity_column", "entity")

        def _class_ok(parts) -> bool:
            if problem_type != ProblemType.CLASSIFICATION:
                return all(p is not None and len(p) > 0 for p in parts)
            nc = df[target].nunique()
            return all(p is not None and len(p) > 0 and p[target].nunique() >= nc for p in parts)

        train = val = test = None
        strategy = "stratified-random"

        # 1) Grouped out-of-time — no entity straddles the boundary AND train is the past.
        if group_col is not None and time_col is not None:
            parts = self._grouped_chronological_split(
                df, time_col, group_col, train_ratio, val_ratio, test_ratio
            )
            if parts is not None and _class_ok(parts):
                train, val, test = parts
                strategy = "out-of-time"
                state.data_quality_flags["group_aware"] = entity_name
                logger.info(
                    f"Grouped out-of-time split on '{entity_name}'+time: "
                    f"train(oldest)={len(train)}, val={len(val)}, test(newest)={len(test)}"
                )

        # 2) Pure out-of-time — time axis, no usable entity grouping.
        if train is None and time_col is not None:
            parts = self._chronological_split(df, time_col, train_ratio, val_ratio, test_ratio)
            if _class_ok(parts):
                train, val, test = parts
                strategy = "out-of-time"
                logger.info(f"Out-of-time split on '{time_col}': train(oldest)={len(train)}, val={len(val)}, test(newest)={len(test)}")
            else:
                logger.warning(
                    "Out-of-time split produced a class-incomplete fold "
                    "(rare class concentrated in time) — trying group-aware / stratified."
                )

        # 3) Group-aware (non-temporal) — keep each entity within one split.
        if train is None and group_col is not None:
            parts = self._group_aware_split(
                df, target, group_col, problem_type, train_ratio, val_ratio, test_ratio
            )
            if parts is not None and _class_ok(parts):
                train, val, test = parts
                strategy = "group-aware"
                state.data_quality_flags["group_aware"] = entity_name
                logger.info(
                    f"Group-aware split on '{entity_name}': "
                    f"train={len(train)}, val={len(val)}, test={len(test)}"
                )

        # 4) Fallback: stratified random.
        if train is None:
            train, val, test = self._random_split(df, target, problem_type, train_ratio, val_ratio, test_ratio)
            strategy = "stratified-random"
            logger.info(f"Stratified-random split: train={len(train)}, val={len(val)}, test={len(test)}")

        # CRITICAL: drop the hidden split keys so they never become model features
        # (the time key proxies the label boundary; the group key proxies identity).
        from core.constants import AXIOM_SPLIT_TIME_COL
        _hidden = [AXIOM_SPLIT_TIME_COL, AXIOM_SPLIT_GROUP_COL]
        train = train.drop(columns=_hidden, errors="ignore")
        val = val.drop(columns=_hidden, errors="ignore")
        test = test.drop(columns=_hidden, errors="ignore")

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

    def _grouped_chronological_split(self, df, time_col, group_col, train_ratio, val_ratio, test_ratio):
        """Out-of-time split at ENTITY granularity: order entities by first-seen time,
        assign whole entities oldest→train / newest→test, then sort each split by time.

        This prevents BOTH future leakage (train is the past) and identity leakage
        (an entity's rows never straddle the boundary). Returns ``None`` if the
        result would be degenerate (any split empty), so the caller can fall back.
        """
        try:
            first_seen = df.groupby(group_col)[time_col].min().sort_values(kind="mergesort")
            sizes = df.groupby(group_col).size()
            n = len(df)
            i_train, i_val = n * train_ratio, n * (train_ratio + val_ratio)
            assign: dict = {}
            cum = 0
            for g in first_seen.index:
                if cum < i_train:
                    assign[g] = "train"
                elif cum < i_val:
                    assign[g] = "val"
                else:
                    assign[g] = "test"
                cum += int(sizes[g])
            bucket = df[group_col].map(assign)
            # Sort each split by time so the rows remain chronological (keeps the
            # out-of-time guarantee and makes downstream time-aware CV valid).
            parts = tuple(
                df[bucket == name].sort_values(time_col, kind="mergesort")
                for name in ("train", "val", "test")
            )
            if any(len(p) == 0 for p in parts):
                return None
            return parts
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Grouped out-of-time split failed ({e}); will fall back")
            return None

    def _group_aware_split(self, df, target, group_col, problem_type, train_ratio, val_ratio, test_ratio):
        """Non-temporal split that keeps each entity entirely within one split.

        Classification uses StratifiedGroupKFold (preserves class balance AND group
        integrity); regression uses GroupShuffleSplit. Returns ``None`` on any
        failure (e.g. too few groups for the requested folds) so the caller falls
        back to a stratified-random split.
        """
        import numpy as np

        groups = df[group_col].to_numpy()
        n = len(df)
        pos = np.arange(n)
        try:
            if problem_type == ProblemType.CLASSIFICATION:
                from sklearn.model_selection import StratifiedGroupKFold
                y = df[target].to_numpy()
                n_test = max(2, int(round(1.0 / test_ratio)))
                sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=self._seed)
                trval_pos, test_pos = next(sgkf.split(pos, y, groups))
                val_frac = val_ratio / (train_ratio + val_ratio)
                n_val = max(2, int(round(1.0 / val_frac)))
                sgkf2 = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=self._seed)
                sub_tr, sub_val = next(sgkf2.split(trval_pos, y[trval_pos], groups[trval_pos]))
                train_pos, val_pos = trval_pos[sub_tr], trval_pos[sub_val]
            else:
                from sklearn.model_selection import GroupShuffleSplit
                gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=self._seed)
                trval_pos, test_pos = next(gss.split(pos, groups=groups))
                val_frac = val_ratio / (train_ratio + val_ratio)
                gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=self._seed)
                sub_tr, sub_val = next(gss2.split(trval_pos, groups=groups[trval_pos]))
                train_pos, val_pos = trval_pos[sub_tr], trval_pos[sub_val]

            parts = (df.iloc[train_pos], df.iloc[val_pos], df.iloc[test_pos])
            if any(len(p) == 0 for p in parts):
                return None
            # Sanity: no entity may appear in more than one split.
            s_tr, s_val, s_te = (set(df[group_col].iloc[p]) for p in (train_pos, val_pos, test_pos))
            if (s_tr & s_val) or (s_tr & s_te) or (s_val & s_te):
                logger.warning("Group-aware split leaked an entity across folds; falling back")
                return None
            return parts
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Group-aware split failed ({e}); will fall back")
            return None

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
