"""Fix #4 — group-aware (entity) splitting prevents identity leakage.

The same entity (card/user/account) must never appear in more than one split, else
the model can memorise the entity instead of learning the pattern — inflated offline
metrics that collapse in production. These tests pin the entity-integrity guarantee
for both the temporal and non-temporal group paths, and that the hidden split keys
never leak into the saved feature matrices.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("crewai", MagicMock())
sys.modules.setdefault("crewai.tools", MagicMock())

import numpy as np
import pandas as pd
import pytest

from core.config import load_settings
from core.constants import (
    AXIOM_SPLIT_GROUP_COL,
    AXIOM_SPLIT_TIME_COL,
    ProblemType,
)
from core.state import PipelineState
from agents.splitting.tools import DataSplittingService


def _entity_df(n_groups=25, per_group=24, seed=0, with_time=True):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        t0 = int(rng.integers(0, 5000))
        for j in range(per_group):
            row = {
                "f1": float(rng.normal()),
                "f2": float(rng.normal()),
                "target": int(rng.random() < 0.3),
                AXIOM_SPLIT_GROUP_COL: f"user_{g}",
            }
            if with_time:
                row[AXIOM_SPLIT_TIME_COL] = t0 + j
            rows.append(row)
    return pd.DataFrame(rows)


def _svc():
    return DataSplittingService(load_settings().splitting, seed=42)


def _no_entity_overlap(parts):
    sets = [set(p[AXIOM_SPLIT_GROUP_COL]) for p in parts]
    return not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_grouped_chronological_no_entity_overlap_and_ordered():
    df = _entity_df(with_time=True)
    parts = _svc()._grouped_chronological_split(
        df, AXIOM_SPLIT_TIME_COL, AXIOM_SPLIT_GROUP_COL, 0.7, 0.15, 0.15
    )
    assert parts is not None
    tr, va, te = parts
    assert _no_entity_overlap(parts)
    assert len(tr) + len(va) + len(te) == len(df)
    # Train is the past: its first-seen entities precede the test entities.
    assert tr[AXIOM_SPLIT_TIME_COL].min() <= te[AXIOM_SPLIT_TIME_COL].min()


def test_group_aware_split_no_entity_overlap_classification():
    df = _entity_df(with_time=False)
    parts = _svc()._group_aware_split(
        df, "target", AXIOM_SPLIT_GROUP_COL, ProblemType.CLASSIFICATION, 0.7, 0.15, 0.15
    )
    assert parts is not None
    assert _no_entity_overlap(parts)
    # Both classes present in every split (StratifiedGroupKFold).
    for p in parts:
        assert p["target"].nunique() == 2


def test_group_aware_split_regression():
    df = _entity_df(with_time=False)
    df["target"] = np.arange(len(df), dtype=float)  # continuous
    parts = _svc()._group_aware_split(
        df, "target", AXIOM_SPLIT_GROUP_COL, ProblemType.REGRESSION, 0.7, 0.15, 0.15
    )
    assert parts is not None
    assert _no_entity_overlap(parts)


def test_run_drops_hidden_keys_and_records_group_strategy(tmp_path):
    """End-to-end: the saved train/val/test must NOT contain either hidden key, and
    the run must record a group-aware strategy."""
    df = _entity_df(with_time=True)
    featured = tmp_path / "featured.csv"
    df.to_csv(featured, index=False)

    state = PipelineState(run_id="grp", target_column="target",
                          featured_data_path=str(featured))
    state.problem_type = "classification"
    state.data_quality_flags["entity_column"] = "cc_num"

    settings = load_settings()
    settings.pipeline.artifact_dir = str(tmp_path / "artifacts")
    _svc_ = DataSplittingService(settings.splitting, seed=42)
    _svc_.run(state, settings)

    for path in (state.train_path, state.val_path, state.test_path):
        cols = pd.read_csv(path).columns
        assert AXIOM_SPLIT_TIME_COL not in cols
        assert AXIOM_SPLIT_GROUP_COL not in cols
    # Entity + time present => grouped out-of-time, recorded as out-of-time.
    assert state.data_quality_flags["split_strategy"] == "out-of-time"
    assert state.data_quality_flags.get("group_aware") == "cc_num"


def test_group_aware_disabled_falls_back(tmp_path):
    """With group_aware_split='off' and no time axis, the entity key is ignored and
    a plain stratified-random split is used."""
    df = _entity_df(with_time=False)
    featured = tmp_path / "featured.csv"
    df.to_csv(featured, index=False)
    state = PipelineState(run_id="grp2", target_column="target",
                          featured_data_path=str(featured))
    state.problem_type = "classification"
    settings = load_settings()
    settings.splitting.group_aware_split = "off"
    settings.splitting.time_aware_split = "off"
    settings.pipeline.artifact_dir = str(tmp_path / "artifacts")
    DataSplittingService(settings.splitting, seed=42).run(state, settings)
    assert state.data_quality_flags["split_strategy"] == "stratified-random"
