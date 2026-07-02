"""Fix #5 — per-entity aggregate features must be time-safe (causal).

Whole-dataset groupby stats (mean/std/max/count over ALL of a card's rows) leak the
future: at prediction time you only know the card's PAST. These tests pin that each
row's aggregate reflects only earlier transactions for that entity.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("crewai", MagicMock())
sys.modules.setdefault("crewai.tools", MagicMock())

import numpy as np
import pandas as pd

from core.config import load_settings
from agents.feature_engineering.tools import FeatureEngineeringService


def _svc():
    return FeatureEngineeringService(load_settings().feature_engineering)


def test_time_safe_aggregates_use_only_the_past():
    # One card, amounts 10,20,30,40 in time order.
    df = pd.DataFrame({
        "cc_num": ["c1", "c1", "c1", "c1"],
        "amt": [10.0, 20.0, 30.0, 40.0],
        "t": [1, 2, 3, 4],
    })
    out = FeatureEngineeringService._time_safe_entity_aggregates(df, "cc_num", "amt", df["t"])
    # Count of PRIOR transactions.
    assert list(out["card_tx_count"]) == [0, 1, 2, 3]
    # Mean over the past only (row 0 has no past -> 0).
    assert list(out["card_amt_mean"]) == [0.0, 10.0, 15.0, 20.0]
    # Max over the past only (never includes the current or future row).
    assert list(out["card_amt_max"]) == [0.0, 10.0, 20.0, 30.0]
    # Crucially: the LAST row's mean is 20 (=mean 10,20,30), NOT 25 (the whole-card
    # mean that a leaky groupby('mean') would produce).
    assert out["card_amt_mean"].iloc[-1] == 20.0


def test_time_safe_aggregates_are_per_entity():
    df = pd.DataFrame({
        "cc_num": ["a", "b", "a", "b"],
        "amt": [100.0, 5.0, 200.0, 7.0],
        "t": [1, 1, 2, 2],
    })
    out = FeatureEngineeringService._time_safe_entity_aggregates(
        df, "cc_num", "amt", df["t"]
    )
    # Row 2 (card a, second txn) sees only card a's first amount (100), not b's.
    assert out["card_amt_mean"].iloc[2] == 100.0
    # Row 3 (card b, second txn) sees only card b's first amount (5), not a's.
    assert out["card_amt_mean"].iloc[3] == 5.0


def test_time_safe_aggregates_realign_to_original_order():
    # Rows given OUT of time order; output must align to the original row positions.
    df = pd.DataFrame({
        "cc_num": ["c1", "c1", "c1"],
        "amt": [30.0, 10.0, 20.0],
        "t": [3, 1, 2],   # true order: row1 (t1,10) -> row2 (t2,20) -> row0 (t3,30)
    })
    out = FeatureEngineeringService._time_safe_entity_aggregates(df, "cc_num", "amt", df["t"])
    # Original row 0 is the LATEST (t=3): its past is {10, 20} -> mean 15, count 2.
    assert out["card_tx_count"].iloc[0] == 2
    assert out["card_amt_mean"].iloc[0] == 15.0
    # Original row 1 is the EARLIEST (t=1): no past.
    assert out["card_tx_count"].iloc[1] == 0
    assert out["card_amt_mean"].iloc[1] == 0.0


def test_add_domain_features_uses_time_safe_path_when_time_present():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "cc_num": rng.choice(["c1", "c2", "c3"], n),
        "amt": rng.uniform(1, 500, n).round(2),
        "unix_time": np.arange(n) * 60,  # numeric time axis
    })
    out_df, created = _svc().add_domain_features(df.copy())
    for col in ("card_tx_count", "card_amt_mean", "card_amt_std", "card_amt_max", "card_amt_zscore"):
        assert col in created and col in out_df.columns
    # First transaction of each card has zero prior count somewhere.
    assert (out_df["card_tx_count"] == 0).sum() >= 3


def test_add_domain_features_fallback_without_time_axis():
    rng = np.random.default_rng(1)
    n = 90
    df = pd.DataFrame({
        "cc_num": rng.choice(["c1", "c2"], n),
        "amt": rng.uniform(1, 100, n).round(2),
    })
    out_df, created = _svc().add_domain_features(df.copy())
    # Still produces the velocity features (whole-card stats fallback).
    for col in ("card_tx_count", "card_amt_mean", "card_amt_max"):
        assert col in out_df.columns
    # Whole-card count is the full per-card size (>= 1 for every row).
    assert (out_df["card_tx_count"] >= 1).all()
