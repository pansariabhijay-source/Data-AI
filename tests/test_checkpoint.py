"""Smoke test for the pipeline checkpoint system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.state import PipelineState
from pipeline.checkpoint import CheckpointManager


def test_checkpoint_save_load(tmp_path):
    mgr = CheckpointManager(str(tmp_path), enabled=True)
    state = PipelineState(run_id="smoke_test", problem_type="classification")
    state.mark_stage_start("preprocessing")
    state.mark_stage_end("preprocessing")

    path = mgr.save(state, "preprocessing")
    assert path is not None
    assert Path(path).exists()

    loaded = mgr.load_latest("smoke_test")
    assert loaded is not None
    assert loaded.run_id == "smoke_test"
    assert "preprocessing" in loaded.completed_stages


def test_checkpoint_disabled(tmp_path):
    mgr = CheckpointManager(str(tmp_path), enabled=False)
    state = PipelineState(run_id="test")
    result = mgr.save(state, "test")
    assert result is None


def test_load_nonexistent(tmp_path):
    mgr = CheckpointManager(str(tmp_path), enabled=True)
    result = mgr.load_latest("nonexistent_run")
    assert result is None


def test_list_checkpoints(tmp_path):
    mgr = CheckpointManager(str(tmp_path), enabled=True)
    state = PipelineState(run_id="multi_test")
    mgr.save(state, "stage1")
    mgr.save(state, "stage2")
    cps = mgr.list_checkpoints("multi_test")
    assert "stage1" in cps
    assert "stage2" in cps
