"""
Pipeline checkpoint — save and restore pipeline state for fault tolerance.

Checkpoints are saved as JSON after each stage completion. On failure,
the pipeline can resume from the latest checkpoint instead of restarting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from core.logging_config import get_logger
from core.state import PipelineState
from core.utils import ensure_directory, get_timestamp

logger = get_logger("checkpoint")


class CheckpointManager:
    """Manages pipeline state checkpoints on disk."""

    def __init__(self, checkpoint_dir: str = "artifacts", enabled: bool = True) -> None:
        self._enabled = enabled
        self._dir = Path(checkpoint_dir)
        if self._enabled:
            ensure_directory(self._dir)

    def save(self, state: PipelineState, stage: str) -> Optional[str]:
        """Save a checkpoint after a stage completes. Returns checkpoint path."""
        if not self._enabled:
            return None
        run_dir = self._dir / state.run_id
        ensure_directory(run_dir)
        cp_path = run_dir / f"checkpoint_{stage}.json"
        try:
            cp_path.write_text(state.to_checkpoint(), encoding="utf-8")
            # Also save a 'latest' symlink-style file
            latest_path = run_dir / "checkpoint_latest.json"
            latest_path.write_text(state.to_checkpoint(), encoding="utf-8")
            logger.info(f"Checkpoint saved: {cp_path}")
            return str(cp_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None

    def load_latest(self, run_id: str) -> Optional[PipelineState]:
        """Load the latest checkpoint for a given run ID."""
        latest_path = self._dir / run_id / "checkpoint_latest.json"
        if not latest_path.exists():
            logger.warning(f"No checkpoint found for run {run_id}")
            return None
        try:
            json_str = latest_path.read_text(encoding="utf-8")
            state = PipelineState.from_checkpoint(json_str)
            logger.info(f"Checkpoint loaded for run {run_id}, stage: {state.current_stage}")
            return state
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def load_stage(self, run_id: str, stage: str) -> Optional[PipelineState]:
        """Load a specific stage checkpoint."""
        cp_path = self._dir / run_id / f"checkpoint_{stage}.json"
        if not cp_path.exists():
            return None
        try:
            return PipelineState.from_checkpoint(cp_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to load stage checkpoint: {e}")
            return None

    def list_checkpoints(self, run_id: str) -> list[str]:
        """List all available checkpoint stages for a run."""
        run_dir = self._dir / run_id
        if not run_dir.exists():
            return []
        return sorted([
            f.stem.replace("checkpoint_", "")
            for f in run_dir.glob("checkpoint_*.json")
            if f.stem != "checkpoint_latest"
        ])
