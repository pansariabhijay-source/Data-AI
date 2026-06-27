"""
Artifact retention / disk hygiene.

Every pipeline run writes a run directory under ``artifacts/<run_id>/`` (model
files, train/val/test split CSVs, SHAP, checkpoints, run_state.json) and a report
under ``reports/<run_id>/``, plus the uploaded dataset under
``data/uploads/<user_id>/``. None of this was ever cleaned up, so on a machine
with limited free space the disk fills and runs start failing with
``OSError: [Errno 28] No space left on device``.

``cleanup_artifacts`` enforces a bounded footprint:

  - Always keep the newest ``min_keep`` runs (so a returning user never loses
    their latest results, regardless of age).
  - Keep at most ``max_runs`` runs total (hard count cap — the primary bound).
  - Additionally delete runs older than ``retention_days`` once past ``min_keep``.
  - Never touch a run that is currently in flight.
  - Prune uploaded dataset files older than ``retention_days``.

All limits are configurable via env vars and every deletion is best-effort
(failures are logged, never raised) so maintenance can never crash the app.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional

from core.logging_config import get_logger

logger = get_logger("maintenance")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


# Defaults chosen to keep the footprint small on low-disk machines while never
# dropping a user's most recent handful of runs.
def _max_runs() -> int:
    return max(1, _int_env("ARTIFACT_MAX_RUNS", 25))


def _min_keep() -> int:
    return max(1, _int_env("ARTIFACT_MIN_KEEP", 5))


def _retention_days() -> int:
    # The count cap (max_runs) is the primary disk bound; this age cap is a
    # gentle secondary so genuinely stale runs eventually go. Kept generous so a
    # user's existing runs aren't deleted out from under them.
    return max(1, _int_env("ARTIFACT_RETENTION_DAYS", 30))


def _safe_rmtree(path: Path) -> bool:
    """Delete a directory tree, swallowing (and logging) any error."""
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=False)
        return True
    except Exception:
        logger.warning("Failed to remove %s", path, exc_info=True)
        return False


def cleanup_artifacts(
    active_run_ids: Optional[Iterable[str]] = None,
    artifact_root: Optional[Path] = None,
    report_root: Optional[Path] = None,
    uploads_root: Optional[Path] = None,
) -> dict:
    """Enforce the retention policy. Returns a small stats dict.

    ``active_run_ids`` are never deleted (runs still in progress).
    """
    active = set(active_run_ids or ())
    artifact_root = artifact_root or Path(os.environ.get("ARTIFACT_DIR", "artifacts"))
    report_root = report_root or Path(os.environ.get("REPORT_DIR", "reports"))
    uploads_root = uploads_root or (Path("data") / "uploads")

    max_runs, min_keep, retention_days = _max_runs(), _min_keep(), _retention_days()
    cutoff = time.time() - retention_days * 86_400

    removed_runs: list[str] = []

    # ── Run directories (artifacts are the authoritative list of runs) ──────
    run_dirs: list[Path] = []
    if artifact_root.exists():
        run_dirs = [d for d in artifact_root.iterdir() if d.is_dir()]
    # Newest first, so index == age rank.
    run_dirs.sort(key=lambda d: _safe_mtime(d), reverse=True)

    for rank, d in enumerate(run_dirs):
        run_id = d.name
        if run_id in active:
            continue
        if rank < min_keep:
            continue  # always keep the most recent few
        too_many = rank >= max_runs
        too_old = _safe_mtime(d) < cutoff
        if too_many or too_old:
            ok = _safe_rmtree(d)
            # Remove the matching report dir alongside the artifacts.
            _safe_rmtree(report_root / run_id)
            if ok:
                removed_runs.append(run_id)

    # ── Orphan report dirs (report exists but artifacts already gone/expired) ─
    if report_root.exists():
        kept_ids = {d.name for d in run_dirs if d.name not in removed_runs}
        for rd in report_root.iterdir():
            if rd.is_dir() and rd.name not in kept_ids and rd.name not in active:
                if _safe_mtime(rd) < cutoff:
                    _safe_rmtree(rd)

    # ── Stale uploaded datasets ─────────────────────────────────────────────
    removed_uploads = 0
    if uploads_root.exists():
        for f in uploads_root.rglob("*"):
            if f.is_file() and _safe_mtime(f) < cutoff:
                try:
                    f.unlink()
                    removed_uploads += 1
                except Exception:
                    logger.warning("Failed to remove upload %s", f, exc_info=True)

    if removed_runs or removed_uploads:
        logger.info(
            "Artifact cleanup: removed %d run(s) and %d stale upload(s) "
            "(keep newest %d, max %d, retention %dd)",
            len(removed_runs), removed_uploads, min_keep, max_runs, retention_days,
        )
    return {
        "removed_runs": removed_runs,
        "removed_uploads": removed_uploads,
        "total_runs_seen": len(run_dirs),
    }


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
