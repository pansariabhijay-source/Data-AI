"""
Common utility functions: memory optimization, path management, serialization helpers.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.logging_config import get_logger

logger = get_logger("utils")


def read_csv_safe(path: Any, **kwargs: Any) -> pd.DataFrame:
    """``pd.read_csv`` that tolerates non-UTF-8 files.

    Real-world CSVs (e.g. exports from Excel) are often Latin-1 / Windows-1252 and
    crash a plain UTF-8 read with a ``UnicodeDecodeError``. Try UTF-8 first, then
    fall back to the common 8-bit encodings so the upload still loads instead of
    failing on a stray byte.
    """
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    # Last resort: replace undecodable bytes rather than fail the whole run.
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **kwargs)


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric dtypes to reduce memory usage. Operates in-place and returns df."""
    start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")
    end_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"Memory optimized: {start_mem:.1f}MB -> {end_mem:.1f}MB ({(1 - end_mem/start_mem)*100:.1f}% reduction)")
    return df


def get_memory_usage_mb(df: pd.DataFrame) -> float:
    return round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)


def ensure_directory(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def generate_run_id() -> str:
    """Generate a unique run ID.

    A timestamp prefix keeps IDs human-sortable, but second precision alone
    collides when two runs start in the same second (concurrent pipelines, fast
    clicks). A short random suffix guarantees uniqueness so runs never share an
    ID — which would otherwise mix up their state, history, and artifacts.
    """
    import secrets
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{secrets.token_hex(3)}"


def get_timestamp() -> str:
    """Return ISO-formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def safe_json_serialize(obj: Any) -> Any:
    """Convert non-serializable objects to JSON-safe types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return {"type": "DataFrame", "shape": list(obj.shape), "columns": list(obj.columns)}
    if isinstance(obj, pd.Series):
        return {"type": "Series", "name": obj.name, "length": len(obj)}
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return str(obj)
    return str(obj)


def get_artifact_dir(base_dir: str, run_id: str) -> Path:
    """Return run-specific artifact directory."""
    p = Path(base_dir) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def infer_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify columns into semantic types."""
    result: dict[str, list[str]] = {
        "numeric": [], "categorical": [], "datetime": [], "boolean": [], "text": [],
    }
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            result["boolean"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            result["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            result["datetime"].append(col)
        else:
            nunique = df[col].nunique()
            if nunique <= 2 and set(df[col].dropna().unique()).issubset({0, 1, True, False, "0", "1", "yes", "no", "true", "false"}):
                result["boolean"].append(col)
            elif nunique / max(len(df), 1) > 0.5:
                result["text"].append(col)
            else:
                result["categorical"].append(col)
    return result
