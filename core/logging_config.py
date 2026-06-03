"""
Structured logging for the autonomous data science pipeline.

Features:
- JSON-formatted log records for machine parsing
- Rotating file handlers to prevent disk exhaustion
- Console handler with human-readable output
- Correlation ID injection via contextvars (per pipeline run)
- Stage timing decorator for automatic duration tracking
"""

from __future__ import annotations

import contextvars
import functools
import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import psutil

_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="no-run-id"
)


def set_correlation_id(run_id: str) -> None:
    _correlation_id.set(run_id)


def get_correlation_id() -> str:
    return _correlation_id.get()


class StructuredJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": get_correlation_id(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        for attr in ("stage", "duration_s", "memory_mb", "detail"):
            val = getattr(record, attr, None)
            if val is not None:
                entry[attr] = val
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class ReadableFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m", "INFO": "\033[32m", "WARNING": "\033[33m",
        "ERROR": "\033[31m", "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        c = self.COLORS.get(record.levelname, self.RESET)
        rid = get_correlation_id()
        ts = self.formatTime(record, "%H:%M:%S")
        stage = getattr(record, "stage", "")
        s = f" [{stage}]" if stage else ""
        return f"{ts} {c}{record.levelname:<8}{self.RESET} [{rid[:8]}]{s} {record.getMessage()}"


def setup_logging(
    log_dir: str = "logs", level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5,
    json_format: bool = True,
) -> logging.Logger:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("pipeline")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    fh = logging.handlers.RotatingFileHandler(
        log_path / "pipeline.log", maxBytes=max_bytes,
        backupCount=backup_count, encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(StructuredJsonFormatter() if json_format else logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"))
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(ReadableFormatter())
    root.addHandler(ch)

    eh = logging.handlers.RotatingFileHandler(
        log_path / "pipeline_errors.log", maxBytes=max_bytes,
        backupCount=backup_count, encoding="utf-8",
    )
    eh.setLevel(logging.ERROR)
    eh.setFormatter(StructuredJsonFormatter())
    root.addHandler(eh)

    root.info("Logging initialized", extra={"stage": "init"})
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"pipeline.{name}")


F = TypeVar("F", bound=Callable[..., Any])


def log_stage_timing(stage: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(stage)
            proc = psutil.Process(os.getpid())
            mem_before = proc.memory_info().rss / (1024 * 1024)
            start = time.perf_counter()
            logger.info(f"Stage '{stage}' started", extra={"stage": stage, "memory_mb": round(mem_before, 2)})
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                mem_after = proc.memory_info().rss / (1024 * 1024)
                logger.info(
                    f"Stage '{stage}' completed in {elapsed:.2f}s (mem: {mem_before:.0f}->{mem_after:.0f} MB)",
                    extra={"stage": stage, "duration_s": round(elapsed, 3), "memory_mb": round(mem_after, 2)},
                )
                return result
            except Exception:
                elapsed = time.perf_counter() - start
                logger.exception(f"Stage '{stage}' failed after {elapsed:.2f}s", extra={"stage": stage, "duration_s": round(elapsed, 3)})
                raise
        return wrapper  # type: ignore[return-value]
    return decorator


def log_memory_usage(logger: Optional[logging.Logger] = None) -> dict[str, float]:
    if logger is None:
        logger = get_logger("memory")
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    info = {"rss_mb": round(mem.rss / (1024**2), 2), "vms_mb": round(mem.vms / (1024**2), 2), "percent": round(proc.memory_percent(), 2)}
    logger.debug(f"Memory usage: {info}")
    return info
