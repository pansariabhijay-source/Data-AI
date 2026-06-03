"""Core module — Foundation layer for the autonomous data science pipeline."""

from core.constants import ProblemType, PipelineStage, Severity, ModelStatus
from core.exceptions import PipelineError
from core.state import PipelineState

__all__ = [
    "ProblemType",
    "PipelineStage",
    "Severity",
    "ModelStatus",
    "PipelineError",
    "PipelineState",
]
