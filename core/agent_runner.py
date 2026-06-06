"""
Axiom Core — Modular Agent Runner

Enables independent agent execution, custom workflows,
and per-agent output capture with timing and memory tracking.
"""

from __future__ import annotations

import time
import traceback
from typing import Optional

import psutil

from core.config import load_settings, Settings
from core.logging_config import get_logger
from core.state import PipelineState
from core.utils import get_timestamp

logger = get_logger("agent_runner")

# Agent name -> (ServiceClass, init_function, config_attr)
AGENT_REGISTRY = {
    "data_collection": (
        "agents.data_collection.tools",
        "DataCollectionService",
        "init_data_collection",
        "data_collection",
    ),
    "preprocessing": (
        "agents.preprocessing.tools",
        "PreprocessingService",
        "init_preprocessing",
        "preprocessing",
    ),
    "feature_engineering": (
        "agents.feature_engineering.tools",
        "FeatureEngineeringService",
        "init_feature_engineering",
        "feature_engineering",
    ),
    "data_splitting": (
        "agents.splitting.tools",
        "DataSplittingService",
        "init_splitting",
        "splitting",
    ),
    "model_training": (
        "agents.training.tools",
        "ModelTrainingService",
        "init_training",
        "training",
    ),
    "error_detection": (
        "agents.error_detection.tools",
        "ErrorDetectionService",
        "init_error_detection",
        "error_detection",
    ),
    "improvement": (
        "agents.improvement.tools",
        "ImprovementService",
        "init_improvement",
        "improvement",
    ),
    "finalization": (
        "agents.finalization.tools",
        "FinalizationService",
        "init_finalization",
        None,
    ),
}

ALL_AGENTS_ORDERED = [
    "data_collection",
    "preprocessing",
    "feature_engineering",
    "data_splitting",
    "model_training",
    "error_detection",
    "improvement",
    "finalization",
]


class AgentOutput:
    """Structured output from a single agent execution."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.status = "idle"
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.duration_seconds: float = 0.0
        self.memory_mb: float = 0.0
        self.summary: dict = {}
        self.metrics: dict = {}
        self.logs: list = []
        self.artifacts: dict = {}
        self.visualizations: list = []
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "memory_mb": round(self.memory_mb, 2),
            "summary": self.summary,
            "metrics": self.metrics,
            "logs": self.logs,
            "artifacts": self.artifacts,
            "visualizations": self.visualizations,
            "error": self.error,
        }


def _get_memory_mb() -> float:
    """Get current process memory in MB."""
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def run_single_agent(
    agent_name: str,
    state: PipelineState,
    settings: Optional[Settings] = None,
) -> AgentOutput:
    """Run a single agent independently and capture structured output."""
    output = AgentOutput(agent_name)

    if agent_name not in AGENT_REGISTRY:
        output.status = "failed"
        output.error = f"Unknown agent: {agent_name}"
        return output

    if settings is None:
        settings = load_settings()

    module_path, service_class_name, init_func_name, config_attr = AGENT_REGISTRY[agent_name]

    output.status = "running"
    output.started_at = get_timestamp()
    output.logs.append({"time": get_timestamp(), "msg": f"Starting {agent_name}..."})

    mem_before = _get_memory_mb()
    t_start = time.time()

    try:
        import importlib
        module = importlib.import_module(module_path)
        ServiceClass = getattr(module, service_class_name)
        init_fn = getattr(module, init_func_name)

        # Initialize the service with shared state
        init_fn(state, settings)

        # Build service with appropriate config
        if agent_name == "data_splitting":
            service = ServiceClass(
                getattr(settings, config_attr),
                settings.pipeline.random_seed,
            )
        elif agent_name == "model_training":
            from core.model_registry import build_default_registry
            service = ServiceClass(
                getattr(settings, config_attr),
                build_default_registry(settings.pipeline.random_seed),
                settings.pipeline.random_seed,
            )
        elif agent_name == "finalization":
            service = ServiceClass()
        else:
            service = ServiceClass(getattr(settings, config_attr))

        # Mark stage start
        state.mark_stage_start(agent_name)

        # Run the service
        if agent_name in ("error_detection",):
            service.run(state)
        elif agent_name == "finalization":
            service.run(state, settings)
        else:
            service.run(state, settings)

        # Capture output
        output.status = "completed"
        output.completed_at = get_timestamp()
        output.duration_seconds = time.time() - t_start
        output.memory_mb = _get_memory_mb() - mem_before

        # Build summary based on agent type
        output.summary = _extract_agent_summary(agent_name, state)
        output.metrics = _extract_agent_metrics(agent_name, state)
        output.artifacts = dict(state.artifacts)

        # Generate visualizations best-effort
        try:
            from pathlib import Path
            if agent_name == "data_collection":
                import os
                import pandas as pd
                from visualization.engine import generate_enterprise_mode_viz
                if state.raw_data_path and Path(state.raw_data_path).exists():
                    df = pd.read_csv(state.raw_data_path)
                    # Charts are statistical summaries — render them on a bounded
                    # sample so viz time stays flat regardless of dataset size.
                    try:
                        _cap = int(os.environ.get("VIZ_SAMPLE_ROWS", "50000"))
                    except ValueError:
                        _cap = 50000
                    if _cap > 0 and len(df) > _cap:
                        df = df.sample(n=_cap, random_state=0)
                    output.visualizations = generate_enterprise_mode_viz(df, state.target_column)
            elif agent_name == "model_training":
                from visualization.engine import generate_model_comparison, generate_feature_importance
                if state.model_results:
                    model_viz = generate_model_comparison([
                        {"name": m.model_name, "status": m.status, "metrics": m.metrics, "is_best": m.is_best}
                        for m in state.model_results
                    ])
                    if model_viz:
                        output.visualizations.append(model_viz)
                if state.feature_importances:
                    fi_viz = generate_feature_importance(state.feature_importances)
                    if fi_viz:
                        output.visualizations.append(fi_viz)
        except Exception as viz_err:
            logger.warning(f"Failed to generate visualizations for agent {agent_name}: {viz_err}")

        output.logs.append({"time": get_timestamp(), "msg": f"✓ {agent_name} completed in {output.duration_seconds:.1f}s"})

        state.mark_stage_end(agent_name)

    except Exception as e:
        output.status = "failed"
        output.error = str(e)
        output.completed_at = get_timestamp()
        output.duration_seconds = time.time() - t_start
        output.memory_mb = _get_memory_mb() - mem_before
        output.logs.append({"time": get_timestamp(), "msg": f"✗ {agent_name} failed: {str(e)[:200]}"})
        logger.exception(f"Agent {agent_name} failed")

    return output


def run_workflow(
    agent_list: list[str],
    state: PipelineState,
    settings: Optional[Settings] = None,
    on_agent_start=None,
    on_agent_complete=None,
) -> dict[str, AgentOutput]:
    """Run a custom sequence of agents."""
    if settings is None:
        settings = load_settings()

    outputs: dict[str, AgentOutput] = {}

    for agent_name in agent_list:
        if on_agent_start:
            on_agent_start(agent_name)

        output = run_single_agent(agent_name, state, settings)
        outputs[agent_name] = output

        if on_agent_complete:
            on_agent_complete(agent_name, output)

        if output.status == "failed":
            # For critical early stages, stop the workflow
            if agent_name in ("data_collection", "preprocessing"):
                logger.error(f"Critical agent {agent_name} failed, stopping workflow")
                state.failed = True
                state.failure_reason = output.error or f"Critical agent {agent_name} failed"
                break

    return outputs


def run_full_pipeline(
    state: PipelineState,
    settings: Optional[Settings] = None,
    on_agent_start=None,
    on_agent_complete=None,
) -> dict[str, AgentOutput]:
    """Run the full pipeline (all agents in order)."""
    return run_workflow(
        ALL_AGENTS_ORDERED, state, settings,
        on_agent_start, on_agent_complete,
    )


def _extract_agent_summary(agent_name: str, state: PipelineState) -> dict:
    """Extract a human-readable summary from pipeline state for a given agent."""
    summary: dict = {}

    if agent_name == "data_collection" and state.dataset_metadata:
        m = state.dataset_metadata
        summary = {
            "rows": m.n_rows,
            "columns": m.n_columns,
            "memory_mb": round(m.memory_usage_mb, 2),
            "numeric_columns": len(m.numeric_columns),
            "categorical_columns": len(m.categorical_columns),
            "problem_type": state.problem_type,
            "target_column": state.target_column,
            "null_columns": len([c for c, v in m.null_counts.items() if v > 0]),
        }
    elif agent_name == "preprocessing" and state.preprocessing_summary:
        p = state.preprocessing_summary
        summary = {
            "rows_before": p.rows_before,
            "rows_after": p.rows_after,
            "columns_before": p.columns_before,
            "columns_after": p.columns_after,
            "duplicates_removed": p.duplicates_removed,
            "quality_score": round(p.quality_score, 4),
            "nulls_filled": len(p.nulls_filled),
            "outliers_handled": len(p.outliers_handled),
        }
    elif agent_name == "feature_engineering" and state.feature_engineering_summary:
        f = state.feature_engineering_summary
        summary = {
            "features_before": f.n_features_before,
            "features_after": f.n_features_after,
            "features_created": len(f.features_created),
            "features_removed": len(f.features_removed),
            "encodings_applied": len(f.encoding_applied),
        }
    elif agent_name == "data_splitting":
        summary = {
            "train_path": state.train_path,
            "val_path": state.val_path,
            "test_path": state.test_path,
            "split_ratios": state.split_ratios,
        }
    elif agent_name == "model_training":
        trained = [m for m in state.model_results if m.status == "trained"]
        summary = {
            "models_trained": len(trained),
            "best_model": state.best_model_name,
            "best_metric_name": state.best_metric_name,
            "best_metric_value": state.best_metric_value,
            "total_training_time": round(sum(m.training_time_seconds for m in trained), 2),
        }
    elif agent_name == "error_detection":
        summary = {
            "issues_found": len(state.error_reports),
            "critical": len([e for e in state.error_reports if e.severity == "critical"]),
            "warnings": len([e for e in state.error_reports if e.severity == "warning"]),
            "retryable": len(state.get_retryable_errors()),
        }
    elif agent_name == "improvement":
        summary = {
            "retry_count": state.retry_count,
            "experiments": len(state.experiment_history),
            "best_model": state.best_model_name,
            "best_metric_value": state.best_metric_value,
        }
    elif agent_name == "finalization":
        summary = {
            "artifacts": list(state.artifacts.keys()),
            "reports": list(state.report_paths.keys()),
            "best_model": state.best_model_name,
        }

    return summary


def _extract_agent_metrics(agent_name: str, state: PipelineState) -> dict:
    """Extract numeric metrics from pipeline state for a given agent."""
    metrics: dict = {}

    if agent_name == "data_collection" and state.dataset_metadata:
        metrics["rows"] = state.dataset_metadata.n_rows
        metrics["columns"] = state.dataset_metadata.n_columns
        metrics["memory_mb"] = state.dataset_metadata.memory_usage_mb
        if state.data_quality_flags.get("quality_score"):
            metrics["quality_score"] = state.data_quality_flags["quality_score"]
    elif agent_name == "preprocessing" and state.preprocessing_summary:
        metrics["quality_score"] = state.preprocessing_summary.quality_score
        metrics["rows_removed"] = state.preprocessing_summary.rows_before - state.preprocessing_summary.rows_after
        metrics["duplicates_removed"] = state.preprocessing_summary.duplicates_removed
    elif agent_name == "feature_engineering" and state.feature_engineering_summary:
        metrics["features_before"] = state.feature_engineering_summary.n_features_before
        metrics["features_after"] = state.feature_engineering_summary.n_features_after
    elif agent_name == "model_training" and state.best_metric_value is not None:
        metrics[state.best_metric_name or "score"] = state.best_metric_value
        metrics["models_trained"] = len([m for m in state.model_results if m.status == "trained"])
    elif agent_name == "error_detection":
        metrics["issues_found"] = len(state.error_reports)
    elif agent_name == "improvement":
        metrics["retry_count"] = state.retry_count
        if state.best_metric_value:
            metrics["final_score"] = state.best_metric_value

    return metrics
