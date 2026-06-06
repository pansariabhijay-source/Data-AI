"""
Out-of-process pipeline execution.

Runs the full pipeline in a CHILD PROCESS so its CPU-heavy work (model training,
SHAP, mutual-info, etc.) never holds the API's GIL. While a pipeline runs the
parent thread only blocks on a ``multiprocessing.Queue`` (which releases the
GIL), so uploads and status polls stay responsive.

Progress is streamed to the parent as queue events; the parent owns all
run-state / DB persistence. This module deliberately imports NO web/FastAPI code
so the spawned child stays lightweight.
"""

from __future__ import annotations

from typing import Optional

from core.state import PipelineState


def build_result_payload(state: PipelineState) -> dict:
    """Build the API result payload from pipeline state (shared by both paths)."""
    return {
        "run_id": state.run_id,
        "status": "failed" if state.failed else "completed",
        "error": state.failure_reason if state.failed else None,
        "problem_type": state.problem_type,
        "target_column": state.target_column,
        "best_model": state.best_model_name,
        "best_metric_name": state.best_metric_name,
        "best_metric_value": state.best_metric_value,
        "retry_count": state.retry_count,
        "dataset": {
            "rows": state.dataset_metadata.n_rows if state.dataset_metadata else None,
            "columns": state.dataset_metadata.n_columns if state.dataset_metadata else None,
            "quality_score": state.data_quality_flags.get("quality_score"),
        },
        "preprocessing": {
            "rows_before": state.preprocessing_summary.rows_before if state.preprocessing_summary else None,
            "rows_after": state.preprocessing_summary.rows_after if state.preprocessing_summary else None,
            "quality_score": state.preprocessing_summary.quality_score if state.preprocessing_summary else None,
            "duplicates_removed": state.preprocessing_summary.duplicates_removed if state.preprocessing_summary else 0,
        },
        "features": {
            "before": state.feature_engineering_summary.n_features_before if state.feature_engineering_summary else None,
            "after": state.feature_engineering_summary.n_features_after if state.feature_engineering_summary else None,
            "selected": state.selected_features[:20],
        },
        "models": [
            {
                "name": r.model_name,
                "status": r.status,
                "metrics": r.metrics,
                "time_s": r.training_time_seconds,
                "is_best": r.is_best,
            }
            for r in state.model_results
        ],
        "errors": [
            {
                "severity": e.severity,
                "type": e.error_type,
                "cause": e.root_cause,
                "fix": e.recommended_fix,
            }
            for e in state.error_reports
        ],
        "artifacts": state.artifacts,
        "reports": state.report_paths,
        "experiment_history": [
            {
                "iteration": e.iteration,
                "best_model": e.best_model,
                "best_value": e.best_metric_value,
            }
            for e in state.experiment_history
        ],
    }


def run_pipeline_child(queue, run_id: str, data_path: str,
                       target_column: Optional[str], mode: str = "free") -> None:
    """Child-process entry point. Streams progress events to ``queue``.

    Event tuples:
      ("start", agent_name, ts)
      ("complete", agent_name, output_dict, ts)
      ("done", payload_dict)        # terminal — full result + post-run viz
      ("error", message, traceback) # terminal — unexpected failure
      ("__end__",)                  # always last, so the parent loop can stop
    """
    try:
        from core.config import load_settings
        from core.logging_config import setup_logging, get_logger, set_correlation_id
        from core.utils import get_timestamp
        from core.agent_runner import run_full_pipeline

        settings = load_settings()
        setup_logging(log_dir=settings.pipeline.log_dir, level=settings.logging.level)
        set_correlation_id(run_id)
        log = get_logger("pipeline_worker")

        state = PipelineState(
            run_id=run_id,
            raw_data_path=data_path,
            target_column=target_column,
            max_retries=settings.pipeline.max_retries,
            started_at=get_timestamp(),
        )

        def on_start(agent_name):
            queue.put(("start", agent_name, get_timestamp()))

        def on_complete(agent_name, output):
            queue.put(("complete", agent_name, output.to_dict(), get_timestamp()))

        run_full_pipeline(state, settings, on_start, on_complete)
        state.completed_at = get_timestamp()

        visualizations: list = []
        if not state.failed:
            try:
                from visualization.engine import (
                    generate_model_comparison, generate_feature_importance,
                )
                mv = generate_model_comparison([
                    {"name": m.model_name, "status": m.status, "metrics": m.metrics, "is_best": m.is_best}
                    for m in state.model_results
                ])
                if mv:
                    visualizations.append(mv)
                if state.feature_importances:
                    fv = generate_feature_importance(state.feature_importances)
                    if fv:
                        visualizations.append(fv)
            except Exception as viz_err:  # noqa: BLE001
                log.warning(f"Post-pipeline visualization failed: {viz_err}")

        queue.put(("done", {
            "failed": bool(state.failed),
            "failure_reason": state.failure_reason,
            "result": build_result_payload(state),
            "visualizations": visualizations,
            "best_model_name": state.best_model_name,
            "best_metric_name": state.best_metric_name,
            "best_metric_value": state.best_metric_value,
            "completed_at": state.completed_at,
        }))
    except Exception as e:  # noqa: BLE001
        import traceback
        try:
            queue.put(("error", str(e), traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            queue.put(("__end__",))
        except Exception:
            pass
