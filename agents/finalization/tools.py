"""
Finalization Tool — save artifacts, generate reports, and SHAP explanations.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from core.config import Settings
from core.logging_config import get_logger, log_stage_timing
from core.state import ErrorReport, PipelineState
from core.utils import ensure_directory, get_timestamp, safe_json_serialize

logger = get_logger("finalization")


class FinalizationService:
    @log_stage_timing("finalization")
    def run(self, state: PipelineState, settings: Settings) -> PipelineState:
        run_dir = ensure_directory(Path(settings.pipeline.artifact_dir) / state.run_id)
        report_dir = ensure_directory(Path(settings.pipeline.report_dir) / state.run_id)

        # Save final metadata
        metadata = {
            "run_id": state.run_id,
            "problem_type": state.problem_type,
            "target_column": state.target_column,
            "best_model": state.best_model_name,
            "best_metric": state.best_metric_name,
            "best_value": state.best_metric_value,
            "total_retries": state.retry_count,
            "completed_stages": state.completed_stages,
            "started_at": state.started_at,
            "completed_at": get_timestamp(),
        }
        meta_path = run_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=safe_json_serialize), encoding="utf-8")
        state.artifacts["metadata"] = str(meta_path)

        # Save all metrics
        metrics_data = [{
            "model": r.model_name, "status": r.status,
            "metrics": r.metrics, "train_metrics": r.train_metrics,
            "time_s": r.training_time_seconds, "params": r.hyperparameters,
        } for r in state.model_results]
        metrics_path = run_dir / "all_metrics.json"
        metrics_path.write_text(json.dumps(metrics_data, indent=2, default=safe_json_serialize), encoding="utf-8")
        state.artifacts["metrics"] = str(metrics_path)

        # Save feature importances
        if state.feature_importances:
            fi_path = run_dir / "feature_importances.json"
            sorted_fi = dict(sorted(state.feature_importances.items(), key=lambda x: x[1], reverse=True))
            fi_path.write_text(json.dumps(sorted_fi, indent=2, default=safe_json_serialize), encoding="utf-8")
            state.artifacts["feature_importances"] = str(fi_path)

        # Generate SHAP explanations (if possible)
        self._generate_shap(state, run_dir)

        # Generate markdown report
        report_path = report_dir / "pipeline_report.md"
        self._generate_markdown_report(state, report_path, run_dir)
        state.report_paths["final_report"] = str(report_path)

        # Save experiment history
        history_path = run_dir / "experiment_history.json"
        history = [e.model_dump() for e in state.experiment_history]
        history_path.write_text(json.dumps(history, indent=2, default=safe_json_serialize), encoding="utf-8")
        state.artifacts["experiment_history"] = str(history_path)

        # Save error reports
        if state.error_reports:
            err_path = run_dir / "error_reports.json"
            errs = [e.model_dump() for e in state.error_reports]
            err_path.write_text(json.dumps(errs, indent=2, default=safe_json_serialize), encoding="utf-8")
            state.artifacts["error_reports"] = str(err_path)

        # Save final state checkpoint
        state_path = run_dir / "final_state.json"
        state_path.write_text(state.to_checkpoint(), encoding="utf-8")

        state.completed_at = get_timestamp()
        state.mark_stage_end("finalization")
        logger.info(f"Finalization complete. Artifacts saved to {run_dir}")
        return state

    def _generate_shap(self, state: PipelineState, run_dir: Path) -> None:
        """Generate SHAP explanations for the best model."""
        if not state.best_model_path or not state.train_path:
            return
        try:
            import shap
            model = joblib.load(state.best_model_path)
            train_df = pd.read_csv(state.train_path, low_memory=False)
            target = state.target_column

            if target and target in train_df.columns:
                X = train_df.drop(columns=[target]).select_dtypes(include=[np.number])
            else:
                X = train_df.select_dtypes(include=[np.number])

            # Sample for performance
            if len(X) > 500:
                X = X.sample(500, random_state=42)

            # Use appropriate explainer
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                try:
                    explainer = shap.LinearExplainer(model, X)
                except Exception:
                    explainer = shap.KernelExplainer(model.predict, X.iloc[:50])

            shap_values = explainer.shap_values(X)

            # Save mean absolute SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            mean_shap = np.abs(shap_values).mean(axis=0)
            shap_dict = dict(sorted(zip(X.columns, [float(v) for v in mean_shap]), key=lambda x: x[1], reverse=True))
            shap_path = run_dir / "shap_importance.json"
            shap_path.write_text(json.dumps(shap_dict, indent=2), encoding="utf-8")
            state.artifacts["shap"] = str(shap_path)
            logger.info(f"SHAP explanations saved to {shap_path}")
        except Exception as e:
            logger.warning(f"SHAP generation failed (non-critical): {e}")

    def _generate_markdown_report(
        self, state: PipelineState, path: Path, run_dir: Optional[Path] = None
    ) -> None:
        """Generate a comprehensive markdown report."""
        # Load SHAP data if available
        shap_data: dict = {}
        if run_dir:
            shap_path = run_dir / "shap_importance.json"
            if shap_path.exists():
                try:
                    shap_data = json.loads(shap_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

        lines: list[str] = []

        # ── Title & metadata ──────────────────────────────────────────────────
        lines += [
            "# Pipeline Report",
            "",
            "| Property | Value |",
            "|----------|-------|",
            f"| **Run ID** | `{state.run_id}` |",
            f"| **Problem Type** | {(state.problem_type or 'Unknown').title()} |",
            f"| **Target Column** | `{state.target_column or 'None (clustering)'}` |",
            f"| **Started** | {state.started_at or 'N/A'} |",
            f"| **Completed** | {get_timestamp()} |",
            f"| **Retries** | {state.retry_count} |",
            "",
            "---",
            "",
        ]

        # ── Executive Summary ─────────────────────────────────────────────────
        n_trained = len([m for m in state.model_results if m.status == "trained"])
        if state.best_model_name and state.best_metric_value is not None:
            metric_label = (state.best_metric_name or "score").upper()
            lines += [
                "## Executive Summary",
                "",
                f"The pipeline successfully trained **{n_trained} model{'s' if n_trained != 1 else ''}** "
                f"for a **{state.problem_type or 'machine learning'}** task. "
                f"The best performing model was **{state.best_model_name}**, achieving "
                f"a **{metric_label} of {state.best_metric_value:.4f}**.",
                "",
                "---",
                "",
            ]

        # ── Dataset Summary ───────────────────────────────────────────────────
        if state.dataset_metadata:
            m = state.dataset_metadata
            null_cols = len([c for c, v in m.null_counts.items() if v > 0])
            lines += [
                "## Dataset Summary",
                "",
                "| Property | Value |",
                "|----------|-------|",
                f"| Rows | {m.n_rows:,} |",
                f"| Columns | {m.n_columns} |",
                f"| Memory Usage | {m.memory_usage_mb:.2f} MB |",
                f"| Numeric Features | {len(m.numeric_columns)} |",
                f"| Categorical Features | {len(m.categorical_columns)} |",
                f"| Datetime Features | {len(m.datetime_columns)} |",
                f"| Columns with Nulls | {null_cols} |",
                "",
                "---",
                "",
            ]

        # ── Preprocessing ─────────────────────────────────────────────────────
        if state.preprocessing_summary:
            p = state.preprocessing_summary
            lines += [
                "## Preprocessing",
                "",
                "| Property | Before | After |",
                "|----------|--------|-------|",
                f"| Rows | {p.rows_before:,} | {p.rows_after:,} |",
                f"| Columns | {p.columns_before} | {p.columns_after} |",
                "",
                "| Action | Count |",
                "|--------|-------|",
                f"| Duplicates Removed | {p.duplicates_removed} |",
                f"| Null Columns Filled | {len(p.nulls_filled)} |",
                f"| Outliers Handled | {len(p.outliers_handled)} |",
                f"| Columns Dropped | {len(p.columns_dropped)} |",
                f"| **Quality Score** | **{p.quality_score:.4f}** |",
                "",
                "---",
                "",
            ]

        # ── Feature Engineering ───────────────────────────────────────────────
        if state.feature_engineering_summary:
            fe = state.feature_engineering_summary
            lines += [
                "## Feature Engineering",
                "",
                "| Property | Value |",
                "|----------|-------|",
                f"| Features Before | {fe.n_features_before} |",
                f"| Features After | {fe.n_features_after} |",
                f"| Features Created | {len(fe.features_created)} |",
                f"| Features Removed | {len(fe.features_removed)} |",
                f"| Encodings Applied | {len(fe.encoding_applied)} |",
                f"| Scaling Applied | {len(fe.scaling_applied)} |",
                "",
            ]
            if fe.selected_features:
                lines += ["**Top Selected Features:**", ""]
                for feat in fe.selected_features[:10]:
                    lines.append(f"- `{feat}`")
                lines.append("")
            lines += ["---", ""]

        # ── Model Results ─────────────────────────────────────────────────────
        lines += ["## Model Results", ""]
        primary = state.best_metric_name or "score"

        all_metric_keys: set[str] = set()
        for r in state.model_results:
            all_metric_keys.update(r.metrics.keys())
        metric_keys = sorted(all_metric_keys)[:5]

        if metric_keys:
            mh = " | ".join(k.upper() for k in metric_keys)
            ms = " | ".join("------" for _ in metric_keys)
            lines.append(f"| Model | Status | {mh} | Time (s) |")
            lines.append(f"|-------|--------|{ms}|---------|")
        else:
            lines += ["| Model | Status | Time (s) |", "|-------|--------|---------|"]

        sorted_results = sorted(
            state.model_results,
            key=lambda r: r.metrics.get(primary, 0) if r.status == "trained" else -1,
            reverse=True,
        )
        for r in sorted_results:
            star = "⭐ " if r.is_best else ""
            if metric_keys:
                vals = " | ".join(
                    f"{r.metrics[k]:.4f}" if r.status == "trained" and k in r.metrics else "—"
                    for k in metric_keys
                )
                lines.append(f"| {star}{r.model_name} | {r.status} | {vals} | {r.training_time_seconds:.1f} |")
            else:
                lines.append(f"| {star}{r.model_name} | {r.status} | {r.training_time_seconds:.1f} |")

        lines += ["", "---", ""]

        # ── SHAP Feature Importance ───────────────────────────────────────────
        if shap_data:
            top_n = min(15, len(shap_data))
            lines += [
                "## SHAP Feature Importance",
                "",
                f"Top {top_n} features ranked by mean absolute SHAP value (higher = more influential):",
                "",
                "| Rank | Feature | Mean |SHAP Value| |",
                "|------|---------|------|",
            ]
            for rank, (feat, val) in enumerate(list(shap_data.items())[:top_n], 1):
                lines.append(f"| {rank} | `{feat}` | {val:.6f} |")
            lines += ["", "---", ""]

        # ── Issues Detected ───────────────────────────────────────────────────
        if state.error_reports:
            lines += [
                "## Issues Detected",
                "",
                "| Severity | Type | Root Cause | Recommended Fix |",
                "|----------|------|------------|-----------------|",
            ]
            for e in state.error_reports:
                cause = (e.root_cause[:80] + "…") if len(e.root_cause) > 80 else e.root_cause
                fix = (e.recommended_fix[:80] + "…") if len(e.recommended_fix) > 80 else e.recommended_fix
                lines.append(f"| {e.severity.upper()} | {e.error_type} | {cause} | {fix} |")
            lines += ["", "---", ""]

        # ── Experiment History ────────────────────────────────────────────────
        if state.experiment_history:
            lines += [
                "## Experiment History",
                "",
                "| Iteration | Best Model | Score | Improvements |",
                "|-----------|------------|-------|-------------|",
            ]
            for exp in state.experiment_history:
                impr = ", ".join(exp.improvements_applied[:3]) if exp.improvements_applied else "—"
                lines.append(
                    f"| {exp.iteration} | {exp.best_model} | {exp.best_metric_value:.4f} | {impr} |"
                )
            lines += ["", "---", ""]

        # ── Artifacts ─────────────────────────────────────────────────────────
        if state.artifacts:
            lines += [
                "## Saved Artifacts",
                "",
                "| Artifact | Path |",
                "|----------|------|",
            ]
            for name, art_path in state.artifacts.items():
                lines.append(f"| {name} | `{art_path}` |")
            lines += ["", "---", ""]

        lines += ["", "*Generated by **Axiom Autonomous Data Scientist Pipeline***"]

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Markdown report saved to {path}")


_service: Optional[FinalizationService] = None
_state: Optional[PipelineState] = None
_settings: Optional[Settings] = None


def init_finalization(state: PipelineState, settings: Settings) -> None:
    global _service, _state, _settings
    _service = FinalizationService()
    _state = state
    _settings = settings


def finalize_pipeline(instruction: str) -> str:
    """Save all artifacts, generate reports, and produce SHAP explanations.

    Saves the best model, metrics, feature importances, experiment history,
    and generates a comprehensive markdown report.

    Args:
        instruction: Description of finalization task.

    Returns:
        JSON summary of saved artifacts.
    """
    global _service, _state, _settings
    if _service is None or _state is None or _settings is None:
        return json.dumps({"error": "Finalization service not initialized"})
    try:
        _state.mark_stage_start("finalization")
        _state = _service.run(_state, _settings)
        return json.dumps({
            "status": "success",
            "artifacts": _state.artifacts,
            "reports": _state.report_paths,
            "best_model": _state.best_model_name,
            "best_metric_value": _state.best_metric_value,
        }, default=str)
    except Exception as e:
        logger.exception("Finalization failed")
        _state.add_error(ErrorReport(
            severity="high", stage="finalization",
            error_type=type(e).__name__, root_cause=str(e),
            traceback_str=traceback.format_exc(),
            recommended_fix="Check file permissions and disk space",
            retryable=False,
        ))
        return json.dumps({"error": str(e)})
