"""
Finalization Tool — save artifacts, generate reports, and SHAP explanations.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from core.config import Settings
from core.constants import ProblemType
from core.logging_config import get_logger, log_stage_timing
from core.metrics import (
    bootstrap_metric_cis,
    compute_metrics,
    confusion_counts,
    positive_class_proba,
    target_inverse_transform,
)
from core.state import ErrorReport, PipelineState
from core.utils import build_model_matrix, ensure_directory, get_timestamp, safe_json_serialize

logger = get_logger("finalization")

# KernelExplainer (used for non-tree/non-linear champions like ensembles & SVC)
# defaults to nsamples="auto" = 2*n_features+2048 coalition evaluations per row,
# which is ~40s for an ensemble. The mean |SHAP| ranking is stable with far
# fewer samples, so bound it. Override via SHAP_KERNEL_NSAMPLES.
try:
    _SHAP_KERNEL_NSAMPLES = int(os.environ.get("SHAP_KERNEL_NSAMPLES", "100"))
except (TypeError, ValueError):
    _SHAP_KERNEL_NSAMPLES = 100

# Number of bootstrap resamples for test-metric confidence intervals (0 disables).
# The test set is resampled with replacement this many times; cost is bounded
# further by a row cap inside bootstrap_metric_cis.
try:
    _BOOTSTRAP_CI_N = int(os.environ.get("BOOTSTRAP_CI_N", "500"))
except (TypeError, ValueError):
    _BOOTSTRAP_CI_N = 500


class FinalizationService:
    @log_stage_timing("finalization")
    def run(self, state: PipelineState, settings: Settings) -> PipelineState:
        if not state.model_results and not state.best_model_name:
            raise RuntimeError(
                "Finalization aborted: no models were trained. "
                "Check model_training logs for the root cause."
            )
        run_dir = ensure_directory(Path(settings.pipeline.artifact_dir) / state.run_id)
        report_dir = ensure_directory(Path(settings.pipeline.report_dir) / state.run_id)

        # Re-score the (possibly improvement-tuned) champion on the untouched test
        # split so the report's headline numbers are an honest generalization estimate.
        self._evaluate_on_test(state)

        # Save final metadata
        metadata = {
            "run_id": state.run_id,
            "problem_type": state.problem_type,
            "target_column": state.target_column,
            "best_model": state.best_model_name,
            "best_metric": state.best_metric_name,
            "best_value": state.best_metric_value,
            "decision_threshold": state.best_threshold,
            "test_metrics": state.test_metrics,
            "test_metric_cis": state.test_metric_cis,
            "test_confusion": state.test_confusion,
            "total_retries": state.retry_count,
            "completed_stages": state.completed_stages,
            "started_at": state.started_at,
            "completed_at": get_timestamp(),
        }
        meta_path = run_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=safe_json_serialize), encoding="utf-8")
        state.artifacts["metadata"] = str(meta_path)

        # ── Inference manifest ────────────────────────────────────────────────
        # Documents exactly what the champion expects so there's no train/serve
        # skew: the ordered feature list, decision threshold, and which columns
        # were dropped as leakage. Full raw->prediction transform replay is the
        # reproduction notebook's job (Report → Reproduce); this records the
        # contract.
        fe = state.feature_engineering_summary
        manifest = {
            "champion_model": state.best_model_name,
            "model_file": state.best_model_path,
            "problem_type": state.problem_type,
            "target": state.target_column,
            "decision_threshold": state.best_threshold,
            "expected_features": list(state.selected_features or []),
            "n_expected_features": len(state.selected_features or []),
            "encodings": getattr(fe, "encodings_applied", None) if fe else None,
            # If the regression target was log-transformed for training, the model
            # predicts in log-space — apply this inverse (expm1) to raw predictions
            # to recover the target's original units. None for untransformed targets.
            "target_transform": state.data_quality_flags.get("log_transformed_target"),
            "dropped_as_leakage": state.data_quality_flags.get("potential_leakage") or [],
            "how_to_use": (
                "Load `model_file` with joblib. It expects `expected_features` (already "
                "preprocessed + feature-engineered, in this order). To score RAW new data, "
                "replay the pipeline transforms — the faithful way is the reproduction "
                "notebook (Report → Reproduce). Apply `decision_threshold` to "
                "predict_proba[:,1] for the positive class."
            ),
        }
        manifest_path = run_dir / "inference_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=safe_json_serialize), encoding="utf-8")
        state.artifacts["inference_manifest"] = str(manifest_path)

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

    def _evaluate_on_test(self, state: PipelineState) -> None:
        """Score the final champion on the held-out test split at its tuned threshold."""
        problem_type = ProblemType(state.problem_type) if state.problem_type else None
        if (
            not state.test_path or not state.best_model_path
            or problem_type in (None, ProblemType.CLUSTERING)
        ):
            return
        try:
            test_df = pd.read_csv(state.test_path, low_memory=False)
            target = state.target_column
            if not target or target not in test_df.columns:
                return
            X_test, _ = build_model_matrix(test_df, target, state.selected_features)
            y_test = test_df[target].values
            model = joblib.load(state.best_model_path)

            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                except Exception:
                    y_prob = None

            if (
                problem_type == ProblemType.CLASSIFICATION
                and y_prob is not None
                and state.best_threshold is not None
                and len(np.unique(y_test)) == 2
            ):
                classes = np.asarray(getattr(model, "classes_", [0, 1]))
                pos = positive_class_proba(y_prob)
                preds = np.where(pos >= state.best_threshold, classes[1], classes[0])
            else:
                preds = model.predict(X_test)

            inv = target_inverse_transform(state.data_quality_flags)
            state.test_metrics = compute_metrics(
                y_test, preds, problem_type, y_prob, inverse_transform=inv,
            )
            cm = confusion_counts(y_test, preds)
            if cm:
                state.test_confusion = cm
            # Bootstrap confidence intervals so the single-slice test score reports
            # its own uncertainty (e.g. "PR-AUC 0.78 [0.74, 0.82]").
            if _BOOTSTRAP_CI_N > 0:
                try:
                    state.test_metric_cis = bootstrap_metric_cis(
                        y_test, preds, problem_type, y_prob,
                        inverse_transform=inv, n_boot=_BOOTSTRAP_CI_N,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Bootstrap CI computation skipped: {e}")
            logger.info(f"Held-out TEST metrics: {state.test_metrics}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Test-set evaluation failed (non-critical): {e}")

    def _generate_shap(self, state: PipelineState, run_dir: Path) -> None:
        """Generate SHAP explanations for the best model."""
        if not state.best_model_path or not state.train_path:
            return
        try:
            import shap
            model = joblib.load(state.best_model_path)
            # A calibrated champion is a CalibratedClassifierCV wrapper — explain
            # its underlying estimator so the fast TreeExplainer/LinearExplainer
            # still applies (the calibrator is monotonic and doesn't change which
            # features matter).
            try:
                from sklearn.calibration import CalibratedClassifierCV
                if isinstance(model, CalibratedClassifierCV):
                    inner = getattr(model, "estimator", model)
                    # Unwrap FrozenEstimator -> the real base; else use inner.
                    model = getattr(inner, "estimator", inner)
            except Exception:
                pass
            train_df = pd.read_csv(state.train_path, low_memory=False)
            target = state.target_column

            # Build the SHAP frame from the same canonical feature list (by name) the
            # model was trained on, so feature names/order line up with the estimator.
            feats = [c for c in (state.selected_features or []) if c in train_df.columns]
            if feats:
                X = train_df.reindex(columns=feats).apply(pd.to_numeric, errors="coerce").fillna(0.0)
            elif target and target in train_df.columns:
                X = train_df.drop(columns=[target]).select_dtypes(include=[np.number])
            else:
                X = train_df.select_dtypes(include=[np.number])

            # Sample for performance
            if len(X) > 500:
                X = X.sample(500, random_state=42)

            # Use appropriate explainer. TreeExplainer/LinearExplainer are cheap;
            # KernelExplainer is model-agnostic but O(n_explain * n_background) model
            # calls, which can take minutes on non-tree models (e.g. SVC). Bound its
            # work so finalization can never appear to hang.
            X_explain = X
            use_kernel = False
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                try:
                    explainer = shap.LinearExplainer(model, X)
                except Exception:
                    background = shap.kmeans(X, min(20, len(X)))
                    explainer = shap.KernelExplainer(model.predict, background)
                    # Explain a capped subset — mean |SHAP| is stable on a sample.
                    X_explain = X.sample(min(100, len(X)), random_state=42)
                    use_kernel = True

            # TreeExplainer/LinearExplainer are exact and fast; only the
            # model-agnostic KernelExplainer needs its coalition budget bounded.
            if use_kernel:
                shap_values = explainer.shap_values(
                    X_explain, nsamples=_SHAP_KERNEL_NSAMPLES, silent=True
                )
            else:
                shap_values = explainer.shap_values(X_explain)
            X = X_explain

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
            ]
            if state.best_threshold is not None:
                lines += [
                    f"Predictions use an F1-optimised decision threshold of "
                    f"**{state.best_threshold:.4f}** on the positive-class probability "
                    f"(not the default 0.5), tuned for the class imbalance in the target.",
                    "",
                ]
            # Make the honest-vs-leaked framing unmissable: if columns that almost
            # perfectly predict the target were removed, the headline score is much
            # lower than a naive model that keeps them — but it's the real one.
            leaked = state.data_quality_flags.get("potential_leakage") or []
            if leaked:
                cols = ", ".join(f"`{c}`" for c in leaked[:8])
                more = f" (+{len(leaked) - 8} more)" if len(leaked) > 8 else ""
                lines += [
                    f"**Leakage-free score.** {len(leaked)} feature(s) that almost perfectly "
                    f"predict the target — {cols}{more} — were detected and **removed before "
                    f"training**, because they would not be known at prediction time in the real "
                    f"world (they effectively encode the answer). The scores in this report are "
                    f"therefore an honest estimate of real-world performance. A model trained "
                    f"*with* those columns can report near-perfect accuracy (~99%) but is useless "
                    f"in production — it is simply reading the label back. Removing leakage is why "
                    f"this honest score is lower than a naive baseline that keeps every column.",
                    "",
                ]
            lines += ["---", ""]

        # ── Pipeline Explanation (plain-English narrative of what happened) ────
        # Never let a formatting bug in the narrative lose the entire report — the
        # model results and metrics are the important payload.
        try:
            lines += self._build_explanation(state)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Explanation section skipped ({e})")

        # ── Held-Out Test Performance ─────────────────────────────────────────
        if state.test_metrics:
            cis = state.test_metric_cis or {}
            lines += [
                "## Held-Out Test Performance",
                "",
                "Honest generalization estimate — the champion scored on the untouched "
                "test split at its tuned threshold.",
                "",
            ]
            if cis:
                lines += [
                    "The **95% CI** column is a percentile bootstrap over the test rows: "
                    "it shows how much the score could move on a different sample of the "
                    "same size, so a single-slice number isn't read as more precise than it is.",
                    "",
                    "| Metric | Value | 95% CI |",
                    "|--------|-------|--------|",
                ]
                for key in ("f1", "precision", "recall", "roc_auc", "pr_auc",
                            "balanced_accuracy", "accuracy", "r2", "rmse", "mae"):
                    if key in state.test_metrics:
                        ci = cis.get(key)
                        ci_txt = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "—"
                        lines.append(f"| {key.upper()} | {state.test_metrics[key]:.4f} | {ci_txt} |")
            else:
                lines += ["| Metric | Value |", "|--------|-------|"]
                for key in ("f1", "precision", "recall", "roc_auc", "pr_auc",
                            "balanced_accuracy", "accuracy", "r2", "rmse", "mae"):
                    if key in state.test_metrics:
                        lines.append(f"| {key.upper()} | {state.test_metrics[key]:.4f} |")
            lines.append("")
            cm = state.test_confusion
            if cm:
                lines += [
                    "**Confusion Matrix (test):**",
                    "",
                    "| | Predicted 0 | Predicted 1 |",
                    "|---|---|---|",
                    f"| **Actual 0** | {cm.get('tn', 0):,} (TN) | {cm.get('fp', 0):,} (FP) |",
                    f"| **Actual 1** | {cm.get('fn', 0):,} (FN) | {cm.get('tp', 0):,} (TP) |",
                    "",
                ]
            # Operating points — precision/recall at fixed review budgets (fraud).
            ops = state.test_operating_points
            if ops:
                lines += [
                    "**Operating points (test).** If you can review a fixed fraction of "
                    "transactions, this is what you catch — pick the row that matches your "
                    "review capacity. *Precision* = of those flagged, how many are fraud; "
                    "*recall* = of all fraud, how many you catch.",
                    "",
                    "| Review budget (flagged) | Score threshold | Precision | Recall | Fraud caught |",
                    "|---|---|---|---|---|",
                ]
                for op in ops:
                    lines.append(
                        f"| {op['alert_rate']*100:.2f}% ({op['flagged']:,}) | {op['threshold']:.4f} | "
                        f"{op['precision']*100:.1f}% | {op['recall']*100:.1f}% | "
                        f"{op['caught']:,} / {op['total_fraud']:,} |"
                    )
                lines.append("")
            lines += ["---", ""]

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
                f"| Columns Dropped (high-null, this stage) | {len(p.columns_dropped)} |",
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
        # Show the most decision-relevant metrics first rather than alphabetical.
        preferred = ["f1", "precision", "recall", "roc_auc", "pr_auc",
                     "balanced_accuracy", "r2", "rmse", "mae", "silhouette_score"]
        metric_keys = [k for k in preferred if k in all_metric_keys][:5]
        if not metric_keys:
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

    @staticmethod
    def _explain_created_feature(name: str) -> str:
        """Plain-English purpose of an engineered feature, inferred from its name."""
        n = name.lower()
        if name.endswith("__was_missing"):
            return f"flag marking rows where `{name[:-13]}` was originally missing (missingness can itself be predictive)"
        if "__x__" in name:
            a, b = name.split("__x__", 1)
            return f"interaction term — `{a}` multiplied by `{b}` — to capture their combined effect"
        if n.startswith("log_"):
            return f"log-transform of `{name[4:]}` to tame a skewed (long-tailed) distribution"
        if name in ("geo_distance_km",):
            return "distance in km between the two latitude/longitude points in the row"
        if name in ("age_years",):
            return "age in years derived from a date-of-birth column"
        if n.startswith("card_") or "velocity" in n:
            return "per-group transaction statistic (count / average / spread) — useful for spotting unusual activity"
        for suf, desc in (("_sin", "cyclical encoding"), ("_cos", "cyclical encoding"),
                          ("_is_weekend", "weekend flag"), ("_recency_days", "days since the most recent date"),
                          ("_year", "calendar part"), ("_month", "calendar part"),
                          ("_dow", "day-of-week part"), ("_hour", "hour-of-day part")):
            if name.endswith(suf):
                return f"{desc} extracted from a date/time column"
        return "engineered feature derived from the raw columns"

    def _build_explanation(self, state: PipelineState) -> list[str]:
        """A plain-English 'what the pipeline did and why' section for the report."""
        m = state.dataset_metadata
        p = state.preprocessing_summary
        fe = state.feature_engineering_summary
        trained = [r for r in state.model_results if r.status == "trained"]
        n_trained = len(trained)

        lines: list[str] = ["## Pipeline Explanation", "",
                            "A plain-English walkthrough of what Axiom did with your data and why.", ""]

        # 1. End-to-end narrative ------------------------------------------------
        nar = "Axiom ran its automated stages end-to-end. "
        if m:
            nar += (f"It profiled the dataset (**{m.n_rows:,} rows × {m.n_columns} columns**) and detected a "
                    f"**{(state.problem_type or 'machine learning')}** problem")
            nar += f" from the `{state.target_column}` column. " if state.target_column else ". "
        if p:
            bits = []
            if p.duplicates_removed:
                bits.append(f"removed **{p.duplicates_removed:,}** duplicate rows")
            n_filled = len([1 for v in p.nulls_filled.values() if "filled" in v])
            if n_filled:
                bits.append(f"filled missing values in **{n_filled}** column(s)")
            if p.dtypes_fixed:
                bits.append(f"corrected the data type of **{len(p.dtypes_fixed)}** column(s)")
            if bits:
                nar += "Preprocessing " + ", ".join(bits) + ". "
        if fe:
            nar += (f"Feature engineering took the data from **{fe.n_features_before} to "
                    f"{fe.n_features_after} columns** ({len(fe.features_created)} added, "
                    f"{len(fe.features_removed)} removed). ")
        if state.best_model_name:
            nar += (f"Finally, **{n_trained}** models were trained and compared on a validation split — "
                    f"**{state.best_model_name}** won and was scored once on the untouched test set.")
        lines += [nar, ""]

        # Evaluation protocol — important for credibility on temporal data (fraud).
        strategy = state.data_quality_flags.get("split_strategy")
        entity = state.data_quality_flags.get("group_aware")
        if strategy == "out-of-time":
            lines += [
                "**Out-of-time evaluation.** A time axis was detected, so the data was split "
                "**chronologically** — the model trains on the oldest transactions and is tested on the "
                "**newest** ones. This mirrors production (you only ever have the past to predict the "
                "future) and avoids the optimistic bias of a random split, which leaks future patterns "
                "and the same entities into training. The reported numbers are therefore a realistic "
                "estimate of how the model performs on tomorrow's data.",
                "",
            ]
        elif strategy == "group-aware":
            lines += [
                f"**Group-aware evaluation.** Rows were grouped by `{entity}` so that every "
                f"{entity} appears in only one of train/validation/test. This prevents identity "
                "leakage — the model can't 'recognise' an entity it already memorised — while "
                "preserving class balance across the splits.",
                "",
            ]
        elif strategy == "stratified-random":
            lines += [
                "**Stratified-random evaluation.** No usable time axis was found, so the data was split "
                "randomly while preserving the class balance across train/validation/test.",
                "",
            ]
        if entity and strategy == "out-of-time":
            lines += [
                f"The split is also **group-aware**: every `{entity}` is confined to a single split, "
                "so no entity straddles train and test — closing the identity-leakage gap that a plain "
                "chronological split leaves open.",
                "",
            ]

        def bullets(title: str, items: list[tuple[str, str]], cap: int = 14) -> None:
            if not items:
                return
            lines.append(f"### {title}")
            lines.append("")
            for col, why in items[:cap]:
                lines.append(f"- `{col}` — {why}")
            if len(items) > cap:
                lines.append(f"- …and {len(items) - cap} more")
            lines.append("")

        # 2. Column types corrected ---------------------------------------------
        if p and p.dtypes_fixed:
            bullets("Column types corrected", list(p.dtypes_fixed.items()))

        # 3. Columns added -------------------------------------------------------
        if fe and fe.features_created:
            bullets("Columns added (and why)",
                    [(c, self._explain_created_feature(c)) for c in fe.features_created])

        # 4. Columns dropped -----------------------------------------------------
        dropped: dict[str, str] = {}
        if p:
            for col, why in p.nulls_filled.items():
                if "dropped" in why and "rows" not in why:
                    dropped[col] = "over 70% of its values were missing, so it carried too little information"
        if fe:
            dropped.update(getattr(fe, "removal_reasons", {}) or {})
        bullets("Columns dropped (and why)", list(dropped.items()), cap=16)

        # 5. Encodings -----------------------------------------------------------
        if fe and fe.encoding_applied:
            enc_items: list[tuple[str, str]] = []
            for col, why in fe.encoding_applied.items():
                if "dropped" in why:
                    continue
                if "one_hot" in why:
                    enc_items.append((col, "one-hot encoded (each category became its own yes/no column)"))
                elif "frequency" in why:
                    enc_items.append((col, "frequency-encoded (each category replaced by how often it appears)"))
                elif "label" in why:
                    enc_items.append((col, "label-encoded (each category mapped to a number)"))
            bullets("Categorical columns encoded", enc_items)

        # 6. Why this model won --------------------------------------------------
        if state.best_model_name and state.best_metric_value is not None:
            primary = state.best_metric_name or "score"
            label = primary.upper()
            others = sorted(
                (r for r in trained if r.model_name != state.best_model_name),
                key=lambda r: r.metrics.get(primary, float("-inf")), reverse=True,
            )
            lines += [f"### Why **{state.best_model_name}** was selected", ""]
            why = (f"Out of **{n_trained}** models trained, **{state.best_model_name}** scored the best "
                   f"validation **{label} ({state.best_metric_value:.4f})**")
            if others and primary in others[0].metrics:
                why += f", ahead of the runner-up **{others[0].model_name}** ({others[0].metrics[primary]:.4f})"
            why += ". Every model saw the same features and the same split, so this is an apples-to-apples comparison."
            lines += [why, ""]
            notes: list[str] = []
            if "ensemble" in state.best_model_name.lower():
                notes.append("It is a soft-voting **ensemble** — it averages the predictions of the top individual "
                             "models, which together generalized better than any one of them alone.")
            if state.best_model_name.endswith("_tuned"):
                notes.append("**Hyperparameter tuning** found settings that beat the model's defaults.")
            if state.best_threshold is not None:
                notes.append(f"Its decision threshold was tuned to **{state.best_threshold:.3f}** (not the default 0.5) "
                             f"to get the best balance of precision and recall on the imbalanced target.")
            tm = state.test_metrics or {}
            for k in (primary, "f1", "r2", "roc_auc"):
                if k in tm:
                    notes.append(f"On the **held-out test set** (data it never saw during training or selection) it "
                                 f"scored **{k.upper()} = {tm[k]:.4f}**, confirming the result is real and not overfit.")
                    break
            for note in notes:
                lines.append(f"- {note}")
            if notes:
                lines.append("")

        lines += ["---", ""]
        return lines


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
