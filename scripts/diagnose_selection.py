"""
Diagnostic: run the pipeline through training on a dataset, then score EVERY
trained model on the untouched test split. Reveals whether champion selection
(by single-split validation F1) is leaving accuracy on the table vs. what a
different selection criterion would have picked.

Usage: venv/Scripts/python.exe scripts/diagnose_selection.py <dataset_key>
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.agent_runner import run_single_agent  # noqa: E402
from core.config import load_settings  # noqa: E402
from core.constants import ProblemType  # noqa: E402
from core.metrics import compute_metrics, find_optimal_threshold, positive_class_proba  # noqa: E402
from core.state import PipelineState  # noqa: E402
from core.utils import generate_run_id  # noqa: E402

UPLOADS = ROOT / "data" / "uploads"
CASES = {
    "creditcard": (UPLOADS / "creditcard.csv", "Class"),
    "sample_classification": (UPLOADS / "sample_classification.csv", "species"),
    "mental_health": (UPLOADS / "mental_health_lifestyle_2000.csv", "Burnout"),
    "weather": (UPLOADS / "weather.csv", "RainTomorrow"),
}


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "creditcard"
    path, target = CASES[key]
    settings = load_settings()
    state = PipelineState(run_id=generate_run_id() + f"_sel_{key}",
                          target_column=target, raw_data_path=str(path))
    for stage in ("data_collection", "preprocessing", "feature_engineering",
                  "data_splitting", "model_training"):
        run_single_agent(stage, state, settings)

    pt = ProblemType(state.problem_type)
    test_df = pd.read_csv(state.test_path, low_memory=False)
    val_df = pd.read_csv(state.val_path, low_memory=False) if state.val_path else None
    Xte = test_df.drop(columns=[target]).select_dtypes(include=[np.number]).values
    yte = test_df[target].values
    if val_df is not None:
        Xva = val_df.drop(columns=[target]).select_dtypes(include=[np.number]).values
        yva = val_df[target].values

    print(f"\n{key}: problem={pt.value}  champion={state.best_model_name} "
          f"(val_{state.best_metric_name}={state.best_metric_value:.4f})")
    print(f"{'model':<32} {'val_f1':>8} {'val_prau':>8} {'val_auc':>8} | "
          f"{'te_f1':>8} {'te_auc':>8} {'te_prauc':>8}")
    print("-" * 92)

    binary = pt == ProblemType.CLASSIFICATION and len(np.unique(yte)) == 2
    rows = []
    for r in state.model_results:
        if r.status != "trained" or not r.model_path or not Path(r.model_path).exists():
            continue
        model = joblib.load(r.model_path)
        yprob = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None
        if binary and yprob is not None and val_df is not None:
            # tune threshold on val (honest), apply to test
            vp = positive_class_proba(model.predict_proba(Xva))
            classes = np.asarray(getattr(model, "classes_", [0, 1]))
            thr = find_optimal_threshold(yva, vp, pos_label=classes[1])
            pos = positive_class_proba(yprob)
            preds = np.where(pos >= thr, classes[1], classes[0])
        else:
            preds = model.predict(Xte)
        m = compute_metrics(yte, preds, pt, yprob)
        rows.append((r.model_name, r.metrics.get("f1", float("nan")), m, r.metrics))
        print(f"{r.model_name:<32} {r.metrics.get('f1', float('nan')):>8.4f} "
              f"{r.metrics.get('pr_auc', float('nan')):>8.4f} {r.metrics.get('roc_auc', float('nan')):>8.4f} | "
              f"{m.get('f1', float('nan')):>8.4f} {m.get('roc_auc', float('nan')):>8.4f} "
              f"{m.get('pr_auc', float('nan')):>8.4f}")

    if rows and binary:
        best_test = max(rows, key=lambda x: x[2].get("f1", 0))
        best_auc = max(rows, key=lambda x: x[2].get("roc_auc", 0))
        pick_valf1 = max(rows, key=lambda x: x[3].get("f1", 0))
        pick_prauc = max(rows, key=lambda x: x[3].get("pr_auc", 0))
        print("-" * 92)
        print(f"Best on TEST f1:        {best_test[0]} (test_f1={best_test[2].get('f1'):.4f})")
        print(f"Best on TEST auc:       {best_auc[0]} (test_auc={best_auc[2].get('roc_auc'):.4f})")
        print(f"Picked by val_f1:       {pick_valf1[0]} -> test_f1={pick_valf1[2].get('f1'):.4f} test_auc={pick_valf1[2].get('roc_auc'):.4f}")
        print(f"Picked by val_pr_auc:   {pick_prauc[0]} -> test_f1={pick_prauc[2].get('f1'):.4f} test_auc={pick_prauc[2].get('roc_auc'):.4f}")

    import shutil
    shutil.rmtree(ROOT / settings.pipeline.artifact_dir / state.run_id, ignore_errors=True)


if __name__ == "__main__":
    main()
