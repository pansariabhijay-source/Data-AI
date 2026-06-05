"""
Multi-dataset benchmark harness for the Axiom pipeline.

Runs the REAL deterministic agent pipeline (data_collection → preprocessing →
feature_engineering → data_splitting → model_training [→ improvement]) on every
configured dataset and reports the honest held-out TEST metrics. This is the
objective yardstick used to decide whether a pipeline change actually improves
accuracy — no LLM / API key required.

Usage:
    venv/Scripts/python.exe scripts/benchmark_suite.py                 # all datasets
    venv/Scripts/python.exe scripts/benchmark_suite.py --quick         # skip creditcard (slow)
    venv/Scripts/python.exe scripts/benchmark_suite.py --only weather mental_health
    venv/Scripts/python.exe scripts/benchmark_suite.py --improve       # also run the tuning agent
    venv/Scripts/python.exe scripts/benchmark_suite.py --json out.json  # write machine-readable results
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.agent_runner import run_single_agent  # noqa: E402
from core.config import load_settings  # noqa: E402
from core.state import PipelineState  # noqa: E402
from core.utils import generate_run_id  # noqa: E402

UPLOADS = ROOT / "data" / "uploads"


@dataclass
class DatasetCase:
    key: str
    path: Path
    target: str
    note: str = ""
    slow: bool = False


CASES: list[DatasetCase] = [
    DatasetCase("sample_classification", UPLOADS / "sample_classification.csv",
                "species", "iris-like multiclass"),
    DatasetCase("weather", UPLOADS / "weather.csv",
                "RainTomorrow", "binary; RISK_MM is leakage"),
    DatasetCase("mental_health", UPLOADS / "mental_health_lifestyle_2000.csv",
                "Burnout", "binary/multiclass lifestyle"),
    DatasetCase("social_friction", UPLOADS / "social_friction_students_vs_workers.csv",
                "Narasi_Coping_Mechanism", "categorical-heavy"),
    DatasetCase("sales_dealsize", UPLOADS / "sales_data_sample.csv",
                "DEALSIZE", "classification, messy text cols"),
    DatasetCase("gold_volatility", UPLOADS / "high_Frequency_Gold_Vola tility_2026.csv",
                "Volume", "regression on OHLCV"),
    DatasetCase("creditcard", UPLOADS / "creditcard.csv",
                "Class", "real fraud, 0.17% positive", slow=True),
]

# Pipeline stages to run for the benchmark. Stop after training to get the
# honest held-out test metrics; optionally include improvement (tuning).
BASE_STAGES = [
    "data_collection",
    "preprocessing",
    "feature_engineering",
    "data_splitting",
    "model_training",
]


@dataclass
class CaseResult:
    key: str
    status: str = "ok"
    problem_type: str | None = None
    best_model: str | None = None
    val_metric_name: str | None = None
    val_metric: float | None = None
    test_metrics: dict = field(default_factory=dict)
    threshold: float | None = None
    seconds: float = 0.0
    error: str | None = None


def run_case(case: DatasetCase, settings, improve: bool) -> CaseResult:
    res = CaseResult(key=case.key)
    if not case.path.exists():
        res.status = "missing"
        res.error = f"file not found: {case.path}"
        return res

    state = PipelineState(
        run_id=generate_run_id() + f"_bench_{case.key}",
        target_column=case.target,
        raw_data_path=str(case.path),
    )

    stages = list(BASE_STAGES)
    if improve:
        stages += ["error_detection", "improvement"]

    t0 = time.perf_counter()
    try:
        for stage in stages:
            out = run_single_agent(stage, state, settings)
            if out.status == "error" and stage == "model_training":
                res.status = "error"
                res.error = f"{stage}: {out.error}"
                res.seconds = time.perf_counter() - t0
                return res
    except Exception as e:  # noqa: BLE001
        res.status = "error"
        res.error = f"{type(e).__name__}: {e}"
        res.seconds = time.perf_counter() - t0
        return res

    # When tuning ran, the champion may have changed after training scored the
    # test split — re-score so the reported metrics reflect the tuned model.
    if improve:
        _rescore_test(state)

    res.seconds = time.perf_counter() - t0
    res.problem_type = state.problem_type
    res.best_model = state.best_model_name
    res.val_metric_name = state.best_metric_name
    res.val_metric = state.best_metric_value
    res.test_metrics = dict(state.test_metrics or {})
    res.threshold = state.best_threshold

    # Benchmark artifacts (trained models, intermediate CSVs) are throwaway and
    # large (creditcard models alone are ~1GB). Remove them so repeated runs
    # don't fill the disk.
    _cleanup_run(state, settings)
    return res


def _rescore_test(state: PipelineState) -> None:
    """Re-evaluate the current champion on the held-out test split (post-tuning)."""
    import joblib
    import numpy as np
    import pandas as pd
    from core.constants import ProblemType
    from core.metrics import compute_metrics, positive_class_proba

    pt = ProblemType(state.problem_type) if state.problem_type else None
    if not state.test_path or not state.best_model_path or pt in (None, ProblemType.CLUSTERING):
        return
    try:
        test_df = pd.read_csv(state.test_path, low_memory=False)
        target = state.target_column
        if not target or target not in test_df.columns:
            return
        X = test_df.drop(columns=[target]).select_dtypes(include=[np.number]).values
        y = test_df[target].values
        model = joblib.load(state.best_model_path)
        y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        if (pt == ProblemType.CLASSIFICATION and y_prob is not None
                and state.best_threshold is not None and len(np.unique(y)) == 2):
            classes = np.asarray(getattr(model, "classes_", [0, 1]))
            pos = positive_class_proba(y_prob)
            preds = np.where(pos >= state.best_threshold, classes[1], classes[0])
        else:
            preds = model.predict(X)
        state.test_metrics = compute_metrics(y, preds, pt, y_prob)
    except Exception:  # noqa: BLE001
        pass


def _cleanup_run(state: PipelineState, settings) -> None:
    for base in (settings.pipeline.artifact_dir, "reports"):
        d = ROOT / base / state.run_id
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    for attr in ("cleaned_data_path", "featured_data_path",
                 "train_path", "val_path", "test_path"):
        p = getattr(state, attr, None)
        if p and Path(p).exists() and "uploads" not in str(p):
            try:
                Path(p).unlink()
            except OSError:
                pass


def fmt_metrics(m: dict) -> str:
    if not m:
        return "(no test metrics)"
    keys = ["f1", "roc_auc", "pr_auc", "precision", "recall", "accuracy", "r2", "rmse", "mae"]
    parts = [f"{k}={m[k]:.4f}" for k in keys if k in m]
    return "  ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip slow datasets (creditcard)")
    ap.add_argument("--only", nargs="*", help="run only these dataset keys")
    ap.add_argument("--improve", action="store_true", help="also run error_detection + improvement")
    ap.add_argument("--json", type=str, default=None, help="write results to this JSON file")
    args = ap.parse_args()

    settings = load_settings()

    cases = CASES
    if args.only:
        cases = [c for c in cases if c.key in set(args.only)]
    if args.quick:
        cases = [c for c in cases if not c.slow]

    print("=" * 92)
    print(f"AXIOM BENCHMARK SUITE  ({len(cases)} datasets, improve={args.improve})")
    print("=" * 92)

    results: list[CaseResult] = []
    for case in cases:
        print(f"\n>> {case.key}  [{case.note}]  target={case.target}")
        r = run_case(case, settings, args.improve)
        results.append(r)
        if r.status != "ok":
            print(f"  [X] {r.status}: {r.error}  ({r.seconds:.1f}s)")
            continue
        thr = f"  thr={r.threshold:.4f}" if r.threshold is not None else ""
        print(f"  problem={r.problem_type}  best={r.best_model}  "
              f"val_{r.val_metric_name}={r.val_metric:.4f}{thr}  ({r.seconds:.1f}s)")
        print(f"  TEST: {fmt_metrics(r.test_metrics)}")

    print("\n" + "=" * 92)
    print("SUMMARY (held-out TEST)")
    print("=" * 92)
    for r in results:
        if r.status != "ok":
            print(f"  {r.key:<22} {r.status.upper():<8} {r.error or ''}")
            continue
        head = r.test_metrics.get("f1", r.test_metrics.get("r2"))
        head_name = "f1" if "f1" in r.test_metrics else ("r2" if "r2" in r.test_metrics else "?")
        head_str = f"{head:.4f}" if head is not None else "n/a"
        print(f"  {r.key:<22} {r.best_model or '?':<28} test_{head_name}={head_str:<10} ({r.seconds:.0f}s)")

    if args.json:
        out = [vars(r) for r in results]
        Path(args.json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
