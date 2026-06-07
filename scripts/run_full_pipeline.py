"""
Run the full deterministic Axiom pipeline (all 8 agents, no LLM/CrewAI required)
on a dataset and report the champion, held-out metrics, and artifact locations.

This is the same agent machinery the API/app drives, so it produces a real run
under artifacts/<run_id>/ (models + metrics + SHAP) and reports/<run_id>/.

Usage:
    venv/Scripts/python.exe scripts/run_full_pipeline.py --data data/uploads/foo.csv --target label
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.agent_runner import ALL_AGENTS_ORDERED, run_single_agent  # noqa: E402
from core.config import load_settings  # noqa: E402
from core.state import PipelineState  # noqa: E402
from core.utils import generate_run_id  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    settings = load_settings()
    run_id = args.run_id or generate_run_id()
    state = PipelineState(run_id=run_id, target_column=args.target, raw_data_path=args.data)

    print(f"\n{'=' * 78}\nAXIOM PIPELINE  run_id={run_id}\n  data={args.data}\n  target={args.target}\n{'=' * 78}")
    t0 = time.perf_counter()
    for agent in ALL_AGENTS_ORDERED:
        ts = time.perf_counter()
        out = run_single_agent(agent, state, settings)
        flag = "OK " if out.status != "error" else "ERR"
        print(f"  [{flag}] {agent:<20} {time.perf_counter() - ts:6.1f}s"
              + (f"   ERROR: {out.error}" if out.error else ""))
        if out.status == "error" and agent in ("data_collection", "data_splitting", "model_training"):
            print("  Aborting — a required stage failed.")
            break

    dt = time.perf_counter() - t0
    print(f"\n{'-' * 78}")
    print(f"problem_type : {state.problem_type}")
    print(f"champion     : {state.best_model_name}  "
          f"({state.best_metric_name}={state.best_metric_value})")
    if state.best_threshold is not None:
        print(f"threshold    : {state.best_threshold:.4f}")
    if state.test_metrics:
        print("held-out TEST:")
        for k in ("f1", "precision", "recall", "roc_auc", "pr_auc",
                  "balanced_accuracy", "accuracy", "r2", "rmse", "mae"):
            if k in state.test_metrics:
                print(f"   {k:<18} {state.test_metrics[k]:.4f}")
    if state.test_confusion:
        print(f"confusion    : {state.test_confusion}")
    print(f"\nartifacts    : artifacts/{run_id}/")
    print(f"report       : reports/{run_id}/report.md")
    print(f"total time   : {dt:.1f}s")


if __name__ == "__main__":
    main()
