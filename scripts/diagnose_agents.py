"""
Diagnostic harness: build a deliberately DIRTY dataset and run every agent
through core.agent_runner, reporting timing and what each agent actually changed.

Run:  venv/Scripts/python.exe scripts/diagnose_agents.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.agent_runner import ALL_AGENTS_ORDERED, run_single_agent  # noqa: E402
from core.config import load_settings  # noqa: E402
from core.state import PipelineState  # noqa: E402
from core.utils import generate_run_id  # noqa: E402


def make_dirty_dataset(path: Path, n: int = 1200) -> None:
    """Create a messy classification dataset with every cleaning challenge AND
    genuine signal, so we can verify cleaning preserves the signal and the model
    is actually good (not just that the pipeline ran)."""
    rng = np.random.RandomState(7)

    # ── genuine predictive features (target is a real function of these) ──────
    age = rng.normal(40, 12, n)
    income = rng.normal(50000, 15000, n)
    risk_flag = rng.binomial(1, 0.3, n)        # predictive BINARY feature
    # latent score drives the target → real, learnable signal
    logit = -1.0 + 0.04 * (age - 40) + 0.00003 * (income - 50000) + 1.6 * risk_flag
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.uniform(0, 1, n) < prob).astype(int)

    # ── now dirty the features up ────────────────────────────────────────────
    age[rng.choice(n, 60, replace=False)] = np.nan          # missing
    age[rng.choice(n, 15, replace=False)] = 999             # extreme outliers
    income[rng.choice(n, 100, replace=False)] = np.nan

    # numeric stored as strings with stray text
    score = rng.uniform(0, 100, n).round(2).astype(object)
    score[rng.choice(n, 40, replace=False)] = "N/A"

    # constant (zero-variance) column
    constant = np.ones(n)

    # high-null column (should be dropped)
    mostly_null = np.full(n, np.nan, dtype=object)
    mostly_null[rng.choice(n, 30, replace=False)] = 1.0

    # categoricals: low + high cardinality
    city = rng.choice(["NY", "LA", "SF", "CHI", None], n)
    user_id = [f"u{idx}" for idx in range(n)]               # ID-like (drop me)

    # boolean-as-string
    is_member = rng.choice(["yes", "no"], n).astype(object)

    # datetime as string
    dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.randint(0, 1000, n), unit="D")
    signup = dates.astype(str)

    # leakage: near-perfect copy of target (must be detected AND removed)
    leaky = target + rng.normal(0, 0.001, n)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "risk_flag": risk_flag,
        "score": score,
        "constant": constant,
        "mostly_null": mostly_null,
        "city": city,
        "user_id": user_id,
        "is_member": is_member,
        "signup": signup,
        "leaky": leaky,
        "target": target,
    })

    # inject duplicate rows
    df = pd.concat([df, df.iloc[:50]], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"[dirty data] wrote {len(df)} rows x {len(df.columns)} cols -> {path}")


def snapshot(path: str | None) -> str:
    if not path or not Path(path).exists():
        return "  (no file)"
    df = pd.read_csv(path, low_memory=False)
    return f"  shape={df.shape}  dtypes={dict(df.dtypes.astype(str).value_counts())}"


def main() -> None:
    data_path = ROOT / "data" / "dirty_test.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    make_dirty_dataset(data_path)

    settings = load_settings()
    state = PipelineState(
        run_id=generate_run_id() + "_diag",
        target_column="target",
        raw_data_path=str(data_path),
    )

    print("\n" + "=" * 70)
    print("RUNNING EACH AGENT IN ORDER")
    print("=" * 70)

    for agent in ALL_AGENTS_ORDERED:
        out = run_single_agent(agent, state, settings)
        print(f"\n── {agent} ── [{out.status}] {out.duration_seconds:.3f}s")
        if out.error:
            print(f"   ERROR: {out.error}")
        if out.summary:
            for k, v in out.summary.items():
                print(f"   {k}: {v}")

    print("\n" + "=" * 70)
    print("STATE TRACE")
    print("=" * 70)
    print("raw     :", snapshot(state.raw_data_path))
    print("cleaned :", snapshot(state.cleaned_data_path))
    print("featured:", snapshot(state.featured_data_path))
    print("train   :", snapshot(state.train_path))
    if state.preprocessing_summary:
        p = state.preprocessing_summary
        print(f"\nPREPROCESSING: rows {p.rows_before}->{p.rows_after}, "
              f"cols {p.columns_before}->{p.columns_after}, "
              f"dups={p.duplicates_removed}, nulls_filled={len(p.nulls_filled)}, "
              f"outliers={len(p.outliers_handled)}, dropped={p.columns_dropped}, "
              f"quality={p.quality_score}")
        print(f"  null strategies: {p.nulls_filled}")
        print(f"  dtype fixes: {p.dtypes_fixed}")
    print(f"\nproblem_type={state.problem_type}")
    print(f"best_model={state.best_model_name} {state.best_metric_name}={state.best_metric_value}")
    print(f"errors detected: {len(state.error_reports)}")
    for e in state.error_reports:
        print(f"  [{e.severity}] {e.error_type}: {e.root_cause[:80]}")


if __name__ == "__main__":
    main()
