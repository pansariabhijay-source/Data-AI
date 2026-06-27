"""
FE / preprocessing ablation — does the pipeline's preprocessing + feature
engineering HELP or HURT model quality vs a minimal baseline?

Motivated by the report that running the pipeline WITHOUT preprocessing + FE gave
better results than WITH them. This isolates the effect by training the same
models on the same train/test split under several feature-prep regimes:

  RAW-MINIMAL : median-impute numerics, one-hot small cats, frequency-encode big
                cats, drop near-unique id cols. No outlier clipping, no variance/
                correlation pruning, no top-K MI cut, no scaling.
  PIPELINE    : the real PreprocessingService + FeatureEngineeringService output
                (artifacts/<run>/featured_data.csv).
  + ablations : PIPELINE with one aggressive step turned off, to find the culprit.

Runs purely offline (no API, no LLM). Usage:
    venv/bin/python scripts/fe_ablation_experiment.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split

from core.config import load_settings
from core.constants import ProblemType
from core.state import PipelineState
from core.utils import generate_run_id

SEED = 42


# ── feature prep regimes ────────────────────────────────────────────────────

def raw_minimal(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Cheapest sane prep: impute, encode, drop pure-id cols. Nothing aggressive."""
    df = df.copy()
    df = df.dropna(subset=[target])
    y = df[target]
    X = df.drop(columns=[target])
    n = max(len(X), 1)
    for col in list(X.columns):
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            if s.nunique() / n > 0.999 and "float" not in str(s.dtype):
                X = X.drop(columns=[col])          # pure integer id
            else:
                X[col] = s.fillna(s.median())
            continue
        # object / categorical
        nun = s.nunique(dropna=True)
        if nun / n > 0.5:                          # id-like string
            X = X.drop(columns=[col])
        elif nun <= 15:
            d = pd.get_dummies(s.astype("object"), prefix=col, dummy_na=True, dtype=int)
            X = pd.concat([X.drop(columns=[col]), d], axis=1)
        else:                                       # frequency encode
            f = s.astype("object").map(s.astype("object").value_counts(normalize=True))
            X[col] = f.astype(float).fillna(0.0)
    out = pd.concat([X, y.rename(target)], axis=1)
    return out


def run_service_prep(raw_path: str, target: str, regime: str) -> pd.DataFrame:
    """Run the real preprocessing + FE services, optionally disabling one step."""
    from agents.preprocessing.tools import PreprocessingService
    from agents.feature_engineering.tools import FeatureEngineeringService

    settings = load_settings()
    state = PipelineState(run_id=generate_run_id(), raw_data_path=raw_path, target_column=target)
    # Problem type via data_collection (sets state.problem_type)
    from core.agent_runner import run_single_agent
    run_single_agent("data_collection", state, settings)

    pre = PreprocessingService(settings.preprocessing)
    fe = FeatureEngineeringService(settings.feature_engineering)

    # Monkeypatch one aggressive step into a no-op for ablations.
    if regime == "no_outlier":
        pre.handle_outliers = lambda df, target=None: (df, {})
    if regime == "no_variance":
        fe.remove_low_variance = lambda df, target=None: (df, [])
    if regime == "no_corr":
        fe.remove_correlated = lambda df, target=None: (df, [])
    if regime == "no_topk":
        fe.select_k_best = lambda df, tgt, pt: (df, {})
    if regime == "no_scale":
        fe.scale_features = lambda df, target=None: (df, {})
    if regime == "no_iddrop":
        fe.drop_numeric_id_columns = lambda df, target=None: (df, {})

    pre.run(state, settings)
    fe.run(state, settings)
    return pd.read_csv(state.featured_data_path, low_memory=False)


# ── evaluation ──────────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame, target: str, problem: ProblemType) -> float:
    if target not in df.columns:
        return float("nan")
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)
    y = df[target]
    if X.shape[1] == 0:
        return float("nan")
    strat = y if problem == ProblemType.CLASSIFICATION else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=strat)
    scores = []
    if problem == ProblemType.CLASSIFICATION:
        models = [
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            HistGradientBoostingClassifier(random_state=SEED),
        ]
        for m in models:
            m.fit(Xtr, ytr)
            avg = "binary" if y.nunique() == 2 else "macro"
            scores.append(f1_score(yte, m.predict(Xte), average=avg))
    else:
        models = [Ridge(), HistGradientBoostingRegressor(random_state=SEED)]
        for m in models:
            m.fit(Xtr, ytr)
            scores.append(r2_score(yte, m.predict(Xte)))
    return float(np.mean(scores))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    datasets = [
        ("sample_classification", root / "data/sample_classification.csv", "species", ProblemType.CLASSIFICATION),
        ("sample_regression", root / "data/sample_regression.csv", "price", ProblemType.REGRESSION),
        ("fraud_detection", root / "Fraud Detection Dataset.csv", "Fraudulent", ProblemType.CLASSIFICATION),
    ]

    regimes = ["raw_minimal", "pipeline", "no_outlier", "no_variance",
               "no_corr", "no_topk", "no_scale", "no_iddrop"]

    for name, path, target, problem in datasets:
        if not path.exists():
            print(f"skip {name}: missing {path}")
            continue
        print(f"\n=== {name}  (target={target}, {problem.value}) ===")
        raw = pd.read_csv(path, low_memory=False)
        print(f"  raw shape: {raw.shape}")
        results = {}
        # baseline
        results["raw_minimal"] = evaluate(raw_minimal(raw, target), target, problem)
        for regime in regimes[1:]:
            try:
                feat = run_service_prep(str(path), target, regime)
                results[regime] = evaluate(feat, target, problem)
            except Exception as e:
                results[regime] = float("nan")
                print(f"    {regime} FAILED: {e}")
        base = results["raw_minimal"]
        metric = "F1" if problem == ProblemType.CLASSIFICATION else "R2"
        for regime in regimes:
            v = results[regime]
            delta = "" if regime == "raw_minimal" else f"  (vs raw {v - base:+.4f})"
            print(f"    {regime:14s} {metric}={v:.4f}{delta}")


if __name__ == "__main__":
    main()
