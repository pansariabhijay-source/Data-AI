"""
Does the FE fix HELP on data that exercises the broken code paths?

The toy CSVs are too clean to trigger the bugs that were fixed. This builds a
synthetic classification set whose signal lives precisely in the features the OLD
pipeline destroyed:

  * `skewed_amount`     — right-skewed but predictive; OLD remove_low_variance
                          (clip+minmax+floor) could drop it as "near-constant".
  * `rare_flag`         — 1% prevalence, predictive; OLD low-variance clip nuked it.
  * `region_code` (int) — low-cardinality coded categorical with "code" in the
                          name; OLD drop_numeric_id_columns dropped it on the name.

Compares the real PreprocessingService + FeatureEngineeringService output against
minimal prep, scoring a fixed model on a held-out split. With the fixes the
pipeline should now retain the signal and match/beat minimal prep.

    venv/bin/python scripts/fe_bugpath_experiment.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from core.constants import ProblemType
from scripts.fe_ablation_experiment import raw_minimal, run_service_prep  # reuse

SEED = 42


def make_dataset(n: int = 8000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    # latent signal
    region = rng.integers(0, 6, size=n)              # 6 regions, coded as int
    region_risk = np.array([0.1, 0.5, 0.2, 0.8, 0.3, 0.6])[region]
    skewed = rng.lognormal(mean=2.0, sigma=1.3, size=n)   # right-skewed amount
    rare = (rng.random(n) < 0.01).astype(int)        # 1% rare flag
    noise = rng.normal(size=(n, 4))
    logit = (
        2.5 * region_risk
        + 0.0015 * skewed
        + 1.8 * rare
        - 2.0
        + 0.2 * noise[:, 0]
    )
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({
        "region_code": region,          # name-pattern trap for the ID dropper
        "skewed_amount": skewed,        # variance-drop trap
        "rare_flag": rare,              # rare one-hot/binary trap
        "noise_0": noise[:, 0], "noise_1": noise[:, 1],
        "noise_2": noise[:, 2], "noise_3": noise[:, 3],
        "target": y,
    })
    return df


def evaluate(df: pd.DataFrame, target: str) -> float:
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)
    y = df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    scores = []
    for m in (LogisticRegression(max_iter=1000, class_weight="balanced"),
              HistGradientBoostingClassifier(random_state=SEED)):
        m.fit(Xtr, ytr)
        scores.append(f1_score(yte, m.predict(Xte)))
    return float(np.mean(scores)), [c for c in X.columns]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    df = make_dataset()
    path = root / "data" / "_bugpath_synth.csv"
    df.to_csv(path, index=False)
    print(f"synthetic set: {df.shape}, positive rate={df['target'].mean():.3f}")

    base, base_cols = evaluate(raw_minimal(df, "target"), "target")
    feat = run_service_prep(str(path), "target", "pipeline")
    pipe, pipe_cols = evaluate(feat, "target")

    print(f"\n  raw_minimal   F1={base:.4f}  features={sorted(base_cols)}")
    print(f"  pipeline      F1={pipe:.4f}  (vs raw {pipe - base:+.4f})")
    print(f"                features kept: {sorted(pipe_cols)}")
    for trap in ("region_code", "skewed_amount", "rare_flag"):
        kept = any(c == trap or c.startswith(trap) for c in pipe_cols)
        print(f"    {'KEPT ' if kept else 'DROPPED'} {trap}")
    path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
