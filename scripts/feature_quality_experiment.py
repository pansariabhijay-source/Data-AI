"""
Prove the two Tier-3 feature-quality changes actually help, on data where the
signal they capture exists:

  1. Missing-value indicators — when *whether* a value is missing is predictive
     (common in fraud), imputation alone erases it. Compare model test AUC with
     indicators ON vs OFF.
  2. Frequency encoding vs label encoding for a high-cardinality category whose
     rarity predicts the target. Compare test AUC.

Run: python scripts/feature_quality_experiment.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

import agents.preprocessing.tools as P  # noqa: E402
from core.config import load_settings  # noqa: E402


def _auc(Xtr, ytr, Xte, yte, linear=False):
    if linear:
        m = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    else:
        m = LGBMClassifier(n_estimators=200, verbose=-1).fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])


def test_missing_indicators():
    rng = np.random.default_rng(0); n = 16000
    df = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(6)})
    # target depends partly on whether `x_inform` is missing
    missing_mask = rng.uniform(size=n) < 0.35
    y = ((df["x0"] * 0.6 + missing_mask * 1.4 + rng.normal(scale=0.7, size=n)) > 0.8).astype(int)
    x_inform = rng.normal(size=n)
    x_inform[missing_mask] = np.nan  # value itself is pure noise; MISSINGNESS carries signal
    df["x_inform"] = x_inform
    df["target"] = y

    svc = P.PreprocessingService(load_settings().preprocessing)
    results = {}
    for flag in (0, 1):
        P._ADD_MISSING_INDICATORS = flag
        out, _ = svc.handle_missing_values(df.copy(), "target")
        feats = [c for c in out.columns if c != "target"]
        ntr = 12000
        # Use a linear model: trees can recover some missingness from the
        # median-imputation spike, but linear models can't — so this isolates the
        # indicator's true contribution (and LogisticRegression is a common champ).
        results[flag] = _auc(out[feats].iloc[:ntr], out["target"].iloc[:ntr],
                             out[feats].iloc[ntr:], out["target"].iloc[ntr:], linear=True)
    P._ADD_MISSING_INDICATORS = 1
    return results[0], results[1]


def test_frequency_encoding():
    rng = np.random.default_rng(1); n = 16000
    # 300 device ids; rare devices are fraudulent
    counts = rng.integers(1, 60, size=300)
    devices = np.repeat(np.arange(300), counts)
    rng.shuffle(devices)
    devices = devices[:n] if len(devices) >= n else np.resize(devices, n)
    freq = pd.Series(devices).map(pd.Series(devices).value_counts())
    p = 1 / (1 + np.exp((freq.values - 25) / 8))  # rarer -> higher fraud prob
    y = (rng.uniform(size=n) < p).astype(int)
    base = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})

    ntr = 12000
    # label encoding (old behaviour): arbitrary integer per device
    lab = pd.Series(devices).astype("category").cat.codes.values
    Xlab = base.assign(device=lab)
    auc_label = _auc(Xlab.iloc[:ntr], y[:ntr], Xlab.iloc[ntr:], y[ntr:])
    # frequency encoding (new behaviour)
    s = pd.Series(devices).astype(str)
    fe = s.map(s.value_counts(normalize=True)).astype(float).values
    Xfe = base.assign(device=fe)
    auc_freq = _auc(Xfe.iloc[:ntr], y[:ntr], Xfe.iloc[ntr:], y[ntr:])
    return auc_label, auc_freq


def main():
    a_off, a_on = test_missing_indicators()
    print("1) Missing-value indicators (informative missingness, linear model)")
    print(f"   test AUC  OFF={a_off:.4f}  ON={a_on:.4f}   delta={a_on - a_off:+.4f}")

    a_lab, a_freq = test_frequency_encoding()
    print("2) High-cardinality encoding (rarity predicts target)")
    print(f"   test AUC  label={a_lab:.4f}  frequency={a_freq:.4f}   delta={a_freq - a_lab:+.4f}")

    ok = (a_on > a_off + 0.005) and (a_freq > a_lab + 0.005)
    print(f"\nVERDICT: {'both changes help on the signal they target.' if ok else 'weak — reconsider.'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
