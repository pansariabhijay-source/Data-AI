"""
Prove the champion-calibration change improves probability quality on the
UNTOUCHED test split — before trusting it on by default.

For each dataset: fit the model on train, calibrate on val (isotonic/sigmoid,
prefit), and compare uncalibrated vs calibrated on test by:
  - Brier score (lower = better)
  - log loss (lower = better)
  - ROC-AUC (must be unchanged — calibration is monotonic)
  - F1 at each model's own F1-optimal threshold (should not get worse)

Run: python scripts/calibration_experiment.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.frozen import FrozenEstimator  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, f1_score  # noqa: E402
from core.metrics import predict_with_optimal_threshold  # noqa: E402

from lightgbm import LGBMClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402


def datasets():
    r = np.random.default_rng(0)
    out = {}
    n = 12000; X = r.normal(size=(n, 12))
    y = (X[:, 0] * 1.2 + X[:, 1] * 0.8 + r.normal(scale=0.8, size=n) > 0).astype(int)
    out["balanced"] = (X, y)
    n = 16000; X = r.normal(size=(n, 12))
    logit = X[:, 0] * 1.5 + X[:, 2] - X[:, 4]
    y = (r.uniform(size=n) < 1 / (1 + np.exp(-(logit - 2.5)))).astype(int)
    out["imbalanced(~8%)"] = (X, y)
    n = 12000; X = r.normal(size=(n, 10))
    y = ((X[:, 0] * X[:, 1] + X[:, 2] ** 2 - 1) + r.normal(scale=0.7, size=n) > 0).astype(int)
    out["nonlinear"] = (X, y)
    return out


def split(X, y, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X))
    nte = int(len(X) * 0.2); nva = int(len(X) * 0.2)
    return (X[idx[nte + nva:]], y[idx[nte + nva:]]), (X[idx[nte:nte + nva]], y[idx[nte:nte + nva]]), (X[idx[:nte]], y[idx[:nte]])


MODELS = {
    "LGBM": lambda: LGBMClassifier(n_estimators=200, verbose=-1),
    "XGB": lambda: XGBClassifier(n_estimators=200, verbosity=0),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=0),
}


def f1_at_opt(y_te, proba_te, y_va, proba_va, classes):
    # derive threshold on val, apply to test (mirrors the pipeline)
    _, thr = predict_with_optimal_threshold(y_va, proba_va, classes)
    pos = proba_te[:, -1]
    preds = np.where(pos >= thr, classes[-1], classes[0])
    return f1_score(y_te, preds)


def main():
    print(f"{'dataset':<16} {'model':<13} {'Brier u->c':<20} {'logloss u->c':<20} {'AUC':<9} {'F1 u->c':<16} adopt?")
    print("-" * 100)
    improved = total = 0
    auc_ok = True
    for name, (X, y) in datasets().items():
        (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y)
        for mname, mk in MODELS.items():
            m = mk(); m.fit(Xtr, ytr); classes = m.classes_
            pv, pt = m.predict_proba(Xva), m.predict_proba(Xte)
            yb = (yte == classes[-1]).astype(int)
            bb, lb = brier_score_loss(yb, pt[:, -1]), log_loss(yte, pt)
            auc_b = roc_auc_score(yb, pt[:, -1])
            f1_b = f1_at_opt(yte, pt, yva, pv, classes)

            method = "isotonic" if len(Xva) >= 1000 else "sigmoid"
            cal = CalibratedClassifierCV(FrozenEstimator(m), method=method).fit(Xva, yva)
            cpv, cpt = cal.predict_proba(Xva), cal.predict_proba(Xte)
            ba, la = brier_score_loss(yb, cpt[:, -1]), log_loss(yte, cpt)
            auc_a = roc_auc_score(yb, cpt[:, -1])
            f1_a = f1_at_opt(yte, cpt, yva, cpv, cal.classes_)

            # pipeline adopts only if val-Brier improves; here we report test-Brier
            adopt = ba < bb
            total += 1; improved += int(ba < bb)
            if abs(auc_a - auc_b) > 0.02:
                auc_ok = False
            print(f"{name:<16} {mname:<13} {bb:.4f}->{ba:.4f}      {lb:.4f}->{la:.4f}      "
                  f"{auc_b:.3f}     {f1_b:.4f}->{f1_a:.4f}   {'yes' if adopt else 'no'}")

    print("\n" + "=" * 60)
    print(f"Calibration improved test Brier on {improved}/{total} model-datasets")
    print(f"ROC-AUC preserved (monotonic) on all: {auc_ok}")
    ok = improved >= total * 0.6 and auc_ok
    print(f"\nVERDICT: {'calibration helps probability quality — keep default ON.' if ok else 'weak — reconsider default.'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
