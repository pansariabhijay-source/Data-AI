"""
Before/after benchmark for the class-imbalance + threshold fixes on the fraud
dataset. Demonstrates why the old pipeline produced high AUC but poor F1, and how
the fix recovers minority-class F1.

Run: python scripts/benchmark_fraud.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from core.constants import ProblemType
from core.metrics import predict_with_optimal_threshold
from core.model_registry import (
    apply_imbalance_handling,
    build_default_registry,
    fit_model,
    get_fitted_n_estimators,
)

DATA = "data/uploads/fraud.csv"
TARGET = "is_fraud"
SEED = 42


def load() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    # Minimal encoding so XGBoost can consume the categoricals.
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def report(name: str, y_true, y_pred, pos_proba) -> None:
    print(
        f"  {name:<34} "
        f"F1={f1_score(y_true, y_pred):.4f}  "
        f"P={precision_score(y_true, y_pred, zero_division=0):.4f}  "
        f"R={recall_score(y_true, y_pred):.4f}  "
        f"AUC={roc_auc_score(y_true, pos_proba):.4f}"
    )


def main() -> None:
    X, y = load()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED
    )
    print(f"train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}  "
          f"fraud rate={y.mean():.3%}\n")

    reg = build_default_registry(SEED)
    spec = reg.get(ProblemType.CLASSIFICATION, "XGBClassifier")

    # ── OLD: no scale_pos_weight, fixed 0.5 threshold ────────────────────────
    old = reg.create_instance(ProblemType.CLASSIFICATION, "XGBClassifier")
    old.set_params(scale_pos_weight=1.0)  # neutralise the new default
    old.fit(X_tr, y_tr)
    old_proba = old.predict_proba(X_te)[:, 1]
    old_pred = (old_proba >= 0.5).astype(int)

    # ── NEW: scale_pos_weight + early stopping + F1-optimal threshold ────────
    new = apply_imbalance_handling(
        reg.create_instance(ProblemType.CLASSIFICATION, "XGBClassifier"), spec, y_tr
    )
    new = fit_model(new, spec, X_tr, y_tr, eval_X=X_val, eval_y=y_val)
    trees = get_fitted_n_estimators(new)
    _, thr = predict_with_optimal_threshold(y_val, new.predict_proba(X_val), new.classes_)
    new_proba = new.predict_proba(X_te)[:, 1]
    new_pred = (new_proba >= thr).astype(int)

    print("XGBoost on held-out TEST set:")
    report("OLD (0.5 thresh, no weighting)", y_te, old_pred, old_proba)
    report(f"NEW (thr={thr:.3f}, {trees} trees, weighted)", y_te, new_pred, new_proba)


if __name__ == "__main__":
    main()
