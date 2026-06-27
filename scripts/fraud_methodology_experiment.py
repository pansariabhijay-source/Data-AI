"""
Fraud methodology experiment — quantify three senior-DS improvements on the
canonical creditcard.csv (0.173% positive, time-ordered).

  1. SPLIT: stratified-random vs chronological (out-of-time). Random splitting
     leaks the future on temporal data; OOT is what a fraud team must report.
  2. SELECTION: champion picked by mean(PR-AUC,ROC-AUC) vs PR-AUC alone, under
     extreme imbalance where ROC-AUC is saturated and uninformative.
  3. REFIT: champion fit on a 100k subsample (current pipeline) vs refit on the
     full training set.

Reports held-out PR-AUC (average precision) and F1 — the metrics that matter for
imbalanced fraud. Uses the project's real registry + metric utilities.

    venv/bin/python scripts/fraud_methodology_experiment.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import average_precision_score, f1_score

from core.constants import ProblemType
from core.metrics import (
    positive_class_proba, predict_with_optimal_threshold, selection_score,
)
from core.model_registry import (
    apply_imbalance_handling, build_default_registry, fit_model, maybe_wrap_scaler,
)

SEED = 42
TARGET = "Class"
SUBSAMPLE = 100_000


def split(df: pd.DataFrame, mode: str):
    n = len(df)
    if mode == "random":
        from sklearn.model_selection import train_test_split
        tv, te = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df[TARGET])
        tr, va = train_test_split(tv, test_size=0.1765, random_state=SEED, stratify=tv[TARGET])
    else:  # chronological — train=oldest, test=newest
        df = df.sort_values("Time")
        i1, i2 = int(n * 0.70), int(n * 0.85)
        tr, va, te = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    return tr, va, te


def xy(d):
    return d.drop(columns=[TARGET]).select_dtypes(include=[np.number]).values, d[TARGET].values


def subsample(X, y, cap=SUBSAMPLE):
    if len(X) <= cap:
        return X, y
    rng = np.random.default_rng(SEED)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        alloc = max(int(round(cap * len(idx) / len(X))), min(len(idx), 2000))
        keep.append(idx if alloc >= len(idx) else rng.choice(idx, alloc, replace=False))
    idx = np.concatenate(keep); rng.shuffle(idx)
    return X[idx], y[idx]


def train_all(Xtr, ytr, Xva):
    reg = build_default_registry(SEED)
    out = {}
    for spec in reg.list_models(ProblemType.CLASSIFICATION):
        if spec.max_train_samples and len(Xtr) > spec.max_train_samples:
            continue
        try:
            m = reg.create_instance(ProblemType.CLASSIFICATION, spec.name)
            m = apply_imbalance_handling(m, spec, ytr)
            m = maybe_wrap_scaler(m, spec)
            m = fit_model(m, spec, Xtr, ytr, eval_X=Xva, eval_y=None if spec.supports_early_stopping else None)
            out[spec.name] = m
        except Exception as e:
            print(f"    ({spec.name} skipped: {str(e)[:50]})")
    return out


def evaluate(df, split_mode, refit_full):
    tr, va, te = split(df, split_mode)
    Xtr_full, ytr = xy(tr); Xva, yva = xy(va); Xte, yte = xy(te)
    Xtr, ytr_s = subsample(Xtr_full, ytr)
    models = train_all(Xtr, ytr_s, Xva)

    # selection candidates: val metrics
    from core.metrics import compute_metrics
    rows = {}
    for name, m in models.items():
        pv = m.predict_proba(Xva)
        vm = compute_metrics(yva, m.predict(Xva), ProblemType.CLASSIFICATION, pv)
        rows[name] = (m, vm)

    def pick(metric_fn):
        return max(rows, key=lambda n: metric_fn(rows[n][1]))

    champ_mean = pick(lambda vm: selection_score(vm, ProblemType.CLASSIFICATION, 2))
    champ_prauc = pick(lambda vm: vm.get("pr_auc", 0))

    def test_scores(name):
        m = rows[name][0]
        if refit_full and len(Xtr_full) > len(Xtr):
            # refit the SAME kind of model on the full training set
            reg = build_default_registry(SEED)
            spec = reg.get(ProblemType.CLASSIFICATION, name) if name in [s.name for s in reg.list_models(ProblemType.CLASSIFICATION)] else None
            if spec is not None:
                try:
                    mm = maybe_wrap_scaler(apply_imbalance_handling(reg.create_instance(ProblemType.CLASSIFICATION, name), spec, ytr), spec)
                    m = fit_model(mm, spec, Xtr_full, ytr)
                except Exception:
                    pass
        pv = m.predict_proba(Xva)
        _, thr = predict_with_optimal_threshold(yva, pv, m.classes_)
        pt = positive_class_proba(m.predict_proba(Xte))
        preds = np.where(pt >= thr, m.classes_[1], m.classes_[0])
        return average_precision_score(yte, pt), f1_score(yte, preds)

    return champ_mean, champ_prauc, test_scores


def main():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "creditcard.csv")
    print(f"creditcard: {df.shape}, positives {int(df.Class.sum())} ({df.Class.mean()*100:.3f}%)\n")

    for split_mode in ("random", "chronological"):
        print(f"=== SPLIT = {split_mode} ===")
        cm, cp, scorer = evaluate(df, split_mode, refit_full=False)
        ap_m, f1_m = scorer(cm)
        ap_p, f1_p = scorer(cp)
        print(f"  champion by mean(PR,ROC) : {cm:24s} -> test PR-AUC={ap_m:.4f}  F1={f1_m:.4f}")
        print(f"  champion by PR-AUC only  : {cp:24s} -> test PR-AUC={ap_p:.4f}  F1={f1_p:.4f}")
        # refit effect on the PR-AUC champion
        _, cp2, scorer_refit = evaluate(df, split_mode, refit_full=True)
        ap_r, f1_r = scorer_refit(cp2)
        print(f"  + refit champion on FULL train ({cp2}) -> test PR-AUC={ap_r:.4f}  F1={f1_r:.4f}")
        print()


if __name__ == "__main__":
    main()
