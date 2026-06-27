"""
Prove (or disprove) that k-fold CV champion selection is more STABLE than the
old single-validation-split selection — before trusting it in the pipeline.

For each dataset and many random train/val/test splits we crown a champion two
ways:
  (A) single-split: max selection_score on one val split (current behaviour)
  (B) cv: max k-fold CV mean (minus a small std penalty) (new behaviour)

We then report, per dataset:
  - stability: how often each method picks its OWN modal champion across seeds
    (higher = more stable / less luck-of-the-split)
  - generalization: mean selection_score of each method's champion on the
    untouched TEST split (higher = better, or at least not worse)

Run: python scripts/cv_selection_experiment.py
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_settings  # noqa: E402
from core.constants import ProblemType  # noqa: E402
from core.metrics import compute_metrics, selection_score  # noqa: E402
from core.model_registry import build_default_registry  # noqa: E402
from agents.training.tools import ModelTrainingService, _CV_STD_PENALTY  # noqa: E402

N_SEEDS = 10


def datasets() -> dict:
    """Numeric datasets where model selection is genuinely close (where stability matters)."""
    out = {}
    r = np.random.default_rng(0)

    # 1) Near-linear: logistic vs trees are close
    n = 3000; X = r.normal(size=(n, 8))
    y = (X @ r.normal(size=8) + r.normal(scale=1.2, size=n) > 0).astype(int)
    out["near_linear"] = (pd.DataFrame(X), pd.Series(y), ProblemType.CLASSIFICATION)

    # 2) Imbalanced (~10% positive)
    n = 4000; X = r.normal(size=(n, 10))
    logit = X[:, 0] * 1.5 + X[:, 2] - X[:, 4]
    p = 1 / (1 + np.exp(-(logit - 2.2)))
    y = (r.uniform(size=n) < p).astype(int)
    out["imbalanced"] = (pd.DataFrame(X), pd.Series(y), ProblemType.CLASSIFICATION)

    # 3) Mild nonlinear: trees vs linear close
    n = 3000; X = r.normal(size=(n, 6))
    y = ((X[:, 0] * X[:, 1] + X[:, 2] ** 2 - 1 + r.normal(scale=0.8, size=n)) > 0).astype(int)
    out["mild_nonlinear"] = (pd.DataFrame(X), pd.Series(y), ProblemType.CLASSIFICATION)

    # 4) Low signal / noisy: selection is genuinely ambiguous (stress test)
    n = 3500; X = r.normal(size=(n, 12))
    y = (X[:, 0] * 0.5 + r.normal(scale=2.0, size=n) > 0).astype(int)
    out["noisy_lowsignal"] = (pd.DataFrame(X), pd.Series(y), ProblemType.CLASSIFICATION)
    return out


def split(X, y, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_te = int(len(X) * 0.2); n_va = int(len(X) * 0.2)
    te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def champion_for_seed(Xdf, ys, problem_type, seed, artifact_dir):
    X = Xdf.to_numpy(dtype=float); y = ys.to_numpy()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y, seed)
    n_classes = len(np.unique(ytr))

    settings = load_settings()
    registry = build_default_registry(seed)
    service = ModelTrainingService(settings.training, registry, seed)
    specs = [s for s in registry.list_models(problem_type)
             if s.max_train_samples is None or len(Xtr) <= s.max_train_samples]

    results = [service._train_single_model(s, Xtr, ytr, Xva, yva, problem_type,
                                           artifact_dir, has_validation=True, resampled=False)
               for s in specs]
    trained = [r for r in results if r.status == "trained" and r.metrics]
    if not trained:
        return None

    # (A) single split
    single = max(trained, key=lambda r: selection_score(r.metrics, problem_type, n_classes))
    # (B) cv
    cv = service._cv_selection_scores(specs, Xtr, ytr, problem_type, n_classes, seed, resampled=False)
    def cv_score(r):
        c = cv.get(r.model_name)
        return (c[0] - _CV_STD_PENALTY * c[1]) if c else selection_score(r.metrics, problem_type, n_classes)
    cvbest = max(trained, key=cv_score)

    # test-set generalization of each champion (threshold-independent)
    def test_score(r):
        import joblib
        m = joblib.load(r.model_path)
        prob = m.predict_proba(Xte) if hasattr(m, "predict_proba") else None
        preds = m.predict(Xte)
        return selection_score(compute_metrics(yte, preds, problem_type, prob), problem_type, n_classes)

    return {
        "single_name": single.model_name, "single_test": test_score(single),
        "cv_name": cvbest.model_name, "cv_test": test_score(cvbest),
    }


def main():
    print(f"Stability of champion selection over {N_SEEDS} random splits "
          f"(single-split vs {load_settings().training and ''}CV)\n")
    print(f"{'dataset':<16} {'single: modal/stability':<30} {'cv: modal/stability':<30} {'test single->cv':<16}")
    print("-" * 95)
    wins = {"cv_more_stable": 0, "cv_not_worse_test": 0, "total": 0}

    with tempfile.TemporaryDirectory() as td:
        adir = Path(td)
        for name, (Xdf, ys, pt) in datasets().items():
            singles, cvs, st, ct = [], [], [], []
            for seed in range(N_SEEDS):
                res = champion_for_seed(Xdf, ys, pt, seed, adir)
                if not res:
                    continue
                singles.append(res["single_name"]); cvs.append(res["cv_name"])
                st.append(res["single_test"]); ct.append(res["cv_test"])

            def stability(names):
                c = Counter(names); modal, freq = c.most_common(1)[0]
                return modal, freq / len(names), len(c)
            sm, sfreq, sdistinct = stability(singles)
            cm, cfreq, cdistinct = stability(cvs)
            mean_st, mean_ct = float(np.mean(st)), float(np.mean(ct))

            wins["total"] += 1
            if cfreq > sfreq:
                wins["cv_more_stable"] += 1
            if mean_ct >= mean_st - 1e-4:
                wins["cv_not_worse_test"] += 1

            print(f"{name:<16} {sm[:18]:<18} {sfreq*100:>3.0f}% ({sdistinct} distinct)   "
                  f"{cm[:18]:<18} {cfreq*100:>3.0f}% ({cdistinct} distinct)   "
                  f"{mean_st:.4f}->{mean_ct:.4f}")

    print("\n" + "=" * 60)
    print(f"CV more stable on {wins['cv_more_stable']}/{wins['total']} datasets")
    print(f"CV champion's test score not worse on {wins['cv_not_worse_test']}/{wins['total']} datasets")
    verdict = (wins["cv_more_stable"] >= wins["total"] - 1
               and wins["cv_not_worse_test"] >= wins["total"] - 1)
    print(f"\nVERDICT: {'CV selection is the better default — commit it.' if verdict else 'Mixed — do NOT commit; investigate.'}")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
