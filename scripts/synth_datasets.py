"""
Synthetic dataset generator for Axiom end-to-end testing.

Produces a wide variety of *adversarial* tabular datasets that exercise every
edge case the upload + pipeline must survive in production:

  - binary / multiclass classification, regression
  - severe class imbalance (fraud-like)
  - heavy missing values, all-NaN columns
  - mixed dtypes, booleans, high-cardinality categoricals
  - datetime columns, unicode text, currency/percent strings
  - constant (zero-variance) columns, duplicate rows, ID/leakage columns
  - tiny datasets, wide datasets, tall datasets
  - messy headers (spaces, symbols, duplicates), numeric-looking strings

Each generator returns ``(DataFrame, target_column_or_None, notes)``.

Usage:
    python scripts/synth_datasets.py            # write all CSVs to data/synth/
    from scripts.synth_datasets import DATASETS  # programmatic access
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

SEED = 7
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "synth"


def _rng(salt: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + salt)


# ── Generators ──────────────────────────────────────────────────────────────


def binary_clean():
    rng = _rng(1)
    n = 1500
    X = rng.normal(size=(n, 8))
    logit = X[:, 0] * 1.2 + X[:, 1] * 0.8 - X[:, 2] * 0.5 + rng.normal(scale=0.4, size=n)
    y = (logit > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
    df["target"] = y
    return df, "target", "clean balanced binary classification"


def binary_imbalanced():
    rng = _rng(2)
    n = 4000
    X = rng.normal(size=(n, 10))
    score = X[:, 0] * 2 + X[:, 3] - X[:, 5]
    p = 1 / (1 + np.exp(-(score - 4)))  # shift to make positives rare
    y = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(10)])
    df["is_fraud"] = y
    return df, "is_fraud", f"imbalanced binary ({y.mean()*100:.1f}% positive)"


def multiclass():
    rng = _rng(3)
    n = 1800
    centers = rng.normal(scale=3, size=(4, 5))
    y = rng.integers(0, 4, size=n)
    X = centers[y] + rng.normal(size=(n, 5))
    df = pd.DataFrame(X, columns=[f"dim_{i}" for i in range(5)])
    df["species"] = pd.Series(y).map({0: "alpha", 1: "beta", 2: "gamma", 3: "delta"})
    return df, "species", "4-class classification with string labels"


def regression_clean():
    rng = _rng(4)
    n = 1500
    X = rng.normal(size=(n, 7))
    y = X @ rng.normal(size=7) + rng.normal(scale=0.5, size=n) + 10
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(7)])
    df["price"] = y
    return df, "price", "clean linear regression"


def regression_skewed():
    rng = _rng(5)
    n = 1600
    X = rng.normal(size=(n, 6))
    base = np.abs(X @ rng.normal(size=6)) + 0.1
    y = np.exp(base) + rng.exponential(scale=2, size=n)  # heavy right skew
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
    df["revenue"] = y
    return df, "revenue", "regression with heavily skewed target"


def wide_many_features():
    rng = _rng(6)
    n, p = 600, 120
    X = rng.normal(size=(n, p))
    signal = X[:, :5] @ rng.normal(size=5)
    y = (signal + rng.normal(scale=0.5, size=n) > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(p)])
    df["label"] = y
    return df, "label", "wide dataset (120 features, few rows)"


def tall_few_features():
    rng = _rng(7)
    n = 80000
    a = rng.normal(size=n)
    b = rng.exponential(size=n)
    y = (a + np.log1p(b) + rng.normal(scale=0.3, size=n) > 0.5).astype(int)
    df = pd.DataFrame({"alpha": a, "beta": b, "gamma": rng.uniform(size=n)})
    df["outcome"] = y
    return df, "outcome", "tall dataset (80k rows, 3 features)"


def missing_heavy():
    rng = _rng(8)
    n = 2000
    X = rng.normal(size=(n, 8))
    df = pd.DataFrame(X, columns=[f"m{i}" for i in range(8)])
    # 30% missing in half the columns; one column almost entirely missing
    for c in df.columns[:4]:
        idx = rng.choice(n, size=int(n * 0.3), replace=False)
        df.loc[idx, c] = np.nan
    df["m0"] = np.nan  # fully empty column
    y = (X[:, 5] + X[:, 6] > 0).astype(int)
    df["target"] = y
    return df, "target", "heavy missing values incl. an all-NaN column"


def mixed_dtypes():
    rng = _rng(9)
    n = 2000
    df = pd.DataFrame({
        "age": rng.integers(18, 90, size=n),
        "income": rng.normal(60000, 20000, size=n).round(2),
        "city": rng.choice(["NYC", "LA", "Chicago", "Houston", "Phoenix"], size=n),
        "is_member": rng.choice([True, False], size=n),
        "score": rng.uniform(0, 100, size=n),
        "grade": rng.choice(list("ABCDEF"), size=n),
    })
    y = ((df["income"] > 60000) & (df["age"] < 50)).astype(int)
    df["approved"] = y
    return df, "approved", "mixed numeric/categorical/bool dtypes"


def datetime_features():
    rng = _rng(10)
    n = 2000
    start = pd.Timestamp("2020-01-01")
    dates = start + pd.to_timedelta(rng.integers(0, 1500, size=n), unit="D")
    df = pd.DataFrame({
        "signup_date": dates.strftime("%Y-%m-%d"),
        "amount": rng.exponential(scale=100, size=n).round(2),
        "channel": rng.choice(["web", "mobile", "store"], size=n),
    })
    y = np.isin(dates.month, [11, 12]).astype(int)  # holiday signal
    df["converted"] = y
    return df, "converted", "datetime string column + features"


def high_cardinality():
    rng = _rng(11)
    n = 3000
    df = pd.DataFrame({
        "user_id": [f"u_{i:06d}" for i in rng.integers(0, 100000, size=n)],
        "zip": rng.integers(10000, 99999, size=n).astype(str),
        "value": rng.normal(size=n),
        "category": rng.choice([f"cat_{i}" for i in range(200)], size=n),
    })
    y = (df["value"] > 0).astype(int)
    df["target"] = y
    return df, "target", "high-cardinality categorical columns"


def unicode_text():
    rng = _rng(12)
    n = 1200
    names = ["café", "naïve", "Zürich", "東京", "Москва", "São Paulo", "🚀rocket", "Ωmega"]
    df = pd.DataFrame({
        "place": rng.choice(names, size=n),
        "measure": rng.normal(size=n),
        "note": rng.choice(["good — ok", "bad, no", "neutral; meh"], size=n),
    })
    y = (df["measure"] > 0).astype(int)
    df["flag"] = y
    return df, "flag", "unicode/emoji text values"


def single_feature():
    rng = _rng(13)
    n = 1000
    x = rng.normal(size=n)
    y = (x > 0).astype(int)
    df = pd.DataFrame({"only_feature": x, "target": y})
    return df, "target", "single predictor column"


def constant_columns():
    rng = _rng(14)
    n = 1500
    df = pd.DataFrame({
        "useful": rng.normal(size=n),
        "constant_int": 5,
        "constant_str": "same",
        "another": rng.normal(size=n),
    })
    y = (df["useful"] + df["another"] > 0).astype(int)
    df["target"] = y
    return df, "target", "zero-variance constant columns present"


def duplicate_rows():
    rng = _rng(15)
    n = 800
    X = rng.normal(size=(n, 5))
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(5)])
    df["target"] = (X[:, 0] > 0).astype(int)
    # duplicate 40% of rows
    dup = df.sample(frac=0.4, random_state=SEED)
    df = pd.concat([df, dup], ignore_index=True)
    return df, "target", "many duplicate rows"


def id_leakage():
    rng = _rng(16)
    n = 1500
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df.insert(0, "row_id", np.arange(n))           # monotonic ID
    df["leak"] = y * 1.0 + rng.normal(scale=1e-6, size=n)  # near-perfect leak
    df["target"] = y
    return df, "target", "ID column + target-leaking feature"


def tiny_dataset():
    rng = _rng(17)
    n = 40
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    return df, "target", "tiny dataset (40 rows)"


def all_categorical():
    rng = _rng(18)
    n = 1500
    df = pd.DataFrame({
        "color": rng.choice(["red", "green", "blue"], size=n),
        "size": rng.choice(["S", "M", "L", "XL"], size=n),
        "shape": rng.choice(["circle", "square", "triangle"], size=n),
    })
    y = (df["color"] == "red").astype(int)
    df["target"] = y
    return df, "target", "no numeric feature columns"


def outliers_negatives():
    rng = _rng(19)
    n = 2000
    X = rng.normal(size=(n, 6))
    X[rng.choice(n, 20, replace=False), 0] = 1e6   # extreme outliers
    X[rng.choice(n, 20, replace=False), 1] = -1e6
    y = (X[:, 2] - X[:, 3] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
    df["target"] = y
    return df, "target", "extreme outliers and negatives"


def messy_headers():
    rng = _rng(20)
    n = 1500
    df = pd.DataFrame({
        "First Name": rng.choice(["a", "b", "c"], size=n),
        "amount ($)": rng.normal(size=n),
        "rate %": rng.uniform(size=n),
        "col.with.dots": rng.normal(size=n),
        "Unnamed: 0": np.arange(n),
    })
    y = (df["amount ($)"] > 0).astype(int)
    df["target?"] = y
    return df, "target?", "messy header names with spaces/symbols"


def numeric_strings():
    rng = _rng(21)
    n = 1500
    df = pd.DataFrame({
        "price_str": ["$" + f"{v:.2f}" for v in rng.uniform(1, 1000, size=n)],
        "pct_str": [f"{v:.1f}%" for v in rng.uniform(0, 100, size=n)],
        "clean_num": rng.normal(size=n),
        "comma_num": [f"{int(v):,}" for v in rng.integers(1000, 9999999, size=n)],
    })
    y = (df["clean_num"] > 0).astype(int)
    df["target"] = y
    return df, "target", "numeric values stored as strings ($, %, commas)"


def no_target_clustering():
    rng = _rng(22)
    n = 1500
    centers = rng.normal(scale=4, size=(3, 5))
    lbl = rng.integers(0, 3, size=n)
    X = centers[lbl] + rng.normal(size=(n, 5))
    df = pd.DataFrame(X, columns=[f"dim{i}" for i in range(5)])
    return df, None, "no target column (unsupervised path)"


GENERATORS: list[Callable] = [
    binary_clean, binary_imbalanced, multiclass, regression_clean,
    regression_skewed, wide_many_features, tall_few_features, missing_heavy,
    mixed_dtypes, datetime_features, high_cardinality, unicode_text,
    single_feature, constant_columns, duplicate_rows, id_leakage,
    tiny_dataset, all_categorical, outliers_negatives, messy_headers,
    numeric_strings, no_target_clustering,
]


def build_all() -> list[tuple[str, pd.DataFrame, Optional[str], str]]:
    """Return [(name, df, target, notes), ...] for every generator."""
    out = []
    for gen in GENERATORS:
        df, target, notes = gen()
        out.append((gen.__name__, df, target, notes))
    return out


def write_all(out_dir: Path = OUT_DIR) -> list[dict]:
    """Write every dataset to ``out_dir`` as CSV and return manifest dicts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, df, target, notes in build_all():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        manifest.append({
            "name": name,
            "path": str(path),
            "rows": len(df),
            "cols": df.shape[1],
            "target": target,
            "notes": notes,
        })
    return manifest


if __name__ == "__main__":
    rows = write_all()
    print(f"Wrote {len(rows)} datasets to {OUT_DIR}\n")
    for r in rows:
        tgt = r["target"] or "(none)"
        print(f"  {r['name']:24s} {r['rows']:>6}x{r['cols']:<4} target={tgt:12s} {r['notes']}")
