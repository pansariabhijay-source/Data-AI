"""Generate sample datasets for testing the pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def generate_classification_dataset(path: str, n_samples: int = 500) -> None:
    """Generate a synthetic classification dataset."""
    rng = np.random.RandomState(42)
    n = n_samples
    df = pd.DataFrame({
        "sepal_length": rng.normal(5.8, 0.8, n),
        "sepal_width": rng.normal(3.0, 0.4, n),
        "petal_length": rng.normal(3.7, 1.7, n),
        "petal_width": rng.normal(1.2, 0.8, n),
        "color": rng.choice(["red", "blue", "green"], n),
        "season": rng.choice(["spring", "summer", "fall", "winter"], n),
        "is_wild": rng.choice([True, False], n),
    })
    # Target based on features with some noise
    score = df["petal_length"] * 0.5 + df["petal_width"] * 0.3 + rng.normal(0, 0.3, n)
    df["species"] = pd.cut(score, bins=3, labels=["setosa", "versicolor", "virginica"])
    # Add some nulls
    null_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    df.loc[null_idx, "sepal_width"] = np.nan
    null_idx2 = rng.choice(n, size=int(n * 0.03), replace=False)
    df.loc[null_idx2, "color"] = np.nan
    # Add some duplicates
    dup_idx = rng.choice(n, size=10, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"Classification dataset saved to {path} ({len(df)} rows)")


def generate_regression_dataset(path: str, n_samples: int = 500) -> None:
    """Generate a synthetic regression dataset."""
    rng = np.random.RandomState(42)
    n = n_samples
    df = pd.DataFrame({
        "sqft": rng.normal(1500, 500, n).clip(300, 5000),
        "bedrooms": rng.choice([1, 2, 3, 4, 5], n),
        "bathrooms": rng.choice([1, 1.5, 2, 2.5, 3], n),
        "year_built": rng.randint(1950, 2024, n),
        "neighborhood": rng.choice(["downtown", "suburb", "rural", "industrial"], n),
        "has_garage": rng.choice([True, False], n),
    })
    df["price"] = (
        df["sqft"] * 150
        + df["bedrooms"] * 20000
        + df["bathrooms"] * 15000
        + (2024 - df["year_built"]) * -500
        + rng.normal(0, 30000, n)
    ).clip(50000)
    null_idx = rng.choice(n, size=int(n * 0.04), replace=False)
    df.loc[null_idx, "year_built"] = np.nan
    df.to_csv(path, index=False)
    print(f"Regression dataset saved to {path} ({len(df)} rows)")


if __name__ == "__main__":
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    generate_classification_dataset(str(data_dir / "sample_classification.csv"))
    generate_regression_dataset(str(data_dir / "sample_regression.csv"))
    print("Done!")
