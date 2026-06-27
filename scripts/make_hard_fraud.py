"""
Generate a deliberately HARD, realistic fraud-detection dataset to stress-test the
pipeline. Stressors (each mirrors a real-world failure mode):

  * Extreme imbalance (~0.8% fraud).
  * Temporal CONCEPT DRIFT — a new "card-testing" fraud tactic (tiny amounts, very
    high velocity) appears only in the most recent period. A model only sees it if
    evaluation respects time; out-of-time test is genuinely harder (the honest truth).
  * Signal lives in INTERACTIONS (high amount × foreign × night; distance × amount),
    not single columns — defeats univariate thinking.
  * Entity/velocity structure (cc_num, prior-activity, velocity_1h).
  * Mixed & dirty types: amount as "$1,234.56" strings; a real datetime; high-card IDs.
  * Informative missingness: unverified email (blank) skews fraud.
  * A LEAKAGE TRAP: `investigation_score` is set post-hoc and ~perfectly predicts the
    label — it must be detected and dropped (it wouldn't exist at prediction time).
  * 12 pure-noise columns + a redundant (correlated) column to test selection/pruning.
  * ~1% label noise.

Usage:  venv/bin/python scripts/make_hard_fraud.py
Writes: hard_fraud_dataset.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N = 90_000
rng = np.random.default_rng(SEED)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    # ── time axis (150 days); last 30% is the "drift" era ────────────────────
    t0 = pd.Timestamp("2024-01-01")
    secs = np.sort(rng.uniform(0, 150 * 24 * 3600, size=N))
    ts = t0 + pd.to_timedelta(secs, unit="s")
    frac = secs / secs.max()
    drift_era = frac > 0.70                      # new tactic only late

    # ── entities ─────────────────────────────────────────────────────────────
    n_cards = 7000
    cc_num = rng.integers(10**15, 10**16, size=n_cards)[rng.integers(0, n_cards, size=N)]
    merchant_id = rng.integers(0, 1800, size=N)

    # ── core features ────────────────────────────────────────────────────────
    amount = np.round(rng.lognormal(mean=4.2, sigma=1.1, size=N), 2)        # ~ $20–2000
    amount_z = (np.log1p(amount) - np.log1p(amount).mean()) / np.log1p(amount).std()
    hour = ts.hour.values
    is_night = ((hour < 6) | (hour >= 23)).astype(int)
    is_foreign = (rng.random(N) < 0.06).astype(int)                          # 6% foreign
    device_type = rng.choice(["mobile", "web", "pos", "atm", "tablet"],
                             p=[0.45, 0.25, 0.18, 0.07, 0.05], size=N)
    merchant_category = rng.choice(
        ["grocery", "fuel", "travel", "electronics", "restaurant", "gaming",
         "jewelry", "pharmacy", "utilities", "crypto", "gift_card", "clothing",
         "rideshare", "charity"], size=N)
    # geo: home base + jitter; foreign txns are far away
    home_lat, home_lon = -34.6, -58.4
    lat = home_lat + rng.normal(0, 0.05, N)
    lon = home_lon + rng.normal(0, 0.05, N)
    merch_lat = lat + rng.normal(0, 0.3, N) + is_foreign * rng.normal(8, 4, N)
    merch_lon = lon + rng.normal(0, 0.3, N) + is_foreign * rng.normal(8, 4, N)
    distance = np.sqrt((merch_lat - lat) ** 2 + (merch_lon - lon) ** 2)

    customer_age = rng.normal(42, 14, N).clip(18, 90)
    account_age_days = rng.integers(1, 3000, N)
    velocity_1h = rng.poisson(1.2, N)            # baseline card activity in last hour
    # informative missingness: ~10% unverified (blank) email — skews fraud
    email_verified = rng.choice(["yes", "no", ""], p=[0.80, 0.10, 0.10], size=N).astype(object)

    # ── card-testing drift ring: late-era, tiny amounts, bursty velocity ─────
    ring = drift_era & (rng.random(N) < 0.012)
    amount = np.where(ring, np.round(rng.uniform(0.5, 5.0, N), 2), amount)
    velocity_1h = np.where(ring, rng.poisson(15, N), velocity_1h)

    # ── fraud label from INTERACTIONS (+ drift pattern) + noise ──────────────
    # Strong, separable interaction signal on a low base rate, so the problem is
    # HARD (signal in conjunctions, invisible to univariate screening) but learnable.
    logit = (
        -6.0
        + 5.0 * ((amount_z > 1.2) & (is_foreign == 1) & (is_night == 1))     # classic ring
        + 4.0 * ((distance > 6) & (amount_z > 0.8))                          # geo × amount
        + 3.0 * (email_verified == "")                                       # missingness
        + 1.6 * (np.isin(merchant_category, ["crypto", "gift_card", "jewelry"]))
        + 5.0 * (ring & (velocity_1h > 8))                                   # DRIFT tactic (late only)
        + rng.normal(0, 0.5, N)                                              # irreducible noise
    )
    p = sigmoid(logit)
    is_fraud = (rng.random(N) < p).astype(int)
    # Realistic but light label noise: flip a tiny fraction (0.1%) — enough to be
    # real-world messy without swamping a 1%-base-rate signal.
    flip = rng.random(N) < 0.001
    is_fraud = np.where(flip, 1 - is_fraud, is_fraud)

    # ── traps & junk ─────────────────────────────────────────────────────────
    # LEAKAGE: post-hoc investigation score (~perfectly predicts label)
    investigation_score = is_fraud * 100 + rng.normal(0, 1.5, N)
    # redundant feature (correlated with amount)
    billed_amount = np.round(amount * (1 + rng.normal(0, 0.01, N)), 2)
    # noise columns
    noise = {f"sensor_{i}": rng.normal(0, 1, N) for i in range(12)}

    df = pd.DataFrame({
        "transaction_id": [f"TX{n:09d}" for n in range(N)],
        "timestamp": ts.astype(str),
        "cc_num": cc_num,
        "merchant_id": merchant_id,
        "amount": [f"${a:,.2f}" for a in amount],         # dirty string $ + commas
        "billed_amount": billed_amount,
        "device_type": device_type,
        "merchant_category": merchant_category,
        "country": np.where(is_foreign == 1, "FOREIGN", "AR"),
        "lat": lat, "long": lon, "merch_lat": merch_lat, "merch_long": merch_lon,
        "customer_age": customer_age,
        "account_age_days": account_age_days,
        "velocity_1h": velocity_1h,
        "email_verified": email_verified,
        "investigation_score": investigation_score,        # LEAK
        **noise,
        "is_fraud": is_fraud,
    })
    # inject the customer_age missingness AFTER (8% missing)
    df.loc[rng.random(N) < 0.08, "customer_age"] = np.nan

    out = Path(__file__).resolve().parents[1] / "hard_fraud_dataset.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  shape={df.shape}")
    print(f"fraud rate: {is_fraud.mean()*100:.3f}%  ({is_fraud.sum()} positives)")
    print(f"drift-era fraud rate: {is_fraud[drift_era].mean()*100:.3f}% vs "
          f"early {is_fraud[~drift_era].mean()*100:.3f}%")
    from sklearn.metrics import roc_auc_score
    print(f"leak column AUC to label: {roc_auc_score(is_fraud, investigation_score):.4f} (should be ~1.0 → must be dropped)")


if __name__ == "__main__":
    main()
