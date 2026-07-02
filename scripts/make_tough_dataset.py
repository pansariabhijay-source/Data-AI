"""Generate a deliberately BRUTAL synthetic fraud dataset to stress-test the pipeline.

Every property here targets a specific failure mode the recent fixes address:

* **Extreme imbalance** (~1% fraud)                → PR-AUC / threshold / SMOTE.
* **Recurring entities** (cards repeat many times) → group-aware split (#4); a naive
  split lets the model memorise a card's identity across train/test.
* **Entity-level confounding** (some cards are "compromised") → without group-aware
  splitting the model cheats on card identity instead of learning behaviour.
* **Concept drift** — a NEW fraud pattern (crypto merchants) appears ONLY in the last
  20% of time → a random split leaks the future; out-of-time (#2 CV, splitting) is honest.
* **Signal hidden in interactions** (big amount AT an odd hour) → univariate MI misses it.
* **Per-entity behavioural signal** (spend spike vs the card's OWN history) → needs
  time-safe expanding aggregates (#5); whole-dataset stats would leak the future.
* **Two leakage traps**:
    - `investigation_score` (numeric)  → single-feature AUC ~1.0  (numeric leak).
    - `case_status`        (string)    → near-perfect categorical leak (#3 — the old
      detector coerced it to NaN and missed it entirely).
* **Messy real-world data**: log-skewed amounts, lat/long (geo distance), a DOB (age),
  a unit-laden text column ("12.3 km"), informative missingness, and pure-noise columns.

Deterministic (seeded). Writes ``tough_fraud_dataset.csv`` to the repo root.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def make(n_cards: int = 1500, seed: int = 20240701) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    horizon_days = 365
    merchant_cats = ["grocery", "gas", "retail", "travel", "dining", "online"]

    for card_id in range(n_cards):
        n_tx = int(rng.integers(12, 45))
        # Each card has its own typical spend and home location.
        home_amt = float(rng.lognormal(mean=3.4, sigma=0.5))     # ~ $30 typical
        home_lat = float(rng.uniform(25, 49))
        home_lon = float(rng.uniform(-124, -67))
        compromised = rng.random() < 0.06                        # entity-level confound
        dob_year = int(rng.integers(1945, 2003))
        start_day = float(rng.uniform(0, horizon_days * 0.8))
        t = start_day
        for _ in range(n_tx):
            t += float(rng.exponential(3.0))                     # days between tx
            t = min(t, horizon_days)
            frac_time = t / horizon_days
            hour = int(rng.integers(0, 24))

            # Amount: usually near the card's norm; compromised cards spike more often.
            spike = rng.random() < (0.18 if compromised else 0.05)
            amt = home_amt * (rng.lognormal(1.6, 0.4) if spike else rng.lognormal(0.0, 0.35))
            amt = float(min(amt, 8000.0))

            merch_cat = rng.choice(merchant_cats)
            # Merchant location: usually near home, sometimes far (card-not-present etc).
            far = rng.random() < 0.12
            mlat = home_lat + (rng.normal(0, 8) if far else rng.normal(0, 0.5))
            mlon = home_lon + (rng.normal(0, 12) if far else rng.normal(0, 0.6))
            dist = float(_haversine_km(home_lat, home_lon, mlat, mlon))

            # NEW drift pattern: crypto merchants only exist in the last 20% of time.
            crypto = (frac_time > 0.8) and (rng.random() < 0.05)
            if crypto:
                merch_cat = "crypto"

            # --- genuine, transaction-level fraud signal (generalises across cards) ---
            amt_ratio = amt / max(home_amt, 1.0)
            logit = -6.4
            logit += 2.6 * (amt_ratio > 3.0)                     # spend spike vs OWN history
            logit += 2.2 * ((hour <= 4) and (amt > 150))          # interaction: odd hour + big
            logit += 1.6 * (dist > 200)                           # far from home
            logit += 3.0 * crypto                                 # drift pattern (late-only)
            logit += 0.8 * compromised                            # entity-level nudge
            p = _sigmoid(logit)
            is_fraud = int(rng.random() < p)

            rows.append({
                "cc_num": f"card_{card_id:05d}",
                "trans_time": t,                 # numeric time axis (days)
                "hour": hour,
                "amt": round(amt, 2),
                "merchant_category": merch_cat,
                "lat": round(home_lat, 4),
                "long": round(home_lon, 4),
                "merch_lat": round(mlat, 4),
                "merch_long": round(mlon, 4),
                "dob": f"{dob_year}-0{rng.integers(1,9)}-15",
                "distance_str": f"{dist:.1f} km",   # unit-laden text
                "is_fraud": is_fraud,
            })

    df = pd.DataFrame(rows)
    n = len(df)
    fr = df["is_fraud"].values

    # ---- Leakage trap #1: numeric near-perfect predictor (post-hoc score) ----
    df["investigation_score"] = np.where(
        fr == 1, rng.uniform(0.90, 1.0, n), rng.uniform(0.0, 0.08, n)
    ).round(4)

    # ---- Leakage trap #2: string categorical near-perfect predictor ----
    df["case_status"] = np.where(
        fr == 1, "confirmed_fraud",
        rng.choice(["cleared", "closed", "auto_cleared"], n),
    )

    # ---- Informative missingness: a risk field is null more often for fraud ----
    risk = rng.uniform(0, 1, n).round(3).astype(object)
    miss_mask = (rng.random(n) < np.where(fr == 1, 0.55, 0.12))
    risk[miss_mask] = np.nan
    df["merchant_risk"] = risk

    # ---- Pure noise columns (distractors) ----
    for i in range(12):
        df[f"noise_{i}"] = rng.normal(size=n).round(4)
    df["noise_cat"] = rng.choice([f"v{k}" for k in range(8)], n)

    # Shuffle rows so the file isn't pre-sorted by time/card (the pipeline must recover it).
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = make()
    out = "tough_fraud_dataset.csv"
    df.to_csv(out, index=False)
    rate = df["is_fraud"].mean()
    print(f"Wrote {out}: {len(df):,} rows x {df.shape[1]} cols")
    print(f"Fraud rate: {rate*100:.2f}%  ({int(df['is_fraud'].sum()):,} positives)")
    print(f"Distinct cards: {df['cc_num'].nunique():,}")
    late = df["merchant_category"].eq("crypto")
    print(f"Drift pattern 'crypto' rows: {int(late.sum())} "
          f"(fraud among them: {df.loc[late,'is_fraud'].mean()*100:.0f}%)")
