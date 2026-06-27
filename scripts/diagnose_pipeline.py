"""
Deep diagnostic: time the upload and inspect what EVERY agent actually does on a
large, deliberately dirty dataset. Prints per-agent duration + the concrete work
each performed (rows cleaned, dupes removed, features engineered, models trained)
so we can see whether the first stages are real or no-ops.

Run the backend first, then: python scripts/diagnose_pipeline.py
"""
from __future__ import annotations

import io
import time
import random
import json
import numpy as np
import pandas as pd
import requests

BASE = "http://127.0.0.1:8000/api"


def make_dirty(n=200_000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "amount": rng.exponential(120, n).round(2),
        "balance": rng.normal(5000, 2500, n).round(2),
        "score": rng.normal(size=n),
        "age": rng.integers(18, 90, n),
        "city": rng.choice(["NYC", "LA", "Chicago", "Houston", "Phoenix", "Boston"], n),
        "device": rng.choice([f"dev_{i}" for i in range(500)], n),   # high cardinality
        "signup": (pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 1200, n), "D")).strftime("%Y-%m-%d"),
        "constant": 1,                                                # zero variance
        "txn_id": np.arange(n),                                       # ID col
    })
    # signal + label
    logit = 0.000004 * df["amount"] + 0.0002 * df["balance"] / 1000 + df["score"] - (df["age"] > 60).astype(int)
    df["is_fraud"] = (logit + rng.normal(scale=0.5, size=n) > logit.mean() + 1.0).astype(int)
    # inject dirtiness
    for c in ["balance", "score", "age", "city"]:
        idx = rng.choice(n, size=n // 12, replace=False)
        df.loc[idx, c] = np.nan
    df.loc[rng.choice(n, 30, replace=False), "amount"] = 1e7          # extreme outliers
    dup = df.sample(frac=0.10, random_state=1)                        # 10% dup rows
    df = pd.concat([df, dup], ignore_index=True)
    return df


def main():
    tok = requests.post(f"{BASE}/auth/signup", json={
        "email": f"diag{random.randint(0, 999999)}@t.com", "username": "d", "password": "pw123456"
    }).json()["token"]
    H = {"Authorization": f"Bearer {tok}", "Origin": "http://localhost:3000"}

    df = make_dirty()
    buf = io.BytesIO(); df.to_csv(buf, index=False); buf.seek(0)
    mb = len(buf.getvalue()) / 1e6
    print(f"Dataset: {len(df):,} rows x {df.shape[1]} cols, {mb:.0f} MB "
          f"(missing, dups, outliers, datetime, high-card, const, id)\n")

    # ── 1) UPLOAD TIMING ──
    t0 = time.time()
    up = requests.post(f"{BASE}/upload", headers=H, files={"file": ("dirty.csv", buf, "text/csv")}, timeout=900)
    up_dt = time.time() - t0
    if not up.ok:
        print("UPLOAD FAILED:", up.status_code, up.text[:200]); return
    upj = up.json()
    print(f"── UPLOAD: {up_dt:.1f}s | rows={upj['n_rows']:,} cols={len(upj['columns'])} "
          f"charts={len(upj['visualizations'])} resp={len(up.content)/1e6:.2f}MB\n")

    # ── 2) RUN PIPELINE ──
    rid = requests.post(f"{BASE}/run", headers=H, data={
        "data_path": upj["path"], "target_column": "is_fraud", "mode": "free"
    }).json()["run_id"]
    print(f"── PIPELINE {rid} ──")

    t0 = time.time()
    seen = set()
    while time.time() - t0 < 900:
        s = requests.get(f"{BASE}/status/{rid}", headers=H, timeout=60).json()
        for name, out in (s.get("agent_outputs") or {}).items():
            key = (name, out.get("status"))
            if key not in seen and out.get("status") in ("completed", "failed"):
                seen.add(key)
                dur = out.get("duration_seconds")
                print(f"  [{time.time()-t0:6.1f}s] {name:20s} {out.get('status'):9s} {dur if dur is not None else '?':>6}s")
        if s["status"] in ("completed", "failed"):
            break
        time.sleep(1.5)
    total = time.time() - t0
    print(f"  TOTAL pipeline wall time: {total:.1f}s\n")

    # ── 3) PER-AGENT DETAIL: what did each actually DO? ──
    s = requests.get(f"{BASE}/status/{rid}", headers=H, timeout=60).json()
    outs = s.get("agent_outputs") or {}
    print("── WHAT EACH AGENT DID (summary/metrics) ──")
    order = ["data_collection", "preprocessing", "feature_engineering", "data_splitting",
             "model_training", "error_detection", "improvement", "finalization"]
    for name in order:
        o = outs.get(name)
        if not o:
            print(f"  {name:20s} — MISSING (did not run)")
            continue
        summary = o.get("summary") or {}
        metrics = o.get("metrics") or {}
        dur = o.get("duration_seconds")
        compact = {k: v for k, v in {**summary, **metrics}.items()
                   if not isinstance(v, (list, dict)) or (isinstance(v, list) and len(v) <= 6)}
        # truncate
        line = json.dumps(compact, default=str)
        if len(line) > 280:
            line = line[:280] + "…"
        print(f"\n  ● {name}  ({o.get('status')}, {dur}s)")
        print(f"    {line}")

    # ── 4) FINAL RESULT ──
    res = requests.get(f"{BASE}/results/{rid}", headers=H, timeout=60).json()
    print("\n── RESULT ──")
    print(f"  problem_type = {res.get('problem_type')}")
    print(f"  best_model   = {res.get('best_model')}  {res.get('best_metric_name')}={res.get('best_metric_value')}")
    print(f"  rows {res.get('preprocessing',{}).get('rows_before')} -> {res.get('preprocessing',{}).get('rows_after')}"
          f"  dupes_removed={res.get('preprocessing',{}).get('duplicates_removed')}")
    print(f"  features {res.get('features',{}).get('before')} -> {res.get('features',{}).get('after')}")
    print(f"  models trained = {len([m for m in res.get('models',[]) if m.get('status')=='trained'])}")


if __name__ == "__main__":
    main()
