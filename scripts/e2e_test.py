"""
Axiom end-to-end production test harness.

Exercises *every* API endpoint against a live backend across 20+ adversarial
synthetic datasets, plus auth and security negative tests. Verifies that:

  - auth works and is enforced (no token / bad token / cross-user access)
  - upload is fast, non-blocking, and returns valid payloads on every dataset
  - the full pipeline runs to completion (or fails gracefully, never crashes)
  - status / results / report / shap / excel / pdf / runs all respond correctly
  - bad input is rejected with 4xx (not 5xx), path traversal is blocked

Run the backend first (python api.py), then:
    python scripts/e2e_test.py            # full run incl. pipelines
    python scripts/e2e_test.py --quick    # skip pipeline execution
    python scripts/e2e_test.py --concurrency 4
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.synth_datasets import build_all  # noqa: E402

BASE = "http://127.0.0.1:8000/api"
TIMEOUT = 600


class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def check(self, name: str, ok: bool, detail: str = ""):
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        return ok

    @property
    def passed(self):
        return sum(1 for c in self.checks if c["ok"])

    @property
    def failed(self):
        return sum(1 for c in self.checks if not c["ok"])

    def failures(self):
        return [c for c in self.checks if not c["ok"]]


R = Report()


def _new_user(prefix: str) -> tuple[str, dict]:
    email = f"{prefix}_{int(time.time()*1000)%10_000_000}@axiomtest.io"
    r = requests.post(f"{BASE}/auth/signup",
                      json={"email": email, "username": prefix, "password": "TestPass123!"},
                      timeout=30)
    r.raise_for_status()
    tok = r.json()["token"]
    return email, {"Authorization": f"Bearer {tok}"}


# ── Auth & security ─────────────────────────────────────────────────────────


def test_auth_and_security():
    print("\n=== AUTH & SECURITY ===")
    email, H = _new_user("alice")
    R.check("signup returns token", "Authorization" in H)

    # login
    r = requests.post(f"{BASE}/auth/login", json={"email": email, "password": "TestPass123!"}, timeout=30)
    R.check("login succeeds", r.status_code == 200 and "token" in r.json())

    r = requests.post(f"{BASE}/auth/login", json={"email": email, "password": "wrong"}, timeout=30)
    R.check("login rejects bad password (401)", r.status_code == 401)

    r = requests.post(f"{BASE}/auth/signup",
                      json={"email": email, "username": "x", "password": "y"}, timeout=30)
    R.check("signup rejects duplicate email (400)", r.status_code == 400)

    # /me
    r = requests.get(f"{BASE}/auth/me", headers=H, timeout=30)
    R.check("/auth/me with token (200)", r.status_code == 200)

    r = requests.get(f"{BASE}/auth/me", timeout=30)
    R.check("/auth/me without token rejected (401/403)", r.status_code in (401, 403))

    r = requests.get(f"{BASE}/auth/me", headers={"Authorization": "Bearer totally-bogus"}, timeout=30)
    R.check("/auth/me with bad token (401)", r.status_code == 401)

    # workspace
    r = requests.get(f"{BASE}/workspace", headers=H, timeout=30)
    R.check("/workspace returns prefs", r.status_code == 200)
    r = requests.post(f"{BASE}/workspace/mode", headers=H, json={"mode": "free"}, timeout=30)
    R.check("/workspace/mode set free", r.status_code == 200)
    r = requests.post(f"{BASE}/workspace/mode", headers=H, json={"mode": "bogus"}, timeout=30)
    R.check("/workspace/mode rejects bad mode (400)", r.status_code == 400)

    # path traversal on run
    r = requests.post(f"{BASE}/run", headers=H,
                      data={"data_path": "../../../../etc/passwd", "mode": "free"}, timeout=30)
    R.check("path traversal blocked on /run (400)", r.status_code == 400)
    r = requests.post(f"{BASE}/run", headers=H,
                      data={"data_path": "C:/Windows/System32/drivers/etc/hosts", "mode": "free"}, timeout=30)
    R.check("absolute path outside uploads blocked (400)", r.status_code == 400)

    # unknown run / cross-user access
    r = requests.get(f"{BASE}/status/run_does_not_exist", headers=H, timeout=30)
    R.check("unknown run returns 404", r.status_code == 404)

    # cross-user: bob cannot read alice's run
    return email, H


def test_cross_user_isolation(alice_H, alice_run_id):
    if not alice_run_id:
        return
    _, bob_H = _new_user("bob")
    r = requests.get(f"{BASE}/status/{alice_run_id}", headers=bob_H, timeout=30)
    R.check("cross-user run access blocked (403/404)", r.status_code in (403, 404),
            f"got {r.status_code}")


def test_bad_uploads(H):
    print("\n=== BAD UPLOAD HANDLING (must be 4xx, never 5xx) ===")
    # unsupported extension
    r = requests.post(f"{BASE}/upload", headers=H,
                      files={"file": ("evil.exe", io.BytesIO(b"MZ..."), "application/octet-stream")}, timeout=60)
    R.check("rejects .exe (400)", r.status_code == 400, f"got {r.status_code}")

    # empty file
    r = requests.post(f"{BASE}/upload", headers=H,
                      files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")}, timeout=60)
    R.check("rejects empty csv (4xx)", 400 <= r.status_code < 500, f"got {r.status_code}")

    # garbage that's not a table
    r = requests.post(f"{BASE}/upload", headers=H,
                      files={"file": ("junk.csv", io.BytesIO(b"\x00\x01\x02not,a,valid\x00csv"), "text/csv")}, timeout=60)
    R.check("garbage csv handled (not 5xx)", r.status_code < 500, f"got {r.status_code}")

    # upload without auth
    r = requests.post(f"{BASE}/upload",
                      files={"file": ("x.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")}, timeout=60)
    R.check("upload without auth rejected (401/403)", r.status_code in (401, 403), f"got {r.status_code}")


# ── Per-dataset upload + pipeline ───────────────────────────────────────────


def upload_dataset(H, name, df) -> dict | None:
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)
    csv_mb = len(buf.getvalue()) / 1e6
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/upload", headers=H,
                          files={"file": (f"{name}.csv", buf, "text/csv")}, timeout=TIMEOUT)
    except Exception as e:
        R.check(f"upload[{name}]", False, f"exception {type(e).__name__}: {e}")
        return None
    dt = time.time() - t0
    if r.status_code != 200:
        R.check(f"upload[{name}]", False, f"status {r.status_code}: {r.text[:120]}")
        return None
    j = r.json()
    ok = (j.get("n_rows") == len(df)
          and isinstance(j.get("columns"), list) and len(j["columns"]) == df.shape[1]
          and isinstance(j.get("preview"), list))
    R.check(f"upload[{name}]", ok,
            f"{dt:.1f}s csv={csv_mb:.1f}MB rows={j.get('n_rows')} viz={len(j.get('visualizations', []))}")
    return j if ok else None


def _poll_status(run_id, H, name):
    """Poll /status until terminal, tolerating transient timeouts.

    Under heavy concurrent CPU load the API event loop can be briefly starved,
    so a single slow poll must not abort the whole run — we retry with backoff.
    """
    consecutive_errors = 0
    while True:
        try:
            s = requests.get(f"{BASE}/status/{run_id}", headers=H, timeout=60)
            consecutive_errors = 0
        except requests.exceptions.RequestException:
            consecutive_errors += 1
            if consecutive_errors >= 6:
                return "error", f"status unreachable after {consecutive_errors} retries"
            time.sleep(3)
            continue
        if s.status_code != 200:
            return "error", f"status poll HTTP {s.status_code}"
        status = s.json()["status"]
        if status in ("completed", "failed"):
            return status, ""
        time.sleep(2)


def run_pipeline(H, name, data_path, target) -> tuple[str, dict]:
    """Start a pipeline and poll to completion. Returns (run_id, results)."""
    data = {"data_path": data_path, "mode": "free"}
    if target:
        data["target_column"] = target
    r = requests.post(f"{BASE}/run", headers=H, data=data, timeout=60)
    if r.status_code != 200:
        R.check(f"start[{name}]", False, f"status {r.status_code}: {r.text[:120]}")
        return "", {}
    run_id = r.json()["run_id"]

    t0 = time.time()
    status, err = "running", ""
    while time.time() - t0 < TIMEOUT:
        status, err = _poll_status(run_id, H, name)
        if status in ("completed", "failed", "error"):
            break
    dur = time.time() - t0

    if status == "error":
        R.check(f"pipeline[{name}]", False, f"poll error after {dur:.0f}s: {err}")
        return run_id, {}

    try:
        res = requests.get(f"{BASE}/results/{run_id}", headers=H, timeout=60).json()
    except requests.exceptions.RequestException as e:
        R.check(f"pipeline[{name}]", False, f"results fetch failed: {e}")
        return run_id, {}
    if status == "completed":
        R.check(f"pipeline[{name}]", True,
                f"{dur:.0f}s best={res.get('best_model')} "
                f"{res.get('best_metric_name')}={res.get('best_metric_value')}")
    else:
        # A graceful failure is still recorded; the harness flags it for review
        # but it is not a server crash. We surface the reason.
        R.check(f"pipeline[{name}]", False,
                f"FAILED in {dur:.0f}s: {str(res.get('error'))[:160]}")
    return run_id, res


def test_post_run_endpoints(H, name, run_id):
    if not run_id:
        return
    # report (may legitimately 404 if generation skipped) — must not 5xx
    r = requests.get(f"{BASE}/report/{run_id}", headers=H, timeout=30)
    R.check(f"report[{name}] no-5xx", r.status_code < 500, f"got {r.status_code}")
    # shap (often 404 for linear models) — must not 5xx
    r = requests.get(f"{BASE}/shap/{run_id}", headers=H, timeout=30)
    R.check(f"shap[{name}] no-5xx", r.status_code < 500, f"got {r.status_code}")
    # visualizations
    r = requests.get(f"{BASE}/visualizations/{run_id}", headers=H, timeout=30)
    R.check(f"visualizations[{name}]", r.status_code == 200, f"got {r.status_code}")
    # pdf export
    r = requests.get(f"{BASE}/report/{run_id}/pdf", headers=H, timeout=120)
    R.check(f"pdf[{name}] no-5xx", r.status_code < 500,
            f"{r.status_code} {len(r.content) if r.ok else ''}")


def test_concurrency_responsiveness(H, df):
    """Fire several uploads at once; health must stay responsive throughout."""
    print("\n=== CONCURRENCY / NON-BLOCKING ===")
    buf = io.BytesIO(); df.to_csv(buf, index=False); payload = buf.getvalue()

    latencies = []

    def probe():
        for _ in range(40):
            t = time.time()
            try:
                requests.get(f"{BASE}/health", timeout=30)
                latencies.append(time.time() - t)
            except Exception:
                latencies.append(99.0)
            time.sleep(0.2)

    import threading
    th = threading.Thread(target=probe, daemon=True); th.start()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(requests.post, f"{BASE}/upload", headers=H,
                          files={"file": (f"conc{i}.csv", io.BytesIO(payload), "text/csv")}, timeout=TIMEOUT)
                for i in range(4)]
        codes = [f.result().status_code for f in as_completed(futs)]
    th.join()
    max_lat = max(latencies) if latencies else 0
    R.check("all concurrent uploads 200", all(c == 200 for c in codes), f"codes={codes}")
    R.check("health stays responsive under load (<1.5s)", max_lat < 1.5,
            f"max health latency {max_lat*1000:.0f}ms")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip pipeline execution")
    ap.add_argument("--concurrency", type=int, default=2, help="parallel pipelines")
    args = ap.parse_args()

    # backend liveness
    try:
        h = requests.get(f"{BASE}/health", timeout=10)
        assert h.status_code == 200
    except Exception as e:
        print(f"Backend not reachable at {BASE}/health: {e}")
        print("Start it with:  venv/Scripts/python.exe api.py")
        sys.exit(2)

    print("Backend healthy:", h.json())

    alice_email, H = test_auth_and_security()
    test_bad_uploads(H)

    datasets = build_all()
    print(f"\n=== UPLOADS ({len(datasets)} datasets) ===")
    uploaded = {}
    for name, df, target, notes in datasets:
        j = upload_dataset(H, name, df)
        if j:
            uploaded[name] = (j["path"], target, df)

    # concurrency test using a mid-size dataset
    mid = next((df for n, df, t, _ in datasets if n == "mixed_dtypes"), datasets[0][1])
    test_concurrency_responsiveness(H, mid)

    first_run_id = ""
    if not args.quick:
        print(f"\n=== PIPELINES (concurrency={args.concurrency}) ===")
        results = {}
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(run_pipeline, H, name, path, target): name
                    for name, (path, target, _) in uploaded.items()}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    run_id, res = fut.result()
                    results[name] = run_id
                    if not first_run_id and run_id:
                        first_run_id = run_id
                except Exception as e:
                    R.check(f"pipeline[{name}]", False, f"harness exception: {e}")

        print("\n=== POST-RUN ENDPOINTS ===")
        for name, run_id in results.items():
            test_post_run_endpoints(H, name, run_id)

        # runs listing
        r = requests.get(f"{BASE}/runs", headers=H, timeout=30)
        R.check("/runs lists user runs", r.status_code == 200 and len(r.json().get("runs", [])) >= len(uploaded),
                f"{len(r.json().get('runs', []))} runs")

    test_cross_user_isolation(H, first_run_id)

    # ── summary ──
    print("\n" + "=" * 60)
    print(f"TOTAL: {R.passed} passed, {R.failed} failed")
    if R.failed:
        print("\nFAILURES:")
        for c in R.failures():
            print(f"  - {c['name']}: {c['detail']}")
    Path("e2e_report.json").write_text(json.dumps(R.checks, indent=2), encoding="utf-8")
    print(f"\nFull report written to e2e_report.json")
    sys.exit(1 if R.failed else 0)


if __name__ == "__main__":
    main()
