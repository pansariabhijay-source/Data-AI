# Axiom — Current Work & TODO

A living log of what's been done and what's next. Keep it updated as work lands.
Newest changes at the top of each section.

---

## ✅ Done (recent)

### Visualizations
- **Fixed a latent pairplot/PCA bug** (`visualization/engine.py`): they used
  positional `.iloc[label_index]` to attach the target, which goes
  out-of-bounds on any non-contiguous index — so on-demand pairplot/PCA *failed*
  for any dataset >50k rows (the on-demand endpoint samples to 50k). Now uses
  label-based `.loc`. Verified: both render on a 70k run.
- **Per-agent viz now renders on a bounded sample** (`core/agent_runner.py`,
  `VIZ_SAMPLE_ROWS`) so data_collection's chart generation stays flat regardless
  of dataset size (was re-rendering the free charts on the full frame).

### Pipeline / ML
- **CV-based champion selection — implemented as an OPT-IN** (`agents/training/tools.py`,
  `_cv_selection_scores`, off by default: `CV_SELECTION_FOLDS=0`). Ranks base
  models by k-fold CV mean − std-penalty instead of a single noisy val split.
  **Evidence** (`scripts/cv_selection_experiment.py`, 10 splits × 4 datasets):
  it only changes the champion when selection is genuinely ambiguous
  (noisy/low-signal — single-split was 90% stable / 2 champions, CV 100% / 1, and
  CV generalised slightly better on test); on separable problems it's a no-op,
  and it never hurt test score. Because it adds per-run fitting cost and only the
  ambiguous minority benefits, it's **off by default** — enable with
  `CV_SELECTION_FOLDS=4` when you value selection stability over speed (e.g.
  fraud with clustered boosting models).
- **Train on a stratified subsample of large data** (`agents/training/tools.py`,
  `_subsample_for_training`). Candidate models fit on at most `TRAIN_SAMPLE_ROWS`
  rows (default 100k), preserving class ratios and keeping ≥2k rows/class so
  minority (fraud) signal survives. Champion is still scored on the **full** val
  split and tested on the **full** untouched test set. Cuts the dominant cost
  (model_training was ~100s on 200k rows). Disable with `TRAIN_SAMPLE_ROWS=0`.
- **Improvement agent crash fixed** — it died with *"Model 'VotingEnsemble' not
  registered"* whenever the champion was an ensemble. Now skips tuning
  gracefully (ensembles have no single hyperparameters to search) and promotes a
  tuned base model only when it actually beats the champion.
- **Cut the fruitless tuning budget** (`core/constants.py`, `configs/default.yaml`):
  iterations 50→15, max rounds 5→3, patience 2→1. Tuning rarely beats defaults
  per our own benchmarks; this took the pipeline 175s→116s on a 16.5k set.

### Uploads / reliability
- **Direct-to-backend uploads** bypass the Next.js dev proxy that truncated
  >300 MB bodies and reset the connection (the "fails to connect" error). Cap
  raised to 1 GB; backend streams to disk in chunks; clean **507** on disk-full.
- **Artifact-retention policy** (`core/maintenance.py`): keep newest 5, cap at 25
  runs, delete runs/uploads older than 30 days; runs at startup + after every
  run. Stops `artifacts/` from filling the disk. Env: `ARTIFACT_MAX_RUNS`,
  `ARTIFACT_MIN_KEEP`, `ARTIFACT_RETENTION_DAYS`.
- Event-loop no longer blocks during upload (profiling/viz moved to a
  threadpool; viz on a 50k sample; matplotlib warmed at startup).

### Reports
- **In-app reports** render via `react-markdown` + `remark-gfm`
  (`frontend/src/components/ReportMarkdown.tsx`) with premium themed tables —
  replaced a hand-rolled parser.
- **PDF appendix** now renders markdown tables as real ReportLab tables
  (`visualization/pdf_report.py`) instead of raw `| pipe |` text.

---

## 🔜 TODO (prioritized)

### Tier 1 — highest leverage
1. **Process isolation for training/tuning.** Training runs in in-process
   `ThreadPoolExecutor` threads that share the GIL with the API, so a running
   pipeline starves uploads/status (the root cause behind "upload takes ages
   while a model trains"). Move heavy fitting to a `ProcessPoolExecutor` /
   subprocess worker. Biggest reliability win; medium-large change.
2. **CV-based champion selection** — DONE as an opt-in (see Done section). If a
   future fairer test on boosting-clustered datasets shows broad benefit,
   reconsider flipping `CV_SELECTION_FOLDS` on by default.
3. **Probability calibration of the champion.** The F1-optimal threshold and any
   probability outputs sit on *uncalibrated* scores. Wrap the champion in
   `CalibratedClassifierCV` (binary classification), re-derive the threshold on
   calibrated probs, adopt only if it doesn't hurt val score. Real PR-AUC gains
   for imbalanced/fraud; keep it gated and fast.

### Tier 2 — scalability & smarter tuning
4. **Gate / cheapen tuning.** Only tune when CV std is high or the top-2 models
   are within noise; otherwise skip. Consider `HalvingRandomSearchCV`
   (successive halving) instead of `RandomizedSearchCV` for the same budget at
   lower cost.
5. **Refit the champion on full data** when subsampling kicked in (currently the
   champion trains on the sample; a final refit on the full train set recovers
   the last bit of accuracy at the cost of one extra fit).

### Tier 3 — feature & data quality
6. **Target/frequency encoding** for high-cardinality categoricals (we have
   500-value columns; one-hot/label loses signal there).
7. **Missing-indicator features** before imputation (missingness is predictive).
8. **Richer datetime decomposition** (cyclical hour/day, is_weekend, recency).
9. **Train/test drift checks** (PSI) in `error_detection`.
10. **Persist preprocessing + FE as one `sklearn.Pipeline`** for reproducible
    inference / no train-serve skew.

### Tier 4 — observability / UX
11. **SSE / WebSocket** live logs instead of status polling.
12. **Surface each agent's "what I did" summary** prominently in the UI (would
    have pre-empted the "the agents do nothing" confusion).

---

## ⚙️ Config / env knobs (quick reference)
| Env var | Default | Effect |
|---|---|---|
| `TRAIN_SAMPLE_ROWS` | 100000 | Cap rows used to fit candidate models (0 = off) |
| `CV_SELECTION_FOLDS` | 0 | k-fold CV champion selection (0 = off; 4 enables) |
| `CV_SELECTION_MAX_ROWS` | 40000 | Row cap for CV ranking sample |
| `MAX_UPLOAD_MB` | 1024 | Upload size cap |
| `VIZ_SAMPLE_ROWS` | 50000 | Rows used to render upload charts |
| `ARTIFACT_MAX_RUNS` | 25 | Max run dirs kept |
| `ARTIFACT_MIN_KEEP` | 5 | Always-kept newest runs |
| `ARTIFACT_RETENTION_DAYS` | 30 | Age cap for runs/uploads |
| `USE_SMOTE` | off | Train-only minority resampling |

## 🧪 How to verify changes
- `python scripts/diagnose_pipeline.py` — times upload + dumps what every agent
  did on a large dirty dataset.
- `python scripts/e2e_test.py --concurrency 2` — full API/auth/upload/pipeline
  harness across 22 synthetic datasets.
