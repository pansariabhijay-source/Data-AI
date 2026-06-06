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

### Reports / Export
- **Jupyter notebook (.ipynb) export** (`visualization/notebook_report.py`,
  `GET /api/report/{run_id}/notebook`, "Notebook" button on the report + free
  results pages). Exports a run as a self-contained notebook: the report
  narrative as markdown cells, the generated charts embedded as images, plus
  runnable code cells to load the trained champion (`joblib`) and score new data.
  Built as plain nbformat-v4 JSON (no extra runtime dep); validated with
  `nbformat.validate`. Verified live: 200, valid .ipynb, correct content-type.

### Pipeline / ML
- **Prominent leakage-removal callout in the report**
  (`agents/finalization/tools.py`). When columns that almost perfectly predict
  the target are detected and dropped, the Executive Summary now explains the
  honest-vs-leaked framing up front. Motivated by the fake-internship dataset:
  it ships a `fraud_score` column with **AUC = 1.0** to the label (the label is
  ~`fraud_score > 50`). A Kaggle notebook kept it and reported 99.9% accuracy —
  pure leakage. Our pipeline correctly drops it (AUC-based detector,
  `core/validation.detect_target_leakage`) and reports the honest ~0.91; verified
  XGBoost-without-`fraud_score` = 0.916 acc / 0.801 f1, matching our champion.
  The callout makes clear the lower score is the *real* one, not a regression.
- **Frequency encoding for high-cardinality categoricals**
  (`agents/feature_engineering/tools.py`). High-card columns were *label-encoded*
  (an arbitrary ordinal models misread as magnitude); now each category maps to
  its relative frequency — meaningful and leakage-safe (rare device/merchant =
  predictive in fraud). Proof (`scripts/feature_quality_experiment.py`): on data
  where rarity predicts the target, test AUC **0.870 → 0.955 (+0.085)**.
- **Missing-value indicator features** (`agents/preprocessing/tools.py`,
  `ADD_MISSING_INDICATORS`, default ON). Adds `<col>__was_missing` before
  imputation, for columns with ≥`MISSING_INDICATOR_MIN_FRAC` nulls (default 1%);
  downstream selection prunes uninformative ones. Whether a value is missing is
  often predictive (fraud). Proof: with informative missingness, **linear-model**
  test AUC **0.716 → 0.891 (+0.175)** (trees gain little — they recover some
  signal from the median-imputation spike). Verified end-to-end: the indicator
  was created and selected as a top feature.
- **Process isolation for the full pipeline** (`core/pipeline_worker.py`,
  `PIPELINE_ISOLATION`, default ON). `/api/run` now executes the pipeline in a
  child process (spawn) that streams progress over a `multiprocessing.Queue`; the
  parent thread blocks on the queue (releasing the GIL) and still owns all
  run-state/DB persistence — so status/results endpoints are unchanged. Falls
  back to the in-thread runner (`_run_pipeline_inthread`) if disabled or if spawn
  fails. Verified end-to-end (all 8 stages, status, results, calibration flow).
  **Measured** (probe during training): isolated avg 13ms/max 162ms vs thread
  avg 21ms/max 294ms — a moderate latency win (numpy/sklearn release the GIL), but
  the real wins are **crash isolation** (a training OOM/segfault kills only the
  child, not the API), **memory hygiene** (child frees big frames/models on exit),
  and better tail latency under load. Cost: ~1-2s child spawn per run.
- **Probability calibration of the champion** (`agents/training/tools.py`,
  `_maybe_calibrate_champion`, `CALIBRATE_CHAMPION`, default ON). Tree ensembles
  /boosters produce miscalibrated probabilities, yet the F1-optimal threshold and
  any risk scores sit on top of them. After selection, the champion is calibrated
  on the val split via `CalibratedClassifierCV(FrozenEstimator(model))` (isotonic
  ≥1k rows else sigmoid — no base refit), the threshold is re-derived on
  calibrated probs, and it's **adopted only if the Brier score improves**.
  Calibration is monotonic so AUC/selection is never affected (verified).
  `scripts/calibration_experiment.py`: test Brier improved on 7/9 model-datasets,
  AUC preserved on all. Made `core/ensemble.ProbabilityAveragingEnsemble` a
  first-class sklearn classifier (BaseEstimator/ClassifierMixin + `fit`) so the
  frequent ensemble champion is calibratable too; finalization unwraps the
  calibrated wrapper so SHAP keeps using the fast explainer.
- **Bounded SHAP KernelExplainer cost** (`agents/finalization/tools.py`,
  `SHAP_KERNEL_NSAMPLES`, default 100). Tree/linear champions use the fast exact
  explainers, but ensembles/SVC fall to the model-agnostic KernelExplainer,
  which defaulted to nsamples="auto" (~2·n_features+2048 coalitions/row) ≈ 39s —
  the cause of the ~49s finalization stage. Capping nsamples=100 keeps the same
  top-feature ranking (verified) at **39s → 4.3s (9× faster)**.
- **Mutual-information feature ranking now runs on a row sample**
  (`agents/feature_engineering/tools.py`, `MI_SAMPLE_ROWS`, default 50k).
  Profiling showed `select_k_best` (mutual_info) was **19.5s of ~21.5s** of
  feature_engineering on 220k rows — it's O(n log n) kNN density estimation, used
  only to RANK features. A sample preserves the ranking of the informative
  features (only zero-MI noise columns reshuffle, verified) and selection still
  runs on the full frame. Measured: select_k_best 19.5s→5.1s, FE ~21.5s→~7s.
- **Halved the forest tree counts** (`core/model_registry.py`): RandomForest &
  ExtraTrees defaulted to **300 estimators with no early stopping** — profiling
  showed RandomForest alone was 58.7s of an ~85s model_training stage on 100k
  rows. Cut to 150 (RandomForestRegressor was already 100). Measured: total
  model fitting ~93s→49s, RF 58.7s→29.1s, with AUC change <0.001 (noise) and F1
  unchanged. Boosting models (which usually win) were never the bottleneck.
  Also dropped LogisticRegression's no-op `n_jobs` (silences a per-run sklearn
  FutureWarning).
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
1. **Process isolation for the full pipeline** — DONE (see Done section). Future:
   extend the same child-process pattern to `/api/workflow/run` (enterprise
   custom workflows still run in-thread).
2. **CV-based champion selection** — DONE as an opt-in (see Done section). If a
   future fairer test on boosting-clustered datasets shows broad benefit,
   reconsider flipping `CV_SELECTION_FOLDS` on by default.
3. **Probability calibration of the champion** — DONE (see Done section).

### Tier 2 — scalability & smarter tuning
4. **Gate / cheapen tuning.** Only tune when CV std is high or the top-2 models
   are within noise; otherwise skip. Consider `HalvingRandomSearchCV`
   (successive halving) instead of `RandomizedSearchCV` for the same budget at
   lower cost.
5. **Refit the champion on full data** when subsampling kicked in (currently the
   champion trains on the sample; a final refit on the full train set recovers
   the last bit of accuracy at the cost of one extra fit).

### Tier 3 — feature & data quality
6. **Frequency encoding** for high-cardinality categoricals — DONE (see Done).
7. **Missing-indicator features** before imputation — DONE (see Done).
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
| `VIZ_SAMPLE_ROWS` | 50000 | Rows used to render charts (upload + per-agent) |
| `MI_SAMPLE_ROWS` | 50000 | Rows used for mutual-info feature ranking (0 = off) |
| `SHAP_KERNEL_NSAMPLES` | 100 | Coalition budget for SHAP KernelExplainer (ensembles/SVC) |
| `CALIBRATE_CHAMPION` | 1 | Calibrate champion probabilities (0 = off) |
| `PIPELINE_ISOLATION` | 1 | Run `/api/run` pipeline in a child process (0 = in-thread) |
| `ADD_MISSING_INDICATORS` | 1 | Add `<col>__was_missing` features (0 = off) |
| `MISSING_INDICATOR_MIN_FRAC` | 0.01 | Min null fraction to add an indicator |
| `ARTIFACT_MAX_RUNS` | 25 | Max run dirs kept |
| `ARTIFACT_MIN_KEEP` | 5 | Always-kept newest runs |
| `ARTIFACT_RETENTION_DAYS` | 30 | Age cap for runs/uploads |
| `USE_SMOTE` | off | Train-only minority resampling |

## 🧪 How to verify changes
- `python scripts/diagnose_pipeline.py` — times upload + dumps what every agent
  did on a large dirty dataset.
- `python scripts/e2e_test.py --concurrency 2` — full API/auth/upload/pipeline
  harness across 22 synthetic datasets.
