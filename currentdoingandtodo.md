# Axiom — Current Work & TODO

A living log of what's been done and what's next. Keep it updated as work lands.
Newest changes at the top of each section.

---

## 📖 Session context / handoff (read this first)

This is the full story of the work session, so anyone (or a fresh chat) can pick
up with complete context. Everything below is **committed and pushed to GitHub
`main`** (latest commit `e7259dc`).

> **NOTE (newest work):** after the backend session narrated below, a full
> **frontend design-system overhaul** was done — see the **"🎨 Frontend"** section
> just below this handoff. The backend narrative (commits up to `739778d`) is
> unchanged and still accurate.

### How it started — the reported bug
The user reported: *"whenever I upload a dataset it takes ages and then fails to
connect to the backend"* and suspected an auth problem. **It was not auth.**

**Root cause #1 (event-loop block):** `/api/upload` rendered 5 matplotlib charts
on the **full** dataset *synchronously on the asyncio event loop*, freezing the
whole server during every upload. Proven: a concurrent request was blocked
**6.6 s** before the fix → **73 ms** after. Fix: moved all heavy work to a
threadpool, stream the upload to disk in chunks, render charts on a 50k sample,
warm up matplotlib at startup, return 4xx (not 500) for bad files, namespace
uploads per-user, and `generate_run_id()` now has a random suffix (it was
colliding under concurrency). Added `GET /api/health`. Built a full test harness
(`scripts/e2e_test.py` + `scripts/synth_datasets.py`, 22 datasets) → **153/153
green**.

**Root cause #2 (surfaced when user retested):** uploading a **>300 MB** file —
the Next.js dev proxy buffers the body and resets the connection past its cap
(`ECONNRESET`), which the UI mislabeled "backend not connected." Fix: uploads now
go **directly to the backend** (bypass the proxy), cap raised to 1 GB, graceful
**507** on disk-full. Also discovered the machine's **C: drive was full** — a
recurring source of fake "failures" (ENOSPC); led to the artifact-retention
policy below.

### The big themes we worked through (in order)
1. **Upload reliability** (above) + report rendering made premium (react-markdown
   in-app; PDF tables fixed).
2. **"The agents do nothing / pipeline runs in 5s"** — *disproven.* A diagnostic
   (`scripts/diagnose_pipeline.py`) showed every agent does real work
   (preprocessing removed 20k dup rows, FE encoded + dropped a constant col,
   training fit 7 models, etc.). Found + fixed a real bug: the **improvement
   agent crashed** when the champion was an ensemble ("VotingEnsemble not
   registered"). Cut the fruitless tuning budget.
3. **Artifact-retention policy** so runs don't fill the disk.
4. **Performance arc (≈2× faster pipeline)** — profile → prove quality-neutral →
   commit, each step: train on a 100k stratified subsample, forest trees 300→150,
   mutual-info on a 50k sample, SHAP KernelExplainer bounded, viz `.iloc` bug fix.
5. **Quality features** — probability calibration (default on, proven), CV-based
   champion selection (proven *not* worth defaulting on → opt-in), frequency
   encoding for high-cardinality cats, missing-value indicators.
6. **Reliability** — process isolation (pipeline runs in a child process).
7. **The leakage investigation** (see below) — the headline ML finding.
8. **Notebook exports** — a report `.ipynb` and a *true reproduction* `.ipynb`.
9. **Reports page** redesigned to show dataset + date/time (+ fixed a gitignore
   bug that had stopped the whole reports route from being tracked).
10. **Cleanup** — deleted build caches/logs/debris and the 651 MB `.next` cache.

### KEY ML FINDING — the fake-internship "accuracy gap" was data leakage
The user compared our report (champion `LogisticRegression_tuned`, **F1 ≈ 0.81 /
acc 0.91**) to a Kaggle notebook claiming **99.9% accuracy** and asked why we
didn't match. **Answer: the notebook leaks.** The dataset ships a `fraud_score`
column with **ROC-AUC = 1.0** to the label (the label is ~`fraud_score > 50`) —
it *is* the answer. The notebook did `X = df.drop("is_fake_posting")`, keeping
`fraud_score`, so XGBoost trivially hit 99.9%. Reproduced exactly: XGB **with**
the leak = 0.9994; XGB **without** = 0.916/0.801 — matching our champion. Our
pipeline's AUC-based leakage detector (`core/validation.detect_target_leakage`,
threshold 0.999) correctly drops it and reports the honest ~0.91. **Our pipeline
is correct; the notebook is wrong.** We did NOT weaken detection to "match" —
instead added a prominent leakage callout to the report. (Memory: `fake_internship_leakage`.)

### `fraud.csv` poor F1 — also not a bug
`fraud.csv` is essentially **noise** (AUC ≈ 0.52 ≈ random). No model can predict
it; a poor F1 is the honest, correct outcome. The real Kaggle credit-card data
(`creditcard.csv`) gets F1 ≈ 0.85. `fraud.csv` isn't in the repo anymore.

### Commits this session (all on `main`)
`497dd4d` upload event-loop fix + e2e harness · `4768578` direct large uploads +
premium reports + agent fixes · `37a3048` artifact retention · `ef696c2` train
subsample · `ef2053e` CV-selection opt-in · `fea112a` SHAP bound · `a53aec7` MI
sample · `6d593ad` forest trees 300→150 · `da2395c` probability calibration ·
`24dfbd8` process isolation · `0eed10d` frequency encoding + missing indicators ·
`f034960` leakage callout · `b080e7b` .ipynb export · `ba39e4c` reproduction
notebook · `65bce31` reports-page clarity · `739778d` gitignore reports-route fix.

### Current state
- Both servers run via `npm run dev` **from the `frontend/` dir** (launches
  FastAPI backend on :8000 + Next.js on :3000). NOTE: `npm run dev` must be run
  from `frontend/`, not the repo root.
- App: http://localhost:3000 · API health: http://127.0.0.1:8000/api/health
- Disk was a recurring problem on this machine — kept it clean; deleted `.next`
  (regenerates), logs, caches, old benchmark JSONs. Kept the 170 MB dataset CSV
  per the user. The real space hogs are `venv` (~1.4 GB) and
  `frontend/node_modules` (~0.5 GB), both required to run.
- Uncommitted-by-design: `.env` + `.claude/settings.local.json` (local agent
  state) and the 166 MB `fake_internship_detection_dataset.csv` (now **gitignored**
  — it exceeds GitHub's 100 MB per-file limit and must never be committed). The
  old `PROJECT_*.md`/`README.pdf` deletions + `dev-all.mjs` change were committed
  in `3e5be1e`.

---

## 🎨 Frontend — full design-system overhaul + "Mission Control" (latest session)

A complete **visual redesign** of the Next.js 16 frontend. **Not a stack change** —
still Next.js 16 + Tailwind v4 + React 19, talking to the same FastAPI backend via
the dev proxy. **All app functionality preserved** (auth, routing, upload, polling,
pipeline, reports). All committed & pushed to `main` (latest `e7259dc`).

### The shared design language (every page now follows this)
- **Deep navy + Instrument Serif + frosted glass, fully monochrome.** Background is
  deep navy (never pure black). Headings/display use **Instrument Serif** (loaded
  via a Google Fonts `@import` at the top of `globals.css`); body is **Inter**.
  Surfaces are frosted, rim-lit glass; buttons are rounded-full pills.
- **No brand hues.** Indigo/violet/teal were all replaced with a **cool-grayscale
  ramp**. The ONLY colors left are **semantic status** (success / warning / error)
  and chart data encodings. If you see indigo creep back, it's a regression.
- Lives in **`frontend/src/app/globals.css`**: the `@theme` tokens (primary / pro /
  accent / agent-* are all grayscale now), `.liquid-glass`, `fade-rise` animations,
  `.mission-atmos` (cosmic radial-glow gradient), `.starfield` (twinkle), and the
  glass / button / badge primitives.

### Page-by-page (file → what changed)
- **`src/app/page.tsx` (landing `/`)** — cinematic **fullscreen looping-video hero**
  (serif headline, liquid-glass nav + CTAs) → a **Free vs Enterprise** "Choose your
  path" section (monochrome tier cards) → footer. "Begin Journey" routes into the
  app (auth-aware). **IMPORTANT:** `/` is now a **PUBLIC** page — the Next 16
  middleware **`src/proxy.ts`** was changed so `/` is no longer login-gated (it used
  to redirect to `/auth`).
- **`src/app/enterprise/page.tsx` (dashboard)** — rebuilt as **"Axiom Mission
  Control"**: cosmic atmosphere + starfield backdrop, a "MISSION CONTROL · ONLINE"
  pill, large serif greeting, premium glass **stat cards** (Active pipelines /
  Reports generated / Agents online=8 / Last execution), glowing **quick-action
  tiles** (Upload→file picker, Workflow Builder, Agent Console, Reports, Run
  History), and a **recent-activity "mission log"**. Reuses `AnimatedCounter` and
  `fadeUp/stagger` from `@/lib/animations`. All upload/polling logic kept.
- **`src/components/layout/Sidebar.tsx`** — now a **floating** rounded glass panel
  (inset from edges), white-glass active state. **`AppShell.tsx`** renders
  `.mission-atmos` + `.starfield` for enterprise routes and sets the main margin for
  the floating sidebar (collapsed 100px / expanded 264px).
- **Console pages** (`enterprise/{reports,runs,visualizations,settings,agents}` +
  `report/[runId]`) — serif page titles (because `--font-display` is now Instrument
  Serif), serif **section headers** and serif **stat numbers**. Done at the
  component level (`StatCard`, `SectionHeader`, settings `Section`) so it propagates.
- **`src/app/auth` + `src/app/welcome`** — serif headings, navy, consistent.
- **Free-tier** (`src/app/free`, `free/results/[runId]`) — same serif/monochrome.
- **`src/app/enterprise/workflow/page.tsx`** — replaced **undefined** CSS classes
  (`pro-glass-xs`, `btn-pro`, `badge-pro-gold` had NO css → rendered as flat boxes)
  with real `glass-panel`/`btn-ghost`; serif hero + section headers; swapped a
  hand-rolled markdown parser for the shared **`ReportMarkdown`**.
- **`src/lib/types.ts`** — `AGENT_META` had **8 hardcoded rainbow hex colors**;
  neutralized to the gray ramp, which fixes agent icons **everywhere** they render.

### Frontend commits this session (all on `main`)
`85fa647` cinematic navy + Instrument Serif redesign · `5aeb29d` retune console
primitives · `8d7390e` full-monochrome palette + Mission Control dashboard ·
`3e5be1e` chore: housekeeping (removed `PROJECT_*.md` + `README.pdf`, updated
`dev-all.mjs`, added `scripts/run_full_pipeline.py`) · `b0eadcc` gitignore the
166 MB CSV · `e7259dc` Workflow Builder polish.

### How to VERIFY frontend changes (for a fresh agent — no Playwright installed)
Pages are gated by auth (client `AppShell` + server `proxy.ts`). To screenshot a
**gated** page headless: (1) create a throwaway user via `POST /api/auth/signup`
(a test user **`ui_tester`, id 65** already exists in the dev DB); (2) navigate to
any same-origin page, then seed `localStorage['axiom-auth-store']` +
`['axiom-app-store']` + the `axiom-auth`/`axiom-workspace` cookies with that token;
(3) drive **system Chrome over CDP** — Node 24 has a global `WebSocket`; launch
`chrome.exe --headless=new --remote-debugging-port=PORT --remote-allow-origins=*`,
attach to the page target, `Page.navigate`, then `Page.captureScreenshot` with
`captureBeyondViewport:true` for a full-page shot. (This is how every screenshot in
this session was produced; the helper scripts were temporary and deleted.)

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
- **Reports page now identifies each run clearly** (`/api/runs` +
  `app/enterprise/reports/page.tsx`). The list was showing only a cryptic
  `run_id`. `/api/runs` now also returns the **dataset filename**, **target**,
  and **completed timestamp**; the cards were redesigned to headline the dataset
  name, with date · time, a Free/Pro badge, target, champion + metric, and the
  run_id demoted to a small reference line. Users can now tell at a glance which
  report is for which dataset and when.
- **True reproduction notebook** (`build_reproduction_notebook`,
  `GET /api/report/{run_id}/notebook?kind=reproduce`, "Reproduce" button). A
  standalone notebook that **re-runs Axiom's actual pipeline agent-by-agent**
  (`run_single_agent` over the 8 agents on a shared `PipelineState`) on the same
  dataset — not a re-implementation that could drift. Deterministic seed → it
  reproduces the same champion, honest score, and leakage handling. Each agent is
  its own cell with inspection of the resulting state. Verified: the exact
  sequence runs all 8 agents end-to-end and produces a champion; notebook is
  strictly valid and every code cell compiles.
- **Jupyter notebook (.ipynb) export** (`visualization/notebook_report.py`,
  `GET /api/report/{run_id}/notebook`, "Notebook" button on the report + free
  results pages). Exports a run as a self-contained notebook: the report
  narrative as markdown cells, the generated charts embedded as images, plus
  runnable code cells to load the trained champion (`joblib`) and score new data.
  Built as plain nbformat-v4 JSON (no extra runtime dep); validated with
  `nbformat.validate`. Verified live: 200, valid .ipynb, correct content-type.

### UX / Reproducibility
- **Per-agent "what it did" surfaced in the live pipeline view**
  (`app/free/pipeline/[runId]/page.tsx`, `agentFacts`). Each completed stage now
  shows the real facts from its summary (e.g. "rows after: 200,000 · duplicates
  removed: 20,000 · features after: 21") instead of a generic description —
  directly answering the "do the agents actually do anything?" question.
- **Inference manifest** (`agents/finalization/tools.py` → `inference_manifest.json`).
  Records exactly what the champion expects — ordered feature list, decision
  threshold, problem type, target, leakage-dropped columns, model file — so
  there's no train/serve skew about *which* features in *what* order. Full
  raw→prediction transform replay is the reproduction notebook (Report →
  Reproduce). NOTE: a single fitted `sklearn.Pipeline` for online serving was
  deliberately NOT built — the agents apply transforms imperatively (encoders/
  scalers aren't fitted-transformers), so that's a large refactor with low ROI
  for a batch tool whose reproducibility is already covered by the repro notebook.

### Pipeline / ML
- **Richer datetime decomposition** (`agents/feature_engineering/tools.py`,
  `extract_datetime_features`). Beyond raw year/month/dow/hour, now adds
  **cyclical** sin/cos for month/dow/hour (so Dec≈Jan, 23h≈0h — matters for
  linear models), **is_weekend**, and **recency_days** (days before the latest
  timestamp). Verified one datetime column expands to 12 features; selection
  prunes the rest.
- **Train→test drift detection (PSI)** (`agents/error_detection/tools.py`,
  `_check_drift`). Computes Population Stability Index per numeric feature
  between the train and test splits (row-capped sample); flags features with
  PSI≥0.25 as a LOW finding + records `data_quality_flags["drift_psi"]`.
  Diagnostic only. Verified: catches a shifted feature (PSI=5.7), ignores a
  stable one.
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
0. **Prediction / inference feature — NOT BUILT (highest-value gap).** The pipeline
   is currently **train → evaluate → report ONLY**. There is **no `/api/predict`**
   and no UI to score new/unseen data — you cannot ask "predict the outcome for
   these new rows." **Blocker:** preprocessing + feature-engineering transforms are
   applied *imperatively* by the agents and are **NOT persisted as reusable fitted
   transformers** (`inference_manifest.json` has `encodings: null`; only the final
   champion model is dumped via joblib). The model expects an already-transformed
   feature space it can't rebuild from raw input — so the only faithful "replay" is
   the heavy reproduction notebook (Report → Reproduce), not lightweight scoring.
   **Proposed scope (smallest-useful-first):** (1) persist the fitted preprocess+FE
   as one saved transform (a `ColumnTransformer`/`Pipeline`) — the enabling work;
   (2) `POST /api/predict/{run_id}` — upload new raw CSV (no target) → rows + class
   + probability, downloadable; (3) a "Predict" tab on the report page; (4) optional
   **top-K-per-group** ranking (pass a group column → top-K rows per group). Use
   case that motivated this: an F1 dataset (`16_F1_Race_Results_2019_2024.csv`) —
   predict each race's **top-3 podium** drivers. Target is derived
   (`Top3_Podium = Position <= 3`); most columns are post-race **leakage** (Points
   AUC≈0.999, Position=1.0) so only pre-race signal is valid (Starting Grid AUC≈0.92,
   Driver/Team). **User decided to DEFER building this for now** (asked, said leave it).
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
8. **Richer datetime decomposition** — DONE (see Done).
9. **Train/test drift checks** (PSI) — DONE (see Done).
10. **Persist preprocessing + FE as one `sklearn.Pipeline`** — PARTIAL/DONE: an
    inference manifest is written (see Done). A full fitted-transformer pipeline
    for online serving is deliberately deferred (large refactor, low ROI; repro
    notebook already gives faithful replay).

### Tier 4 — observability / UX
11. **SSE / WebSocket live logs** — EVALUATED, deferred. Browser `EventSource`
    can't send the `Authorization: Bearer` header the app uses, so SSE needs
    token-in-URL or cookie auth + proxy streaming — real complexity for marginal
    gain over the 1.5s polling that already works. Revisit only if true real-time
    streaming becomes a requirement.
12. **Surface each agent's "what I did" summary** — DONE (see Done).

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
