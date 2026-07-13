# Axiom — Autonomous Data Scientist
### Internship Project Report

---

## 1. Executive summary (read this first)

**Axiom is an automated machine-learning (ML) platform.** A user uploads a spreadsheet
(CSV file), points at the column they want to predict, and Axiom does everything a data
scientist would normally do by hand — clean the data, build useful features, train many
prediction models, pick the best one, check it for mistakes, and write a full report —
**automatically, with no coding required by the user.**

It is built as two parts that work together:

- A **backend** (the "brain") written in **Python** that does all the data work.
- A **website** (the "face") written in **Next.js / React** that lets a person use it
  through a clean, modern interface in their browser.

The project is aimed especially at **fraud detection** (catching fraudulent
transactions), and includes the methods a serious fraud team would expect: correct
handling of rare events, time-based evaluation, and reports that say how many frauds you
catch at a given review budget.

> **One important honesty note up front.** Axiom uses the word "agent" for each step of
> its pipeline, but **these agents are ordinary, deterministic Python programs** doing
> statistics and machine learning. **No large language model (no ChatGPT-style AI) runs
> during a normal pipeline run** — so each run costs **zero AI tokens** and makes **zero
> external AI calls**. The "intelligence" comes from classic, well-understood ML
> algorithms running on the user's own machine. (There is an optional, unused AI config
> stub left in the code, but it is not part of the running pipeline.)

---

## 2. The problem it solves

Building a good ML model normally requires an expert and many manual, error-prone steps:
loading messy data, fixing missing values, encoding text columns into numbers, choosing
and tuning models, and — most importantly — **avoiding subtle mistakes that make a model
look great in testing but fail in the real world.**

Axiom packages all of that expertise into an automatic pipeline so that **a non-expert
can get a trustworthy model and a clear, honest report** from a raw CSV in a few minutes.

---

## 3. How it is organised (architecture)

```
   Browser (the website the user sees)
        │  sends requests over the internet (HTTP)
        ▼
   FastAPI backend  (api.py)  ── handles login, file upload, starting runs
        │  starts a separate child process for each run (so the website never freezes)
        ▼
   Pipeline Worker  ── runs the 8 agents in order
        ▼
   [1] Data Collection → [2] Preprocessing → [3] Feature Engineering →
   [4] Data Splitting → [5] Model Training → [6] Error Detection →
   [7] Improvement → [8] Finalization
        ▼
   Saved results: trained model, charts, and a full report (Markdown + PDF + Notebook)
```

**Why a separate child process per run?** Training a model can use the computer's full
processor for minutes. If that ran inside the website's server, the whole website would
freeze and look broken. By running each pipeline in its own isolated process, the website
stays fast and responsive, and if a run crashes it only kills that one process — not the
whole server.

---

## 4. The 8-agent pipeline — every step in plain words

This is the heart of the project. The data flows through eight steps ("agents"), one
after another. Each one does a specific job and hands its results to the next.

### Agent 1 — Data Collection (`agents/data_collection/`)
**Job: read and understand the file.**
- Loads the CSV (in chunks if it is very large, so it doesn't run out of memory).
- Tolerates files that aren't in standard text encoding (e.g. exported from Excel) so they
  don't crash on a stray character.
- Looks at every column and decides its type: number, category (text), date, true/false, or
  free text.
- **Decides what kind of problem this is** automatically:
  - few distinct values to predict → **Classification** (e.g. fraud / not fraud),
  - a continuous number → **Regression** (e.g. predict a price),
  - no target chosen → **Clustering** (group similar rows).
- Computes a **data-quality score** (how clean the data is).
- **Checks for "leakage"** — a feature that secretly *is* the answer (explained in the
  glossary). If found, it is flagged so it can be removed later.
- Warns if the classes are very imbalanced (e.g. only 0.2% fraud), or if no feature has any
  real relationship to the target (a sign the data is essentially noise).

### Agent 2 — Preprocessing (`agents/preprocessing/`)
**Job: clean the data.**
- Drops rows where the answer (target) is missing — they're useless for learning.
- Removes exact duplicate rows.
- Fills in missing values sensibly (numbers → the middle value; categories → the most common
  value), and **adds a small flag column** recording where a value *was* missing — because
  "this field was blank" is sometimes itself a clue (common in fraud).
- Fixes column types: converts text that is really numbers into numbers, including
  **unit-laden text** like `"$1,234.56"`, `"963 hp"`, or `"2.5 sec"` → proper numbers; and
  recognises date columns.
- Outlier handling (clipping extreme values) is **available but off by default**, because
  the tree-based models Axiom uses are naturally robust to outliers, and clipping can throw
  away the very large values (e.g. a huge fraudulent purchase) that carry the signal.

### Agent 3 — Feature Engineering (`agents/feature_engineering/`)
**Job: turn raw columns into better signals for the model.**
- **Removes ID-like columns** (e.g. `transaction_id`) — they're just labels and a model can
  "memorise" them instead of learning.
- **Turns dates into useful features**: year, month, day-of-week, hour, plus smart "cyclical"
  versions so the model knows December is next to January and 23:00 is next to 00:00.
- **Adds domain features** known to help fraud detection: distance between cardholder and
  merchant locations, age from a date of birth, a log-transform of money amounts (to tame
  big numbers), and per-card spending statistics (how much/how often this card is used).
- **Encodes text categories into numbers**: small categories become yes/no columns
  (one-hot); large ones are replaced by how often each value appears (frequency encoding),
  which is meaningful and safe.
- **Removes useless features**: columns that are essentially constant, and one of every pair
  of near-duplicate columns (keeping the one more related to the target).
- **Removes leakage columns** that Agent 1 flagged.
- Keeps the most informative features (up to 100) so the model isn't drowned in noise.

### Agent 4 — Data Splitting (`agents/splitting/`)
**Job: divide the data into Train / Validation / Test (70% / 15% / 15%).**
- **Train** = what the model learns from. **Validation** = used to pick the best model and
  tune settings. **Test** = locked away and only used once at the very end, to get an honest
  score on data the model has never seen.
- **Out-of-time splitting for time-based data (important for fraud):** if the data has a time
  column, Axiom trains on the **oldest** transactions and tests on the **newest** — exactly
  like real life, where you only have the past to predict the future. A naive random split
  would let the model "peek at the future" and report a falsely high score.
- For classification without a time column, it uses **stratified** splitting so the rare class
  (e.g. fraud) appears in the same proportion in every split.

### Agent 5 — Model Training (`agents/training/`)
**Job: train several models and choose the best.**
- Trains a whole set of models **in parallel** (at the same time): Logistic Regression,
  Random Forest, Extra Trees, Histogram Gradient Boosting, XGBoost, LightGBM, and (on small
  data) a Support Vector Machine. For regression it uses the matching set.
- **Handles rare classes**: it can create synthetic examples of the rare class (SMOTE) in the
  training data only, and it weights the rare class more heavily, so the model doesn't simply
  ignore the 1%-of-data fraud cases.
- **Scaling done the right way**: models that need their inputs on a common scale (like linear
  models) get their own scaler that is fitted *only* on training data — so no information
  leaks from the test set. Tree models, which don't need scaling, are left untouched.
- **Picks a smart decision threshold** instead of the naive 50%: for imbalanced data it finds
  the cut-off that best balances catching fraud vs false alarms.
- **Builds an ensemble**: averages the predictions of the top models, which often beats any
  single model, and uses it if it wins.
- **Calibrates probabilities**: adjusts the model so that "80% confident" really means 80%,
  which matters when those probabilities drive decisions.
- Champion selection uses **threshold-independent ranking scores** (PR-AUC and ROC-AUC) rather
  than a single noisy number, so the chosen model generalises better.

### Agent 6 — Error Detection (`agents/error_detection/`)
**Job: audit the results and flag problems.** It checks for:
- performance that is too low to be useful,
- **overfitting** (a model that memorised training data but fails on new data) — and it is
  smart about this: it only raises the flag when validation performance is *also* weak, so it
  doesn't cry wolf for tree models that naturally score perfectly on training data,
- too many features, class imbalance, all-models-failed,
- **drift** between train and test (using a standard "PSI" measure) — useful confirmation that
  a time-based split is doing its job.

### Agent 7 — Improvement (`agents/improvement/`)
**Job: try to make the best model even better through tuning.**
- Only runs if Agent 6 found fixable problems.
- Uses **RandomizedSearchCV** (tries many combinations of model settings and cross-checks
  each) and keeps a tuned model **only if it actually beats the current champion**.
- Skips gracefully when the champion is an ensemble (which has no single set of knobs to turn).

### Agent 8 — Finalization (`agents/finalization/`)
**Job: produce the honest final score, explanations, and all output files.**
- Scores the champion **once** on the locked-away Test set — the honest real-world estimate.
- **Operating-point table (built for fraud teams):** shows, for several review budgets,
  *"if you investigate the top X% riskiest transactions, you catch Y% of fraud at Z%
  precision"* — so a team can pick a threshold matching their capacity.
- **SHAP explanations:** ranks which features most influenced the model's decisions, so the
  model isn't a black box.
- **Plain-English explanation section:** the report describes, in words, what the pipeline
  did and **why each column was dropped or added**, and **why the chosen model won**.
- Saves the trained model, a machine-readable summary, and the full report.

---

## 5. The supporting engine (the `core/` folder)

These modules are the shared toolbox every agent relies on:

- **`config.py` + `configs/default.yaml`** — all the adjustable settings in one place (split
  ratios, thresholds, which behaviours are on/off), with sensible defaults.
- **`constants.py`** — every "magic number" and threshold, named and documented in one file.
- **`state.py`** — the single shared record of a run (the "PipelineState"): the data paths,
  the detected problem type, all results and summaries. Every agent reads and updates it.
- **`model_registry.py`** — the catalogue of available models, their default settings, their
  tuning ranges, and the logic for fitting them (including early-stopping for boosters and
  the per-model scaling wrapper).
- **`metrics.py`** — all scoring in one place: F1, precision/recall, ROC-AUC, PR-AUC, R², the
  best-threshold finder, the champion-selection score, and the new operating-points table.
- **`ensemble.py`** — the soft-voting ensemble that averages model probabilities.
- **`resampling.py`** — the SMOTE logic for rare classes (training data only).
- **`validation.py`** — data-quality scoring, **leakage detection**, and imbalance detection.
- **`pipeline_worker.py`** — runs a pipeline in a separate child process and streams progress
  back so the website can show a live progress bar.
- **`agent_runner.py`** — the actual engine that runs the 8 agents in order (or any custom
  subset), capturing timing, memory, and per-agent summaries.
- **`maintenance.py`** — housekeeping: automatically deletes old runs so the disk never fills
  up (keeps the newest few, caps the total, removes anything too old).
- **`utils.py`** — helpers (safe CSV reading, memory optimisation, ID generation, etc.).
- **`logging_config.py` / `exceptions.py`** — structured logs for debugging and clean,
  specific error types.
- **`pipeline/` (orchestrator + checkpoint)** — an alternative CrewAI-based runner and a
  save/resume mechanism. (The live path is `agent_runner.py`; the CrewAI orchestrator is
  present but not the default route.)

---

## 6. The web application

### Backend — FastAPI (`api.py`)
A REST API that the website talks to. It handles:
- **Accounts and security:** sign up / log in, passwords stored as secure hashes (bcrypt),
  short-lived random session tokens, and strict cross-origin rules (CORS). Run data is
  scoped to the owning user.
- **Upload:** streams big files straight to disk (up to 1 GB), returns a preview and starter
  charts. Heavy work runs off the main thread so uploads never freeze the server.
- **Running pipelines:** `POST /api/run` (full pipeline), `POST /api/workflow/run` (custom
  agent sequences for enterprise users), and per-agent endpoints.
- **Live status & results:** the website polls `GET /api/status/{run_id}` to show progress.
- **Reports & exports:** Markdown report, PDF download, and Jupyter notebook export
  (both a results notebook and a true "reproduce this run" notebook).
- **Health check** and disk-full handling (returns a clean error instead of crashing).

### Database — SQLite (`database.py`)
A small, file-based database storing: **users** (email, name, hashed password), **sessions**
(login tokens), **prompt/run history** (what each user ran and the outcome), and **user
preferences** (free vs enterprise mode, theme). It uses WAL mode for safe concurrent access.

### Frontend — Next.js / React (`frontend/`)
The browser interface, recently redesigned into a clean "Mission Control" look (deep navy,
elegant serif headings, frosted-glass panels). It has **two tiers**:
- **Free** — a simple flow: upload → watch the pipeline → see results.
- **Enterprise** — a dashboard with a workflow builder, agent console, run history, reports,
  and visualizations.

It talks to the backend through a development proxy, manages login/app state with small
stores (Zustand), and — for large uploads — sends files **directly** to the backend to avoid
a proxy size limit that previously caused confusing "backend unavailable" errors.

### Command line (`main.py`)
For developers, the whole pipeline can also be run headless from a terminal:
`python main.py --data file.csv --target column_name`.

---

## 7. What you get out (outputs)

For every run, Axiom saves (under `artifacts/<run_id>/` and `reports/<run_id>/`):
- the **trained champion model** (ready to load and use),
- a **Markdown report** and a **downloadable PDF** with all the tables, charts, the
  plain-English explanation, the operating-point table, and SHAP feature importances,
- **Jupyter notebooks** to reproduce or extend the run,
- machine-readable JSON summaries (metrics, feature importances, run state),
- an **inference manifest** describing exactly what features the model expects.

---

## 8. Special focus: fraud detection (and why it is done correctly)

Fraud is hard because frauds are rare, patterns change over time, and a careless evaluation
can look great while being useless in production. Axiom handles this properly:

- **Rare-class handling:** class weighting + optional SMOTE + a tuned decision threshold, so
  the 0.2%-fraud cases are not ignored.
- **The right metric:** it reports **PR-AUC** (the standard metric for rare-event detection),
  not just accuracy — because with 0.2% fraud, a model that flags *nothing* is "99.8%
  accurate" but catches no fraud.
- **Out-of-time evaluation:** train on the past, test on the future — no peeking.
- **Automatic leakage removal:** columns that secretly encode the answer are detected and
  dropped, so the reported score is honest.
- **Operating points:** the report tells a fraud team exactly what they catch at their chosen
  review budget — the number that actually matters operationally.

This was validated on real and synthetic fraud data (see Section 9).

---

## 9. Engineering work done in this project (with evidence)

A core principle followed throughout: **decide by measurement, not by assumption.** Every
change below was proven with an experiment, and several tempting ideas were *rejected*
because the data showed they didn't help.

1. **Fixed preprocessing & feature engineering that were silently hurting accuracy.**
   It was found that running the pipeline *without* these two agents sometimes beat running
   *with* them. Root causes were found and fixed: a global data scaler that leaked
   information and hurt tree models (moved to a correct per-model scaler); outlier clipping
   that erased signal (turned off by default); a variance filter that deleted useful
   skewed/rare columns (made conservative); a correlation filter that dropped the wrong
   column (made target-aware); an ID-detector that removed legitimate coded columns
   (tightened); and explicit interaction features that mostly added noise (turned off,
   because boosting models already capture interactions).

2. **Added out-of-time (chronological) evaluation.** Measured on the standard credit-card
   fraud dataset: a random split reported PR-AUC **0.83**, but the honest out-of-time number
   was **0.78** — the random split was hiding ~5 points of optimism. Axiom now uses the
   honest method automatically when a time column exists.

3. **Made the pipeline handle messy real-world data:** reads non-standard file encodings,
   and extracts numbers from unit-laden text like `"$1,234.56"` or `"963 hp"`. (A subtle bug
   where this extractor damaged date columns was caught and fixed by parsing dates first.)

4. **Made the report explain itself** in plain English (what changed and why, why the model
   won) and **crash-proof** (a problem in one report section can no longer lose the whole
   report).

5. **Made the overfitting warning smarter** — it now only flags a model when its validation
   score is genuinely weak, removing false alarms for tree models.

6. **Added the operating-point table** for fraud decisioning.

7. **Built a deliberately hard synthetic fraud dataset** (`scripts/make_hard_fraud.py`) with
   extreme imbalance, concept drift (a new fraud tactic appearing only recently), signal
   hidden in feature interactions, a leakage trap, and noise. The pipeline handled it well
   (out-of-time PR-AUC matched/beat a hand-built reference model), confirming robustness.

8. **Measured and rejected** three plausible changes because they didn't help: a different
   champion-selection metric (picked the same model), refitting the champion on all data
   (inconsistent — helped one model, hurt another), and a velocity-feature leakage fix
   (no measurable impact on the tested data). This disciplined "test, then decide" approach
   is itself a key result.

---

## 10. Testing and quality

- **88 automated unit tests** (`tests/`) cover every core module and agent, and all pass.
- **Reproducible experiments** (`scripts/`) back every claim: feature-engineering ablation,
  fraud methodology comparison, calibration, and CV-selection studies.
- **End-to-end checks** confirm the full 8-agent pipeline runs cleanly on classification,
  regression, and fraud datasets, that no internal "split key" leaks into the model, and
  that the reports render correctly.

---

## 11. How to run it

**Backend setup**
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then add any keys/origins
```

**Run the website (backend + frontend together)** — from the `frontend/` folder:
```bash
npm install
npm run dev
```
Then open **http://localhost:3000**. The API runs at **http://127.0.0.1:8000**.

**Run headless from the terminal**
```bash
python main.py --data path/to/data.csv --target target_column
```

**Run the tests**
```bash
pytest
```

---

## 12. Key settings you can adjust (quick reference)

| Setting | Default | What it does |
|---|---|---|
| `splitting.time_aware_split` | `auto` | Use chronological (out-of-time) split when a time column exists |
| `feature_engineering.scaling_method` | `none` | Global scaling off (scaling is done per-model instead) |
| `preprocessing.outlier_method` | `none` | Outlier clipping off (trees are robust to outliers) |
| `feature_engineering.select_k_best` | 100 | Max features kept by importance |
| `feature_engineering.enable_interactions` | off | Explicit feature-product features off |
| `error_detection.overfitting_min_val_score` | 0.80 | Only flag overfitting when validation is also weak |
| `training.use_smote` | on | Create synthetic rare-class examples (training only) |
| `TRAIN_SAMPLE_ROWS` | 100000 | Cap rows used to fit candidate models (for speed) |
| `CALIBRATE_CHAMPION` | on | Make the model's probabilities trustworthy |
| `PIPELINE_ISOLATION` | on | Run each pipeline in its own crash-proof child process |
| `ARTIFACT_MAX_RUNS` / `_MIN_KEEP` / `_RETENTION_DAYS` | 25 / 5 / 30 | Auto-cleanup of old runs |

---

## 13. Glossary (every ML term used, in one line)

- **Classification** — predicting a category (fraud / not fraud).
- **Regression** — predicting a number (a price).
- **Clustering** — grouping similar rows when there is no target.
- **Feature** — an input column the model learns from.
- **Target** — the column we want to predict.
- **Train / Validation / Test split** — separate data slices for learning, tuning, and a
  final honest score.
- **Class imbalance** — when one class (e.g. fraud) is very rare.
- **SMOTE** — a technique that creates realistic synthetic examples of the rare class.
- **Leakage** — when a feature secretly contains the answer, giving a fake-high score that
  collapses in real use.
- **Out-of-time evaluation** — testing on the most recent data after training on older data,
  to mimic real deployment.
- **Overfitting** — memorising the training data instead of learning patterns that generalise.
- **One-hot / frequency encoding** — two ways of turning text categories into numbers.
- **Scaling** — putting numeric features on a common range (needed by some models).
- **Accuracy** — fraction of predictions correct (misleading for rare events).
- **Precision** — of the items we flagged, how many were truly positive.
- **Recall** — of all true positives, how many we caught.
- **F1 score** — a single balance of precision and recall.
- **ROC-AUC** — overall ranking ability; can look deceptively high for rare events.
- **PR-AUC (Average Precision)** — the better ranking metric for rare events like fraud.
- **R²** — for regression, the fraction of variation in the target the model explains.
- **Decision threshold** — the probability cut-off for saying "positive".
- **Calibration** — making predicted probabilities match real-world frequencies.
- **Ensemble** — combining several models for a stronger one.
- **SHAP** — a method that explains which features drove each prediction.
- **Operating point** — a chosen threshold tied to a real budget (e.g. review the top 0.5%).
- **PSI (Population Stability Index)** — a measure of how different two data slices are.
- **Hyperparameter tuning** — searching for a model's best internal settings.

---

## 14. File-by-file map (so every part is accounted for)

**Top level**
- `api.py` — the web backend (FastAPI): accounts, upload, run, status, reports.
- `main.py` — the command-line entry point.
- `database.py` — the SQLite database (users, sessions, history, preferences).
- `requirements.txt` — the Python libraries needed.
- `README.md` — quick-start guide. `working.md` — a plain-English deep dive. `currentdoingandtodo.md` — the running engineering log.
- `configs/default.yaml` — the default settings file.

**`agents/`** — the 8 pipeline steps (each has `agent.py` = its definition, `tools.py` = its
actual logic): `data_collection`, `preprocessing`, `feature_engineering`, `splitting`,
`training`, `error_detection`, `improvement`, `finalization`, plus `manager` and `base.py`.

**`core/`** — the shared engine: `config.py`, `constants.py`, `state.py`, `model_registry.py`,
`metrics.py`, `ensemble.py`, `resampling.py`, `validation.py`, `agent_runner.py`,
`pipeline_worker.py`, `maintenance.py`, `utils.py`, `logging_config.py`, `exceptions.py`.

**`pipeline/`** — `orchestrator.py` (alternative runner) and `checkpoint.py` (save/resume).

**`visualization/`** — chart and report generation: `engine.py` (charts), `pdf_report.py`
(PDF), `notebook_report.py` (Jupyter export), `theme.py` (styling).

**`frontend/`** — the Next.js website: `src/app/` (pages for landing, auth, free tier,
enterprise dashboard, reports), `src/components/` (shared UI), `src/lib/` (API client),
`src/store/` (app/auth state), `src/proxy.ts` (auth/routing middleware).

**`scripts/`** — reproducible experiments and tools: `make_hard_fraud.py` (hard fraud dataset
generator), `fraud_methodology_experiment.py`, `fe_ablation_experiment.py`,
`fe_bugpath_experiment.py`, `calibration_experiment.py`, `cv_selection_experiment.py`,
`e2e_test.py`, `diagnose_*.py`, `benchmark_*.py`, `synth_datasets.py`.

**`tests/`** — the 88 automated tests, one file per module.

**`artifacts/` and `reports/`** — generated outputs per run (not part of the source code).

---

## 15. Limitations and possible future work

- **Live prediction service is not built yet.** Axiom trains, evaluates, and reports; it does
  not yet expose a `/predict` endpoint to score brand-new rows directly. The faithful way to
  reproduce predictions today is the auto-generated reproduction notebook. Building a saved,
  reusable transform pipeline for one-click scoring is the highest-value next step.
- **Entity-level aggregate features** (per-card statistics) are currently computed over the
  whole dataset; a fully leak-free version would compute them only from each transaction's
  past. Measured impact is small, but it is a known refinement for very strict settings.
- **The CrewAI/LLM path** exists in the code but is intentionally not used at runtime; it
  could be wired up if natural-language orchestration is ever desired.

---

*Prepared as the internship project report for Axiom — Autonomous Data Scientist.
The full source code and history are available in the project's GitHub repository.*
