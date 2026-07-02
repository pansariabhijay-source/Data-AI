# Axiom — Complete Project Context

> **Single source of truth for the whole project.** This file consolidates and reconciles
> `PROJECT_REPORT.md`, `working.md`, `frontend/context.md`, and `currentdoingandtodo.md`
> against the actual code. If you read one file to understand Axiom, read this one.
>
> Companion docs (deeper dives): `working.md` (backend ML internals, plain English),
> `frontend/context.md` (every route/component/store), `currentdoingandtodo.md` (session
> handoff + TODO), `PROJECT_REPORT.md` (narrative report). When those drift from code,
> **the code wins** — paths below are the authoritative anchors.

---

## 1. What Axiom is

An **automated machine-learning (AutoML) platform**. A user uploads a CSV, names the column
to predict, and Axiom runs an 8-step pipeline that cleans the data, engineers features,
trains many models, picks the best, audits it, tunes it, and emits a model + report —
no coding required. Strong focus on **fraud detection** (rare-event handling, out-of-time
evaluation, operating-point reporting).

**Honesty note:** each pipeline "agent" is **deterministic Python** (pandas / scikit-learn /
XGBoost / LightGBM). **No LLM runs during a normal web run** — zero tokens, zero external AI
calls. The `llm:` block in `configs/default.yaml` and the CrewAI path feed only the optional
CLI orchestrator (see §3), not the live web pipeline.

Stack: **Python 3.11+ / FastAPI** backend, **Next.js 16 + React 19 + TypeScript + Tailwind v4**
frontend, **SQLite** (SQLAlchemy) for auth/history.

---

## 2. The two execution paths (read this first)

There are **two independent ways the pipeline runs**. This is the most common source of
confusion — keep them distinct:

| | **Web/API path (the live one)** | **CLI / CrewAI path** |
|---|---|---|
| Entry | `api.py` → child process | `main.py` |
| Engine | `core/pipeline_worker.py` → `core/agent_runner.py` | `pipeline/orchestrator.py` (CrewAI `Crew`) |
| Agents | deterministic `*Service` classes run in sequence | same services, wrapped as CrewAI `Agent`s |
| LLM? | **No** — 0 tokens | Yes — Cerebras LLM orchestrates |
| Used by | the website, default | developers / headless, optional |

Both paths call the **same underlying service logic** in `agents/<name>/tools.py`. The
CrewAI path adds an LLM "manager" on top; the web path skips it. `core/pipeline_worker.py`
runs each web pipeline in a **separate child process** so heavy CPU work never freezes the API.

---

## 3. Architecture map

```
Browser (Next.js :3000)
   │  HTTP, Authorization: Bearer <token>   (uploads go DIRECT to :8000, bypassing the proxy)
   ▼
FastAPI backend (api.py :8000) ── auth · upload · run · status · reports · exports
   │  spawns ONE child process per run; streams progress via multiprocessing.Queue
   ▼
core/pipeline_worker.run_pipeline_child
   │
   ▼
core/agent_runner.run_full_pipeline  ── runs the 8 agents in order (AGENT_REGISTRY)
   ▼
[1] data_collection → [2] preprocessing → [3] feature_engineering → [4] data_splitting
→ [5] model_training → [6] error_detection → [7] improvement → [8] finalization
   ▼
artifacts/<run_id>/  (model, metrics, SHAP, inference_manifest)  +  reports/<run_id>/  (md/PDF/notebook)
```

Single shared record threaded through every stage: **`core/state.py::PipelineState`** (a
Pydantic model — paths, problem type, summaries, model results, flags). Agents only mutate
state via their tool functions; the LLM never mutates it directly.

---

## 4. The 8 agents (`agents/<name>/`, logic in `tools.py`)

Agent IDs are defined once in `core/agent_runner.py::AGENT_REGISTRY` and must match the
frontend `AGENT_META`/`AGENT_ORDER` in `frontend/src/lib/types.ts`.

1. **data_collection** — load CSV (chunked, encoding-tolerant), profile every column, detect
   problem type (classification / regression / clustering), compute quality score, flag
   **target leakage** (`core/validation.py::detect_target_leakage`) and **class imbalance**.
2. **preprocessing** — drop rows with missing target, dedupe, impute (numeric→median,
   categorical→mode) + add `<col>__was_missing` flags, fix dtypes (incl. unit-laden text like
   `"$1,234.56"`, `"963 hp"`), parse dates. Outlier clipping **off by default**
   (`outlier_method: none`).
3. **feature_engineering** (`agents/feature_engineering/tools.py`) — the densest stage, in order:
   drop leakage cols → encode target → log-transform skewed regression target → drop numeric
   ID cols → datetime + cyclical (sin/cos) features → domain fraud features (geo_distance,
   age, log_amount, per-card velocity/zscore) → encode categoricals (one-hot ≤15 card.,
   else frequency) → remove (near-)constant → remove correlated (keep more target-relevant) →
   MI feature selection (top `select_k_best=100`) → optional interactions (off) → cast bool→int8
   (so bools survive the `select_dtypes(number)` model matrix — see §11) → attach hidden
   out-of-time split key (`__axiom_split_time__`).
4. **data_splitting** — 70/15/15. **Out-of-time chronological split** when a time axis exists
   (train=oldest, test=newest), else **stratified** random. **Group-aware** when an entity
   column (card/user) is detected — each entity is confined to one split (identity-leakage
   guard): *grouped out-of-time* if a time axis also exists, else StratifiedGroupKFold /
   GroupShuffleSplit. Hidden split keys (time + group) dropped before saving.
5. **model_training** (`agents/training/tools.py`) — stratified subsample for fit speed
   (`TRAIN_SAMPLE_ROWS=100k`, min 2k/class), optional **SMOTE** on train only, train all models
   in parallel (ThreadPool), **F1-optimal decision threshold** (precision-floored), soft-voting
   **ensemble** of top-3, optional **CV-based champion selection** (off), **probability
   calibration** (adopted only if Brier improves), then **honest held-out test eval** +
   **operating points**. Champion chosen by `selection_score` (PR-AUC+ROC-AUC), not thresholded F1.
6. **error_detection** — rule audit: low F1/R², overfitting (only when val score also weak),
   feature explosion, low-signal (ROC-AUC<0.55), leakage, imbalance, train/test **PSI drift**.
   Retryable findings trigger agent 7.
7. **improvement** — `RandomizedSearchCV` tuning (15 iters/round, ≤3 rounds, patience 1); keeps
   a tuned model only if it beats the champion; skips gracefully for ensembles.
8. **finalization** (`agents/finalization/tools.py`) — re-score champion on test set, **SHAP**
   (TreeExplainer / Linear / Kernel), write `metadata.json`, `inference_manifest.json`,
   `champion_model.joblib`, `shap_importance.json`, and the Markdown/PDF/notebook reports with a
   plain-English "what we did and why" narrative.

---

## 5. The core engine (`core/`)

| File | Role |
|---|---|
| `state.py` | `PipelineState` + nested schemas — the single shared run record |
| `config.py` | Layered settings: `configs/default.yaml` → env vars → programmatic overrides (Pydantic) |
| `constants.py` | Every threshold/magic number, each with rationale comments |
| `model_registry.py` | Model catalogue, default params, search spaces, `fit_model` (early stopping), `maybe_wrap_scaler` (leakage-safe per-fit scaling), `apply_imbalance_handling` |
| `metrics.py` | All scoring; `selection_score` (champion pick), `find_optimal_threshold`, `operating_points`, `confusion_counts` |
| `ensemble.py` | `ProbabilityAveragingEnsemble` (soft voting) |
| `resampling.py` | SMOTE / RandomOverSampler (train split only) |
| `validation.py` | Quality scoring, leakage detection, imbalance detection |
| `agent_runner.py` | **The live engine** — `AGENT_REGISTRY`, `run_single_agent`, `run_workflow`, `run_full_pipeline` |
| `pipeline_worker.py` | Child-process entry point + `build_result_payload` (shared API payload shape) |
| `maintenance.py` | Auto-cleanup of old runs (keep newest, cap total, purge old) |
| `utils.py`, `logging_config.py`, `exceptions.py` | Helpers, structured logging, typed exceptions |

**Registered models** (`build_default_registry`):
- Classification: LogisticRegression, RandomForest, ExtraTrees, HistGradientBoosting, XGBoost,
  LightGBM, SVC (≤20k rows). + soft-voting ensemble.
- Regression: LinearRegression, Ridge, RandomForest, ExtraTrees, HistGradientBoosting, XGBoost, LightGBM.
- Clustering: KMeans, DBSCAN, AgglomerativeClustering.

---

## 6. Web application

### Backend — `api.py` (FastAPI, ~1600 lines)
Auth (bcrypt hashes, `secrets.token_urlsafe` session tokens, `get_current_user` dependency),
streamed upload to disk (≤1 GB) with path-traversal sanitization + run-scoped authorization,
run/workflow/single-agent endpoints, status polling, results/report/PDF/notebook/Excel export,
SHAP, per-user run history, health check, disk-full handling.

Key routes: `POST /api/auth/{signup,login,logout}`, `GET /api/auth/me`, `POST /api/upload`,
`POST /api/init-run`, `POST /api/run`, `POST /api/workflow/run`,
`POST /api/agent/{name}/run` + `GET /api/agent/{name}/output/{run_id}`, `GET /api/status/{run_id}`,
`GET /api/results/{run_id}`, `GET /api/report/{run_id}[ /pdf | /notebook ]`,
`GET /api/shap/{run_id}`, `GET /api/runs`, `GET/POST /api/visualizations/{run_id}`,
`GET /api/export/{run_id}/excel`, `GET/POST /api/workspace`.

### Database — `database.py` (SQLite + SQLAlchemy, WAL mode)
Tables: `users`, `sessions`, `prompt_history` (run log per user), `user_preferences`
(free/enterprise mode, theme, onboarding). Auto-creates tables on import.

### Frontend — `frontend/` (Next.js 16 App Router + React 19 + Tailwind v4)
Two tiers chosen on `/welcome` and persisted via `/api/workspace`:
- **Free** (`/free/*`): upload → live pipeline → results. One-click.
- **Enterprise** (`/enterprise/*`, wrapped in `AppShell` sidebar): dashboard, **workflow
  builder** (`@xyflow/react`), agent console, runs, reports, visualizations, settings.

Dev: `npm run dev` from `frontend/` launches API (:8000) + Next (:3000) via `dev-all.mjs`.
`next.config.ts` rewrites `/api/*` → `:8000`; **large uploads go direct to the backend**
(`src/lib/api.ts::backendOrigin()`) to dodge the proxy body cap. Auth token lives in
`useAuthStore` (localStorage + `axiom-auth` cookie). State: Zustand (`useAuthStore`,
`useAppStore`, `useAgentStore`). All backend calls + TS types in `src/lib/api.ts`.
UI polls `GET /api/status` every 1.5 s for live progress.

> ⚠️ **Frontend Next.js is a non-standard build** — `frontend/AGENTS.md` warns APIs/conventions
> differ from training data; consult `node_modules/next/dist/docs/` before editing frontend code.
> Current visual design is the **monochrome "Mission Control"** look (latest commits); older docs
> mention navy/electric-blue — that has been superseded.

### CLI — `main.py`
`python main.py --data file.csv --target col [--config ... --resume --verbose]` → runs the
CrewAI orchestrator path (§2).

---

## 7. Outputs (per run)
`artifacts/<run_id>/`: `champion_model.joblib`, per-model `.joblib`, `metadata.json`,
`inference_manifest.json`, `shap_importance.json`, cleaned/featured/train/val/test CSVs.
`reports/<run_id>/`: `pipeline_report.md` (+ PDF + Jupyter notebooks: a results notebook and a
true "reproduce this run" notebook).

---

## 8. Fraud-correctness principles (the design throughline)
No global scaling (avoids test-stat leakage; per-model scaler instead) · out-of-time splits ·
**group-aware (entity) splits** (no card/user straddles train/test) · **time-safe per-entity
aggregates** (each row sees only its own past) · **time-aware CV** for tuning/selection on
out-of-time runs · automatic leakage-column removal (numeric **and** string-categorical) ·
class weighting + SMOTE + tuned threshold for rare classes · **PR-AUC** as the headline
rare-event metric · **bootstrap CIs** on the test score · probability calibration ·
**operating-point table** ("review top X% → catch Y% at Z% precision") · log-target metrics
reported in original units · threshold-independent champion selection.
Every such choice was validated by an experiment in `scripts/` (decide by measurement, not
assumption); several tempting changes were measured and **rejected**.

---

## 9. Configuration & env knobs

Defaults in `configs/default.yaml`; override via env (`SECTION_FIELD`) or programmatic dict.
Notable settings:

| Setting | Default | Effect |
|---|---|---|
| `splitting.time_aware_split` | `auto` | Chronological split when a time axis exists |
| `splitting.group_aware_split` | `auto` | Keep each entity (card/user) within one split (identity-leakage guard); grouped out-of-time when a time axis also exists |
| `BOOTSTRAP_CI_N` (env) | `500` | Bootstrap resamples for 95% CIs on held-out test metrics (0 disables) |
| `feature_engineering.scaling_method` | `none` | No global scaling (per-model instead) |
| `feature_engineering.select_k_best` | `100` | Max features kept by mutual information |
| `feature_engineering.enable_interactions` | `false` | Explicit pairwise products off |
| `preprocessing.outlier_method` | `none` | Outlier clipping off (trees are robust) |
| `error_detection.overfitting_min_val_score` | `0.80` | Only flag overfitting when val is also weak |
| `training.use_smote` | `true` | Synthetic minority examples (train only) |
| `TRAIN_SAMPLE_ROWS` (env) | `100000` | Cap rows used to fit candidates |
| `MI_SAMPLE_ROWS` (env) | `50000` | Cap rows for mutual-info ranking |
| `CALIBRATE_CHAMPION` (env) | `1` | Calibrate champion probabilities |
| `CV_SELECTION_FOLDS` (env) | `0` | CV-based champion selection (off) |
| `VIZ_SAMPLE_ROWS` (env) | `50000` | Cap rows for chart rendering |
| Secrets in `.env` | — | `CEREBRAS_API_KEY` (CLI path only), `ALLOWED_ORIGINS` (CORS) |

---

## 10. Repo layout & how to run

```
api.py main.py database.py            # backend entry + DB
configs/default.yaml                  # settings
agents/<name>/{agent.py,tools.py}     # 8 agents (+ manager, base.py)
core/                                 # shared engine (§5)
pipeline/{orchestrator.py,checkpoint.py}  # CrewAI runner + save/resume
visualization/{engine,pdf_report,notebook_report,theme}.py
frontend/src/{app,components,lib,store}/  # Next.js app
scripts/                              # 19 experiment/benchmark/diagnostic scripts
tests/                                # 88 unit tests (one file per module)
artifacts/  reports/  logs/  data/    # generated outputs (not source)
```

Run web: `cd frontend && npm install && npm run dev` → http://localhost:3000 (API :8000).
Run headless: `python main.py --data file.csv --target col`. Tests: `pytest`.

Backend setup: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cp .env.example .env`.

---

## 11. Current state & known gaps

- **Branch `fix/bool-feature-drop`**: fixed a real bug — boolean feature columns were silently
  dropped from the model matrix (`select_dtypes(include=[np.number])` excludes `bool`) while
  staying in `selected_features`, desyncing the inference manifest. Fix casts bool→`int8` in
  `agents/feature_engineering/tools.py` (~line 611). Latest commit also aligns SWC to Next.
- **Correctness & fraud-rigor pass (uncommitted, this branch):** 6 more fixes — log-target
  metrics inverse-transformed to original units, time-aware tuning/selection CV, string-
  categorical leakage detection, group-aware (entity) splits, time-safe entity aggregates,
  bootstrap CIs on the test score — plus by-name model-matrix selection
  (`core.utils.build_model_matrix`) that closes the manifest-desync class of bug entirely.
  118 unit tests green. Details: top section of `currentdoingandtodo.md`.
- **No live `/predict` endpoint yet** — reproduce predictions via the generated reproduction
  notebook + `inference_manifest.json`. Building a reusable scoring pipeline is the top next step.
- **Per-card aggregate features** are now **time-safe** (causal expanding windows over
  each entity's past only) when a time axis exists — the earlier whole-dataset
  look-ahead is fixed (`fix/bool-feature-drop` branch, uncommitted).
- **CrewAI/LLM path** present but intentionally unused at runtime.
- **Frontend**: logs are polled not streamed (SSE deferred — `EventSource` can't send Bearer);
  Settings page is thin; inference manifest not yet surfaced in the UI.

> Keep this file current when architecture, agents, routes, or key settings change.
