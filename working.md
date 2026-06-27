# How Axiom Works — Full Pipeline Explained

> Plain-English walkthrough of every part of the system. No ML degree required.

---

## The Big Picture

Axiom is an **automated machine learning platform**. You give it a CSV file, tell it which column you want to predict, and it figures out the rest — cleaning the data, engineering features, training multiple models, picking the best one, and generating a report. All of that happens automatically through a chain of 8 "agents" that run one after another.

**Does it use any LLM / ChatGPT-style AI?**

No — not per run. There is an `llm:` section in `configs/default.yaml` (pointing to a Cerebras Llama model) but it is **not wired into the pipeline**. Zero tokens are consumed, zero API calls are made, zero cost per run from an LLM provider. The word "agent" here just means "a self-contained processing step" — each one is a Python class doing deterministic math (statistics, scikit-learn, gradient boosting), not a language model.

---

## System Architecture

```
Browser (Next.js frontend)
        │  HTTP / REST
        ▼
FastAPI backend  (api.py — port 8000)
        │  multiprocessing.Queue  ← spawns a CHILD PROCESS per run
        ▼
Pipeline Worker  (core/pipeline_worker.py)
        │  runs 8 agents in sequence
        ▼
  [Agent 1] Data Collection
  [Agent 2] Preprocessing
  [Agent 3] Feature Engineering
  [Agent 4] Data Splitting
  [Agent 5] Model Training
  [Agent 6] Error Detection
  [Agent 7] Improvement
  [Agent 8] Finalization
        │
        ▼
Artifacts (saved to /artifacts/<run_id>/)
Reports   (saved to /reports/<run_id>/)
```

The API runs in one process. Each pipeline run spawns a **separate child process** so that heavy CPU work (training, SHAP calculations) never freezes the server. The parent and child communicate through a queue — the child sends back progress events as each agent finishes, which the frontend polls to show the live progress bar.

---

## The 8 Agents — One by One

---

### Agent 1 — Data Collection

**What it does:** Loads and profiles your CSV.

**How it runs:**
- Reads the CSV (chunked if > 100 MB)
- Detects each column's type: numeric, categorical, datetime, boolean, text
- Counts nulls, unique values, memory usage
- Figures out the **problem type** automatically:
  - If the target column has ≤ 20 unique string/int values → **Classification**
  - If it's a continuous number → **Regression**
  - If no target is given → **Clustering**
- Checks for class imbalance (are the fraud cases only 0.3% of data?)
- Checks for target leakage (is any feature a near-perfect copy of the target?)

**Output:**
- A dataset profile (rows, columns, null counts, data types)
- The detected problem type
- A quality score (0–1) estimating how clean the data is
- Warning flags for imbalance or leakage

---

### Agent 2 — Preprocessing

**What it does:** Cleans the data so it's fit for modeling.

**How it runs:**

1. **Drop rows with missing target** — rows where the label is blank are useless
2. **Remove duplicates** — keeps only the first copy of any repeated row
3. **Handle missing values** per column:
   - Column with > 70% nulls → drop it entirely
   - Numeric column with some nulls → fill with median
   - Categorical column with some nulls → fill with `"missing"` placeholder
   - Also adds a binary `<column>__was_missing` flag before filling, because *whether* a value was missing is sometimes predictive (e.g. fraud records often have blank fields)
4. **Fix data types** — converts strings that look like numbers, parses date columns
5. **Handle outliers** (IQR method) — **OFF by default.** Tree and gradient-boosting
   models (which usually win) split on rank, so they're robust to outliers, and
   clipping the extreme tail erases exactly the high-leverage values that carry
   signal (large fraud amounts, high-priced homes). Set `outlier_method="iqr"` to
   re-enable; when on it clips to the 1.5×IQR fence but skips discrete columns
   (<20 unique values) and columns where >5% of rows would be clipped (that's the
   real distribution, not noise).
6. **Recompute quality score** on the cleaned data

**Output:**
- A cleaned CSV saved to `artifacts/<run_id>/cleaned_data.csv`
- A preprocessing summary (rows before/after, columns before/after, duplicates removed)

---

### Agent 3 — Feature Engineering

**What it does:** Turns raw columns into better signals for the model.

**How it runs, in order:**

**Step 1 — Drop ID columns**
Columns that are just row identifiers (e.g. `transaction_id`, `user_id`) carry no predictive signal. They are detected two ways:
- Column name matches a pattern: `id`, `idx`, `index`, `key`, `no`, `num`, `code`, `seq`
- Column values are nearly all unique (> 90% unique for numbers, > 50% for strings) — a pure ID

**Step 2 — Parse datetime columns**
Columns that are stored as strings but contain dates are converted to actual datetime objects. Then cyclic time features are extracted:
- `hour_sin`, `hour_cos` — time of day as a circle (so midnight = 23:59, mathematically)
- `dayofweek_sin`, `dayofweek_cos` — day of week
- `month_sin`, `month_cos` — month of year
- `is_weekend` — binary flag
- Elapsed days since epoch

**Step 3 — Encode categoricals**
- ≤ 15 unique values → **one-hot encoding** (creates a binary column per category)
- > 15 unique values → **frequency encoding** (replaces category with its row count)

**Step 4 — Domain feature engineering** (specialized features for common dataset types)
- If lat/lon columns exist → `geo_distance_km` (Haversine distance between merchant and cardholder)
- If a `dob` (date of birth) column exists → `age_years`
- If an `amt` or `amount` column exists → `log_amt` (log-transform reduces the skew of money values)
- If a `card` number column exists → per-card velocity features: `card_tx_count`, `card_amt_mean`, `card_amt_std`, `card_amt_max`, `card_amt_zscore`

**Step 5 — Remove (near-)constant features**
Features with only one value (or a single value covering ≥99% of rows) give a model
nothing to learn from and are dropped. This is deliberately conservative: it removes
ONLY degenerate columns, never a rare-but-present signal. (An earlier version min-max
scaled each column and applied a variance floor, which wrongly deleted skewed-but-
predictive features and rare one-hot flags — the most informative columns in fraud-
type data. Mutual-information selection downstream prunes anything truly uninformative.)

**Step 6 — Remove highly correlated features**
If two features have > 0.95 correlation, one is redundant. The system keeps the one
**more correlated with the target** and drops the other (so the more predictive member
of the pair survives).

**Step 7 — Log-transform skewed regression targets**
For regression, if the target (e.g. `price`) has a skew > 1.0 and is always positive, apply `log(1 + price)`. This prevents the model from being dominated by extreme values. The inverse transform is applied before reporting final predictions.

**Step 8 — Select top K features**
Uses Mutual Information (measures how much knowing a feature reduces uncertainty about the target) to rank all features. Keeps the top 100 (raised from 50, which was discarding useful mid-ranked features). This only bites on genuinely wide datasets, preventing feature explosion while keeping the signal.

**A note on scaling:** numeric features are **not** scaled into the shared dataset.
A single scaler fit on the full data before the train/test split would leak test-set
statistics, and scaling hurts the tree models that usually win. Instead, the few
scale-sensitive models (LogisticRegression, SVC, LinearRegression, Ridge) are wrapped
in their own `StandardScaler` at training time — fit per training fold (no leakage),
while tree/boosting models keep raw, interpretable features.

**Output:**
- A feature-engineered CSV saved to `artifacts/<run_id>/featured_data.csv`
- A summary of features created, removed, and encoded

---

### Agent 4 — Data Splitting

**What it does:** Splits data into train / validation / test sets.

**How it runs:**
- Default split: **70% train / 15% validation / 15% test**
- **Out-of-time (chronological) split when a time axis is detected** (fraud and most
  transactional data are temporal). The model trains on the **oldest** rows and is
  tested on the **newest** — exactly like production, where you only have the past to
  predict the future. A random split would leak future patterns (and the same
  card/user) into training and inflate the offline score. The time axis is found from
  the primary timestamp (threaded from feature engineering) or a numeric time column
  (e.g. creditcard's `Time`); the sort key is dropped before saving so the model never
  sees it. Controlled by `splitting.time_aware_split` (`auto`/`on`/`off`).
- **Falls back to stratified splitting** when there's no time axis (or if a
  chronological slice would lose the rare class) — every split keeps the same class
  ratio as the original (so a 0.3% fraud rate is preserved, not accidentally
  concentrated).
- For clustering, no split — the full dataset is used
- The test set is locked away and never seen during training or tuning

**Output:**
- Three CSV files: `train.csv`, `val.csv`, `test.csv`

---

### Agent 5 — Model Training

**What it does:** Trains multiple models and picks the best one.

**How it runs:**

**Step 1 — Optionally subsample for speed**
If training data > 100,000 rows, a stratified subsample of 100,000 rows is used to *fit* the models. Each class keeps at least 2,000 rows so minority classes survive. Models are still scored on the full validation set, so metric estimates are honest.

**Step 2 — Handle class imbalance (SMOTE)**
For imbalanced classification (minority class < 15% of data):
- **SMOTE** (Synthetic Minority Oversampling Technique) generates synthetic minority-class examples by interpolating between existing ones. Think of it as: "here are 500 real fraud examples — let's create 1,000 more plausible-looking ones."
- Target ratio after SMOTE: minority becomes 25% of majority (not fully balanced — empirically, full 50/50 hurts gradient boosters)
- Falls back to RandomOverSampler if there aren't enough minority examples for SMOTE
- When SMOTE runs, model-level class weighting is disabled to avoid double-correcting

**Step 3 — Train all models in parallel**
Using a thread pool, all models are trained simultaneously:

**Classification models:**
| Model | Notes |
|---|---|
| LogisticRegression | Simple linear baseline. `class_weight="balanced"` |
| RandomForestClassifier | 150 trees. `class_weight="balanced"` |
| ExtraTreesClassifier | Like Random Forest but more randomized. `class_weight="balanced"` |
| HistGradientBoostingClassifier | sklearn's fast gradient booster. `class_weight="balanced"` |
| XGBClassifier | XGBoost. 500 trees, `scale_pos_weight` set to minority/majority ratio |
| LGBMClassifier | LightGBM. 500 trees, `is_unbalance=True`, 63 leaves |
| SVC | Support Vector Machine. Only used when training set < 20,000 rows (slow) |

**Regression models:**
| Model | Notes |
|---|---|
| LinearRegression | Simple OLS baseline |
| Ridge | Regularized linear regression |
| RandomForestRegressor | 100 trees |
| ExtraTreesRegressor | 150 trees |
| HistGradientBoostingRegressor | Fast gradient booster with early stopping |
| XGBRegressor | XGBoost |
| LGBMRegressor | LightGBM |

**Clustering models:**
| Model | Notes |
|---|---|
| KMeans | Tries k = 3, 4, 5 clusters |
| DBSCAN | Density-based, auto-detects cluster count |
| AgglomerativeClustering | Hierarchical clustering |

**Step 4 — Score each model**
- Classification: primary metric is **F1** (harmonic mean of precision and recall). For extreme imbalance, also tracks ROC-AUC and PR-AUC.
- Regression: primary metric is **R²** (what fraction of variance the model explains). Also reports RMSE and MAE.
- Clustering: Silhouette score (how well-separated the clusters are).

**Step 5 — Find the optimal decision threshold (classification only)**
By default, models predict "positive" when probability > 0.5. But for imbalanced data, a different threshold gives better F1. The system scans the Precision-Recall curve to find the threshold that maximizes F1, with a guard: the chosen threshold must achieve at least 2× the base fraud rate in precision (prevents the model from predicting everyone as fraud).

**Step 6 — Build a soft-voting ensemble**
Takes the top 3 models and averages their predicted probabilities. If the ensemble beats the best single model, it becomes the new champion.

**Step 7 — Calibrate probabilities**
Tree-based models output poorly calibrated probabilities (e.g. "94% confident" when it's really 70%). Isotonic or Sigmoid calibration corrects this so probabilities are trustworthy. Adopted only if it improves the Brier score.

**Output:**
- All trained models saved as `.joblib` files in `artifacts/<run_id>/`
- Metrics for every model
- Best model name and metric value
- Feature importances

---

### Agent 6 — Error Detection

**What it does:** Reviews the results and flags problems.

**Checks it runs:**

| Check | Threshold | Severity |
|---|---|---|
| Low classification F1 | < 0.30 | High |
| Low regression R² | < 0.10 | High |
| Overfitting | train metric - val metric > 0.15 | Medium |
| Feature explosion | > 500 features | Medium |
| Low signal warning | ROC-AUC < 0.55 | Warning |
| Data leakage | Column with > 0.95 correlation to target | Critical |
| Class imbalance | Minority < 5% of rows | Warning |

Any flagged error marked `retryable=True` triggers Agent 7 (Improvement) to try to fix it.

**Output:**
- A list of error reports with severity, root cause, and recommended fix

---

### Agent 7 — Improvement

**What it does:** Tries to improve the best model through hyperparameter tuning.

**How it runs:**
- Only runs if Agent 6 found retryable errors (e.g. low F1, overfitting)
- Up to 3 tuning rounds, stopping after 1 round with no improvement (patience=1)
- Each round uses **RandomizedSearchCV** — randomly tries 15 combinations of hyperparameters, evaluates each with 5-fold cross-validation, picks the best
- The tuned model is only kept if it beats the current champion

**Search spaces (examples):**
- XGBoost: tries different `n_estimators` (300–800), `max_depth` (4–8), `learning_rate` (0.01–0.1)
- LightGBM: tries different `num_leaves` (31–127), `min_child_samples` (10–50)
- RandomForest: tries different `n_estimators` (100–500), `max_depth` (5–None)

**Output:**
- Tuned model saved as `<ModelName>_tuned.joblib`
- Experiment history (what was tried, what scored what)

---

### Agent 8 — Finalization

**What it does:** Scores the champion on the held-out test set, generates all output files, and explains the model.

**How it runs:**

1. **Evaluate on test set** — the test set has never been seen before. This gives the honest, real-world generalization estimate.

2. **SHAP explanations** — uses SHAP (SHapley Additive exPlanations) to explain why the model makes each prediction:
   - For tree models (XGBoost, LightGBM, RandomForest): uses the fast TreeExplainer
   - For other models: uses KernelExplainer (slower, capped at 100 samples per row to keep it fast)
   - Output: global feature importance ranked by mean |SHAP value|

3. **Save all artifacts:**
   - `metadata.json` — run summary (best model, metric, threshold, test score)
   - `inference_manifest.json` — the exact feature list and preprocessing the champion expects (needed to reproduce predictions)
   - `champion_model.joblib` — the winning model, ready to load and call `.predict()`
   - Charts and visualizations

4. **Generate reports** (PDF + Markdown):
   - Dataset overview
   - Preprocessing summary
   - Feature importances
   - Model comparison table
   - Test set metrics and confusion matrix
   - SHAP explanation plots

**Output:**
- Everything in `artifacts/<run_id>/` and `reports/<run_id>/`

---

## How Data Flows Through the Pipeline

```
Your CSV
  ↓
[1] Load + Profile → raw_data.csv
  ↓
[2] Clean + Impute → cleaned_data.csv
  ↓
[3] Engineer Features → featured_data.csv
  ↓
[4] Split → train.csv  val.csv  test.csv
                ↓         ↓       (locked)
[5] Train Models  ←  score on val.csv
         ↓
[6] Audit Results
         ↓
[7] Tune Champion (if needed)
         ↓
[8] Score on test.csv → Report + champion_model.joblib
```

---

## Token Usage — The Complete Answer

| What | Tokens Used |
|---|---|
| Per pipeline run | **0 tokens** |
| LLM API calls | **None** |
| OpenAI / Anthropic / Cerebras calls per run | **None** |
| Computation per run | Pure Python + scikit-learn + XGBoost + LightGBM on your CPU |

The `llm:` section in `configs/default.yaml` (`cerebras/llama3.1-8b`) is a config stub that was never wired into the running pipeline. No inference calls are made. The "intelligence" in Axiom comes from classical ML algorithms running locally on your machine.

---

## What a Typical Run Looks Like (timing)

On a mid-range laptop with a ~50,000-row classification dataset:

| Agent | What's happening | Typical time |
|---|---|---|
| Data Collection | Reading CSV, profiling columns | 2–5s |
| Preprocessing | Cleaning, imputing, outliers | 3–8s |
| Feature Engineering | Encoding, MI scoring, selection | 5–15s |
| Data Splitting | Stratified split | < 1s |
| Model Training | Training 6–7 models in parallel | 30–120s |
| Error Detection | Rule checks on state | < 1s |
| Improvement | RandomizedSearchCV (if triggered) | 30–90s |
| Finalization | SHAP, reports | 10–30s |
| **Total** | | **~2–5 minutes** |

Large datasets (500k+ rows) can take 15–30 minutes, mostly in the training stage.

---

## Key Design Decisions Explained Simply

**Why a child process?**
Model training holds the CPU for minutes. If it ran in the same process as the API server, every HTTP request (including status polls) would queue behind the training job. The child process keeps the API responsive.

**Why multiple models instead of one?**
Different models are good at different data shapes. XGBoost is often best at tabular data with complex interactions. Logistic Regression is surprisingly competitive on linearly separable problems and is much faster. By training all of them and letting the validation set decide, you don't have to guess.

**Why not fully balance the classes with SMOTE?**
Fully balancing (50/50 fraud vs. non-fraud) makes gradient boosters overfit to the synthetic examples. Empirically, partial balancing (minority becomes ~25% of majority) works better end-to-end.

**Why tune the decision threshold instead of using 0.5?**
With 0.3% fraud rate, a model that says "not fraud" for everyone achieves 99.7% accuracy but F1 = 0. By moving the threshold down (e.g. predict fraud when probability > 0.12), you accept more false positives to catch more real fraud. The threshold is chosen to maximize F1 on the validation set with a precision floor so the model doesn't just flag everyone.

**Why SHAP?**
It answers "why did the model say this?". For fraud detection: "this transaction was flagged because the amount was 8.3 standard deviations above this cardholder's average and it happened at 3 AM." Without SHAP, you'd have a black box.
