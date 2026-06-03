# Axiom — Complete Project Documentation
### Everything you need to understand, explain, and present this project

---

## Table of Contents

1. [What is Axiom?](#1-what-is-axiom)
2. [The Big Picture — How it All Connects](#2-the-big-picture)
3. [The Pipeline State — The Shared Memory](#3-the-pipeline-state)
4. [The 8 Agents — In Depth](#4-the-8-agents-in-depth)
   - Agent 1: Data Collection
   - Agent 2: Preprocessing
   - Agent 3: Feature Engineering
   - Agent 4: Data Splitting
   - Agent 5: Model Training
   - Agent 6: Error Detection
   - Agent 7: Improvement
   - Agent 8: Finalization
5. [The Model Registry](#5-the-model-registry)
6. [The Backend API — Every Endpoint Explained](#6-the-backend-api)
7. [The Frontend API Client — How Calls Are Made](#7-the-frontend-api-client)
8. [The Two Modes — Free vs Pro](#8-the-two-modes)
9. [What Gets Saved to Disk](#9-what-gets-saved-to-disk)
10. [Technology Stack Summary](#10-technology-stack-summary)

---

## 1. What is Axiom?

Axiom is an **autonomous data scientist**. It is a software system that takes a raw CSV file
as input and, without any manual intervention, performs the entire machine learning process
end-to-end and hands you back a trained model, accuracy metrics, and a full report.

Think of it like this: normally a data science team would spend days cleaning data,
engineering features, picking models, and tuning them. Axiom does all of that automatically
in minutes using a pipeline of 8 specialized AI agents.

### Core Design Philosophy

> **"LLM reasons, code executes."**

- The **LLM (Llama 3.1 on Cerebras)** acts as the decision-maker. It reads task descriptions
  and decides what strategy to use.
- The **Python code** (scikit-learn, pandas, XGBoost etc.) actually runs the math — training
  models, computing metrics, transforming data.
- The LLM never directly manipulates data. It calls Python tool functions, which do the
  actual deterministic work.

---

## 2. The Big Picture

Here is the flow of the entire system from a user clicking a button to getting results:

```
USER (browser)
    |
    | uploads CSV file, selects target column, clicks "Launch"
    v
FRONTEND (Next.js — runs on port 3000)
    |
    | sends HTTP POST request to /api/run
    v
BACKEND (FastAPI — runs on port 8000)
    |
    | creates a unique run_id
    | starts a background thread
    | returns run_id to frontend immediately
    v
BACKGROUND THREAD
    |
    | creates a PipelineState object (shared memory for the whole run)
    | calls agent_runner.run_full_pipeline()
    v
CREWAI ORCHESTRATOR
    |
    | runs all 8 agents in order, one after the other
    | each agent receives the same PipelineState, modifies it, passes it forward
    v
AGENT 1 → AGENT 2 → AGENT 3 → ... → AGENT 8
    |
    | each agent writes its results into PipelineState
    | data files (CSV) are saved to disk between stages
    v
FINALIZATION
    |
    | saves best model file (.joblib)
    | generates SHAP explanations
    | writes markdown report
    v
FRONTEND POLLS /api/status/{run_id} every 3 seconds
    |
    | gets live stage updates and logs
    | when status = "completed", fetches /api/results/{run_id}
    v
RESULTS PAGE shown to user
```

The frontend and backend communicate **entirely through HTTP API calls**.
The backend runs the ML pipeline in a background thread and exposes live status
through a polling endpoint. The frontend checks that endpoint every few seconds
and updates the UI in real time.

---

## 3. The Pipeline State

**File:** `core/state.py`

The `PipelineState` is the single most important object in the entire system.
It is a Python object (Pydantic model) that carries all information about one
pipeline run from start to finish. Every agent reads from it and writes to it.

Think of it as a relay baton — each agent picks it up, adds its results, and
hands it to the next agent.

### What the PipelineState Contains

```
PipelineState
├── Identity
│   ├── run_id              → unique ID for this run (e.g. "run_20240512_abc123")
│   ├── problem_type        → "classification", "regression", or "clustering"
│   └── target_column       → column name the user wants to predict
│
├── Data Paths (files saved to disk between stages)
│   ├── raw_data_path       → original uploaded CSV
│   ├── cleaned_data_path   → CSV after preprocessing
│   ├── featured_data_path  → CSV after feature engineering
│   ├── train_path          → training split CSV
│   ├── val_path            → validation split CSV
│   └── test_path           → test split CSV
│
├── Dataset Info
│   └── dataset_metadata    → row count, column names, dtypes, null counts etc.
│
├── Stage Summaries (each agent fills this in)
│   ├── preprocessing_summary      → rows removed, nulls filled, quality score
│   └── feature_engineering_summary → features created/removed, encoding used
│
├── Model Results
│   ├── model_results       → list of all models trained with their metrics
│   ├── best_model_name     → e.g. "XGBClassifier"
│   ├── best_model_path     → path to the .joblib file
│   ├── best_metric_name    → e.g. "f1"
│   └── best_metric_value   → e.g. 0.9234
│
├── Error Tracking
│   └── error_reports       → list of issues found (overfitting, leakage, etc.)
│
├── Progress Tracking
│   ├── current_stage       → which agent is currently running
│   ├── completed_stages    → list of stages that finished successfully
│   ├── stage_timestamps    → start/end time of each stage
│   └── failed              → True if the pipeline crashed
│
└── Retry / Experiment History
    ├── retry_count         → how many improvement iterations ran
    └── experiment_history  → record of each tuning attempt and its score
```

### Why This Design?

Instead of passing data through function arguments and return values, everything
flows through this single shared state object. This makes:
- **Checkpointing** easy: just serialize the whole state to JSON and save it
- **Debugging** easy: one object tells you everything that happened
- **Retrying** easy: restore state from checkpoint and resume
- **The API** easy: the backend just reads the state to answer status/results queries

---

## 4. The 8 Agents In Depth

Each agent follows the same structure:
1. A **Service class** that contains the actual Python logic
2. A **`run()` method** that takes `PipelineState` and returns updated `PipelineState`
3. A **tool function** (e.g. `collect_data()`) that CrewAI calls and that wraps the service
4. An **`init_*()`** function that sets up the service with config before the pipeline starts

---

### Agent 1: Data Collection

**File:** `agents/data_collection/tools.py`
**Purpose:** Load the raw data, understand what's in it, detect what kind of ML problem it is.

#### What it does, step by step:

**Step 1 — Load the CSV**
- Reads the file using pandas
- If the file is over 100MB, reads it in chunks to avoid running out of RAM
- Supports .csv, .tsv, and .txt formats

**Step 2 — Optimize memory**
- Downcasts numeric columns to smaller types (e.g. float64 → float32)
- This can reduce RAM usage by 50-70% on large datasets

**Step 3 — Validate the data**
- Checks for common problems: too many columns, completely empty columns, etc.
- Any issues are stored in `state.data_quality_flags["validation_issues"]`

**Step 4 — Detect Problem Type**
This is the most important decision in the whole pipeline. The agent looks at the
target column (if the user gave one) and uses these rules:

```
No target column given → CLUSTERING (unsupervised, no labels)

Target column is text (strings) → CLASSIFICATION

Target column is numeric:
  - 20 or fewer unique values → CLASSIFICATION (e.g. 0/1, or 1-5 ratings)
  - More than 20 unique values AND ratio > 5% of dataset → REGRESSION (e.g. house prices)
  - Otherwise → CLASSIFICATION
```

This decision flows through the entire rest of the pipeline — it determines which
models get trained, which metrics are used, and how the data gets split.

**Step 5 — Profile the dataset**
Creates a `DatasetMetadata` object containing:
- Number of rows and columns
- Data type of each column (numeric, categorical, datetime, boolean)
- How many null (missing) values are in each column
- Number of unique values per column

**Step 6 — Compute quality score**
A number from 0.0 to 1.0 measuring how clean the data is.
Penalties for: missing values, duplicate rows, constant columns.

**Step 7 — Check for target leakage**
Looks for columns that are suspiciously correlated (>0.95) with the target column.
If a feature almost perfectly predicts the target, it might be a data leak
(e.g. "payment_made" predicting "loan_defaulted" — that's circular information).

**Step 8 — Check for class imbalance** (classification only)
If one class has way fewer samples than others (e.g. 99% "not fraud", 1% "fraud"),
it logs a warning because this will skew model training.

**What it writes to PipelineState:**
- `state.problem_type` → "classification", "regression", or "clustering"
- `state.dataset_metadata` → full profile of the dataset
- `state.data_quality_flags` → quality score, issues found

---

### Agent 2: Preprocessing

**File:** `agents/preprocessing/tools.py`
**Purpose:** Clean the data so it's ready for machine learning. Handles all the
messy real-world issues that raw data always has.

#### What it does, step by step:

**Step 1 — Fix data types**
Pandas sometimes reads a numeric column as text. This step:
- Tries to convert each text column to a number (if >80% of values look numeric)
- Tries to parse date columns (if >80% of values look like dates)
- Detects true/false columns written as "yes/no", "true/false", "1/0"

**Step 2 — Remove duplicates**
Finds rows that are exact copies of each other and removes them.
Keeps the first occurrence.

**Step 3 — Handle missing values**
For each column with missing data, it picks a strategy:

```
Column missing > 70% of values → DROP the column entirely
Column is numeric with some nulls → FILL with median value
Column is text with some nulls → FILL with most common value (mode)
Target column has nulls → DROP those rows (can't train on unlabeled data)
```

**Step 4 — Handle outliers (IQR Winsorization)**
For numeric columns, extreme values (outliers) are clipped to a safe range.

The IQR method:
```
Q1 = 25th percentile value
Q3 = 75th percentile value
IQR = Q3 - Q1
Lower bound = Q1 - (1.5 × IQR)
Upper bound = Q3 + (1.5 × IQR)
Any value below lower bound → set to lower bound
Any value above upper bound → set to upper bound
```
This doesn't delete outliers — it clips them to be less extreme. This is called "winsorization."

**Step 5 — Detect high cardinality**
Flags categorical columns with too many unique values (e.g. a "user_id" column
with 50,000 unique values). These columns need special handling in feature engineering.

**Step 6 — Save cleaned data**
Saves the cleaned DataFrame to `artifacts/{run_id}/cleaned_data.csv`
Updates `state.cleaned_data_path` to point to this new file.

**What it writes to PipelineState:**
- `state.cleaned_data_path` → path to the saved cleaned CSV
- `state.preprocessing_summary` → rows before/after, duplicates removed, nulls filled,
  outliers handled, quality score after cleaning

---

### Agent 3: Feature Engineering

**File:** `agents/feature_engineering/tools.py`
**Purpose:** Transform raw columns into a format that ML models can learn from.
Remove useless features. Keep only the most informative ones.

#### What it does, step by step:

**Step 1 — Encode the target column** (classification only)
If the target is text (e.g. "cat", "dog", "bird"), it gets converted to numbers
(0, 1, 2) using a LabelEncoder. ML models need numbers.

**Step 2 — Extract datetime features**
If any column contains dates/times (e.g. "2024-03-15 14:30:00"), it extracts:
- Year → new column `colname_year`
- Month → new column `colname_month`
- Day of week → new column `colname_dow` (Monday=0, Sunday=6)
- Hour → new column `colname_hour`
Then the original datetime column is dropped. Models can't use raw timestamps
but they can learn from "purchases spike on Fridays" or "failures peak in January."

**Step 3 — Encode categorical columns**
Text columns need to become numbers. There are two approaches:

```
Column has ≤ 15 unique values → ONE-HOT ENCODING
  Example: "color" with values ["red", "blue", "green"]
  Becomes 3 new columns: color_red (0/1), color_blue (0/1), color_green (0/1)

Column has > 15 unique values → LABEL ENCODING
  Example: "city" with 200 unique city names
  Each city gets assigned a number: New York=0, London=1, Tokyo=2...
```

One-hot is more accurate for low-cardinality columns. Label encoding is used
for high-cardinality columns to avoid creating hundreds of new columns.

**Step 4 — Remove low-variance features**
A feature that is the same value (or nearly the same) for almost every row
doesn't help the model learn anything. Example: a column "country" that is
"India" for 99.9% of rows. This step removes such features.

**Step 5 — Remove highly correlated features**
If two features are almost identical (correlation > 0.95), one is redundant.
Example: "height_cm" and "height_inches" — keeping both would double-count
the same information. One gets dropped.

**Step 6 — Feature selection (SelectKBest)**
From the remaining features, picks the top 20 most informative ones using
mutual information — a statistical measure of how much knowing a feature's
value reduces uncertainty about the target.
This prevents the model from being overwhelmed by too many weak features.

**Step 7 — Scale numeric features**
ML models like Logistic Regression and SVM work better when all features
are on the same scale. Options:
- **Standard scaling:** subtract mean, divide by standard deviation → range roughly [-3, 3]
- **MinMax scaling:** shift to range [0, 1]
- **Robust scaling:** uses median and IQR, resistant to outliers

Tree models (Random Forest, XGBoost) don't need scaling, but it doesn't hurt them either.

**Step 8 — Save engineered data**
Saves to `artifacts/{run_id}/featured_data.csv`

**What it writes to PipelineState:**
- `state.featured_data_path` → path to the engineered CSV
- `state.selected_features` → list of final feature column names
- `state.feature_importances` → mutual information score for each feature
- `state.feature_engineering_summary` → full summary of what changed

---

### Agent 4: Data Splitting

**File:** `agents/splitting/tools.py`
**Purpose:** Divide the data into training set, validation set, and test set so
that model performance can be measured honestly on data the model never saw.

#### Why we split data:

If you train a model on all your data and then test it on the same data,
it will look amazing but will fail on new real data. You must test on
held-out data the model was never shown.

#### The split ratios:

```
Training set    70%  → Used to train the model
Validation set  15%  → Used during development to compare models and tune them
Test set        15%  → Used once at the very end to get final honest performance
```

#### Stratified vs Random split:

```
CLASSIFICATION → Stratified split
  Makes sure each class appears in each split at the same ratio as the full dataset.
  Example: If 30% of data is class=1, then 30% of train, 30% of val, 30% of test is class=1.
  Without stratification, you could get bad splits where val has no rare class examples.

REGRESSION → Random split
  No stratification needed — target is continuous (not discrete classes).

CLUSTERING → No split at all
  There are no labels, so you train on all the data.
```

#### What it saves:
Three separate CSV files:
- `artifacts/{run_id}/train.csv`
- `artifacts/{run_id}/val.csv`
- `artifacts/{run_id}/test.csv`

**What it writes to PipelineState:**
- `state.train_path`, `state.val_path`, `state.test_path`
- `state.split_ratios` → {"train": 0.70, "val": 0.15, "test": 0.15}

---

### Agent 5: Model Training

**File:** `agents/training/tools.py`
**Purpose:** Train multiple ML models simultaneously, evaluate each one, and
pick the best performer.

#### The Model Registry

Before understanding training, you need to understand the Model Registry
(`core/model_registry.py`). It's a dictionary of all available models,
organized by problem type. Each entry contains:
- How to create the model (the factory function)
- Default hyperparameters
- Search space for hyperparameter tuning (used by Agent 7)

**For Classification problems, these models are trained:**
- `LogisticRegression` — a simple linear model, fast, good baseline
- `RandomForestClassifier` — ensemble of 100 decision trees, robust
- `SVC` — Support Vector Machine, good for smaller datasets
- `XGBClassifier` — gradient boosting, often the best performer
- `LGBMClassifier` — LightGBM, another gradient boosting, very fast

**For Regression problems:**
- `LinearRegression` — simplest regression model
- `Ridge` — linear regression with regularization to prevent overfitting
- `RandomForestRegressor`
- `XGBRegressor`
- `LGBMRegressor`

**For Clustering:**
- `KMeans` — groups data into K clusters (default K=3)
- `DBSCAN` — density-based clustering, finds arbitrarily shaped clusters
- `AgglomerativeClustering` — hierarchical clustering

#### How training works:

**Step 1 — Load data**
Reads `train.csv` and `val.csv` into memory.

**Step 2 — Parallel training**
All models are trained at the same time using Python's `ThreadPoolExecutor`.
If you have 5 models to train, they all start simultaneously and the
system waits for all of them to finish. This saves significant time.

Each model training records:
- Validation metrics (the real performance measure)
- Training metrics (to detect overfitting later)
- Training time in seconds
- RAM used (in MB)

**Step 3 — Evaluate with the right metric**
Different problem types use different "primary metrics" to decide which model is best:
```
Classification → F1 score (weighted average across all classes)
Regression     → R² score (how much variance the model explains, 1.0 = perfect)
Clustering     → Silhouette score (how well-separated the clusters are)
```

For classification, additional metrics are also computed:
accuracy, precision, recall, F1, ROC-AUC (if binary)

For regression: R², RMSE (Root Mean Square Error), MAE (Mean Absolute Error)

**Step 4 — Save every model**
Every trained model is saved to disk as a `.joblib` file:
`artifacts/{run_id}/models/RandomForestClassifier.joblib`
`artifacts/{run_id}/models/XGBClassifier.joblib`
etc.

**Step 5 — Select the best model**
The model with the highest primary metric on the validation set is chosen.
Its path is stored in `state.best_model_path` and its name in `state.best_model_name`.

**What it writes to PipelineState:**
- `state.model_results` → list of ModelResult objects (one per model)
- `state.best_model_name` → name of the winning model
- `state.best_model_path` → file path to the winning model's .joblib file
- `state.best_metric_name` → e.g. "f1"
- `state.best_metric_value` → e.g. 0.9234

---

### Agent 6: Error Detection

**File:** `agents/error_detection/tools.py`
**Purpose:** Act as a quality auditor. Review everything that happened in the
pipeline so far and flag any problems with their severity and suggested fix.

This agent doesn't fix problems — it only **detects and reports them**.
The report goes into `state.error_reports` and some issues trigger Agent 7 (Improvement).

#### Checks it performs:

**Check 1 — Low performance**
```
Classification: If best F1 < 0.30 → HIGH severity warning
Regression: If best R² < 0.10 → HIGH severity warning
```
Suggests: hyperparameter tuning, better feature engineering, class balancing.

**Check 2 — Overfitting**
Compares training score vs validation score for each model.
```
If train_f1 - val_f1 > 0.15 → MEDIUM severity warning for that model
```
Overfitting means the model memorized the training data instead of learning patterns.
Suggests: add regularization, simplify the model, get more training data.

**Check 3 — Feature explosion**
```
If number of features after engineering > 500 → MEDIUM severity warning
```
Too many features slow down training and can hurt performance (curse of dimensionality).
Suggests: increase aggressiveness of feature selection.

**Check 4 — Class imbalance** (passed from Agent 1)
If the minority class is very rare (e.g. only 1% of samples), flags it.
Suggests: use SMOTE (Synthetic Minority Oversampling Technique), class weights.

**Check 5 — All models failed**
If every single model crashed during training:
```
CRITICAL severity — the whole experiment is in trouble
```

**Check 6 — Low data quality**
If the post-preprocessing quality score < 0.50:
```
MEDIUM severity warning — data may still have serious issues
```

**What it writes to PipelineState:**
- Appends `ErrorReport` objects to `state.error_reports`
- Each report has: severity level, stage it belongs to, root cause description,
  recommended fix, and whether it can be retried

---

### Agent 7: Improvement

**File:** `agents/improvement/tools.py`
**Purpose:** Take the best model from Agent 5 and try to make it even better
through hyperparameter tuning.

Hyperparameters are the configuration settings of a model that you set before
training starts. Examples:
- Random Forest: `n_estimators` (how many trees), `max_depth` (how deep each tree is)
- XGBoost: `learning_rate`, `max_depth`, `n_estimators`
- Logistic Regression: `C` (regularization strength), `solver`

The default hyperparameters from Agent 5 are a good starting point.
Agent 7 searches for better combinations.

#### Two tuning methods:

**Method 1 — RandomizedSearchCV** (always available)
```
Takes the search space defined in the model registry.
Example for RandomForest:
  n_estimators: [50, 100, 200, 300]
  max_depth: [5, 10, 20, None]
  min_samples_split: [2, 5, 10]

Randomly samples combinations from this space (50 trials by default).
Uses 3-fold cross-validation on training data.
Picks the combination with the best CV score.
```

**Method 2 — Optuna** (if installed)
```
A smarter search algorithm that learns from previous trials.
Uses Bayesian optimization to focus the search on promising regions.
More efficient than random search for complex search spaces.
```

#### Does improvement always help?

No. After tuning, the agent compares the tuned model's score against the
original best score. If the tuned model doesn't improve, the original is kept.
The agent logs "No improvement" and moves on.

#### Retry cap:
The improvement agent checks `state.retry_count` vs `state.max_retries` (default: 3).
If max retries is reached, it skips tuning. This prevents infinite loops.

**What it writes to PipelineState:**
- Updates `state.best_model_path` if an improved model is found
- Updates `state.best_metric_value` if score improved
- Increments `state.retry_count`
- Appends to `state.experiment_history`

---

### Agent 8: Finalization

**File:** `agents/finalization/tools.py`
**Purpose:** Package everything up. Save all artifacts, generate the SHAP
explainability analysis, and write the final human-readable report.

#### What it saves:

**1. metadata.json**
A summary of the whole run: run ID, problem type, target column, best model name,
best score, timestamps, all completed stages.

**2. all_metrics.json**
Every model trained, their full metrics (val + train), training time, and
hyperparameters. Useful for deep analysis.

**3. feature_importances.json**
The mutual information scores from Agent 3, sorted from most to least important.

**4. SHAP explanations (shap_importance.json)**
SHAP = SHapley Additive exPlanations. This answers the question:
*"For each feature, how much does it push the prediction up or down?"*

Implementation:
```
Load the best model (.joblib file)
Load a sample of training data (max 500 rows for speed)
Create a SHAP explainer appropriate for the model type:
  - TreeExplainer → for tree-based models (Random Forest, XGBoost, LightGBM) — very fast
  - LinearExplainer → for linear models (Logistic Regression, Ridge)
  - KernelExplainer → universal fallback, works on any model, but slow

Compute SHAP values for each sample
Average the absolute SHAP values across all samples
The result: a ranked list of features by importance
```

Example output: `{"age": 0.342, "income": 0.289, "credit_score": 0.187, ...}`
This tells you: "age" had the biggest average impact on predictions.

**5. pipeline_report.md**
A full markdown report written in plain language covering:
- Executive summary (best model, score)
- Dataset statistics
- What preprocessing did (rows before/after, nulls filled)
- What feature engineering did (features created/removed)
- Model comparison table (all models, all metrics, training times)
- SHAP feature importance ranking (top 15 features)
- Issues detected (overfitting warnings, etc.)
- Experiment history (iterations of tuning)
- File paths of all saved artifacts

**6. experiment_history.json**
Record of every tuning iteration: which model, what score, what improvements applied.

**7. error_reports.json**
All ErrorReport objects serialized to JSON for analysis.

**8. final_state.json**
The entire PipelineState object serialized to JSON — a complete snapshot of
everything that happened in this run. Useful for debugging or resuming.

---

## 5. The Model Registry

**File:** `core/model_registry.py`

The Model Registry is a central dictionary of all available ML models.
Instead of hardcoding model instantiation everywhere, all models are registered
once and looked up by problem type and name.

Each registered model has:
- `name` — identifier (e.g. "XGBClassifier")
- `problem_type` — which problems it works on (classification/regression/clustering)
- `factory` — the Python class to call to create the model
- `default_params` — the parameters used during Agent 5 (initial training)
- `search_space` — the parameter ranges used by Agent 7 (tuning)
- `requires_scaling` — whether this model needs StandardScaler (linear models do)
- `supports_probabilities` — whether it can output prediction probabilities (needed for ROC-AUC)

This design means adding a new ML model is just adding one `registry.register()` call.

---

## 6. The Backend API

**File:** `api.py` (FastAPI, runs on port 8000)

The backend exposes all functionality as HTTP REST endpoints.
Every response is JSON. Authentication uses Bearer tokens (JWT-style).

### Authentication Endpoints

---

#### `POST /api/auth/signup`
Creates a new user account.

**Request body (JSON):**
```json
{
  "email": "user@example.com",
  "username": "john",
  "password": "mypassword123"
}
```

**What happens internally:**
- Checks email and username are not already taken
- Hashes the password using bcrypt (never stores plain text)
- Creates a User record in SQLite database
- Generates a session token
- Returns the token and user info

**Response:**
```json
{
  "status": "ok",
  "token": "abc123...",
  "user": { "id": 1, "username": "john", "email": "user@example.com" }
}
```

---

#### `POST /api/auth/login`
Authenticates an existing user.

**Request body (JSON):**
```json
{ "email": "user@example.com", "password": "mypassword123" }
```

**What happens:**
- Looks up user by email in database
- Verifies password hash matches using bcrypt
- Creates a new session token (UUID) and stores it in the database
- Returns token

**Response:** Same structure as signup.

---

#### `POST /api/auth/logout`
Invalidates the current session token.

**Headers required:** `Authorization: Bearer <token>`

Deletes the session from the database.

---

#### `GET /api/auth/history`
Returns the user's past pipeline runs.

**Headers required:** `Authorization: Bearer <token>`

Queries the `PromptHistory` table in SQLite for all runs associated with the user.

---

### Pipeline Endpoints

---

#### `POST /api/upload`
Uploads a CSV file, profiles it, and generates initial visualizations.

**Request:** `multipart/form-data` with a `file` field containing the CSV.

**What happens:**
1. Saves the file to `uploads/` directory
2. Reads the first 5 rows for preview
3. Generates automatic visualizations (correlation heatmap, distributions,
   missing value chart, box plots, target distribution if target provided)
4. Returns all of this to the frontend immediately

**Response:**
```json
{
  "filename": "titanic.csv",
  "path": "uploads/titanic_abc123.csv",
  "columns": ["survived", "pclass", "age", "fare", ...],
  "dtypes": { "survived": "int64", "age": "float64", ... },
  "n_rows": 891,
  "preview": [ {"survived": 0, "age": 22.0, ...}, ... ],
  "visualizations": [
    {
      "name": "Correlation Heatmap",
      "type": "correlation",
      "description": "Pearson correlation between numeric columns",
      "base64_png": "iVBORw0KGgoA...",
      "category": "basic"
    },
    ...
  ]
}
```

The `base64_png` is the chart image encoded as a Base64 string.
The frontend decodes it and displays it directly in the browser using:
`<img src="data:image/png;base64,{base64_png}" />`

---

#### `POST /api/run`
Starts the full 8-agent pipeline in the background.

**Request:** `multipart/form-data`
```
data_path: "uploads/titanic_abc123.csv"
target_column: "survived"
```

**What happens:**
1. Validates the file exists
2. Generates a unique `run_id` (e.g. "run_20240512_143022_abc")
3. Creates an entry in `_active_runs` dict (in-memory store for all active runs)
4. Records run in SQLite `PromptHistory` table
5. **Starts a background thread** — returns immediately, does NOT wait for pipeline
6. The background thread calls `run_full_pipeline()` which runs all 8 agents

**Response (immediate):**
```json
{ "status": "started", "run_id": "run_20240512_143022_abc" }
```

The pipeline is still running at this point. The frontend uses the `run_id`
to poll for updates.

---

#### `GET /api/status/{run_id}`
Returns the current status of a running or completed pipeline.

**Headers required:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "run_id": "run_20240512_143022_abc",
  "status": "running",
  "current_stage": "model_training",
  "completed_stages": ["data_collection", "preprocessing", "feature_engineering", "data_splitting"],
  "started_at": "2024-05-12T14:30:22Z",
  "logs": [
    { "time": "14:30:22", "msg": "Starting data_collection..." },
    { "time": "14:30:24", "msg": "✓ data_collection completed (1.8s)" },
    { "time": "14:30:25", "msg": "Starting preprocessing..." }
  ],
  "error": null
}
```

The frontend calls this endpoint every 3 seconds to update the progress UI.
`completed_stages` is used to show which agents have a checkmark.
`current_stage` is used to show a spinner on the currently running agent.

---

#### `GET /api/results/{run_id}`
Returns the full results after the pipeline completes.

**Headers required:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "run_id": "...",
  "status": "completed",
  "problem_type": "classification",
  "target_column": "survived",
  "best_model": "XGBClassifier",
  "best_metric_name": "f1",
  "best_metric_value": 0.8734,
  "dataset": { "rows": 891, "columns": 12, "quality_score": 0.82 },
  "preprocessing": { "rows_before": 891, "rows_after": 889, "quality_score": 0.91, "duplicates_removed": 2 },
  "features": { "before": 12, "after": 8, "selected": ["pclass", "age", "fare", ...] },
  "models": [
    { "name": "XGBClassifier", "status": "trained", "metrics": {"f1": 0.8734, "accuracy": 0.8876}, "time_s": 3.2, "is_best": true },
    { "name": "RandomForestClassifier", "status": "trained", "metrics": {"f1": 0.8512}, "time_s": 5.1 },
    ...
  ],
  "errors": [
    { "severity": "MEDIUM", "type": "overfitting", "cause": "RandomForest: train_f1=0.98, val_f1=0.85", "fix": "Add regularization" }
  ],
  "artifacts": { "metadata": "artifacts/run_.../metadata.json", "shap": "artifacts/run_.../shap_importance.json" },
  "reports": { "final_report": "reports/run_.../pipeline_report.md" }
}
```

---

#### `GET /api/report/{run_id}`
Returns the full markdown report text.

**Response:**
```json
{ "report": "# Pipeline Report\n\n| Property | Value |...\n\n## Executive Summary..." }
```

The frontend renders this markdown as HTML in the results page.

---

#### `GET /api/shap/{run_id}`
Returns SHAP feature importance values.

**Response:**
```json
{
  "shap_importance": {
    "pclass": 0.342,
    "age": 0.289,
    "fare": 0.187,
    "sex_male": 0.156,
    ...
  }
}
```

Frontend uses these values to render a bar chart showing feature importance.

---

### Enterprise-Only Endpoints

---

#### `POST /api/init-run`
Creates a run context WITHOUT starting any pipeline.
Used by the Agent Console so a user can initialize a run and then
trigger individual agents one at a time.

**Request:** `multipart/form-data` — same as `/api/run`

**Response:** `{ "run_id": "..." }`

The PipelineState is created and stored but no agents run yet.

---

#### `POST /api/agent/{agentName}/run`
Runs a single named agent on an existing run.

**URL parameter:** `agentName` — one of: `data_collection`, `preprocessing`,
`feature_engineering`, `data_splitting`, `model_training`, `error_detection`,
`improvement`, `finalization`

**Request body (JSON):**
```json
{
  "run_id": "run_20240512_143022_abc",
  "config": {}
}
```

**What happens:**
- Finds the run in `_active_runs` (must exist from `/api/init-run`)
- Restores the PipelineState
- Runs ONLY that specific agent
- Returns the agent's output

**Response:**
```json
{
  "status": "completed",
  "agent_output": {
    "agent_id": "preprocessing",
    "status": "completed",
    "duration_seconds": 2.3,
    "memory_mb": 45.2,
    "summary": { "rows_before": 891, "rows_after": 889 },
    "metrics": { "quality_score": 0.91 },
    "logs": [ ... ],
    "artifacts": { ... }
  }
}
```

---

#### `POST /api/workflow/run`
Runs a custom sequence of agents (not necessarily all 8, not necessarily in default order).
This is what the Workflow Builder uses.

**Request body (JSON):**
```json
{
  "agents": ["data_collection", "preprocessing", "model_training"],
  "data_path": "uploads/titanic_abc123.csv",
  "target_column": "survived",
  "problem_type": "classification"
}
```

The `problem_type` field is optional — if not provided and `data_collection` is not
in the agent list, the pipeline will fail with a clear error asking you to either
include `data_collection` or specify `problem_type` manually.

**Response:** Same as `/api/run` — returns `run_id` immediately.

---

#### `GET /api/runs`
Returns a list of all past and active runs for the current user.

**Response:**
```json
{
  "runs": [
    {
      "run_id": "...",
      "status": "completed",
      "started_at": "2024-05-12T14:30:22Z",
      "mode": "enterprise",
      "best_model": "XGBClassifier",
      "best_metric_value": 0.8734,
      "completed_stages": ["data_collection", "preprocessing", ...]
    },
    ...
  ]
}
```

---

#### `POST /api/visualizations/{run_id}/generate`
Generates a specific visualization on demand.

**Request body (JSON):**
```json
{ "type": "pca_projection", "params": { "n_components": 2 } }
```

**Supported types:**
- `correlation` — Pearson correlation heatmap
- `distributions` — histograms for all numeric columns
- `missing_values` — bar chart of missing value percentages
- `box_plots` — box plots for outlier visualization
- `target_distribution` — class distribution for classification
- `pair_plot` — scatter matrix of top features (Pro only)
- `pca_projection` — PCA 2D projection of the dataset (Pro only)

**Response:**
```json
{
  "name": "PCA Projection",
  "type": "pca_projection",
  "description": "2D PCA projection of features colored by target",
  "base64_png": "iVBORw0KGgo...",
  "category": "advanced"
}
```

---

#### `GET /api/export/{run_id}/excel`
Downloads all agent outputs for a run as an Excel file (.xlsx).

Returns the file as a binary download. The frontend creates a temporary link
and triggers the browser's download dialog.

---

## 7. The Frontend API Client

**File:** `frontend/src/lib/api.ts`

This TypeScript file is the single place where all HTTP calls from the frontend
to the backend are made. Every API call goes through here — no component
directly calls `fetch()`.

### Authentication pattern

Every request (except upload and login/signup) includes the user's token:

```typescript
// Gets the token from Zustand auth store
const token = useAuthStore.getState().token;

// Adds it to every request header
headers.set("Authorization", `Bearer ${token}`);
```

This is done automatically by the `authFetch()` helper function.
Components never have to think about authentication — they just call
functions like `startPipeline()` and authentication is handled transparently.

### Error handling pattern

All API calls go through `handleResponse()`:

```typescript
async function handleResponse(res: Response, defaultError: string) {
  if (!res.ok) {
    // Try to get error message from JSON body
    const data = await res.json();
    throw new Error(data.detail || data.message || defaultError);
  }
  return res.json();
}
```

If the backend returns an error (status 400, 401, 500, etc.), this function
extracts the error message and throws it as a JavaScript Error.
The calling component catches it and shows it to the user.

If the backend is down entirely (502, 504), it shows:
"Backend server is unavailable. Please ensure the Python backend is running."

### File upload

```typescript
export async function uploadDataset(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`/api/upload`, { method: "POST", body: form });
  return handleResponse(res, "Upload failed");
}
```

Note: upload uses plain `fetch`, not `authFetch` — uploading is intentionally
kept public (no auth required) so the server doesn't need to verify a token
just to receive a file.

### Starting a pipeline

```typescript
export async function startPipeline(dataPath: string, targetColumn?: string) {
  const form = new FormData();
  form.append("data_path", dataPath);
  if (targetColumn) form.append("target_column", targetColumn);
  const res = await authFetch(`/api/run`, { method: "POST", body: form });
  return handleResponse(res, "Failed to start");
}
```

`dataPath` is the path the backend returned from the upload call.
The frontend never holds the file data after upload — it just passes the
server-side path back.

### Live status polling

The pipeline monitor page does this:

```typescript
useEffect(() => {
  const poll = setInterval(async () => {
    const status = await getStatus(runId);
    setCurrentStage(status.current_stage);
    setCompletedStages(status.completed_stages);
    setLogs(status.logs);

    if (status.status === "completed" || status.status === "failed") {
      clearInterval(poll);
      if (status.status === "completed") {
        const results = await getResults(runId);
        setResults(results);
      }
    }
  }, 3000); // poll every 3 seconds

  return () => clearInterval(poll);
}, [runId]);
```

The component starts an interval that calls `/api/status/{run_id}` every 3 seconds.
When the status becomes "completed", it stops polling and fetches the full results.
When the component unmounts (user navigates away), the interval is cleared.

### Running a workflow (Pro mode)

```typescript
export async function runWorkflow(
  agentList: string[],
  dataPath: string,
  targetColumn?: string,
  config?: Record<string, unknown>
) {
  const res = await authFetch(`/api/workflow/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      agents: agentList,      // e.g. ["data_collection", "model_training"]
      data_path: dataPath,
      target_column: targetColumn,
      config,
    }),
  });
  return handleResponse(res, "Workflow failed");
}
```

The user selects which agents they want from the Workflow Builder UI.
Their selections are put into `agentList` and sent to the backend.

### Running a single agent (Pro Agent Console)

```typescript
export async function runSingleAgent(agentName: string, runId: string) {
  const res = await authFetch(`/api/agent/${agentName}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId }),
  });
  return handleResponse(res, "Agent run failed");
}
```

The Agent Console first calls `initRun()` to create the run context, gets back
a `run_id`, and then calls `runSingleAgent()` for each agent the user triggers.

---

## 8. The Two Modes — Free vs Pro

### Free Mode

- Single page at `/free`
- User uploads CSV, picks target, clicks one button
- Calls `POST /api/run` → pipeline runs all 8 agents automatically
- User is taken to `/free/pipeline/{run_id}` which shows a live progress screen
- When done, shown at `/free/results/{run_id}`
- No configuration options — fully automatic
- The "Pro features" section on the page shows locked feature cards to encourage upgrade

### Pro Mode (Enterprise)

Enabled by the mode toggle switch. Uses a sidebar layout with these pages:

| Page | URL | Purpose |
|------|-----|---------|
| Command Center | `/enterprise` | Dashboard with run launcher + experiment history |
| Workflow Builder | `/enterprise/workflow` | Select custom agent sequences |
| Agent Console | `/enterprise/agents` | Run individual agents step by step |
| Experiments | `/enterprise/experiments` | List and compare all past runs |
| Analytics | `/enterprise/analytics` | Generate visualizations on demand |
| Artifacts | `/enterprise/artifacts` | Browse saved models and reports |

Pro mode uses the exact same backend API — there is no separate Pro backend.
The difference is that the frontend gives users more control and flexibility.

---

## 9. What Gets Saved to Disk

After a complete pipeline run, the following directory structure is created:

```
artifacts/{run_id}/
├── cleaned_data.csv          ← output of Agent 2 (Preprocessing)
├── featured_data.csv         ← output of Agent 3 (Feature Engineering)
├── train.csv                 ← output of Agent 4 (70% of data)
├── val.csv                   ← output of Agent 4 (15% of data)
├── test.csv                  ← output of Agent 4 (15% of data)
├── metadata.json             ← run summary (best model, score, timestamps)
├── all_metrics.json          ← every model's metrics and hyperparameters
├── feature_importances.json  ← mutual information scores per feature
├── shap_importance.json      ← SHAP values for the best model
├── experiment_history.json   ← record of tuning iterations
├── error_reports.json        ← all issues detected
├── final_state.json          ← complete PipelineState snapshot
└── models/
    ├── LogisticRegression.joblib
    ├── RandomForestClassifier.joblib
    ├── XGBClassifier.joblib         ← each trained model saved separately
    ├── LGBMClassifier.joblib
    └── XGBClassifier_tuned_iter0.joblib  ← tuned version (if Agent 7 improved it)

reports/{run_id}/
└── pipeline_report.md        ← human-readable markdown report

logs/
└── axiom.log                 ← structured JSON logs for all runs

data/
└── axiom.db                  ← SQLite database (users, sessions, history)
```

---

## 10. Technology Stack Summary

| Component | Technology | Why |
|-----------|-----------|-----|
| **Backend framework** | FastAPI (Python) | Auto-generates docs, async support, easy REST APIs |
| **AI orchestration** | CrewAI | Multi-agent coordination with tool use |
| **LLM** | Llama 3.1 via Cerebras | Fast inference, free tier available |
| **ML models** | scikit-learn | Industry standard, wide model selection |
| **Gradient boosting** | XGBoost + LightGBM | Best-in-class tabular ML models |
| **Explainability** | SHAP | Industry standard for ML interpretability |
| **Hyperparameter tuning** | Optuna + RandomizedSearchCV | Best modern tuning approaches |
| **Data processing** | Pandas + NumPy | Standard Python data science stack |
| **Model persistence** | Joblib | Efficient serialization of scikit-learn models |
| **State validation** | Pydantic v2 | Type-safe data models, JSON serialization |
| **Database** | SQLite + SQLAlchemy | Simple, file-based, no server required |
| **Auth** | bcrypt + UUID tokens | Secure password hashing, stateless tokens |
| **Frontend framework** | Next.js 14 (TypeScript) | React-based, server-side rendering, fast |
| **Styling** | Tailwind CSS | Utility-first, consistent design system |
| **State management** | Zustand | Lightweight, persisted across browser sessions |
| **Animations** | Framer Motion | Production-quality UI animations |

---

*This document covers every major component of Axiom — from the moment a user uploads a file
to the moment they see their results. Use this as the foundation for your presentation script.*
