# Axiom: Autonomous Data Scientist
## Executive Project Summary

**Axiom** is an enterprise-grade, multi-agent artificial intelligence platform designed to automate the entire machine learning lifecycle. It acts as an autonomous data scientist, taking raw datasets (e.g., CSV files) and transforming them into production-ready, highly optimized machine learning models with zero human intervention.

This report outlines the architecture, technology stack, and operational flow of the Axiom platform.

---

### 1. Core Architecture & Tech Stack

Axiom bridges the gap between complex data science workflows and intuitive user experiences through a modern, decoupled architecture.

- **Backend (API & ML Engine):** Built with **FastAPI** (Python). It provides a high-performance REST API that handles file uploads, state management, and triggers the ML pipelines.
- **Frontend (User Interface):** Built with **Next.js 14**, React, and TailwindCSS. It features a "glassmorphic", dark-themed UI with fluid animations (using Framer Motion). It includes an Agent Console, a Workflow Builder, and a Visualizations Studio.
- **Orchestration Layer:** Powered by **CrewAI**. This layer manages the swarm of AI agents, delegating tasks and ensuring agents collaborate to solve the ML problem.
- **LLM Provider:** **Cerebras** (Llama 3.1 8B). The LLM is used strictly for reasoning, planning, and code generation, while the actual ML execution is handled by deterministic Python services.
- **Database:** **SQLite** (`axiom.db`) for storing user sessions, metadata, and run history.

---

### 2. The Multi-Agent Swarm & Their Roles

Axiom does not rely on a single monolithic script. Instead, it utilizes a cooperative swarm of **8 specialized AI agents**, each with a distinct role in the data science lifecycle. Below is a detailed breakdown of each agent, how it operates, and the underlying algorithms it utilizes.

#### 2.1 Manager Agent
- **Role:** The overarching orchestrator of the pipeline.
- **How it works:** Analyzes the dataset metadata, identifies the target variable, and determines if the problem requires **Classification** or **Regression**. It then sequences the execution of all downstream worker agents and compiles their findings.

#### 2.2 Data Collection Agent
- **Role:** Data ingestion, profiling, and memory optimization.
- **How it works:** Loads the raw dataset (CSV) into a Pandas DataFrame. It calculates a "Data Quality Score" based on missing values, duplicate rows, and cardinality. It then performs memory optimization by downcasting data types (e.g., converting `float64` to `float32` and mapping low-cardinality strings to `category` types).
- **Outputs:** Optimized dataset, initial summary statistics, and basic visualizations (e.g., feature distributions).

#### 2.3 Preprocessing Agent
- **Role:** Data cleaning and standardization.
- **How it works:**
  - **Imputation:** Handles missing data using SimpleImputer (median for continuous, mode for categorical).
  - **Outlier Handling:** Uses the IQR (Interquartile Range) method to perform **Winsorization**, capping extreme outliers rather than dropping them.
  - **Encoding & Scaling:** Applies `StandardScaler` to numerical features and `OrdinalEncoder` to categorical features, ensuring all data is machine-readable and normalized.

#### 2.4 Feature Engineering Agent
- **Role:** Dimensionality reduction and feature selection.
- **How it works:** Uses statistical tests to drop features that do not contribute predictive power, preventing the "curse of dimensionality".
- **Algorithms Used:**
  - Identifies highly correlated feature pairs (Pearson correlation > 0.85) and removes redundant columns to prevent multicollinearity.
  - Employs **Mutual Information Score** (`mutual_info_classif` or `mutual_info_regression` from scikit-learn) to rank features.
  - Automatically selects the top $K$ features (defaulting to the top 20 or dropping the bottom 20% of low-importance features).

#### 2.5 Data Splitting Agent
- **Role:** Creating robust validation sets.
- **How it works:** Partitions the dataset into **Training (70%)**, **Validation (15%)**, and **Testing (15%)** sets.
- **Methods:** Uses `train_test_split`. For classification tasks, it enforces **Stratified Sampling** to ensure the class distribution (e.g., 90% healthy, 10% sick) remains identical across all splits, preventing biased evaluation.

#### 2.6 Model Training Agent
- **Role:** Training a diverse suite of algorithms and selecting the champion.
- **How it works:** Takes the preprocessed training set and trains an entire registry of models. It evaluates them against the validation set and logs metrics.
- **Models Used (Regression):**
  - **Linear Models:** `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
  - **Tree-Based Ensembles:** `RandomForestRegressor`, `GradientBoostingRegressor`
  - **Advanced Boosters:** `XGBRegressor` (XGBoost), `LGBMRegressor` (LightGBM), `CatBoostRegressor` (if configured)
- **Models Used (Classification):**
  - **Linear/Distance:** `LogisticRegression`, `KNeighborsClassifier`
  - **Tree-Based Ensembles:** `RandomForestClassifier`, `GradientBoostingClassifier`
  - **Advanced Boosters:** `XGBClassifier`, `LGBMClassifier`
- **Evaluation Metrics:** Uses RMSE, MAE, and $R^2$ for regression. Uses Accuracy, F1-Score, Precision, Recall, and ROC-AUC for classification.

#### 2.7 Error Detection Agent (Auditor)
- **Role:** Quality assurance and bias detection.
- **How it works:** Analyzes the Champion Model's metrics to find critical flaws.
  - **Overfitting Check:** Compares Training $R^2$ vs Validation $R^2$. If the gap is > 20%, it flags the model for overfitting.
  - **Class Imbalance Check:** For classification, checks if the minority class represents < 20% of the data.
  - **Leakage Check:** Flags if the model achieves an unrealistic perfect score ($R^2$ or Accuracy > 0.99).

#### 2.8 Improvement Agent (Tuner)
- **Role:** Hyperparameter optimization.
- **How it works:** Attempts to squeeze extra performance out of the Champion Model by tuning its internal parameters.
- **Algorithms Used:** Employs **GridSearchCV** or **RandomizedSearchCV**. For example, if the champion is `RandomForest`, it will systematically test different values for `n_estimators`, `max_depth`, and `min_samples_split` using 3-fold cross-validation. It replaces the champion model only if the new tuned model achieves a strictly higher validation score.

#### 2.9 Finalization Agent
- **Role:** Explainable AI (XAI) and artifact packaging.
- **How it works:**
  - **Explainability:** Employs the **SHAP (SHapley Additive exPlanations)** library to calculate the marginal contribution of every single feature. It generates Feature Importance charts explaining exactly *why* the model makes specific decisions.
  - **Reporting:** Compiles all agent findings into a final Markdown report.
  - **Serialization:** Saves the final trained models as `.pkl` (pickle/joblib) files inside the `artifacts/` directory so they can be deployed to production instantly.

---

### 3. The Visualization Engine

A crucial component of Axiom is its ability to make data and models understandable. The platform features a custom-built **Visualization Engine** (`visualization/engine.py`) leveraging **Seaborn** and **Matplotlib**.

- **Visualizations Studio:** A dedicated UI where users can generate and inspect 10 different types of charts.
- **Chart Types:**
  - *Basic:* Missing Values Heatmaps, Feature Distributions, Outlier Box Plots, Correlation Matrices.
  - *Advanced:* PCA 2D Projections, Pair Plots.
  - *Model-Specific:* Model Performance Comparisons, Feature Importance Bar Charts, Confusion Matrices.
- All charts are rendered in a sleek dark theme that matches the UI and are served to the frontend as base64-encoded PNGs.

---

### 4. Security & Enterprise Readiness

Axiom is built with strict security constraints suitable for enterprise deployment:
- **Deterministic Execution:** The LLM does not execute arbitrary code. It utilizes strict "Tool Services" (pre-written, secure Python functions) to manipulate data.
- **Data Isolation:** All runs and datasets are isolated. API endpoints are secured via JWT-like URL-safe tokens and bcrypt password hashing.
- **Explainable AI (XAI):** Axiom doesn't just output a "black box" model. It uses **SHAP** to explain exactly *why* a model made a decision, ranking features by their predictive contribution.
- **Comprehensive Reporting:** Every pipeline run automatically generates a detailed Markdown and PDF report documenting every decision made by the agents, the metrics of all trained models, and the final XAI insights.

---

### Conclusion

Axiom represents a paradigm shift in machine learning. By combining the reasoning capabilities of Large Language Models with deterministic Python execution and a highly polished Next.js interface, Axiom allows organizations to rapidly prototype, train, and understand machine learning models at a fraction of the traditional time and cost.
