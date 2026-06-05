"""
Constants and enumerations for the autonomous data science pipeline.

All magic numbers, default thresholds, and categorical values are centralized here
to prevent hardcoding across the codebase. Enums enforce type safety at boundaries.
"""

from enum import Enum, unique


@unique
class ProblemType(str, Enum):
    """ML problem type detected or configured for the pipeline run."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    UNKNOWN = "unknown"


@unique
class PipelineStage(str, Enum):
    """Ordered stages of the ML pipeline. Used for logging, checkpointing, and state tracking."""

    DATA_COLLECTION = "data_collection"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_SPLITTING = "data_splitting"
    MODEL_TRAINING = "model_training"
    ERROR_DETECTION = "error_detection"
    IMPROVEMENT = "improvement"
    FINALIZATION = "finalization"


@unique
class Severity(str, Enum):
    """Severity levels for error reports and data quality flags."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@unique
class ModelStatus(str, Enum):
    """Lifecycle status of a trained model artifact."""

    PENDING = "pending"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"
    SELECTED = "selected"
    ARCHIVED = "archived"


@unique
class DataType(str, Enum):
    """Semantic data types for column classification."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


@unique
class ScalingMethod(str, Enum):
    """Feature scaling methods."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


@unique
class OutlierMethod(str, Enum):
    """Outlier detection and handling methods."""

    IQR = "iqr"
    ZSCORE = "zscore"
    NONE = "none"


# ── Default thresholds ──────────────────────────────────────────────────────

DEFAULT_RANDOM_SEED: int = 42
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_TRAIN_RATIO: float = 0.70
DEFAULT_VAL_RATIO: float = 0.15
DEFAULT_TEST_RATIO: float = 0.15

# Preprocessing
DEFAULT_IQR_MULTIPLIER: float = 1.5
DEFAULT_MAX_NULL_THRESHOLD: float = 0.70  # Drop column if > 70% null
DEFAULT_MAX_CARDINALITY: int = 50  # Flag high-cardinality categoricals
DEFAULT_DUPLICATE_KEEP: str = "first"
# Outlier handling guards — prevent winsorization from erasing real signal.
# Columns with <= this many unique values are treated as discrete (binary flags,
# encoded categoricals) and skipped — clipping them would destroy class structure.
DEFAULT_MIN_UNIQUE_FOR_OUTLIER: int = 20
# If more than this fraction of a column would be clipped, the "outliers" are the
# distribution (e.g. a rare-event cluster / minority class), not noise — skip it.
DEFAULT_MAX_OUTLIER_FRACTION: float = 0.05

# Feature engineering
DEFAULT_MAX_ONEHOT_CARDINALITY: int = 15
DEFAULT_VARIANCE_THRESHOLD: float = 0.01
DEFAULT_CORRELATION_THRESHOLD: float = 0.95
DEFAULT_SELECT_K_BEST: int = 20
# Categorical columns whose unique-value ratio exceeds this are ID-like (e.g. user_id,
# transaction_id) and are dropped rather than label-encoded into pure noise.
DEFAULT_ID_UNIQUENESS_RATIO: float = 0.5

# Problem-type detection — a numeric target with more distinct values than this
# is treated as regression, never classification (479-class "Volume" is not a
# classification target). Strings that are >= this fraction numerically parseable
# are coerced and judged as numbers (handles numeric columns dirtied by junk rows).
DEFAULT_MAX_CLASSIFICATION_UNIQUE: int = 20
DEFAULT_NUMERIC_COERCE_FRACTION: float = 0.95
# Integer-coded targets up to this many levels (with a low unique-to-rows ratio)
# are still treated as discrete classes rather than regression.
DEFAULT_MAX_INT_CLASSIFICATION_UNIQUE: int = 50

# Training
DEFAULT_CV_FOLDS: int = 5
DEFAULT_N_JOBS: int = -1
DEFAULT_TIMEOUT_PER_MODEL: int = 300  # seconds

# SMOTE / imbalanced-target resampling (train split only).
# Default OFF by evidence: in isolation SMOTE lifts a single XGB/LGBM on raw
# features, but end-to-end on the pipeline's feature-engineered space it did NOT
# beat the existing handling (class_weight / scale_pos_weight + F1-optimal
# threshold + PR-AUC selection) on creditcard — champion f1 0.846 (weighting) vs
# 0.830 (SMOTE). Kept fully wired and configurable: set training.use_smote=True
# (or env USE_SMOTE=true) to enable for datasets where it helps.
DEFAULT_USE_SMOTE: bool = False
# Target minority/majority ratio after oversampling. PARTIAL on purpose: moderate
# oversampling (0.1-0.5) helps in isolation, but fully balancing (1.0) empirically
# hurts gradient boosters. 0.25 is a safe middle ground when enabled.
DEFAULT_SMOTE_SAMPLING_STRATEGY: float = 0.25
# Only resample when the minority class is below this fraction of all rows; data
# that is already reasonably balanced is left untouched.
DEFAULT_SMOTE_MIN_MINORITY_FRACTION: float = 0.20
# Skip SMOTE above this many training rows (synthetic blow-up / cost guard).
DEFAULT_SMOTE_MAX_TRAIN_SAMPLES: int = 500_000
DEFAULT_SMOTE_K_NEIGHBORS: int = 5

# Error detection
DEFAULT_MIN_CLASSIFICATION_F1: float = 0.30
DEFAULT_MIN_REGRESSION_R2: float = 0.10
DEFAULT_OVERFITTING_THRESHOLD: float = 0.15
DEFAULT_MAX_FEATURE_COUNT: int = 500

# Improvement
DEFAULT_TUNING_ITERATIONS: int = 50
DEFAULT_EARLY_STOPPING_ROUNDS: int = 10
# Iterative tuning loop: stop early once the champion's validation selection score
# reaches this target (0 disables the early stop); otherwise run up to
# DEFAULT_TUNING_MAX_ITERATIONS rounds, stopping after DEFAULT_TUNING_PATIENCE
# consecutive rounds without improvement.
DEFAULT_TUNING_TARGET_METRIC: float = 0.0
DEFAULT_TUNING_MAX_ITERATIONS: int = 5
DEFAULT_TUNING_PATIENCE: int = 2

# Data loading
DEFAULT_CHUNK_SIZE: int = 50_000
DEFAULT_MAX_COLUMNS: int = 500
DEFAULT_SAMPLE_ROWS: int = 10_000

# Logging
DEFAULT_LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
DEFAULT_LOG_BACKUP_COUNT: int = 5
