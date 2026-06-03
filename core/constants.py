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

# Feature engineering
DEFAULT_MAX_ONEHOT_CARDINALITY: int = 15
DEFAULT_VARIANCE_THRESHOLD: float = 0.01
DEFAULT_CORRELATION_THRESHOLD: float = 0.95
DEFAULT_SELECT_K_BEST: int = 20

# Training
DEFAULT_CV_FOLDS: int = 5
DEFAULT_N_JOBS: int = -1
DEFAULT_TIMEOUT_PER_MODEL: int = 300  # seconds

# Error detection
DEFAULT_MIN_CLASSIFICATION_F1: float = 0.30
DEFAULT_MIN_REGRESSION_R2: float = 0.10
DEFAULT_OVERFITTING_THRESHOLD: float = 0.15
DEFAULT_MAX_FEATURE_COUNT: int = 500

# Improvement
DEFAULT_TUNING_ITERATIONS: int = 50
DEFAULT_EARLY_STOPPING_ROUNDS: int = 10

# Data loading
DEFAULT_CHUNK_SIZE: int = 50_000
DEFAULT_MAX_COLUMNS: int = 500
DEFAULT_SAMPLE_ROWS: int = 10_000

# Logging
DEFAULT_LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
DEFAULT_LOG_BACKUP_COUNT: int = 5
