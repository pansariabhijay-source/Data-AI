"""
Configuration management for the autonomous data science pipeline.

Layered config resolution:
  1. configs/default.yaml      (base defaults)
  2. Environment variables     (override per deployment)
  3. CLI / programmatic        (override per run)

Uses Pydantic v2 Settings for validation, type coercion, and documentation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from core.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_CV_FOLDS,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_ID_UNIQUENESS_RATIO,
    DEFAULT_IQR_MULTIPLIER,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_MAX_CARDINALITY,
    DEFAULT_MAX_COLUMNS,
    DEFAULT_MAX_FEATURE_COUNT,
    DEFAULT_MAX_NULL_THRESHOLD,
    DEFAULT_MAX_ONEHOT_CARDINALITY,
    DEFAULT_MAX_OUTLIER_FRACTION,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MIN_CLASSIFICATION_F1,
    DEFAULT_MIN_REGRESSION_R2,
    DEFAULT_MIN_UNIQUE_FOR_OUTLIER,
    DEFAULT_N_JOBS,
    DEFAULT_OVERFITTING_MIN_VAL_SCORE,
    DEFAULT_OVERFITTING_THRESHOLD,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_ROWS,
    DEFAULT_SELECT_K_BEST,
    DEFAULT_SMOTE_K_NEIGHBORS,
    DEFAULT_SMOTE_MAX_TRAIN_SAMPLES,
    DEFAULT_SMOTE_MIN_MINORITY_FRACTION,
    DEFAULT_SMOTE_SAMPLING_STRATEGY,
    DEFAULT_TEST_RATIO,
    DEFAULT_TIMEOUT_PER_MODEL,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_TUNING_ITERATIONS,
    DEFAULT_TUNING_MAX_ITERATIONS,
    DEFAULT_TUNING_PATIENCE,
    DEFAULT_TUNING_TARGET_METRIC,
    DEFAULT_USE_SMOTE,
    DEFAULT_VAL_RATIO,
    DEFAULT_VARIANCE_THRESHOLD,
)


# ── Nested config sections ──────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    """Top-level pipeline execution settings."""

    random_seed: int = DEFAULT_RANDOM_SEED
    max_retries: int = DEFAULT_MAX_RETRIES
    checkpoint_enabled: bool = True
    artifact_dir: str = "artifacts"
    report_dir: str = "reports"
    data_dir: str = "data"
    log_dir: str = "logs"


class LLMConfig(BaseModel):
    """LLM provider configuration (Cerebras via LiteLLM)."""

    model: str = "cerebras/llama3.1-8b"
    temperature: float = 0.1
    max_tokens: int = 4096

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature must be in [0.0, 2.0], got {v}")
        return v


class DataCollectionConfig(BaseModel):
    """Data ingestion settings."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    max_columns: int = DEFAULT_MAX_COLUMNS
    sample_rows_for_profiling: int = DEFAULT_SAMPLE_ROWS


class PreprocessingConfig(BaseModel):
    """Data cleaning and preprocessing settings."""

    # Outlier winsorization is OFF by default. The model registry is dominated by
    # tree / gradient-boosting models whose splits are rank-based and therefore
    # robust to outliers, and clipping the extreme tail (e.g. very large fraud
    # amounts, high-priced homes) erases exactly the high-leverage values that
    # carry signal. Ablation showed IQR clipping lowered held-out R²/F1 vs leaving
    # it off. Scale-sensitive linear models get a per-fit StandardScaler in the
    # training stage instead. Set outlier_method="iqr" to re-enable.
    outlier_method: str = "none"
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER
    max_null_threshold: float = DEFAULT_MAX_NULL_THRESHOLD
    max_cardinality: int = DEFAULT_MAX_CARDINALITY
    duplicate_keep: str = "first"
    min_unique_for_outlier: int = DEFAULT_MIN_UNIQUE_FOR_OUTLIER
    max_outlier_fraction: float = DEFAULT_MAX_OUTLIER_FRACTION

    @field_validator("max_null_threshold")
    @classmethod
    def validate_null_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"max_null_threshold must be in [0.0, 1.0], got {v}")
        return v


class FeatureEngineeringConfig(BaseModel):
    """Feature engineering and selection settings."""

    max_onehot_cardinality: int = DEFAULT_MAX_ONEHOT_CARDINALITY
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD
    select_k_best: int = DEFAULT_SELECT_K_BEST
    # Scaling is NOT applied to the shared feature matrix. A single scaler fit on
    # the full dataset before the split leaks test statistics, and scaling hurts
    # the tree/boosting models that usually win. Instead, scale-sensitive models
    # are wrapped in a per-fit StandardScaler at training time (see
    # core.model_registry.maybe_wrap_scaler), which is leakage-free and keeps trees
    # on raw, interpretable features. Set scaling_method to standard/minmax/robust
    # to also scale the shared matrix (legacy behaviour).
    scaling_method: str = "none"
    id_uniqueness_ratio: float = DEFAULT_ID_UNIQUENESS_RATIO
    drop_leakage_columns: bool = True
    # Explicit pairwise multiplicative interactions are OFF by default. The
    # gradient-boosting / tree models that usually win capture feature interactions
    # natively, so adding O(top_n^2) products mostly injects noise columns (e.g.
    # products of weak/noise features) that dilute the signal and can lower the
    # held-out score. Enable for linear-only problems where interactions help.
    enable_interactions: bool = False


class SplittingConfig(BaseModel):
    """Data splitting ratios and settings."""

    train_ratio: float = DEFAULT_TRAIN_RATIO
    val_ratio: float = DEFAULT_VAL_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO
    # Out-of-time splitting: "auto" uses a chronological split when a usable time
    # axis is detected (train=oldest, val=middle, test=newest), else falls back to a
    # stratified random split. "off" forces stratified random; "on" requires a time
    # column. Out-of-time evaluation is the correct, leakage-free protocol for
    # temporal data like fraud (random splitting leaks the future).
    time_aware_split: str = "auto"

    @field_validator("test_ratio")
    @classmethod
    def validate_ratios_sum(cls, v: float, info) -> float:  # noqa: ANN001
        train = info.data.get("train_ratio", DEFAULT_TRAIN_RATIO)
        val = info.data.get("val_ratio", DEFAULT_VAL_RATIO)
        total = train + val + v
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f} "
                f"(train={train}, val={val}, test={v})"
            )
        return v


class TrainingConfig(BaseModel):
    """Model training settings."""

    n_jobs: int = DEFAULT_N_JOBS
    cv_folds: int = DEFAULT_CV_FOLDS
    timeout_per_model_seconds: int = DEFAULT_TIMEOUT_PER_MODEL

    # ── Imbalanced-target resampling (SMOTE) ─────────────────────────────────
    # Applied to the TRAIN split only (never val/test), before fitting. Only
    # triggers on binary classification whose minority fraction is below
    # ``smote_min_minority_fraction``. Validated to beat class-weighting alone on
    # extreme imbalance (creditcard). Partial oversampling — full balancing
    # (strategy 1.0) empirically HURTS boosters, so the default is moderate.
    use_smote: bool = DEFAULT_USE_SMOTE
    smote_sampling_strategy: float = DEFAULT_SMOTE_SAMPLING_STRATEGY  # target minority/majority ratio
    smote_min_minority_fraction: float = DEFAULT_SMOTE_MIN_MINORITY_FRACTION
    smote_max_train_samples: int = DEFAULT_SMOTE_MAX_TRAIN_SAMPLES
    smote_k_neighbors: int = DEFAULT_SMOTE_K_NEIGHBORS


class ErrorDetectionConfig(BaseModel):
    """Thresholds for automated error / anomaly detection."""

    min_classification_f1: float = DEFAULT_MIN_CLASSIFICATION_F1
    min_regression_r2: float = DEFAULT_MIN_REGRESSION_R2
    overfitting_threshold: float = DEFAULT_OVERFITTING_THRESHOLD
    overfitting_min_val_score: float = DEFAULT_OVERFITTING_MIN_VAL_SCORE
    max_feature_count: int = DEFAULT_MAX_FEATURE_COUNT


class ImprovementConfig(BaseModel):
    """Improvement loop / hyperparameter tuning settings."""

    tuning_iterations: int = DEFAULT_TUNING_ITERATIONS  # inner search budget per round
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS
    use_optuna: bool = False

    # Iterative tuning loop: keep tuning the champion across rounds (each with a
    # fresh search seed) until the validation selection score reaches
    # ``tuning_target_metric`` (0 disables that early stop), capped at
    # ``tuning_max_iterations`` rounds and stopping after ``tuning_patience``
    # consecutive rounds without improvement.
    tuning_target_metric: float = DEFAULT_TUNING_TARGET_METRIC
    tuning_max_iterations: int = DEFAULT_TUNING_MAX_ITERATIONS
    tuning_patience: int = DEFAULT_TUNING_PATIENCE


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    max_bytes: int = DEFAULT_LOG_MAX_BYTES
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT
    json_format: bool = True


# ── Root settings ───────────────────────────────────────────────────────────


class Settings(BaseModel):
    """Root configuration object aggregating all config sections.

    Resolution order: YAML defaults → environment variable overrides.
    """

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    feature_engineering: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig
    )
    splitting: SplittingConfig = Field(default_factory=SplittingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    error_detection: ErrorDetectionConfig = Field(
        default_factory=ErrorDetectionConfig
    )
    improvement: ImprovementConfig = Field(default_factory=ImprovementConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ── Loading utilities ───────────────────────────────────────────────────────


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, preferring override values."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(data: dict) -> dict:
    """Apply environment variable overrides to the config dict.

    Convention: ``LLM_MODEL`` overrides ``data["llm"]["model"]``.
    Only known top-level sections are scanned.
    """
    env_map = {
        "LLM_MODEL": ("llm", "model"),
        "LLM_TEMPERATURE": ("llm", "temperature"),
        "LLM_MAX_TOKENS": ("llm", "max_tokens"),
        "LOG_LEVEL": ("logging", "level"),
        "RANDOM_SEED": ("pipeline", "random_seed"),
        "MAX_RETRIES": ("pipeline", "max_retries"),
        "ARTIFACT_DIR": ("pipeline", "artifact_dir"),
        "REPORT_DIR": ("pipeline", "report_dir"),
        "LOG_DIR": ("pipeline", "log_dir"),
        "USE_SMOTE": ("training", "use_smote"),
        "SMOTE_SAMPLING_STRATEGY": ("training", "smote_sampling_strategy"),
    }
    for env_key, (section, field) in env_map.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            data.setdefault(section, {})[field] = env_val
    return data


def load_settings(
    config_path: Optional[str | Path] = None,
    overrides: Optional[dict] = None,
) -> Settings:
    """Load and validate pipeline settings.

    Args:
        config_path: Path to YAML config file. Defaults to ``configs/default.yaml``.
        overrides: Programmatic overrides applied last.

    Returns:
        Fully validated ``Settings`` instance.

    Raises:
        core.exceptions.ConfigurationError: If the config file is missing or invalid.
    """
    from core.exceptions import ConfigurationError

    if config_path is None:
        config_path = Path(os.environ.get("CONFIG_PATH", "configs/default.yaml"))
    else:
        config_path = Path(config_path)

    data: dict = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
                if isinstance(raw, dict):
                    data = raw
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Failed to parse config file {config_path}: {exc}"
            ) from exc
    else:
        # Warn but don't fail — defaults are sane
        import warnings

        warnings.warn(
            f"Config file not found at {config_path}, using built-in defaults.",
            UserWarning,
            stacklevel=2,
        )

    # Layer: env vars
    data = _apply_env_overrides(data)

    # Layer: programmatic overrides
    if overrides:
        data = _deep_merge(data, overrides)

    try:
        return Settings(**data)
    except Exception as exc:
        raise ConfigurationError(
            f"Configuration validation failed: {exc}",
            context={"raw_data": data},
        ) from exc
