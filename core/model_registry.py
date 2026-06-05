"""
Model registry — abstraction over ML model instantiation and hyperparameter spaces.

Models are registered by problem type. The registry provides:
- Lazy instantiation (only import/create when needed)
- Default hyperparameters
- Search spaces for tuning
- Unified interface for training and prediction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.constants import ProblemType
from core.exceptions import ModelRegistryError
from core.logging_config import get_logger

logger = get_logger("model_registry")


@dataclass
class ModelSpec:
    """Specification for a registered model."""
    name: str
    problem_type: ProblemType
    factory: Callable[..., Any]  # callable that returns a model instance
    default_params: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    requires_scaling: bool = False
    supports_probabilities: bool = False
    # Name of a data-dependent imbalance hyperparameter to inject at fit time
    # (e.g. XGBoost's "scale_pos_weight"). Models that accept ``class_weight``
    # set it directly in ``default_params`` instead.
    imbalance_param: Optional[str] = None
    # Gradient boosters that can train with early stopping against a validation
    # set — lets them grow many more trees but stop before overfitting.
    supports_early_stopping: bool = False
    # Skip this model when the training set is larger than this (e.g. kernel SVC
    # is O(n^2)+ and will hang / waste time on big data). None = no limit.
    max_train_samples: Optional[int] = None


class ModelRegistry:
    """Registry of available ML models keyed by problem type."""

    def __init__(self) -> None:
        self._registry: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        key = f"{spec.problem_type.value}:{spec.name}"
        self._registry[key] = spec
        logger.debug(f"Registered model: {key}")

    def get(self, problem_type: ProblemType, name: str) -> ModelSpec:
        key = f"{problem_type.value}:{name}"
        if key not in self._registry:
            raise ModelRegistryError(f"Model '{name}' not registered for {problem_type.value}")
        return self._registry[key]

    def list_models(self, problem_type: ProblemType) -> list[ModelSpec]:
        return [s for k, s in self._registry.items() if k.startswith(problem_type.value)]

    def create_instance(self, problem_type: ProblemType, name: str, params: Optional[dict[str, Any]] = None) -> Any:
        spec = self.get(problem_type, name)
        merged = {**spec.default_params, **(params or {})}
        try:
            return spec.factory(**merged)
        except Exception as e:
            raise ModelRegistryError(f"Failed to instantiate '{name}': {e}") from e


def build_default_registry(seed: int = 42) -> ModelRegistry:
    """Build and return the default model registry with all supported models."""
    registry = ModelRegistry()

    # ── Classification ──────────────────────────────────────────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.svm import SVC

    registry.register(ModelSpec(
        name="LogisticRegression", problem_type=ProblemType.CLASSIFICATION,
        factory=LogisticRegression,
        default_params={"max_iter": 1000, "random_state": seed, "n_jobs": -1,
                        "class_weight": "balanced"},
        search_space={"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l1", "l2"], "solver": ["liblinear", "saga"]},
        requires_scaling=True, supports_probabilities=True,
    ))
    registry.register(ModelSpec(
        name="RandomForestClassifier", problem_type=ProblemType.CLASSIFICATION,
        factory=RandomForestClassifier,
        default_params={"n_estimators": 300, "random_state": seed, "n_jobs": -1,
                        "class_weight": "balanced"},
        search_space={"n_estimators": [100, 200, 300, 500], "max_depth": [5, 10, 20, None], "min_samples_split": [2, 5, 10]},
        supports_probabilities=True,
    ))
    registry.register(ModelSpec(
        name="SVC", problem_type=ProblemType.CLASSIFICATION,
        factory=SVC,
        default_params={"random_state": seed, "probability": True, "max_iter": 5000,
                        "class_weight": "balanced"},
        search_space={"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
        requires_scaling=True, supports_probabilities=True,
        max_train_samples=20000,
    ))
    registry.register(ModelSpec(
        name="ExtraTreesClassifier", problem_type=ProblemType.CLASSIFICATION,
        factory=ExtraTreesClassifier,
        default_params={"n_estimators": 300, "random_state": seed, "n_jobs": -1,
                        "class_weight": "balanced"},
        search_space={"n_estimators": [200, 300, 500], "max_depth": [10, 20, None],
                      "min_samples_split": [2, 5, 10]},
        supports_probabilities=True,
    ))
    registry.register(ModelSpec(
        name="HistGradientBoostingClassifier", problem_type=ProblemType.CLASSIFICATION,
        factory=HistGradientBoostingClassifier,
        default_params={"random_state": seed, "class_weight": "balanced",
                        "learning_rate": 0.1, "max_iter": 300, "early_stopping": True,
                        "validation_fraction": 0.1, "n_iter_no_change": 20},
        search_space={"learning_rate": [0.03, 0.05, 0.1], "max_iter": [200, 300, 500],
                      "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0.0, 1.0, 10.0]},
        supports_probabilities=True,
    ))

    try:
        from xgboost import XGBClassifier
        registry.register(ModelSpec(
            name="XGBClassifier", problem_type=ProblemType.CLASSIFICATION,
            factory=XGBClassifier,
            default_params={"n_estimators": 300, "random_state": seed, "eval_metric": "aucpr",
                            "use_label_encoder": False, "verbosity": 0, "n_jobs": -1},
            search_space={"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7],
                          "learning_rate": [0.01, 0.05, 0.1, 0.3], "subsample": [0.8, 1.0],
                          "colsample_bytree": [0.8, 1.0]},
            supports_probabilities=True,
            imbalance_param="scale_pos_weight",
            supports_early_stopping=True,
        ))
    except ImportError:
        logger.warning("XGBoost not installed, skipping XGBClassifier")

    try:
        from lightgbm import LGBMClassifier
        registry.register(ModelSpec(
            name="LGBMClassifier", problem_type=ProblemType.CLASSIFICATION,
            factory=LGBMClassifier,
            default_params={"n_estimators": 300, "random_state": seed, "verbose": -1, "n_jobs": -1,
                            "class_weight": "balanced"},
            search_space={"n_estimators": [100, 200, 300], "num_leaves": [15, 31, 63],
                          "learning_rate": [0.01, 0.05, 0.1, 0.3]},
            supports_probabilities=True,
            supports_early_stopping=True,
        ))
    except ImportError:
        logger.warning("LightGBM not installed, skipping LGBMClassifier")

    # ── Regression ──────────────────────────────────────────────────────
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )

    registry.register(ModelSpec(
        name="LinearRegression", problem_type=ProblemType.REGRESSION,
        factory=LinearRegression,
        default_params={"n_jobs": -1},
        search_space={},
        requires_scaling=True,
    ))
    registry.register(ModelSpec(
        name="Ridge", problem_type=ProblemType.REGRESSION,
        factory=Ridge,
        default_params={"random_state": seed},
        search_space={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        requires_scaling=True,
    ))
    registry.register(ModelSpec(
        name="RandomForestRegressor", problem_type=ProblemType.REGRESSION,
        factory=RandomForestRegressor,
        default_params={"n_estimators": 100, "random_state": seed, "n_jobs": -1},
        search_space={"n_estimators": [50, 100, 200, 300], "max_depth": [5, 10, 20, None], "min_samples_split": [2, 5, 10]},
    ))
    registry.register(ModelSpec(
        name="ExtraTreesRegressor", problem_type=ProblemType.REGRESSION,
        factory=ExtraTreesRegressor,
        default_params={"n_estimators": 300, "random_state": seed, "n_jobs": -1},
        search_space={"n_estimators": [200, 300, 500], "max_depth": [10, 20, None], "min_samples_split": [2, 5, 10]},
    ))
    registry.register(ModelSpec(
        name="HistGradientBoostingRegressor", problem_type=ProblemType.REGRESSION,
        factory=HistGradientBoostingRegressor,
        default_params={"random_state": seed, "learning_rate": 0.1, "max_iter": 300,
                        "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 20},
        search_space={"learning_rate": [0.03, 0.05, 0.1], "max_iter": [200, 300, 500],
                      "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0.0, 1.0, 10.0]},
    ))

    try:
        from xgboost import XGBRegressor
        registry.register(ModelSpec(
            name="XGBRegressor", problem_type=ProblemType.REGRESSION,
            factory=XGBRegressor,
            default_params={"n_estimators": 300, "random_state": seed, "verbosity": 0, "n_jobs": -1},
            search_space={"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1, 0.3]},
            supports_early_stopping=True,
        ))
    except ImportError:
        logger.warning("XGBoost not installed, skipping XGBRegressor")

    try:
        from lightgbm import LGBMRegressor
        registry.register(ModelSpec(
            name="LGBMRegressor", problem_type=ProblemType.REGRESSION,
            factory=LGBMRegressor,
            default_params={"n_estimators": 300, "random_state": seed, "verbose": -1, "n_jobs": -1},
            search_space={"n_estimators": [100, 200, 300], "num_leaves": [15, 31, 63], "learning_rate": [0.01, 0.05, 0.1, 0.3]},
            supports_early_stopping=True,
        ))
    except ImportError:
        logger.warning("LightGBM not installed, skipping LGBMRegressor")

    # ── Clustering ──────────────────────────────────────────────────────
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

    registry.register(ModelSpec(
        name="KMeans", problem_type=ProblemType.CLUSTERING,
        factory=KMeans,
        default_params={"n_clusters": 3, "random_state": seed, "n_init": 10},
        search_space={"n_clusters": [2, 3, 4, 5, 6, 7, 8]},
        requires_scaling=True,
    ))
    registry.register(ModelSpec(
        name="DBSCAN", problem_type=ProblemType.CLUSTERING,
        factory=DBSCAN,
        default_params={},
        search_space={"eps": [0.3, 0.5, 1.0, 1.5], "min_samples": [3, 5, 10]},
        requires_scaling=True,
    ))
    registry.register(ModelSpec(
        name="AgglomerativeClustering", problem_type=ProblemType.CLUSTERING,
        factory=AgglomerativeClustering,
        default_params={"n_clusters": 3},
        search_space={"n_clusters": [2, 3, 4, 5], "linkage": ["ward", "complete", "average"]},
        requires_scaling=True,
    ))

    logger.info(f"Model registry built with {len(registry._registry)} models")
    return registry


def fit_model(
    model: Any,
    spec: ModelSpec,
    X_train: Any,
    y_train: Any,
    eval_X: Any = None,
    eval_y: Any = None,
    early_stopping_rounds: int = 50,
    max_estimators: int = 2000,
) -> Any:
    """Fit ``model``, using early stopping for gradient boosters when a validation
    set is supplied.

    For XGBoost/LightGBM this raises ``n_estimators`` to ``max_estimators`` and
    stops once the validation metric plateaus — empirically more accurate than a
    fixed tree count (it trains more where it helps, and not where it overfits).
    All other estimators get a plain ``fit``. Falls back to plain ``fit`` if the
    early-stopping call errors for any reason.
    """
    if spec.supports_early_stopping and eval_X is not None and len(eval_X) > 0:
        module = type(model).__module__
        try:
            if module.startswith("xgboost"):
                model.set_params(n_estimators=max_estimators, early_stopping_rounds=early_stopping_rounds)
                model.fit(X_train, y_train, eval_set=[(eval_X, eval_y)], verbose=False)
                return model
            if module.startswith("lightgbm"):
                from lightgbm import early_stopping as _lgb_early_stopping

                eval_metric = "average_precision" if hasattr(model, "predict_proba") else "rmse"
                model.set_params(n_estimators=max_estimators)
                model.fit(
                    X_train, y_train,
                    eval_set=[(eval_X, eval_y)], eval_metric=eval_metric,
                    callbacks=[_lgb_early_stopping(early_stopping_rounds, verbose=False)],
                )
                return model
        except Exception as e:  # noqa: BLE001 — never let tuning of one model kill the run
            logger.warning(f"Early-stopping fit failed for {spec.name} ({e}); using plain fit")
    model.fit(X_train, y_train)
    return model


def get_fitted_n_estimators(model: Any) -> Optional[int]:
    """Best-effort actual tree count after (possibly early-stopped) fitting."""
    bi = getattr(model, "best_iteration", None)
    if bi is not None:
        return int(bi) + 1
    bi = getattr(model, "best_iteration_", None)  # lightgbm
    if bi is not None:
        return int(bi)
    return getattr(model, "n_estimators", None)


def apply_imbalance_handling(model: Any, spec: ModelSpec, y_train: Any) -> Any:
    """Inject data-dependent class-imbalance parameters into a model instance.

    Models exposing ``class_weight`` already carry ``class_weight="balanced"`` via
    their default params. Tree-boosters like XGBoost instead need a numeric
    ``scale_pos_weight`` (= n_negative / n_positive) computed from the training
    labels — this is set here just before fitting. Multiclass targets are left
    untouched. Returns the same model for convenience.
    """
    if not spec.imbalance_param:
        return model
    import numpy as np

    y = np.asarray(y_train)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return model
    neg, pos = float(counts[0]), float(counts[1])
    if pos <= 0:
        return model
    try:
        model.set_params(**{spec.imbalance_param: neg / pos})
        logger.debug(f"Set {spec.imbalance_param}={neg / pos:.3f} for imbalance handling")
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not set {spec.imbalance_param}: {e}")
    return model
