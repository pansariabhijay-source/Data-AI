"""
Feature Engineering Tool — encoding, scaling, feature generation, and selection.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.config import FeatureEngineeringConfig, Settings
from core.constants import ProblemType
from core.exceptions import FeatureEngineeringError
from core.logging_config import get_logger, log_stage_timing
from core.state import ErrorReport, FeatureEngineeringSummary, PipelineState
from core.utils import ensure_directory

logger = get_logger("feature_engineering")

# Mutual-information feature ranking is O(n log n) (kNN density estimation) and
# dominates feature_engineering time on large data. It's only used to RANK
# features, and a row sample preserves the ranking of the genuinely informative
# columns (only zero-MI noise columns reshuffle). Cap the rows fed to MI here;
# selection is still applied to the full frame. Override via MI_SAMPLE_ROWS
# (0 disables sampling).
try:
    _MI_SAMPLE_ROWS = int(os.environ.get("MI_SAMPLE_ROWS", "50000"))
except (TypeError, ValueError):
    _MI_SAMPLE_ROWS = 50000


class FeatureEngineeringService:
    """Deterministic feature engineering and selection logic."""

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        self._config = config

    def drop_numeric_id_columns(self, df: pd.DataFrame, target: Optional[str] = None) -> tuple[pd.DataFrame, dict[str, str]]:
        """Drop numeric columns that are ID-like (e.g. User_ID, transaction_id as int).

        A numeric column is treated as an ID when BOTH hold:
        - Its name matches an ID-like pattern (contains 'id', 'index', 'key', 'num' at a
          word boundary) OR its unique-value ratio exceeds the config threshold.
        - It is integer-typed (floats with fractional parts carry real signal).

        IDs injected as features let models memorise row identity rather than learning
        the underlying distribution — the result is a near-perfect train score and
        random-chance generalisation.
        """
        from core.constants import (
            DEFAULT_NUMERIC_ID_NAME_MIN_RATIO,
            DEFAULT_NUMERIC_ID_UNIQUENESS_RATIO,
        )
        import re

        # "number" is excluded: "Number_of_X" columns are count features, not IDs.
        # True ID patterns use "num"/"no" as suffix (e.g. account_num, card_no).
        id_pattern = re.compile(r"(?i)(^|_)(id|idx|index|key|no|num|code|seq)($|_)")
        dropped_map: dict[str, str] = {}
        n_rows = max(len(df), 1)

        int_cols = df.select_dtypes(include=["int8", "int16", "int32", "int64",
                                             "uint8", "uint16", "uint32", "uint64"]).columns.tolist()
        if target and target in int_cols:
            int_cols.remove(target)

        to_drop = []
        for col in int_cols:
            nunique = df[col].nunique()
            ratio = nunique / n_rows
            name_is_id = bool(id_pattern.search(col))
            # Near-unique → an ID regardless of name. A matching name is only enough
            # when the column is ALSO fairly high-cardinality — otherwise a low-card
            # coded categorical (e.g. integer area/product code with a few levels)
            # would be wrongly discarded just for having "code"/"num" in its name.
            ratio_is_id = ratio > DEFAULT_NUMERIC_ID_UNIQUENESS_RATIO
            name_and_card_is_id = name_is_id and ratio > DEFAULT_NUMERIC_ID_NAME_MIN_RATIO
            if ratio_is_id or name_and_card_is_id:
                to_drop.append(col)
                reason = f"name pattern + unique ratio {ratio:.2f}" if name_and_card_is_id else f"unique ratio {ratio:.2f}"
                dropped_map[col] = f"dropped (numeric ID-like: {reason}, {nunique} unique)"
                logger.info(f"Dropped numeric ID column '{col}' ({reason})")

        if to_drop:
            df = df.drop(columns=to_drop)
        return df, dropped_map

    def encode_categoricals(self, df: pd.DataFrame, target: Optional[str] = None) -> tuple[pd.DataFrame, dict[str, str]]:
        """One-hot or label encode categorical columns based on cardinality.

        ID-like columns (near-unique values, e.g. user_id) are dropped rather than
        label-encoded: encoding them injects a high-cardinality noise feature that
        models latch onto and overfit, while carrying no generalizable signal.
        """
        encoding_map: dict[str, str] = {}
        cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
        if target and target in cat_cols:
            cat_cols.remove(target)

        n_rows = max(len(df), 1)
        for col in cat_cols:
            nunique = df[col].nunique()
            if nunique <= self._config.max_onehot_cardinality:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                encoding_map[col] = f"one_hot ({nunique} categories)"
            elif nunique / n_rows > self._config.id_uniqueness_ratio:
                df = df.drop(columns=[col])
                encoding_map[col] = f"dropped (ID-like, {nunique} unique)"
                logger.info(f"Dropped ID-like column '{col}' ({nunique} unique values)")
            else:
                # Frequency-encode high-cardinality columns: map each category to
                # its relative frequency. Unlike label encoding (which injects an
                # arbitrary ordinal that models misread as magnitude), this is a
                # meaningful, leakage-safe signal — rare categories (low frequency)
                # are often the predictive ones (e.g. a rare device/merchant in
                # fraud). Selection downstream drops it if uninformative.
                s = df[col].astype(str)
                freq = s.map(s.value_counts(normalize=True))
                df[col] = freq.astype(float).fillna(0.0)
                encoding_map[col] = f"frequency_encoded ({nunique} categories)"

        return df, encoding_map

    def encode_target(self, df: pd.DataFrame, target: str, problem_type: ProblemType) -> pd.DataFrame:
        """Label encode the target column if it's categorical classification."""
        if problem_type != ProblemType.CLASSIFICATION:
            return df
        if target not in df.columns:
            return df
        if pd.api.types.is_numeric_dtype(df[target]):
            return df
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target].astype(str))
        logger.info(f"Label encoded target '{target}' with {len(le.classes_)} classes")
        return df

    def extract_datetime_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Extract useful features from datetime columns.

        Beyond the raw parts (year/month/dow/hour), add:
        * **cyclical** sin/cos for month, day-of-week and hour, so models (esp.
          linear ones) see that December is adjacent to January and 23:00 to 00:00
          rather than maximally far apart;
        * **is_weekend** (a common, strong signal);
        * **recency_days** — days before the most recent timestamp in the column
          (captures "how old / how recent", e.g. freshly-created scam postings).

        Downstream feature selection prunes any of these that aren't informative.
        """
        created: list[str] = []
        # Datetime columns that were parsed in preprocessing are serialized as
        # strings when saved to CSV and re-read. Re-parse them here so downstream
        # feature extraction works even after a CSV round-trip.
        # include both legacy object and newer pandas StringDtype
        obj_cols = [c for c in df.columns
                    if not pd.api.types.is_numeric_dtype(df[c])
                    and not pd.api.types.is_datetime64_any_dtype(df[c])]
        for col in obj_cols:
            sample = df[col].dropna().head(200)
            try:
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    logger.info(f"Re-parsed string column '{col}' as datetime")
            except Exception:
                pass
        dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        for col in dt_cols:
            s = df[col]
            # Capture the first usable datetime (widest-spanning) as the out-of-time
            # split key, in epoch seconds and aligned to df.index, before the raw
            # column is dropped. The splitter uses it to sort chronologically.
            try:
                if getattr(self, "_split_time", None) is None and s.notna().mean() > 0.5:
                    self._split_time = (s.astype("int64") / 1e9).where(s.notna())
            except Exception:
                pass
            for attr, suffix in [("year", "_year"), ("month", "_month"), ("dayofweek", "_dow"), ("hour", "_hour")]:
                try:
                    df[col + suffix] = getattr(s.dt, attr)
                    created.append(col + suffix)
                except Exception:
                    pass
            # Cyclical encodings (period-aware): sin/cos so the wrap-around is smooth.
            for attr, period, suffix in [("month", 12, "_month"), ("dayofweek", 7, "_dow"), ("hour", 24, "_hour")]:
                try:
                    vals = getattr(s.dt, attr).astype(float)
                    df[col + suffix + "_sin"] = np.sin(2 * np.pi * vals / period)
                    df[col + suffix + "_cos"] = np.cos(2 * np.pi * vals / period)
                    created += [col + suffix + "_sin", col + suffix + "_cos"]
                except Exception:
                    pass
            try:
                df[col + "_is_weekend"] = (s.dt.dayofweek >= 5).astype(int)
                created.append(col + "_is_weekend")
            except Exception:
                pass
            try:
                df[col + "_recency_days"] = (s.max() - s).dt.days
                created.append(col + "_recency_days")
            except Exception:
                pass
            df = df.drop(columns=[col])
        return df, created

    def scale_features(self, df: pd.DataFrame, target: Optional[str] = None) -> tuple[pd.DataFrame, dict[str, str]]:
        """Scale numeric features."""
        scaling_map: dict[str, str] = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in numeric_cols:
            numeric_cols.remove(target)

        if not numeric_cols or self._config.scaling_method == "none":
            return df, scaling_map

        if self._config.scaling_method == "standard":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif self._config.scaling_method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif self._config.scaling_method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            return df, scaling_map

        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        for col in numeric_cols:
            scaling_map[col] = self._config.scaling_method

        return df, scaling_map

    def remove_low_variance(self, df: pd.DataFrame, target: Optional[str] = None) -> tuple[pd.DataFrame, list[str]]:
        """Drop columns that are (near-)constant — they give a model nothing.

        Deliberately conservative: it removes ONLY degenerate columns and never a
        rare-but-present signal. The previous implementation min-max scaled each
        column and applied a variance floor, which misfired badly:

          * a right-skewed feature (transaction amount, counts) got compressed by
            the scaling so its variance fell under the floor and the whole column —
            often the most predictive one — was dropped;
          * a rare one-hot dummy or binary flag (e.g. a 0.8%-prevalence fraud
            indicator) was clipped to all-zeros and dropped, deleting exactly the
            rare category that carries the signal.

        A constant column has one distinct value; a near-constant one has a single
        value covering ≥ (1 − variance_threshold) of rows. We test that directly
        (scale-invariant, skew-proof) — but a column with ≥ 2 distinct values whose
        minority share is meaningful is kept, and mutual-information selection
        downstream prunes anything genuinely uninformative.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in numeric_cols:
            numeric_cols.remove(target)
        if not numeric_cols:
            return df, []

        n_rows = len(df)
        # Treat a value covering all-but-an-epsilon of the rows as "constant".
        # variance_threshold (default 0.01) sets that epsilon at 1% — but never
        # higher than the floor below, so we don't delete a 1%-prevalence flag.
        near_const_frac = max(1.0 - float(self._config.variance_threshold), 0.999)
        removed: list[str] = []
        for col in numeric_cols:
            s = df[col]
            nun = s.nunique(dropna=True)
            if nun <= 1:
                removed.append(col)
                continue
            if n_rows and s.value_counts(dropna=True).iloc[0] / n_rows >= near_const_frac:
                removed.append(col)

        if removed:
            df = df.drop(columns=removed)
            logger.info(f"Removed {len(removed)} (near-)constant features: {removed}")
        return df, removed

    def remove_correlated(self, df: pd.DataFrame, target: Optional[str] = None) -> tuple[pd.DataFrame, list[str]]:
        """Remove one of each pair of highly correlated features.

        When two features are near-duplicates, keep the one MORE correlated with
        the target and drop the other. The old version dropped purely by column
        order, so it could discard the more predictive member of the pair — a
        silent accuracy loss. With no usable target signal it falls back to
        order-based dropping.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in numeric_cols:
            numeric_cols.remove(target)
        if len(numeric_cols) < 2:
            return df, []

        corr_matrix = df[numeric_cols].corr().abs()

        # Relevance to target (|corr|), used to decide which of a pair to keep.
        target_rel: dict[str, float] = {}
        if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
            for c in numeric_cols:
                try:
                    target_rel[c] = abs(float(df[c].corr(df[target])))
                except Exception:
                    target_rel[c] = 0.0
                if pd.isna(target_rel[c]):
                    target_rel[c] = 0.0

        thresh = self._config.correlation_threshold
        dropped: set[str] = set()
        cols = list(corr_matrix.columns)
        for i, a in enumerate(cols):
            if a in dropped:
                continue
            for b in cols[i + 1:]:
                if b in dropped:
                    continue
                if corr_matrix.loc[a, b] > thresh:
                    # Drop the less target-relevant of the pair (b on ties / no target).
                    if target_rel.get(a, 0.0) < target_rel.get(b, 0.0):
                        dropped.add(a)
                        break  # a is gone; stop pairing it
                    dropped.add(b)

        to_drop = [c for c in cols if c in dropped]
        if to_drop:
            df = df.drop(columns=to_drop)
            logger.info(f"Removed {len(to_drop)} correlated features (kept the more target-relevant of each pair): {to_drop}")
        return df, to_drop

    def add_domain_features(self, df: pd.DataFrame, target: Optional[str] = None) -> tuple[pd.DataFrame, list[str]]:
        """Add high-value domain features that generic pipelines miss.

        Covers four proven fraud-detection signals:
        1. Haversine distance between cardholder location and merchant location.
        2. Cardholder age derived from a date-of-birth column.
        3. Log-transformed transaction amount (skewed distributions hurt linear models).
        4. Per-cardholder transaction velocity (count and total spend in the window).
        """
        created: list[str] = []

        # ── 1. Geo distance (cardholder ↔ merchant) ──────────────────────────
        lat_cols  = [c for c in df.columns if c.lower() in ("lat", "latitude")]
        long_cols = [c for c in df.columns if c.lower() in ("long", "longitude", "lng")]
        mlat_cols = [c for c in df.columns if "merch" in c.lower() and ("lat" in c.lower())]
        mlong_cols= [c for c in df.columns if "merch" in c.lower() and any(x in c.lower() for x in ("long", "lng"))]
        if lat_cols and long_cols and mlat_cols and mlong_cols:
            try:
                lat  = np.radians(df[lat_cols[0]].astype(float))
                lon  = np.radians(df[long_cols[0]].astype(float))
                mlat = np.radians(df[mlat_cols[0]].astype(float))
                mlon = np.radians(df[mlong_cols[0]].astype(float))
                dlat = mlat - lat
                dlon = mlon - lon
                a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(mlat) * np.sin(dlon / 2) ** 2
                df["geo_distance_km"] = 6371.0 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
                created.append("geo_distance_km")
                logger.info("Added haversine geo_distance_km feature")
            except Exception as e:
                logger.warning(f"geo_distance_km failed: {e}")

        # ── 2. Age from date-of-birth ─────────────────────────────────────────
        dob_cols = [c for c in df.columns if c.lower() in ("dob", "date_of_birth", "birthdate", "birth_date")]
        for col in dob_cols:
            if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    dob = pd.to_datetime(df[col], errors="coerce")
                    if dob.notna().mean() > 0.5:
                        # Use the most recent transaction timestamp as the reference
                        # date so ages reflect the cardholder's age at transaction time.
                        tx_dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
                        ref = df[tx_dt_cols[0]].max() if tx_dt_cols else pd.Timestamp.now()
                        df["age_years"] = ((ref - dob).dt.days / 365.25).clip(0, 120).fillna(40)
                        created.append("age_years")
                        df = df.drop(columns=[col])
                        logger.info(f"Derived age_years from '{col}'")
                        break
                except Exception as e:
                    logger.warning(f"age_years from '{col}' failed: {e}")

        # ── 3. Log-transform of amount / price columns ────────────────────────
        amt_cols = [c for c in df.columns if c.lower() in ("amt", "amount", "transaction_amount", "price", "value")
                    and c != target and pd.api.types.is_numeric_dtype(df[c])]
        for col in amt_cols:
            log_col = f"log_{col}"
            if log_col not in df.columns:
                try:
                    df[log_col] = np.log1p(df[col].clip(lower=0))
                    created.append(log_col)
                except Exception as e:
                    logger.warning(f"log_{col} failed: {e}")

        # ── 4. Per-cardholder transaction velocity ────────────────────────────
        card_cols = [c for c in df.columns if c.lower() in ("cc_num", "card_num", "card_number",
                                                             "user_id", "cardholder_id", "account_id")]
        if card_cols and amt_cols:
            card_col = card_cols[0]
            amt_col  = amt_cols[0]
            try:
                grp = df.groupby(card_col)[amt_col]
                df["card_tx_count"]    = grp.transform("count").astype(np.float32)
                df["card_amt_mean"]    = grp.transform("mean").astype(np.float32)
                df["card_amt_std"]     = grp.transform("std").fillna(0).astype(np.float32)
                df["card_amt_max"]     = grp.transform("max").astype(np.float32)
                # Z-score: how unusual is this transaction for this cardholder?
                std = df["card_amt_std"].replace(0, np.nan)
                df["card_amt_zscore"]  = ((df[amt_col] - df["card_amt_mean"]) / std).fillna(0).clip(-5, 5).astype(np.float32)
                new_cols = ["card_tx_count", "card_amt_mean", "card_amt_std",
                            "card_amt_max", "card_amt_zscore"]
                created.extend(new_cols)
                logger.info(f"Added {len(new_cols)} per-cardholder velocity features from '{card_col}'")
            except Exception as e:
                logger.warning(f"Velocity features failed: {e}")

        return df, created

    def generate_interactions(self, df: pd.DataFrame, target: Optional[str], importances: dict[str, float], top_n: int = 6) -> tuple[pd.DataFrame, list[str]]:
        """Generate pairwise multiplicative interactions for the top-N features.

        On datasets where individual features have weak signal, non-linear interactions
        (e.g. amount × velocity_score) often surface patterns that no single feature
        encodes. We limit to top-N ranked features to avoid O(n²) explosion, and only
        generate products (not squares, which just scale the original).
        Downstream MI selection prunes any that aren't informative.
        """
        created: list[str] = []
        if not importances:
            return df, created

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in numeric_cols:
            numeric_cols.remove(target)

        # Rank by importance, keep top_n that are actually in df
        ranked = [f for f, _ in sorted(importances.items(), key=lambda x: x[1], reverse=True)
                  if f in numeric_cols][:top_n]
        if len(ranked) < 2:
            return df, created

        try:
            for i, a in enumerate(ranked):
                for b in ranked[i + 1:]:
                    col_name = f"{a}__x__{b}"
                    if col_name not in df.columns:
                        df[col_name] = df[a] * df[b]
                        created.append(col_name)
            if created:
                logger.info(f"Generated {len(created)} pairwise interaction features from top-{top_n} ranked features")
        except Exception as e:
            logger.warning(f"Interaction generation failed (non-critical): {e}")
            for col in created:
                if col in df.columns:
                    df = df.drop(columns=[col])
            created = []

        return df, created

    def select_k_best(self, df: pd.DataFrame, target: str, problem_type: ProblemType) -> tuple[pd.DataFrame, dict[str, float]]:
        """Select top K features using mutual information."""
        if target not in df.columns:
            return df, {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
        if not numeric_cols:
            return df, {}

        k = min(self._config.select_k_best, len(numeric_cols))
        X = df[numeric_cols].fillna(0)
        y = df[target]

        # Rank on a row sample to bound MI cost on large data (selection below is
        # still applied to the full frame). The sample preserves the ordering of
        # the informative features; only zero-MI noise columns reshuffle.
        if _MI_SAMPLE_ROWS > 0 and len(X) > _MI_SAMPLE_ROWS:
            samp = np.random.default_rng(42).choice(len(X), size=_MI_SAMPLE_ROWS, replace=False)
            X_mi, y_mi = X.iloc[samp], y.iloc[samp]
        else:
            X_mi, y_mi = X, y

        try:
            if problem_type == ProblemType.CLASSIFICATION:
                from sklearn.feature_selection import mutual_info_classif
                scores = mutual_info_classif(X_mi, y_mi, random_state=42)
            else:
                from sklearn.feature_selection import mutual_info_regression
                scores = mutual_info_regression(X_mi, y_mi, random_state=42)

            importance = dict(zip(numeric_cols, [float(s) for s in scores]))
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            selected = [f for f, _ in sorted_features[:k]]
            # Keep target + selected features + any non-numeric features
            non_numeric = [c for c in df.columns if c not in numeric_cols and c != target]
            keep = [target] + selected + non_numeric
            df = df[[c for c in keep if c in df.columns]]
            logger.info(f"Selected top {k} features by mutual information")
            return df, importance
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return df, {}

    @log_stage_timing("feature_engineering")
    def run(self, state: PipelineState, settings: Settings) -> PipelineState:
        data_path = state.cleaned_data_path
        if not data_path:
            raise FeatureEngineeringError("No cleaned data path available")

        df = pd.read_csv(data_path, low_memory=False)
        n_before = len(df.columns)
        target = state.target_column
        problem_type = ProblemType(state.problem_type) if state.problem_type else ProblemType.UNKNOWN
        self._split_time = None  # reset per run; captured in extract_datetime_features

        # Act on leakage detected during data collection: a feature that is ~perfectly
        # correlated with the target leaks the answer, inflating training metrics while
        # the model fails to generalize. Detection without removal is a silent trap.
        leakage_dropped: list[str] = []
        if self._config.drop_leakage_columns:
            leaked = state.data_quality_flags.get("potential_leakage") or []
            leakage_dropped = [c for c in leaked if c in df.columns and c != target]
            if leakage_dropped:
                df = df.drop(columns=leakage_dropped)
                logger.warning(f"Dropped {len(leakage_dropped)} leakage columns: {leakage_dropped}")

        # Encode target first
        if target:
            df = self.encode_target(df, target, problem_type)

        # For regression, detect and log-transform a heavily skewed target.
        # Skewness > 1 means the distribution is right-tailed (e.g. price, income,
        # counts) and models that minimise squared error will over-fit to the outliers.
        # log1p maps the target to a roughly normal space where MSE is meaningful,
        # and the inverse (expm1) can be applied at inference time.
        if problem_type == ProblemType.REGRESSION and target and target in df.columns:
            try:
                tgt = df[target].dropna()
                if pd.api.types.is_numeric_dtype(tgt) and (tgt > 0).all() and tgt.skew() > 1.0:
                    df[target] = np.log1p(df[target])
                    state.data_quality_flags["log_transformed_target"] = {
                        "column": target,
                        "original_skew": round(float(tgt.skew()), 3),
                        "note": "Target was log1p-transformed due to high skewness. Predictions are in log-space.",
                    }
                    logger.info(
                        f"Log-transformed skewed target '{target}' (skew={tgt.skew():.2f}). "
                        "Metrics are computed in log-space; expm1 for original units."
                    )
            except Exception as e:
                logger.warning(f"Log-target transform failed (non-critical): {e}")

        # Drop numeric ID columns before any other engineering
        df, numeric_id_dropped_map = self.drop_numeric_id_columns(df, target)

        # Datetime features (also re-parses string columns that look like datetimes)
        df, dt_created = self.extract_datetime_features(df)

        # Domain-specific features: geo distance, age, log-amount, velocity
        df, domain_created = self.add_domain_features(df, target)
        dt_created = dt_created + domain_created

        # Encode categoricals
        df, encoding_map = self.encode_categoricals(df, target)
        encoding_map.update(numeric_id_dropped_map)

        # Remove low variance
        df, low_var_removed = self.remove_low_variance(df, target)

        # Remove correlated
        df, corr_removed = self.remove_correlated(df, target)

        # Feature selection (first pass — rank before interactions)
        importances: dict[str, float] = {}
        if target and target in df.columns and problem_type != ProblemType.CLUSTERING:
            df, importances = self.select_k_best(df, target, problem_type)

        # Generate pairwise interactions from top-ranked features, then re-select.
        # Off by default (boosters model interactions natively; explicit products
        # mostly add noise) — see FeatureEngineeringConfig.enable_interactions.
        interaction_created: list[str] = []
        if (
            getattr(self._config, "enable_interactions", False)
            and importances
            and problem_type == ProblemType.CLASSIFICATION
        ):
            df, interaction_created = self.generate_interactions(df, target, importances, top_n=6)
            if interaction_created:
                # Re-rank with interactions included so selection is fair
                df, importances = self.select_k_best(df, target, problem_type)

        # Scale features (after selection)
        df, scaling_map = self.scale_features(df, target)

        # Cast boolean feature columns to int so they survive the model-matrix build.
        # Downstream (training/improvement/finalization) assembles X via
        # `select_dtypes(include=[np.number])`, which EXCLUDES bool — so a bool feature
        # would be silently dropped from the model while remaining in
        # `selected_features`, desyncing the inference manifest from the actual model
        # (and discarding genuine signal). Casting to int keeps them numeric everywhere.
        bool_cols = [c for c in df.select_dtypes(include=["bool"]).columns if c != target]
        if bool_cols:
            df[bool_cols] = df[bool_cols].astype("int8")
            logger.info(f"Cast {len(bool_cols)} boolean feature(s) to int: {bool_cols}")

        # selected_features is computed from df BEFORE attaching the hidden split-time
        # key, so the key is never treated as a model feature.
        selected = [c for c in df.columns if c != target]

        # Attach the out-of-time split key (epoch seconds from the primary datetime)
        # as a hidden column the splitter uses to sort chronologically, then drops.
        from core.constants import AXIOM_SPLIT_TIME_COL
        if getattr(self, "_split_time", None) is not None:
            try:
                df[AXIOM_SPLIT_TIME_COL] = self._split_time.reindex(df.index)
            except Exception as e:
                logger.warning(f"Could not attach split-time key (non-critical): {e}")

        # Save
        artifact_dir = ensure_directory(Path(settings.pipeline.artifact_dir) / state.run_id)
        featured_path = artifact_dir / "featured_data.csv"
        df.to_csv(featured_path, index=False)

        all_removed = leakage_dropped + list(numeric_id_dropped_map.keys()) + low_var_removed + corr_removed
        # Record WHY each feature was removed, for the report's explanation section.
        removal_reasons: dict[str, str] = {}
        for c in leakage_dropped:
            removal_reasons[c] = ("almost perfectly predicts the target (data leakage) — removed "
                                  "so the score reflects real-world performance, not the answer leaking in")
        for c, why in numeric_id_dropped_map.items():
            removal_reasons[c] = "looks like a row identifier, which carries no generalizable signal"
        for c in low_var_removed:
            removal_reasons[c] = "(near-)constant — the same value in almost every row, so nothing to learn from"
        for c in corr_removed:
            removal_reasons[c] = "nearly a duplicate of another feature (>0.95 correlated); kept the more target-relevant one"
        # Categoricals dropped during encoding (recorded in the encoding map as "dropped ...").
        for c, why in encoding_map.items():
            if "dropped" in why and c not in removal_reasons:
                removal_reasons[c] = "high-cardinality identifier-like column — encoding it would only add noise"

        state.featured_data_path = str(featured_path)
        state.selected_features = selected  # excludes the hidden split-time key
        state.feature_importances = importances
        state.feature_engineering_summary = FeatureEngineeringSummary(
            features_created=dt_created + interaction_created,
            features_removed=all_removed,
            removal_reasons=removal_reasons,
            encoding_applied=encoding_map,
            scaling_applied=scaling_map,
            feature_importances=importances,
            selected_features=state.selected_features,
            n_features_before=n_before,
            n_features_after=len(df.columns),
        )
        state.mark_stage_end("feature_engineering")
        logger.info(f"Feature engineering: {n_before}->{len(df.columns)} columns")
        return state


_service: Optional[FeatureEngineeringService] = None
_state: Optional[PipelineState] = None
_settings: Optional[Settings] = None


def init_feature_engineering(state: PipelineState, settings: Settings) -> None:
    global _service, _state, _settings
    _service = FeatureEngineeringService(settings.feature_engineering)
    _state = state
    _settings = settings


def engineer_features(instruction: str) -> str:
    """Engineer and select features for ML modeling.

    Encodes categoricals, extracts datetime features, removes low-variance
    and correlated features, selects top features, and scales numerics.

    Args:
        instruction: Natural language description of feature engineering task.

    Returns:
        JSON summary of feature engineering results.
    """
    global _service, _state, _settings
    if _service is None or _state is None or _settings is None:
        return json.dumps({"error": "Feature engineering service not initialized"})
    try:
        _state.mark_stage_start("feature_engineering")
        _state = _service.run(_state, _settings)
        s = _state.feature_engineering_summary
        return json.dumps({
            "status": "success",
            "features_before": s.n_features_before if s else 0,
            "features_after": s.n_features_after if s else 0,
            "features_created": len(s.features_created) if s else 0,
            "features_removed": len(s.features_removed) if s else 0,
        }, default=str)
    except Exception as e:
        logger.exception("Feature engineering failed")
        _state.add_error(ErrorReport(
            severity="critical", stage="feature_engineering",
            error_type=type(e).__name__, root_cause=str(e),
            traceback_str=traceback.format_exc(),
            recommended_fix="Check encoding and scaling settings",
            retryable=True,
        ))
        return json.dumps({"error": str(e)})
