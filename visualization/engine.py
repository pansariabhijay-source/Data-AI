"""
Axiom Visualization Engine — Core renderer.

Generates Matplotlib/Seaborn visualizations as base64 PNG strings
with the Axiom dark theme. Supports both Free and Enterprise modes.
"""

from __future__ import annotations

import base64
import io
import traceback
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from visualization.theme import apply_axiom_theme, PALETTE, ACCENT, SUCCESS, WARNING, DESTRUCTIVE, TEXT_SECONDARY, VOID, SURFACE, get_axiom_cmap, get_diverging_cmap

from core.logging_config import get_logger

logger = get_logger("viz_engine")


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _safe_generate(fn, *args, **kwargs) -> Optional[dict]:
    """Safely run a visualization generator, catching errors."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Visualization generation failed ({fn.__name__}): {e}")
        return None


# ── FREE MODE VISUALIZATIONS ───────────────────────────────────────────────


def generate_missing_values_heatmap(df: pd.DataFrame) -> dict:
    """Generate a missing values heatmap."""
    apply_axiom_theme()
    null_data = df.isnull()
    if null_data.sum().sum() == 0:
        # Still generate showing no missing values
        fig, ax = plt.subplots(figsize=(10, max(3, len(df.columns) * 0.3)))
        ax.text(0.5, 0.5, "No Missing Values Detected",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=16, color=SUCCESS, fontweight="bold")
        ax.set_facecolor(SURFACE)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.suptitle("Missing Values Analysis", fontsize=14, fontweight="bold", color="#fafafa")
    else:
        cols_with_nulls = [c for c in df.columns if df[c].isnull().any()]
        fig, ax = plt.subplots(figsize=(max(8, len(cols_with_nulls) * 0.8), 6))
        subset = null_data[cols_with_nulls].head(100)
        sns.heatmap(subset.T, cbar=True, cmap=["#1c1c21", DESTRUCTIVE],
                    yticklabels=True, xticklabels=False, ax=ax,
                    cbar_kws={"shrink": 0.5, "label": "Missing"})
        ax.set_title("Missing Values Heatmap", fontsize=14, fontweight="bold", pad=15)
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=9)

    return {
        "name": "Missing Values Heatmap",
        "type": "missing_values",
        "description": f"Missing values analysis across {len(df.columns)} features",
        "base64_png": _fig_to_base64(fig),
        "category": "basic",
    }


def generate_correlation_heatmap(df: pd.DataFrame) -> dict:
    """Generate correlation matrix heatmap for numeric columns."""
    apply_axiom_theme()
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Not enough numeric columns",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color=TEXT_SECONDARY)
        ax.set_facecolor(SURFACE)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        n = min(numeric_df.shape[1], 20)
        corr = numeric_df.iloc[:, :n].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)))
        sns.heatmap(corr, mask=mask, annot=n <= 15, fmt=".2f",
                    cmap=get_diverging_cmap(), center=0,
                    square=True, linewidths=0.5, linecolor=VOID,
                    ax=ax, annot_kws={"size": 8},
                    cbar_kws={"shrink": 0.7, "label": "Correlation"})
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
        ax.tick_params(axis="both", labelsize=9)

    return {
        "name": "Correlation Heatmap",
        "type": "correlation",
        "description": f"Pearson correlation for {numeric_df.shape[1]} numeric features",
        "base64_png": _fig_to_base64(fig),
        "category": "basic",
    }


def generate_distributions(df: pd.DataFrame, max_features: int = 12) -> dict:
    """Generate feature distribution histograms."""
    apply_axiom_theme()
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_features]
    n = len(numeric_cols)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No numeric features found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color=TEXT_SECONDARY)
        ax.set_facecolor(SURFACE)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            data = df[col].dropna()
            ax.hist(data, bins=30, color=PALETTE[i % len(PALETTE)], alpha=0.8, edgecolor=VOID)
            ax.set_title(col, fontsize=10, fontweight="600", pad=6)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()

    return {
        "name": "Feature Distributions",
        "type": "distributions",
        "description": f"Distribution histograms for {n} numeric features",
        "base64_png": _fig_to_base64(fig),
        "category": "basic",
    }


def generate_boxplots(df: pd.DataFrame, max_features: int = 12) -> dict:
    """Generate boxplots for outlier detection."""
    apply_axiom_theme()
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_features]
    n = len(numeric_cols)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No numeric features", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=TEXT_SECONDARY)
        ax.set_facecolor(SURFACE)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        fig, ax = plt.subplots(figsize=(max(8, n * 0.8), 5))
        # Normalize for comparison
        from sklearn.preprocessing import StandardScaler
        try:
            scaled = pd.DataFrame(
                StandardScaler().fit_transform(df[numeric_cols].dropna()),
                columns=numeric_cols
            )
            bp = ax.boxplot([scaled[c].values for c in numeric_cols],
                           labels=[c[:15] for c in numeric_cols],
                           patch_artist=True, showfliers=True,
                           flierprops=dict(marker="o", markersize=3, markerfacecolor=DESTRUCTIVE, alpha=0.5))
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(PALETTE[i % len(PALETTE)])
                patch.set_alpha(0.6)
            for median in bp["medians"]:
                median.set_color("#fafafa")
                median.set_linewidth(1.5)
            ax.set_title("Outlier Detection (Box Plots)", fontsize=14, fontweight="bold", pad=15)
            ax.set_ylabel("Standardized Value", fontsize=10)
            ax.tick_params(axis="x", rotation=45, labelsize=8)
        except Exception:
            ax.text(0.5, 0.5, "Could not generate boxplots", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color=TEXT_SECONDARY)
            ax.set_facecolor(SURFACE)

    return {
        "name": "Outlier Box Plots",
        "type": "boxplots",
        "description": f"Box plots for {n} features showing outlier distribution",
        "base64_png": _fig_to_base64(fig),
        "category": "basic",
    }


def generate_target_distribution(df: pd.DataFrame, target_column: str) -> Optional[dict]:
    """Generate target variable distribution."""
    if target_column not in df.columns:
        return None

    apply_axiom_theme()
    target = df[target_column].dropna()
    n_unique = target.nunique()

    fig, ax = plt.subplots(figsize=(8, 5))

    if n_unique <= 20:
        # Classification - bar chart
        counts = target.value_counts()
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]
        bars = ax.bar(range(len(counts)), counts.values, color=colors, alpha=0.8, edgecolor=VOID)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([str(l)[:20] for l in counts.index], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Target Distribution: {target_column}", fontsize=14, fontweight="bold", pad=15)

        # Add count labels
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", fontsize=9, color=TEXT_SECONDARY)
    else:
        # Regression - histogram with KDE
        ax.hist(target, bins=40, color=ACCENT, alpha=0.7, edgecolor=VOID, density=True)
        try:
            from scipy.stats import gaussian_kde
            kde_x = np.linspace(target.min(), target.max(), 200)
            kde = gaussian_kde(target)
            ax.plot(kde_x, kde(kde_x), color=SUCCESS, linewidth=2)
        except Exception:
            pass
        ax.set_xlabel(target_column, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"Target Distribution: {target_column}", fontsize=14, fontweight="bold", pad=15)

    fig.tight_layout()

    return {
        "name": "Target Distribution",
        "type": "target_distribution",
        "description": f"Distribution of target variable '{target_column}' ({n_unique} unique values)",
        "base64_png": _fig_to_base64(fig),
        "category": "basic",
    }


def generate_dtype_chart(df: pd.DataFrame) -> dict:
    """Generate data type distribution chart."""
    apply_axiom_theme()
    dtype_counts = df.dtypes.astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(dtype_counts))]
    wedges, texts, autotexts = ax.pie(
        dtype_counts.values, labels=dtype_counts.index,
        colors=colors, autopct="%1.0f%%",
        pctdistance=0.85, startangle=90,
        wedgeprops=dict(width=0.4, edgecolor=VOID, linewidth=2)
    )
    for t in texts:
        t.set_color(TEXT_SECONDARY)
        t.set_fontsize(10)
    for t in autotexts:
        t.set_color("#fafafa")
        t.set_fontsize(9)
        t.set_fontweight("bold")

    ax.set_title("Data Type Distribution", fontsize=14, fontweight="bold", pad=15)

    return {
        "name": "Data Types",
        "type": "dtype_distribution",
        "description": f"Data type breakdown across {len(df.columns)} columns",
        "base64_png": _fig_to_base64(fig),
        "category": "basic",
    }


# ── ENTERPRISE MODE VISUALIZATIONS ─────────────────────────────────────────


def generate_pairplot(df: pd.DataFrame, target: Optional[str] = None, max_features: int = 5) -> Optional[dict]:
    """Generate pairplot for top numeric features."""
    apply_axiom_theme()
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns[:max_features])
    if len(numeric_cols) < 2:
        return None

    plot_df = df[numeric_cols].dropna().head(500)
    if target and target in df.columns and target not in numeric_cols:
        # Align by label (.loc), not position (.iloc): plot_df.index holds the
        # original row labels, which only coincide with positions for a clean
        # RangeIndex. On a sampled/filtered df, .iloc[labels] goes out of bounds.
        plot_df[target] = df.loc[plot_df.index, target]

    g = sns.pairplot(
        plot_df, hue=target if target and target in plot_df.columns else None,
        diag_kind="kde", plot_kws={"alpha": 0.6, "s": 15},
        palette=PALETTE[:plot_df[target].nunique()] if target and target in plot_df.columns else None,
        height=2.2
    )
    g.figure.suptitle("Feature Pair Plot", y=1.02, fontsize=14, fontweight="bold")

    return {
        "name": "Pair Plot",
        "type": "pairplot",
        "description": f"Pairwise relationships for top {len(numeric_cols)} features",
        "base64_png": _fig_to_base64(g.figure),
        "category": "advanced",
    }


def generate_pca_plot(df: pd.DataFrame, target: Optional[str] = None) -> Optional[dict]:
    """Generate PCA 2D scatter plot."""
    apply_axiom_theme()
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 10:
        return None

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(numeric_df.head(2000))
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))

    if target and target in df.columns:
        # Label-based alignment (.loc); .iloc would break on a non-RangeIndex df.
        labels = df.loc[numeric_df.head(2000).index, target]
        unique_labels = labels.unique()
        for i, label in enumerate(unique_labels[:10]):
            mask = labels == label
            ax.scatter(components[mask, 0], components[mask, 1],
                      c=PALETTE[i % len(PALETTE)], label=str(label)[:20],
                      alpha=0.6, s=20, edgecolors="none")
        ax.legend(fontsize=8, framealpha=0.3)
    else:
        ax.scatter(components[:, 0], components[:, 1],
                  c=ACCENT, alpha=0.5, s=15, edgecolors="none")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=11)
    ax.set_title("PCA — 2D Projection", fontsize=14, fontweight="bold", pad=15)

    return {
        "name": "PCA Projection",
        "type": "pca",
        "description": f"2D PCA explaining {sum(pca.explained_variance_ratio_):.1%} of total variance",
        "base64_png": _fig_to_base64(fig),
        "category": "advanced",
    }


def generate_feature_importance(importances: dict, top_n: int = 20) -> Optional[dict]:
    """Generate feature importance bar chart."""
    if not importances:
        return None
    apply_axiom_theme()

    sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names = [n[:25] for n, _ in sorted_imp]
    values = [v for _, v in sorted_imp]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    colors = [ACCENT if v >= 0 else DESTRUCTIVE for v in values]
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.8, edgecolor=VOID, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold", pad=15)

    return {
        "name": "Feature Importance",
        "type": "feature_importance",
        "description": f"Top {len(names)} features by importance score",
        "base64_png": _fig_to_base64(fig),
        "category": "advanced",
    }


def generate_confusion_matrix(y_true, y_pred, labels=None) -> Optional[dict]:
    """Generate confusion matrix heatmap."""
    apply_axiom_theme()
    from sklearn.metrics import confusion_matrix as cm_func

    try:
        matrix = cm_func(y_true, y_pred, labels=labels)
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(matrix) * 1.2), max(5, len(matrix))))
    sns.heatmap(matrix, annot=True, fmt="d", cmap=get_axiom_cmap(),
                xticklabels=labels or "auto", yticklabels=labels or "auto",
                ax=ax, linewidths=1, linecolor=VOID,
                annot_kws={"size": 12, "fontweight": "bold"})
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual", fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)

    return {
        "name": "Confusion Matrix",
        "type": "confusion_matrix",
        "description": "Classification confusion matrix",
        "base64_png": _fig_to_base64(fig),
        "category": "advanced",
    }


def generate_model_comparison(models: list) -> Optional[dict]:
    """Generate model comparison bar chart from model results."""
    if not models:
        return None
    apply_axiom_theme()

    trained = [m for m in models if m.get("status") == "trained" and m.get("metrics")]
    if not trained:
        return None

    metric_key = list(trained[0]["metrics"].keys())[0] if trained[0]["metrics"] else None
    if not metric_key:
        return None

    trained.sort(key=lambda m: m["metrics"].get(metric_key, 0), reverse=True)

    names = [m["name"].replace("Classifier", "").replace("Regressor", "").strip()[:20] for m in trained]
    values = [m["metrics"].get(metric_key, 0) for m in trained]
    is_best = [m.get("is_best", False) for m in trained]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.45)))
    colors = [SUCCESS if b else ACCENT for b in is_best]
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.8, edgecolor=VOID, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(metric_key, fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", pad=15)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color=TEXT_SECONDARY)

    fig.tight_layout()

    return {
        "name": "Model Comparison",
        "type": "model_comparison",
        "description": f"Performance comparison across {len(trained)} trained models",
        "base64_png": _fig_to_base64(fig),
        "category": "basic",
    }


# ── BATCH GENERATORS ───────────────────────────────────────────────────────


def generate_free_mode_viz(df: pd.DataFrame, target_column: Optional[str] = None) -> list[dict]:
    """Generate all Free mode visualizations for a dataset."""
    results = []

    generators = [
        lambda: generate_missing_values_heatmap(df),
        lambda: generate_correlation_heatmap(df),
        lambda: generate_distributions(df),
        lambda: generate_boxplots(df),
        lambda: generate_dtype_chart(df),
    ]

    if target_column:
        generators.append(lambda: generate_target_distribution(df, target_column))

    for gen in generators:
        result = _safe_generate(gen)
        if result:
            results.append(result)

    return results


def generate_enterprise_mode_viz(df: pd.DataFrame, target_column: Optional[str] = None) -> list[dict]:
    """Generate all Enterprise mode visualizations for a dataset."""
    results = generate_free_mode_viz(df, target_column)

    advanced_generators = [
        lambda: generate_pairplot(df, target_column),
        lambda: generate_pca_plot(df, target_column),
    ]

    for gen in advanced_generators:
        result = _safe_generate(gen)
        if result:
            results.append(result)

    return results
