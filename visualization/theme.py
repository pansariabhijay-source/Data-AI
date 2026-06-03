"""
Axiom Visualization Theme — Matplotlib/Seaborn dark theme
matching the Axiom void/glass aesthetic.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np

# ── Axiom Color Palette ─────────────────────────────────────────────────────

VOID = "#09090b"
SURFACE = "#111114"
ELEVATED = "#18181c"
PANEL = "#1c1c21"
GLASS_BORDER = "#1f1f24"

TEXT_PRIMARY = "#fafafa"
TEXT_SECONDARY = "#a1a1aa"
TEXT_MUTED = "#52525b"
TEXT_GHOST = "#3f3f46"

ACCENT = "#818cf8"
ACCENT_MUTED = "#6366f1"
SUCCESS = "#34d399"
WARNING = "#fbbf24"
DESTRUCTIVE = "#f87171"
INFO = "#38bdf8"

# Agent palette
PALETTE = [
    "#818cf8", "#a78bfa", "#c084fc", "#e879f9",
    "#f472b6", "#fb923c", "#34d399", "#38bdf8",
    "#fbbf24", "#f87171", "#4ade80", "#22d3ee",
]

SEQUENTIAL = ["#312e81", "#4338ca", "#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe", "#e0e7ff"]


def apply_axiom_theme():
    """Apply the Axiom dark theme to matplotlib and seaborn."""
    plt.rcParams.update({
        # Figure
        "figure.facecolor": VOID,
        "figure.edgecolor": VOID,
        "figure.dpi": 150,
        "figure.figsize": (10, 6),

        # Axes
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GLASS_BORDER,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Grid
        "grid.color": GLASS_BORDER,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,

        # Ticks
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        # Text
        "text.color": TEXT_PRIMARY,

        # Legend
        "legend.facecolor": ELEVATED,
        "legend.edgecolor": GLASS_BORDER,
        "legend.fontsize": 10,
        "legend.labelcolor": TEXT_SECONDARY,

        # Lines
        "lines.linewidth": 2,
        "lines.antialiased": True,

        # Patches
        "patch.edgecolor": GLASS_BORDER,

        # Savefig
        "savefig.facecolor": VOID,
        "savefig.edgecolor": VOID,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })

    # Seaborn style
    sns.set_style("darkgrid", {
        "axes.facecolor": SURFACE,
        "figure.facecolor": VOID,
        "grid.color": GLASS_BORDER,
        "axes.edgecolor": GLASS_BORDER,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
    })

    sns.set_palette(PALETTE)


def get_axiom_cmap():
    """Get Axiom sequential colormap."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("axiom", SEQUENTIAL, N=256)


def get_diverging_cmap():
    """Get Axiom diverging colormap (red-void-blue)."""
    from matplotlib.colors import LinearSegmentedColormap
    colors = [DESTRUCTIVE, VOID, ACCENT]
    return LinearSegmentedColormap.from_list("axiom_div", colors, N=256)
