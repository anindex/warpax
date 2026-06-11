"""Shared matplotlib style configuration.

Sets rcParams for CQG (Classical and Quantum Gravity) compatible figures:
- Computer Modern fonts via ``text.usetex`` (with mathtext fallback for CI)
- CQG single-column (3.39") and double-column (6.69") figure widths
- 300 DPI for rasterised elements in saved figures, 150 DPI for screen display
"""
from __future__ import annotations

import shutil
import warnings

import matplotlib.pyplot as plt


def _latex_available() -> bool:
    """Check whether ``latex`` and ``dvipng`` are on PATH."""
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


USE_TEX: bool = _latex_available()

if not USE_TEX:
    warnings.warn(
        "LaTeX or dvipng not found on PATH; falling back to matplotlib mathtext. "
        "For full CQG typesetting install: texlive-latex-extra, cm-super, dvipng",
        stacklevel=2,
    )


SINGLE_COL: float = 3.39   # inches (CQG single column = 8.6 cm)
DOUBLE_COL: float = 6.69   # inches (CQG double column = 17.0 cm)


STYLE_PARAMS: dict[str, object] = {
    # Font / LaTeX
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    # When usetex is unavailable (e.g. tectonic-only machines), render math in
    # Computer Modern so mathtext figures match the LaTeX-built panels.
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # Display / save resolution
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
    # Line widths
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    # Tick direction & placement
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    # Legend styling
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
}


LINE_STYLES: list[dict[str, object]] = [
    {"linestyle": "-",  "marker": "o", "markersize": 5},     # solid + circle
    {"linestyle": "--", "marker": "s", "markersize": 5},     # dashed + square
    {"linestyle": "-.", "marker": "^", "markersize": 5.5},   # dash-dot + triangle
    {"linestyle": ":",  "marker": "D", "markersize": 4.5},   # dotted + diamond
]

COLORS: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # amber
    "#56B4E9",  # sky blue
]

# Per-metric colors (Okabe-Ito, colorblind-safe). Keys match the names used by
# the analysis scripts, including accents, so a metric keeps one color across
# every figure. ``metric_color`` tolerates accent/spelling variants.
METRIC_COLORS: dict[str, str] = {
    "Alcubierre":     COLORS[0],  # blue
    "Natário":        COLORS[4],  # amber
    "Van den Broeck": COLORS[2],  # green
    "Rodal":          COLORS[3],  # reddish purple
    "WarpShell":      COLORS[5],  # sky blue
    "Schwarzschild":  "#555555",  # neutral gray (vacuum control)
    "Lentz":          COLORS[1],  # vermilion (appendix / excluded)
}

# Accent- and spelling-tolerant lookups for the same colors.
_METRIC_COLOR_ALIASES: dict[str, str] = {
    "natario": "Natário",
    "van den broeck": "Van den Broeck",
    "vandenbroeck": "Van den Broeck",
    "vdb": "Van den Broeck",
    "warp shell": "WarpShell",
}


def metric_color(name: str, default: str = "#000000") -> str:
    """Return the canonical color for a metric, tolerant of accents/spelling."""
    if name in METRIC_COLORS:
        return METRIC_COLORS[name]
    key = _METRIC_COLOR_ALIASES.get(name.strip().lower())
    if key is not None:
        return METRIC_COLORS[key]
    return default


# Sequential map for strictly-negative worst-EC margin heatmaps (~8 decade
# dynamic range); diverging maps waste the unused positive half. Crameri et al. 2024.
PHASE_COLORMAP: str = "viridis"


def apply_style() -> None:
    """Apply matplotlib style settings."""
    plt.rcParams.update(STYLE_PARAMS)
