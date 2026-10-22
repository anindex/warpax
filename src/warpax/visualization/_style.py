"""Shared matplotlib style configuration.

Sets rcParams for CQG (Classical and Quantum Gravity) compatible figures:
- Computer Modern fonts via ``text.usetex`` (with mathtext fallback for CI)
- CQG single-column (3.39") and double-column (6.69") figure widths
- 300 DPI for rasterised elements in saved figures, 150 DPI for screen display

``SINGLE_COL`` and ``DOUBLE_COL`` are inches: the CQG single column is 8.6 cm and
the double column 17.0 cm. ``mathtext.fontset`` is ``cm`` so that, when usetex is
unavailable (tectonic-only machines), mathtext figures match the LaTeX-built panels.
"""

from __future__ import annotations

import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np


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


SINGLE_COL: float = 3.39
DOUBLE_COL: float = 6.69


STYLE_PARAMS: dict[str, object] = {
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
}


LINE_STYLES: list[dict[str, object]] = [
    {"linestyle": "-", "marker": "o", "markersize": 5},
    {"linestyle": "--", "marker": "s", "markersize": 5},
    {"linestyle": "-.", "marker": "^", "markersize": 5.5},
    {"linestyle": ":", "marker": "D", "markersize": 4.5},
]

COLORS: list[str] = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
]
"""Okabe-Ito palette: blue, vermilion, green, pink, amber, sky blue."""

METRIC_COLORS: dict[str, str] = {
    "Alcubierre": COLORS[0],
    "Natário": COLORS[4],
    "Van den Broeck": COLORS[2],
    "Rodal": COLORS[3],
    "WarpShell": COLORS[5],
    "Schwarzschild": "#555555",
    "Lentz": COLORS[1],
}
"""Per-metric Okabe-Ito (colorblind-safe) colors.

Keys match the metric names used by the analysis scripts, including accents, so a
metric keeps one color across every figure; Schwarzschild is a neutral gray vacuum
control and Lentz (appendix / excluded) is vermilion. ``metric_color`` tolerates
accent and spelling variants.
"""

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


PHASE_COLORMAP: str = "viridis"
"""Sequential perceptually-uniform map for strictly-negative worst-EC margin heatmaps
(about 8 decades of dynamic range); a diverging map would waste the unused positive half."""


def apply_style() -> None:
    """Apply matplotlib style settings."""
    plt.rcParams.update(STYLE_PARAMS)


def save_or_return(fig: plt.Figure, save_path: str | None) -> plt.Figure:
    """Save *fig* to *save_path*, close it and return the closed figure; otherwise return *fig*.

    The format follows the path extension; PDF when it has none.
    """
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def slice_axes(grid_shape, grid_bounds, slice_axis: int):
    """``(x_ax, y_ax, labels)`` of the two in-plane axes of the middle slice along ``slice_axis``."""
    remaining = [i for i in range(len(grid_shape)) if i != slice_axis]
    axis_labels = ["x", "y", "z"]
    extents = [np.linspace(*grid_bounds[dim], grid_shape[dim]) for dim in remaining]
    x_ax, y_ax = extents
    return x_ax, y_ax, [axis_labels[dim] for dim in remaining]


def style_table(table, n_cols: int, fontsize: float, cell_colors=None) -> None:
    """Blue bold header row, alternating body shading, thin edges.

    ``cell_colors[i][j]`` overrides the body shading except where it is ``"white"``.
    """
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.auto_set_column_width(list(range(n_cols)))
    table.scale(1, 1.4)
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold", fontsize=fontsize)
        cell.set_edgecolor("white")
        cell.set_linewidth(0.5)
    n_rows = max(row for row, _ in table.get_celld())
    for i in range(n_rows):
        alt_bg = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(n_cols):
            color = cell_colors[i][j] if cell_colors is not None else "white"
            cell = table[i + 1, j]
            cell.set_facecolor(color if color != "white" else alt_bg)
            cell.set_edgecolor("#D0D0D0")
            cell.set_linewidth(0.5)


def extract_slice(data, slice_axis: int = 2):
    """Middle 2D slice of a 3D array along ``slice_axis``."""
    idx = data.shape[slice_axis] // 2
    slc = [slice(None)] * data.ndim
    slc[slice_axis] = idx
    return data[tuple(slc)]
