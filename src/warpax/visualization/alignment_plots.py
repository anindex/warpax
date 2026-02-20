"""Worst-observer alignment angle histogram plots.

Renders multi-panel histograms showing the distribution of alignment angles
between the BFGS worst-case observer boost direction and the eigenvector
predicted by algebraic analysis.  Each panel corresponds to a different
warp velocity, demonstrating that the worst-case observer is typically
misaligned with the eigenvector prediction.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._style import apply_style, DOUBLE_COL


def plot_alignment_histogram(
    angle_arrays: dict[float, np.ndarray],
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot multi-panel alignment angle histograms.

    Parameters
    ----------
    angle_arrays : dict[float, np.ndarray]
        Mapping from warp velocity (v_s) to array of alignment angles
        in degrees.  Each array contains angles at DEC-violation points.
    save_path : str or None
        If provided, save the figure to this path.
    figsize : tuple[float, float] or None
        Figure size in inches.  Defaults to (DOUBLE_COL, DOUBLE_COL * 0.3).

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    apply_style()

    n_panels = len(angle_arrays)
    if n_panels == 0:
        fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL, 2))
        ax.text(0.5, 0.5, "No alignment data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    if figsize is None:
        figsize = (DOUBLE_COL, DOUBLE_COL * 0.45)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=figsize,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    sorted_velocities = sorted(angle_arrays.keys())

    for i, v_s in enumerate(sorted_velocities):
        ax = axes[i]
        angles = angle_arrays[v_s]
        n_pts = len(angles)

        if n_pts > 0:
            ax.hist(
                angles,
                bins=18,
                range=(0, 90),
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7,
                color="#0072B2",
            )
            median_val = float(np.median(angles))
            ax.axvline(
                median_val,
                color="#D55E00",
                linestyle="--",
                linewidth=1.0,
                label=f"median = {median_val:.0f}$^\\circ$",
            )
            ax.legend(fontsize=7, loc="upper right")
        else:
            ax.text(
                0.5, 0.5, "No violations",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=8,
            )

        ax.set_title(f"$v_s = {v_s}$ ({n_pts} pts)", fontsize=9)
        ax.set_xlim(0, 90)
        ax.set_xlabel("Alignment angle (deg)", fontsize=8)

        if i == 0:
            ax.set_ylabel("Count", fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    return fig
