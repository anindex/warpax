"""Combined report generation for spacetime analysis.

Produces multi-panel figures with:
- Energy condition margins
- Curvature invariants
- Energy density profiles
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from numpy.typing import NDArray

from ..geometry.metric import GridSpec


def generate_report(
    margins: dict[str, NDArray],
    invariants: dict[str, NDArray],
    grid: GridSpec,
    metric_name: str = "Spacetime",
    slice_axis: int = 2,
    slice_index: int | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Generate a comprehensive analysis report figure.

    Parameters
    ----------
    margins : dict
        Energy condition margin fields {name: NDArray}.
    invariants : dict
        Curvature invariant fields {name: NDArray}.
    grid : GridSpec
        Grid specification.
    metric_name : str
        Name for the figure title.
    slice_axis : int
        Axis to slice.
    slice_index : int
        Index along slice axis.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
    """
    n_margins = len(margins)
    n_invariants = len(invariants)
    n_cols = max(n_margins, n_invariants)
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 10))
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    if slice_index is None:
        slice_index = grid.shape[slice_axis] // 2

    remaining = [i for i in range(len(grid.shape)) if i != slice_axis]
    x_axis = grid.axes[remaining[0]]
    y_axis = grid.axes[remaining[1]]
    axis_labels = ["x", "y", "z"]

    # Row 1: Energy condition margins
    for i, (name, margin) in enumerate(margins.items()):
        ax = axes[0, i]
        slc = [slice(None)] * margin.ndim
        slc[slice_axis] = slice_index
        data = margin[tuple(slc)]

        vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1e-15)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.pcolormesh(x_axis, y_axis, data.T, cmap="RdBu", norm=norm, shading="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{name} Margin")
        ax.set_xlabel(axis_labels[remaining[0]])
        ax.set_ylabel(axis_labels[remaining[1]])
        ax.set_aspect("equal")

    # Hide unused margin axes
    for i in range(n_margins, n_cols):
        axes[0, i].set_visible(False)

    # Row 2: Curvature invariants
    for i, (name, inv_field) in enumerate(invariants.items()):
        ax = axes[1, i]
        slc = [slice(None)] * inv_field.ndim
        slc[slice_axis] = slice_index
        data = inv_field[tuple(slc)]

        im = ax.pcolormesh(x_axis, y_axis, data.T, cmap="inferno", shading="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(name)
        ax.set_xlabel(axis_labels[remaining[0]])
        ax.set_ylabel(axis_labels[remaining[1]])
        ax.set_aspect("equal")

    for i in range(n_invariants, n_cols):
        axes[1, i].set_visible(False)

    fig.suptitle(f"{metric_name} Analysis Report", fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
