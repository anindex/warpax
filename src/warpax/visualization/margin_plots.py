"""Energy condition margin heatmaps.

Diverging colormap: red = violated (negative), blue = satisfied (positive).
Plots 2D slices (typically z=0 plane) of margin fields.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from numpy.typing import NDArray

from ..geometry.metric import GridSpec


def plot_margin_slice(
    margin: NDArray,
    grid: GridSpec,
    slice_axis: int = 2,
    slice_index: int | None = None,
    title: str = "Energy Condition Margin",
    ax: plt.Axes | None = None,
    cmap: str = "RdBu",
) -> plt.Axes:
    """Plot a 2D slice of an energy condition margin field.

    Parameters
    ----------
    margin : NDArray
        Margin field, shape (*grid.shape,). Negative = violated.
    grid : GridSpec
        Grid specification.
    slice_axis : int
        Axis to slice through (default 2 = z).
    slice_index : int
        Index along slice_axis (default: middle).
    title : str
        Plot title.
    ax : plt.Axes
        Matplotlib axes (created if None).
    cmap : str
        Colormap name (diverging recommended).

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if slice_index is None:
        slice_index = margin.shape[slice_axis] // 2

    # Extract 2D slice
    slc = [slice(None)] * margin.ndim
    slc[slice_axis] = slice_index
    data = margin[tuple(slc)]

    # Get axes for the remaining dimensions
    remaining = [i for i in range(len(grid.shape)) if i != slice_axis]
    x_axis = grid.axes[remaining[0]]
    y_axis = grid.axes[remaining[1]]
    axis_labels = ["x", "y", "z"]

    # Diverging normalization centered at 0
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    if vmax < 1e-15:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.pcolormesh(x_axis, y_axis, data.T, cmap=cmap, norm=norm, shading="auto")
    plt.colorbar(im, ax=ax, label="Margin")
    ax.set_xlabel(axis_labels[remaining[0]])
    ax.set_ylabel(axis_labels[remaining[1]])
    ax.set_title(title)
    ax.set_aspect("equal")

    return ax


def plot_all_conditions(
    margins: dict[str, NDArray],
    grid: GridSpec,
    slice_axis: int = 2,
    slice_index: int | None = None,
    suptitle: str = "Energy Conditions",
) -> plt.Figure:
    """Plot all four energy condition margins in a 2×2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    for ax, (name, margin) in zip(axes_flat, margins.items()):
        plot_margin_slice(
            margin,
            grid,
            slice_axis=slice_axis,
            slice_index=slice_index,
            title=f"{name} Margin",
            ax=ax,
        )

    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig
