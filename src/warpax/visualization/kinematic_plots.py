"""Kinematic scalar field heatmap figures.

Produces figures showing expansion (theta), shear-squared,
and vorticity-squared for the Eulerian congruence across warp metrics.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from numpy.typing import NDArray

from ._style import DOUBLE_COL, SINGLE_COL, apply_style

apply_style()


def _save_or_return(fig: plt.Figure, save_path: str | None) -> plt.Figure:
    """Save figure as PDF if save_path given, otherwise return for interactive use."""
    if save_path is not None:
        fig.savefig(save_path, format="pdf")
        plt.close(fig)
    return fig


def _extract_slice(data: NDArray, slice_axis: int = 2) -> NDArray:
    """Extract the middle 2D slice along the given axis."""
    idx = data.shape[slice_axis] // 2
    slc = [slice(None)] * data.ndim
    slc[slice_axis] = idx
    return data[tuple(slc)]


def plot_kinematic_scalars(
    theta: NDArray,
    sigma_sq: NDArray,
    omega_sq: NDArray,
    grid_bounds: list[tuple[float, float]],
    grid_shape: tuple[int, ...],
    title: str = "",
    slice_axis: int = 2,
    save_path: str | None = None,
) -> plt.Figure:
    """1x3 panel figure: expansion | shear-squared | vorticity-squared.

    Parameters
    ----------
    theta : NDArray
        Expansion scalar, shape (*grid_shape,).
    sigma_sq : NDArray
        Shear-squared scalar, shape (*grid_shape,).
    omega_sq : NDArray
        Vorticity-squared scalar, shape (*grid_shape,).
    grid_bounds : list of (float, float)
        Spatial bounds for each dimension.
    grid_shape : tuple of int
        Grid shape.
    title : str
        Super-title for the figure.
    slice_axis : int
        Axis to slice through (default 2 = z).
    save_path : str or None
        If provided, save as PDF.

    Returns
    -------
    plt.Figure
    """
    remaining = [i for i in range(len(grid_shape)) if i != slice_axis]
    axis_labels = ["x", "y", "z"]

    extents = []
    for dim in remaining:
        lo, hi = grid_bounds[dim]
        extents.append(np.linspace(lo, hi, grid_shape[dim]))
    x_ax, y_ax = extents

    theta_2d = _extract_slice(theta, slice_axis)
    sigma_2d = _extract_slice(sigma_sq, slice_axis)
    omega_2d = _extract_slice(omega_sq, slice_axis)
    omega_2d = np.maximum(omega_2d, 0.0)

    # GridSpec: 3 panel-colorbar groups
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    outer = fig.add_gridspec(1, 3, wspace=0.2,
                             top=0.82 if title else 0.95, bottom=0.15)
    axes, caxes = [], []
    for i in range(3):
        inner = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0, i],
            width_ratios=[1, 0.04], wspace=0.06,
        )
        axes.append(fig.add_subplot(inner[0, 0]))
        caxes.append(fig.add_subplot(inner[0, 1]))

    # Panel 1: Expansion (diverging, signed)
    vmax_t = max(abs(float(np.nanmin(theta_2d))), abs(float(np.nanmax(theta_2d))))
    if vmax_t < 1e-15:
        vmax_t = 1.0
    norm_t = TwoSlopeNorm(vmin=-vmax_t, vcenter=0, vmax=vmax_t)
    im1 = axes[0].pcolormesh(
        x_ax, y_ax, theta_2d.T, cmap="coolwarm", norm=norm_t, shading="auto"
    )
    axes[0].set_title(r"$\theta$", fontsize=9)
    axes[0].set_ylabel(axis_labels[remaining[1]])
    cb1 = fig.colorbar(im1, cax=caxes[0])
    cb1.ax.tick_params(labelsize=6)

    # Panel 2: Shear-squared (positive, inferno)
    im2 = axes[1].pcolormesh(
        x_ax, y_ax, sigma_2d.T, cmap="inferno", shading="auto"
    )
    axes[1].set_title(r"$\sigma^2$", fontsize=9)
    cb2 = fig.colorbar(im2, cax=caxes[1])
    cb2.ax.tick_params(labelsize=6)

    # Panel 3: Vorticity-squared (non-negative by definition)
    omega_max = float(np.nanmax(np.abs(omega_2d)))
    if omega_max < 1e-15:
        # Identically zero (Frobenius theorem for Eulerian congruence)
        im3 = axes[2].pcolormesh(
            x_ax, y_ax, omega_2d.T, cmap="inferno", vmin=0, vmax=1,
            shading="auto",
        )
        axes[2].text(
            0.5, 0.5, r"$\equiv 0$", transform=axes[2].transAxes,
            ha="center", va="center", fontsize=12, color="white",
        )
    else:
        im3 = axes[2].pcolormesh(
            x_ax, y_ax, omega_2d.T, cmap="inferno", vmin=0,
            shading="auto",
        )
    axes[2].set_title(r"$\omega^2$", fontsize=9)
    cb3 = fig.colorbar(im3, cax=caxes[2])
    cb3.ax.tick_params(labelsize=6)

    # Shared x-label (middle panel), y-ticks on leftmost only
    axes[1].set_xlabel(axis_labels[remaining[0]])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    if title:
        fig.suptitle(title, fontsize=9)

    return _save_or_return(fig, save_path)


def plot_kinematic_comparison(
    results_dir: str,
    metrics: list[str],
    v_s: float = 0.5,
    slice_axis: int = 2,
    save_path: str | None = None,
) -> plt.Figure:
    """Multi-row figure: one row per metric showing expansion/shear/vorticity.

    Parameters
    ----------
    results_dir : str
        Directory containing cached kinematic .npz files.
    metrics : list[str]
        Metric names to include.
    v_s : float
        Warp velocity (for filename matching).
    slice_axis : int
        Axis to slice.
    save_path : str or None
        If provided, save as PDF.

    Returns
    -------
    plt.Figure
    """
    # Load available data
    available = []
    for name in metrics:
        path = os.path.join(results_dir, f"{name}_kinematic_vs{v_s}.npz")
        if os.path.exists(path):
            data = np.load(path)
            available.append((name, data))

    if not available:
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2))
        ax.text(0.5, 0.5, "No kinematic data found", ha="center", va="center")
        ax.axis("off")
        return _save_or_return(fig, save_path)

    n_metrics = len(available)

    # GridSpec: 3 equal panels per row, no per-panel colorbars (for even sizing)
    fig = plt.figure(figsize=(DOUBLE_COL, 1.5 * n_metrics))
    gs = fig.add_gridspec(n_metrics, 3, wspace=0.08, hspace=0.25,
                          top=0.93, bottom=0.08)

    axis_labels = ["x", "y", "z"]
    axes_grid = [[fig.add_subplot(gs[r, c]) for c in range(3)]
                 for r in range(n_metrics)]

    for row, (name, data) in enumerate(available):
        theta = data["theta"]
        sigma_sq = data["sigma_sq"]
        omega_sq = data["omega_sq"]
        bounds = [tuple(b) for b in data["grid_bounds"]]
        shape = tuple(data["grid_shape"])

        remaining = [i for i in range(len(shape)) if i != slice_axis]
        extents = []
        for dim in remaining:
            lo, hi = bounds[dim]
            extents.append(np.linspace(lo, hi, shape[dim]))
        x_ax, y_ax = extents

        theta_2d = _extract_slice(theta, slice_axis)
        sigma_2d = _extract_slice(sigma_sq, slice_axis)
        omega_2d = _extract_slice(omega_sq, slice_axis)
        omega_2d = np.maximum(omega_2d, 0.0)

        # Expansion
        vmax = max(abs(float(np.nanmin(theta_2d))), abs(float(np.nanmax(theta_2d))))
        if vmax < 1e-15:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        axes_grid[row][0].pcolormesh(
            x_ax, y_ax, theta_2d.T, cmap="coolwarm", norm=norm, shading="auto"
        )
        axes_grid[row][0].set_ylabel(f"{name}\n{axis_labels[remaining[1]]}")
        if row == 0:
            axes_grid[row][0].set_title(r"$\theta$", fontsize=9)

        # Shear
        axes_grid[row][1].pcolormesh(
            x_ax, y_ax, sigma_2d.T, cmap="inferno", shading="auto"
        )
        axes_grid[row][1].tick_params(labelleft=False)
        if row == 0:
            axes_grid[row][1].set_title(r"$\sigma^2$", fontsize=9)

        # Vorticity (non-negative; annotate if identically zero)
        omega_max = float(np.nanmax(np.abs(omega_2d)))
        if omega_max < 1e-15:
            axes_grid[row][2].pcolormesh(
                x_ax, y_ax, omega_2d.T, cmap="inferno", vmin=0, vmax=1,
                shading="auto",
            )
            axes_grid[row][2].text(
                0.5, 0.5, r"$\equiv 0$", transform=axes_grid[row][2].transAxes,
                ha="center", va="center", fontsize=10, color="white",
            )
        else:
            axes_grid[row][2].pcolormesh(
                x_ax, y_ax, omega_2d.T, cmap="inferno", vmin=0,
                shading="auto",
            )
        axes_grid[row][2].tick_params(labelleft=False)
        if row == 0:
            axes_grid[row][2].set_title(r"$\omega^2$", fontsize=9)

        # x-labels only on bottom row, no y-ticks on cols 1-2
        if row == n_metrics - 1:
            axes_grid[row][1].set_xlabel(axis_labels[remaining[0]])
        else:
            for col in range(3):
                axes_grid[row][col].tick_params(labelbottom=False)

    return _save_or_return(fig, save_path)
