"""Kinematic scalar field heatmap figures.

Produces figures showing expansion (theta), shear-squared,
and vorticity-squared for the Eulerian congruence across warp metrics.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from numpy.typing import NDArray

from ._style import DOUBLE_COL, apply_style, extract_slice, save_or_return, slice_axes

apply_style()


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

    Vorticity-squared is identically zero for the Eulerian congruence (Frobenius
    theorem); when its slice maximum is below 1e-15 the panel uses a fixed
    [0, 1] range and is annotated with an identity-zero label.

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
    x_ax, y_ax, (xlabel, ylabel) = slice_axes(grid_shape, grid_bounds, slice_axis)

    theta_2d = extract_slice(theta, slice_axis)
    sigma_2d = extract_slice(sigma_sq, slice_axis)
    omega_2d = extract_slice(omega_sq, slice_axis)
    omega_2d = np.maximum(omega_2d, 0.0)

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    outer = fig.add_gridspec(1, 3, wspace=0.2, top=0.82 if title else 0.95, bottom=0.15)
    axes, caxes = [], []
    for i in range(3):
        inner = GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=outer[0, i],
            width_ratios=[1, 0.04],
            wspace=0.06,
        )
        axes.append(fig.add_subplot(inner[0, 0]))
        caxes.append(fig.add_subplot(inner[0, 1]))

    vmax_t = max(abs(float(np.nanmin(theta_2d))), abs(float(np.nanmax(theta_2d))))
    if vmax_t < 1e-15:
        vmax_t = 1.0
    norm_t = TwoSlopeNorm(vmin=-vmax_t, vcenter=0, vmax=vmax_t)
    im1 = axes[0].pcolormesh(x_ax, y_ax, theta_2d.T, cmap="coolwarm", norm=norm_t, shading="auto")
    axes[0].set_title(r"$\theta$", fontsize=9)
    axes[0].set_ylabel(ylabel)
    cb1 = fig.colorbar(im1, cax=caxes[0])
    cb1.ax.tick_params(labelsize=6)

    im2 = axes[1].pcolormesh(x_ax, y_ax, sigma_2d.T, cmap="inferno", shading="auto")
    axes[1].set_title(r"$\sigma^2$", fontsize=9)
    cb2 = fig.colorbar(im2, cax=caxes[1])
    cb2.ax.tick_params(labelsize=6)

    omega_max = float(np.nanmax(np.abs(omega_2d)))
    if omega_max < 1e-15:
        im3 = axes[2].pcolormesh(
            x_ax,
            y_ax,
            omega_2d.T,
            cmap="inferno",
            vmin=0,
            vmax=1,
            shading="auto",
        )
        axes[2].text(
            0.5,
            0.5,
            r"$\equiv 0$",
            transform=axes[2].transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="white",
        )
    else:
        im3 = axes[2].pcolormesh(
            x_ax,
            y_ax,
            omega_2d.T,
            cmap="inferno",
            vmin=0,
            shading="auto",
        )
    axes[2].set_title(r"$\omega^2$", fontsize=9)
    cb3 = fig.colorbar(im3, cax=caxes[2])
    cb3.ax.tick_params(labelsize=6)

    axes[1].set_xlabel(xlabel)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    if title:
        fig.suptitle(title, fontsize=9)

    return save_or_return(fig, save_path)
