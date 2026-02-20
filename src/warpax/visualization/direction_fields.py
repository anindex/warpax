"""Worst-observer direction field plots.

Quiver plots showing the spatial direction of the observer boost that
achieves the worst-case (minimum) energy condition value at each point.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ..geometry.types import GridSpec
from ._style import DOUBLE_COL, SINGLE_COL


def plot_worst_observer_field(
    worst_params: NDArray,
    grid: GridSpec,
    slice_axis: int = 2,
    slice_index: int | None = None,
    title: str = r"Worst-Observer Boost Direction (bubble along $x$)",
    ax: plt.Axes | None = None,
    subsample: int = 1,
    significance_mask: NDArray | None = None,
) -> plt.Axes:
    """Plot quiver field of worst-observer boost directions.

    Arrows are normalized to unit length to show *direction* only.
    Color encodes rapidity magnitude (sinh ζ).  Points where the observer
    choice is physically irrelevant (flat spacetime) are suppressed via
    the significance_mask.

    Parameters
    ----------
    worst_params : NDArray
        Observer parameters (ζ, θ, φ) at each grid point, shape (*grid.shape, 3).
    grid : GridSpec
        Grid specification.
    slice_axis : int
        Axis to slice.
    slice_index : int
        Index along slice axis.
    title : str
        Plot title.
    ax : plt.Axes
        Matplotlib axes.
    subsample : int
        Subsample factor for quiver plot readability.
    significance_mask : NDArray or None
        Boolean mask, shape (*grid.shape,). True where the observer direction
        is physically meaningful (e.g., where robust != eulerian margin).
        Arrows at False points are suppressed to avoid noise.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(
            1, 1, figsize=(0.7 * DOUBLE_COL, 0.7 * DOUBLE_COL * 0.8),
        )

    if slice_index is None:
        slice_index = worst_params.shape[slice_axis] // 2

    slc = [slice(None)] * (worst_params.ndim - 1)
    slc[slice_axis] = slice_index
    slc.append(slice(None))
    params_2d = worst_params[tuple(slc)]

    # Extract rapidity, theta, phi
    zeta = params_2d[..., 0]
    theta = params_2d[..., 1]
    phi = params_2d[..., 2]

    # Project boost direction to 2D
    remaining = [i for i in range(len(grid.shape)) if i != slice_axis]
    boost_mag = np.sinh(zeta)
    dx = boost_mag * np.sin(theta) * np.cos(phi)
    dy = boost_mag * np.sin(theta) * np.sin(phi)
    dz = boost_mag * np.cos(theta)

    components = [dx, dy, dz]
    u = components[remaining[0]]
    v = components[remaining[1]]

    # Compute per-point 2D magnitude for normalization and coloring
    mag_2d = np.sqrt(u**2 + v**2)

    # Apply significance mask: zero out arrows in physically trivial regions
    if significance_mask is not None:
        slc_mask = [slice(None)] * significance_mask.ndim
        slc_mask[slice_axis] = slice_index
        mask_2d = significance_mask[tuple(slc_mask)]
        u = np.where(mask_2d, u, 0.0)
        v = np.where(mask_2d, v, 0.0)
        mag_2d = np.where(mask_2d, mag_2d, 0.0)
        boost_mag = np.where(mask_2d, boost_mag, 0.0)

    # Normalize arrows to unit length (direction only); color encodes magnitude
    norm_factor = np.where(mag_2d > 1e-30, mag_2d, 1.0)
    u_norm = u / norm_factor
    v_norm = v / norm_factor
    # Zero out normalized arrows where magnitude was zero
    u_norm = np.where(mag_2d > 1e-30, u_norm, 0.0)
    v_norm = np.where(mag_2d > 1e-30, v_norm, 0.0)

    x_axis = grid.axes[remaining[0]]
    y_axis = grid.axes[remaining[1]]
    X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")

    # Subsample
    s = subsample
    mag_sub = mag_2d[::s, ::s]
    max_mag = float(np.nanmax(mag_sub)) if mag_sub.size > 0 else 0.0

    # If all arrows are zero (e.g., entire slice is masked), show empty field
    if max_mag < 1e-30:
        ax.text(
            0.5, 0.5, "No significant observer dependence\nin this slice",
            transform=ax.transAxes, ha="center", va="center", fontsize=9,
        )
    else:
        # Ensure small arrows are visible by setting a minimum magnitude
        # (at least 20% of max) while preserving direction
        min_vis = 0.2 * max_mag
        mag_display = np.where(mag_sub > 1e-30,
                               np.maximum(mag_sub, min_vis), 0.0)
        scale_factor = mag_display / np.where(mag_sub > 1e-30, mag_sub, 1.0)
        q = ax.quiver(
            X[::s, ::s],
            Y[::s, ::s],
            u_norm[::s, ::s] * scale_factor,
            v_norm[::s, ::s] * scale_factor,
            mag_sub,           # color still reflects true magnitude
            cmap="viridis",
            clim=(0, max_mag),
            scale=25,          # lower scale -> bigger arrows overall
            width=0.005,       # slightly thicker shaft
            headwidth=3.0,
            headlength=3.5,
            pivot="mid",       # center arrows on grid points
        )
        cb = plt.colorbar(q, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label(r"Boost magnitude $|\sinh\zeta|$")

    # Max rapidity annotation (only over significant region)
    if significance_mask is not None:
        slc_m2 = [slice(None)] * significance_mask.ndim
        slc_m2[slice_axis] = slice_index
        m2d = significance_mask[tuple(slc_m2)]
        max_zeta = float(np.nanmax(zeta[m2d])) if m2d.any() else 0.0
    else:
        max_zeta = float(np.nanmax(zeta))
    ax.text(
        0.02, 0.98, rf"$\zeta_{{\max}} = {max_zeta:.2f}$",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    axis_labels = ["x", "y", "z"]
    ax.set_xlabel(axis_labels[remaining[0]])
    ax.set_ylabel(axis_labels[remaining[1]])
    ax.set_title(title)
    ax.set_aspect("equal")

    # Auto-zoom to significant region if mask provided
    if significance_mask is not None:
        slc_m3 = [slice(None)] * significance_mask.ndim
        slc_m3[slice_axis] = slice_index
        m2d = significance_mask[tuple(slc_m3)]
        if m2d.any():
            Xm, Ym = X[m2d], Y[m2d]
            xlo, xhi = float(Xm.min()), float(Xm.max())
            ylo, yhi = float(Ym.min()), float(Ym.max())
            domain_area = (x_axis[-1] - x_axis[0]) * (y_axis[-1] - y_axis[0])
            feat_area = max(xhi - xlo, 1e-30) * max(yhi - ylo, 1e-30)
            if feat_area / domain_area < 0.6:
                pad_x = max((xhi - xlo) * 0.3, x_axis[1] - x_axis[0])
                pad_y = max((yhi - ylo) * 0.3, y_axis[1] - y_axis[0])
                ax.set_xlim(max(xlo - pad_x, x_axis[0]), min(xhi + pad_x, x_axis[-1]))
                ax.set_ylim(max(ylo - pad_y, y_axis[0]), min(yhi + pad_y, y_axis[-1]))

    return ax
