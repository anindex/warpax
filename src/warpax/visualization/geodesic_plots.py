"""Geodesic trajectory visualization: tidal eigenvalue evolution and blueshift profiles.

Produces figures for:
- Tidal eigenvalue evolution along timelike geodesics (spaghettification diagnostic)
- Blueshift factor profiles along null geodesics (frequency shift through warp bubbles)

Both functions accept an optional ``ax`` parameter for subplot embedding and an
optional ``save_path`` for direct PDF export.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ._style import COLORS, DOUBLE_COL, LINE_STYLES, apply_style, save_or_return

apply_style()


def plot_tidal_evolution(
    tidal_eigenvalues: NDArray,
    proper_times: NDArray,
    *,
    title: str | None = None,
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot tidal eigenvalue evolution along a geodesic trajectory.

    Renders the 3 spatial tidal eigenvalues as a function of proper time (or
    coordinate time). A vertical dashed line marks the peak total tidal
    magnitude when it exceeds 1e-10; its location need not coincide with a
    bubble wall.

    Parameters
    ----------
    tidal_eigenvalues : NDArray, shape (N, 3) or (N, 4)
        Tidal eigenvalues at each saved point along the geodesic. If 4
        columns are provided, the near-zero eigenvalue (associated with the
        velocity direction) is dropped automatically by keeping the three
        largest in absolute value per row.
    proper_times : NDArray, shape (N,)
        Proper time (or coordinate time / affine parameter) at each point.
    title : str or None
        Optional figure title.
    save_path : str or None
        If given, save the figure as PDF to this path.
    ax : matplotlib Axes or None
        If given, plot on this axes (for subplot embedding).

    Returns
    -------
    matplotlib Figure
    """
    eigs = np.asarray(tidal_eigenvalues)
    tau = np.asarray(proper_times)

    if eigs.ndim == 2 and eigs.shape[1] == 4:
        sorted_idx = np.argsort(np.abs(eigs), axis=1)
        eigs = np.array([eigs[i, sorted_idx[i, 1:]] for i in range(len(eigs))])

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL, 0.5 * DOUBLE_COL))
    else:
        fig = ax.get_figure()

    n_eigs = eigs.shape[1] if eigs.ndim == 2 else 1
    labels = [f"$\\lambda_{{{i + 1}}}$" for i in range(n_eigs)]

    if eigs.ndim == 1:
        ax.plot(tau, eigs, color=COLORS[0], label=labels[0], linewidth=1.2)
    else:
        mevery = max(1, len(tau) // 15)
        for i in range(n_eigs):
            ls = LINE_STYLES[i % len(LINE_STYLES)]
            ax.plot(
                tau,
                eigs[:, i],
                color=COLORS[i % len(COLORS)],
                linestyle=ls["linestyle"],
                marker=ls["marker"],
                markersize=ls["markersize"],
                markevery=mevery,
                label=labels[i],
                linewidth=1.2,
            )

    if eigs.ndim == 2:
        total_tidal = np.sum(np.abs(eigs), axis=1)
    else:
        total_tidal = np.abs(eigs)
    peak_idx = np.argmax(total_tidal)
    if total_tidal[peak_idx] > 1e-10:
        ax.axvline(
            tau[peak_idx], color="gray", linestyle="--", alpha=0.7, label="Peak tidal magnitude"
        )

    ax.set_xlabel(r"Proper time $\tau$")
    ax.set_ylabel("Tidal eigenvalue")
    ax.legend(loc="best", framealpha=0.9)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    if title:
        ax.set_title(title)

    if own_fig:
        fig.tight_layout(pad=0.5)

    return save_or_return(fig, save_path)


def plot_blueshift_profile(
    blueshift_factors: NDArray,
    positions_x: NDArray,
    *,
    title: str | None = None,
    save_path: str | None = None,
    ax: plt.Axes | None = None,
    bubble_center: float = 0.0,
    bubble_radius: float = 1.0,
    bubble_sigma: float = 8.0,
) -> plt.Figure:
    """Plot frequency ratio profile along a null geodesic.

    Renders the frequency ratio nu_obs/nu_emit vs x-position, with log
    scale on the y-axis when the maximum ratio exceeds 10. The warp bubble
    wall region is shown as a shaded band of half-width 2/sigma around
    r = R on each side of the bubble center (a top-hat profile).

    Parameters
    ----------
    blueshift_factors : NDArray, shape (N,)
        Frequency ratio nu_obs / nu_emit at each point.
    positions_x : NDArray, shape (N,)
        x-coordinate at each saved point along the null geodesic.
    title : str or None
        Optional figure title.
    save_path : str or None
        If given, save the figure as PDF to this path.
    ax : matplotlib Axes or None
        If given, plot on this axes (for subplot embedding).
    bubble_center : float
        x-coordinate of the bubble center (default 0.0).
    bubble_radius : float
        Characteristic radius R of the bubble (default 1.0).
    bubble_sigma : float
        Steepness parameter sigma (default 8.0).

    Returns
    -------
    matplotlib Figure
    """
    bs = np.asarray(blueshift_factors)
    x = np.asarray(positions_x)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL, 0.5 * DOUBLE_COL))
    else:
        fig = ax.get_figure()

    ax.plot(x, bs, color=COLORS[0], linewidth=1.2, label=r"$\nu_{\rm obs}/\nu_{\rm emit}$")

    max_bs = np.nanmax(np.abs(bs))
    if max_bs > 10:
        ax.set_yscale("log")

    wall_half = 2.0 / bubble_sigma if bubble_sigma > 0 else 0.2
    wall_lo = bubble_center - bubble_radius - wall_half
    wall_hi = bubble_center - bubble_radius + wall_half
    wall_lo2 = bubble_center + bubble_radius - wall_half
    wall_hi2 = bubble_center + bubble_radius + wall_half
    ax.axvspan(wall_lo, wall_hi, alpha=0.15, color="orange", label="Bubble wall")
    ax.axvspan(wall_lo2, wall_hi2, alpha=0.15, color="orange")

    ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.3, linestyle="--")
    ax.set_xlabel("$x$ position")
    ax.set_ylabel(r"Frequency ratio $\nu_{\rm obs}/\nu_{\rm emit}$")
    ax.legend(loc="best", framealpha=0.9)
    if title:
        ax.set_title(title)

    if own_fig:
        fig.tight_layout(pad=0.5)

    return save_or_return(fig, save_path)
