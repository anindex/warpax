"""EC-admissible transport as a function of shell design parameters.

Figures for the design-space sweep: transport heatmap, EC boundary with
hatching, contour isolines, and annotated optimum.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from numpy.typing import NDArray

from ._style import DOUBLE_COL, SINGLE_COL, apply_style

apply_style()

_TRANSPORT_CMAP = "viridis"
_MARGIN_CMAP = "RdBu"
_CONSTRAINT_CMAP = "YlGnBu"
_TIDAL_CMAP = "magma"


def _save_or_return(fig: plt.Figure, save_path: str | None) -> plt.Figure | None:
    """Save figure as PDF if save_path given, otherwise return."""
    if save_path is not None:
        fig.savefig(save_path, format="pdf")
        plt.close(fig)
        return None
    return fig


def _add_ec_boundary(
    ax: plt.Axes,
    x: NDArray,
    y: NDArray,
    ec_feasible: NDArray,
    *,
    hatch_inadmissible: bool = True,
) -> None:
    """Overlay EC admissibility boundary and hatching."""
    feas_float = ec_feasible.astype(float)
    try:
        ax.contour(
            x, y, feas_float.T,
            levels=[0.5],
            colors=["#222222"],
            linewidths=[1.8],
            linestyles=["solid"],
        )
    except ValueError:
        pass

    if hatch_inadmissible:
        inadmissible = (~ec_feasible).astype(float)
        try:
            ax.contourf(
                x, y, inadmissible.T,
                levels=[0.5, 1.5],
                colors=["none"],
                hatches=["///"],
            )
        except ValueError:
            pass


def _annotate_optimum(
    ax: plt.Axes,
    x: NDArray,
    y: NDArray,
    transport: NDArray,
    ec_feasible: NDArray,
) -> None:
    """Mark the global optimum (max transport in EC-admissible region)."""
    masked = np.where(ec_feasible, transport, -np.inf)
    if np.all(np.isinf(masked)):
        return

    idx = np.unravel_index(np.argmax(masked), masked.shape)
    x_opt = float(x[idx[0]])
    y_opt = float(y[idx[1]])
    t_opt = float(masked[idx])

    ax.plot(
        x_opt, y_opt,
        marker="*", markersize=14, color="#E69F00",
        markeredgecolor="black", markeredgewidth=0.8,
        zorder=10,
    )
    ax.annotate(
        rf"$\beta^x_{{\max}} = {t_opt:.4f}$",
        xy=(x_opt, y_opt),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )


def plot_phase_diagram(
    sweep_result,
    *,
    transport_metric: str = "transport",
    figsize: tuple[float, float] | None = None,
    show_contours: bool = True,
    show_boundary: bool = True,
    annotate_optimum: bool = True,
    n_contours: int = 8,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure | None:
    r"""EC-admissible transport vs shell design parameters.

    Parameters
    ----------
    sweep_result : SweepResult from optimization.sweep.
    transport_metric : field name for color axis.
    figsize : figure size (default: single-column CQG).
    show_contours : draw transport isolines.
    show_boundary : draw EC admissibility boundary.
    annotate_optimum : mark the maximum-transport admissible point.
    n_contours : number of contour levels.
    title : figure title (default: auto from metric name).
    save_path : if provided, save as PDF.
    """
    grids = sweep_result.to_grids()
    x = np.asarray(sweep_result.compactness_values)
    y = np.asarray(sweep_result.thickness_values)

    transport = grids[transport_metric]
    ec_feasible = grids["ec_feasible"].astype(bool)

    if figsize is None:
        figsize = (SINGLE_COL * 1.25, SINGLE_COL * 1.1)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    transport_masked = np.ma.masked_invalid(transport)
    vmin = float(np.nanmin(transport_masked))
    vmax = float(np.nanmax(transport_masked))
    if vmax - vmin < 1e-15:
        vmax = vmin + 1.0

    im = ax.pcolormesh(
        x, y, transport_masked.T,
        cmap=_TRANSPORT_CMAP,
        norm=Normalize(vmin=vmin, vmax=vmax),
        shading="nearest",
        rasterized=True,
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.03, aspect=25)
    cbar.set_label(r"$\max|\beta^x|$", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    if show_boundary:
        _add_ec_boundary(ax, x, y, ec_feasible)

    if show_contours:
        levels = np.linspace(vmin, vmax, n_contours + 2)[1:-1]
        try:
            cs = ax.contour(
                x, y, transport.T,
                levels=levels,
                colors=["#333333"],
                linewidths=[0.6],
                linestyles=["dashed"],
                alpha=0.7,
            )
            ax.clabel(cs, inline=True, fontsize=6, fmt="%.4f")
        except ValueError:
            pass

    if annotate_optimum:
        _annotate_optimum(ax, x, y, transport, ec_feasible)

    ax.set_xlabel(r"Compactness $M/R_2$", fontsize=11)
    ax.set_ylabel(r"Thickness ratio $\Delta R / R_2$", fontsize=11)
    ax.minorticks_on()
    if title is not None:
        ax.set_title(title, fontsize=11)

    legend_handles = [
        Patch(facecolor="white", edgecolor="black", hatch="///",
              label="EC-violated"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#E69F00",
                   markeredgecolor="black", markersize=12, label="Optimum"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.9, edgecolor="gray")

    return _save_or_return(fig, save_path)


def plot_phase_summary(
    sweep_result,
    *,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
) -> plt.Figure | None:
    r"""Multi-panel summary (2x2): transport, EC margin, constraint, tidal.

    Parameters
    ----------
    sweep_result : SweepResult from optimization.sweep.
    figsize : figure size (default: double-column CQG).
    save_path : if provided, save as PDF.
    """
    grids = sweep_result.to_grids()
    x = np.asarray(sweep_result.compactness_values)
    y = np.asarray(sweep_result.thickness_values)
    ec = grids["ec_feasible"].astype(bool)

    if figsize is None:
        figsize = (DOUBLE_COL, DOUBLE_COL * 0.85)

    fig, axes = plt.subplots(
        2, 2, figsize=figsize, sharex=True, sharey=True, layout="constrained",
    )
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    # (a) Transport
    ax = axes[0, 0]
    t_data = np.ma.masked_invalid(grids["transport"])
    vmin_t = float(np.nanmin(t_data))
    vmax_t = float(np.nanmax(t_data))
    if vmax_t - vmin_t < 1e-15:
        vmax_t = vmin_t + 1.0
    im_a = ax.pcolormesh(
        x, y, t_data.T, cmap=_TRANSPORT_CMAP,
        norm=Normalize(vmin=vmin_t, vmax=vmax_t),
        shading="nearest", rasterized=True,
    )
    _add_ec_boundary(ax, x, y, ec)
    _annotate_optimum(ax, x, y, grids["transport"], ec)
    cbar_a = fig.colorbar(im_a, ax=ax, pad=0.03)
    cbar_a.set_label(r"$\max|\beta^x|$", fontsize=9)
    cbar_a.ax.tick_params(labelsize=7)
    ax.set_ylabel(r"$\Delta R / R_2$", fontsize=10)
    ax.set_title(r"Transport utility", fontsize=10)
    ax.minorticks_on()

    # (b) Worst EC margin
    ax = axes[0, 1]
    m_data = np.ma.masked_invalid(grids["worst_ec_margin"])
    vabs = max(abs(float(np.nanmin(m_data))), abs(float(np.nanmax(m_data))), 1e-6)
    im_b = ax.pcolormesh(
        x, y, m_data.T, cmap=_MARGIN_CMAP,
        norm=Normalize(vmin=-vabs, vmax=vabs),
        shading="nearest", rasterized=True,
    )
    _add_ec_boundary(ax, x, y, ec, hatch_inadmissible=False)
    cbar_b = fig.colorbar(im_b, ax=ax, pad=0.03)
    cbar_b.set_label("Worst EC margin", fontsize=9)
    cbar_b.ax.tick_params(labelsize=7)
    ax.set_title(r"Observer-robust EC margin", fontsize=10)
    ax.minorticks_on()

    # (c) Constraint residual
    ax = axes[1, 0]
    c_data = np.ma.masked_invalid(grids["constraint_residual"])
    c_data_log = np.ma.log10(np.ma.maximum(c_data, 1e-15))
    vmin_c = float(np.nanmin(c_data_log))
    vmax_c = float(np.nanmax(c_data_log))
    if vmax_c - vmin_c < 1e-15:
        vmax_c = vmin_c + 1.0
    im_c = ax.pcolormesh(
        x, y, c_data_log.T, cmap=_CONSTRAINT_CMAP,
        norm=Normalize(vmin=vmin_c, vmax=vmax_c),
        shading="nearest", rasterized=True,
    )
    _add_ec_boundary(ax, x, y, ec)
    cbar_c = fig.colorbar(im_c, ax=ax, pad=0.03)
    cbar_c.set_label(r"$\log_{10}\,\langle\epsilon^2\rangle$", fontsize=9)
    cbar_c.ax.tick_params(labelsize=7)
    ax.set_xlabel(r"$M/R_2$", fontsize=10)
    ax.set_ylabel(r"$\Delta R / R_2$", fontsize=10)
    ax.set_title(r"Constraint residual", fontsize=10)
    ax.minorticks_on()

    # (d) Tidal force
    ax = axes[1, 1]
    tid_data = np.ma.masked_invalid(grids["tidal"])
    tid_log = np.ma.log10(np.ma.maximum(tid_data, 1e-15))
    vmin_tid = float(np.nanmin(tid_log))
    vmax_tid = float(np.nanmax(tid_log))
    if vmax_tid - vmin_tid < 1e-15:
        vmax_tid = vmin_tid + 1.0
    im_d = ax.pcolormesh(
        x, y, tid_log.T, cmap=_TIDAL_CMAP,
        norm=Normalize(vmin=vmin_tid, vmax=vmax_tid),
        shading="nearest", rasterized=True,
    )
    _add_ec_boundary(ax, x, y, ec)
    cbar_d = fig.colorbar(im_d, ax=ax, pad=0.03)
    cbar_d.set_label(r"$\log_{10} A_{\rm geo}$", fontsize=9)
    cbar_d.ax.tick_params(labelsize=7)
    ax.set_xlabel(r"$M/R_2$", fontsize=10)
    ax.set_title(r"Tidal force (interior)", fontsize=10)
    ax.minorticks_on()

    for label, ax in zip(panel_labels, axes.ravel()):
        ax.text(
            0.03, 0.95, label,
            transform=ax.transAxes,
            fontsize=11, fontweight="bold",
            va="top", ha="left",
        )

    return _save_or_return(fig, save_path)
