"""Eulerian vs robust EC margin comparison figures.

Produces figures showing:
- Side-by-side Eulerian and robust margin heatmaps with missed-violation overlay
- Multi-condition comparison grids (NEC/WEC/SEC/DEC)
- Comparison summary tables
- Velocity sweep line plots
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from numpy.typing import NDArray

from ._style import COLORS, DOUBLE_COL, LINE_STYLES, SINGLE_COL, apply_style

apply_style()


def _save_or_return(fig: plt.Figure, save_path: str | None) -> plt.Figure:
    """Save figure as PDF if save_path given, otherwise return for interactive use."""
    if save_path is not None:
        fig.savefig(save_path, format="pdf")
        plt.close(fig)
    return fig


def _diverging_norm(data: NDArray) -> TwoSlopeNorm | SymLogNorm:
    """Create a diverging norm centered at 0 for margin data.

    Uses percentile-based vmax to avoid outlier domination, and falls
    back to SymLogNorm when the dynamic range exceeds 10^4 (e.g.
    WarpShell NEC margins spanning ~10^35).
    """
    absdata = np.abs(data)
    vmax_full = float(np.nanmax(absdata))
    if vmax_full < 1e-30:
        return TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)

    # Use 99.5th percentile to suppress extreme outliers
    vmax = float(np.nanpercentile(absdata, 99.5))
    if vmax < 1e-30:
        vmax = vmax_full  # fall back to true max if percentile is ~0

    # Detect high dynamic range: switch to SymLogNorm
    nonzero = absdata[absdata > 1e-30]
    if nonzero.size > 0:
        vmin_nz = float(np.nanpercentile(nonzero, 5))
        if vmin_nz > 0 and vmax / vmin_nz > 1e4:
            linthresh = float(np.nanmedian(nonzero))
            return SymLogNorm(
                linthresh=linthresh, vmin=-vmax, vmax=vmax,
            )

    return TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)


def _feature_extent(
    data_2d: NDArray,
    x_ax: NDArray,
    y_ax: NDArray,
    threshold_frac: float = 1e-4,
    pad_frac: float = 0.3,
    min_extent_frac: float = 0.15,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Compute axis limits that tightly frame the non-trivial region.

    Returns (xlim, ylim) or None if the feature fills >60% of the domain.

    Handles bimodal features (e.g. Lentz with two separated bubble walls)
    by detecting large gaps and zooming to one cluster.  Enforces a
    minimum zoom extent so point-like features are shown in sufficient
    spatial context.
    """
    absdata = np.abs(data_2d)
    dmax = float(np.nanmax(absdata))
    if dmax < 1e-30:
        return None
    mask = absdata > threshold_frac * dmax
    if not mask.any():
        return None

    X, Y = np.meshgrid(x_ax, y_ax, indexing="ij")
    x_feat = X[mask]
    y_feat = Y[mask]
    xlo, xhi = float(x_feat.min()), float(x_feat.max())
    ylo, yhi = float(y_feat.min()), float(y_feat.max())

    # Only zoom if feature occupies <60% of the domain area
    domain_w = x_ax[-1] - x_ax[0]
    domain_h = y_ax[-1] - y_ax[0]
    feat_area = max(xhi - xlo, 1e-30) * max(yhi - ylo, 1e-30)
    if feat_area / (domain_w * domain_h) > 0.6:
        return None

    # Detect bimodal features: large gap in x -> zoom to one cluster
    x_unique = np.sort(np.unique(x_feat))
    if len(x_unique) > 2:
        gaps = np.diff(x_unique)
        max_gap_idx = int(np.argmax(gaps))
        max_gap = gaps[max_gap_idx]
        feat_span = x_unique[-1] - x_unique[0]
        if feat_span > 0 and max_gap > 0.4 * feat_span:
            gap_x = (x_unique[max_gap_idx] + x_unique[max_gap_idx + 1]) / 2.0
            right_mask = mask & (X > gap_x)
            left_mask = mask & (X <= gap_x)
            chosen = right_mask if right_mask.sum() >= left_mask.sum() else left_mask
            x_feat = X[chosen]
            y_feat = Y[chosen]
            xlo, xhi = float(x_feat.min()), float(x_feat.max())
            ylo, yhi = float(y_feat.min()), float(y_feat.max())

    # Pad with fraction of feature extent (at least one grid cell)
    dx = max((xhi - xlo) * pad_frac, x_ax[1] - x_ax[0])
    dy = max((yhi - ylo) * pad_frac, y_ax[1] - y_ax[0])
    xlim = (max(xlo - dx, x_ax[0]), min(xhi + dx, x_ax[-1]))
    ylim = (max(ylo - dy, y_ax[0]), min(yhi + dy, y_ax[-1]))

    # Enforce minimum zoom extent so compact features have context
    min_w = domain_w * min_extent_frac
    min_h = domain_h * min_extent_frac
    w, h = xlim[1] - xlim[0], ylim[1] - ylim[0]
    if w < min_w:
        xmid = (xlim[0] + xlim[1]) / 2.0
        xlim = (max(xmid - min_w / 2, x_ax[0]), min(xmid + min_w / 2, x_ax[-1]))
    if h < min_h:
        ymid = (ylim[0] + ylim[1]) / 2.0
        ylim = (max(ymid - min_h / 2, y_ax[0]), min(ymid + min_h / 2, y_ax[-1]))

    return xlim, ylim


def _extract_slice(data: NDArray, slice_axis: int = 2) -> NDArray:
    """Extract the middle 2D slice along the given axis."""
    idx = data.shape[slice_axis] // 2
    slc = [slice(None)] * data.ndim
    slc[slice_axis] = idx
    return data[tuple(slc)]


def plot_comparison_panel(
    eulerian_margin: NDArray,
    robust_margin: NDArray,
    missed: NDArray,
    grid_bounds: list[tuple[float, float]],
    grid_shape: tuple[int, ...],
    slice_axis: int = 2,
    title: str = "",
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Create a 1x3 panel: Eulerian margin | Robust margin | Missed overlay.

    Parameters
    ----------
    eulerian_margin : NDArray
        Eulerian-frame margin field, shape (*grid_shape,).
    robust_margin : NDArray
        Observer-robust margin field, shape (*grid_shape,).
    missed : NDArray
        Boolean mask of missed violations, shape (*grid_shape,).
    grid_bounds : list of (float, float)
        Spatial bounds for each dimension.
    grid_shape : tuple of int
        Grid shape.
    slice_axis : int
        Axis to slice through (default 2 = z).
    title : str
        Super-title for the figure.
    figsize : tuple or None
        Figure size. Defaults to DOUBLE_COL width.
    save_path : str or None
        If provided, save as PDF.

    Returns
    -------
    plt.Figure
    """
    if figsize is None:
        figsize = (DOUBLE_COL, DOUBLE_COL * 0.36)

    remaining = [i for i in range(len(grid_shape)) if i != slice_axis]
    axis_labels = ["x", "y", "z"]

    # Build 1D coordinate axes for the remaining dimensions
    extents = []
    for dim in remaining:
        lo, hi = grid_bounds[dim]
        extents.append(np.linspace(lo, hi, grid_shape[dim]))
    x_ax, y_ax = extents

    # Extract 2D slices
    eul_2d = _extract_slice(eulerian_margin, slice_axis)
    rob_2d = _extract_slice(robust_margin, slice_axis)
    miss_2d = _extract_slice(missed.astype(float), slice_axis)

    # Compute auto-zoom extent from the union of both margin fields
    union_2d = np.maximum(np.abs(eul_2d), np.abs(rob_2d))
    zoom = _feature_extent(union_2d, x_ax, y_ax)

    # Per-panel normalization - decide whether to share colorbar
    if zoom is not None:
        xlim, ylim = zoom
        xi = (x_ax >= xlim[0]) & (x_ax <= xlim[1])
        yi = (y_ax >= ylim[0]) & (y_ax <= ylim[1])
        eul_vis = eul_2d[np.ix_(xi, yi)]
        rob_vis = rob_2d[np.ix_(xi, yi)]
    else:
        eul_vis = eul_2d
        rob_vis = rob_2d

    # Check if Eulerian and Robust ranges are close enough to share a colorbar
    eul_absmax = max(abs(float(np.nanmin(eul_vis))),
                     abs(float(np.nanmax(eul_vis))))
    rob_absmax = max(abs(float(np.nanmin(rob_vis))),
                     abs(float(np.nanmax(rob_vis))))
    # Share if the two ranges are within 10x of each other
    share_cbar = (eul_absmax > 1e-30 and rob_absmax > 1e-30
                  and max(eul_absmax, rob_absmax)
                  / min(eul_absmax, rob_absmax) < 10.0)

    if share_cbar:
        # Unified norm from the union of both visible regions
        combined_vis = np.concatenate([eul_vis.ravel(), rob_vis.ravel()])
        shared_norm = _diverging_norm(combined_vis)
        eul_norm = rob_norm = shared_norm
    else:
        eul_norm = _diverging_norm(eul_vis)
        rob_norm = _diverging_norm(rob_vis)

    # --- Layout: use subplots with shared y-axis for clean spacing ---
    from matplotlib.gridspec import GridSpec as GS
    from matplotlib.patches import Patch

    if share_cbar:
        # 3 panels + 1 shared colorbar on the far right
        fig = plt.figure(figsize=figsize)
        gs = GS(1, 4, figure=fig,
                width_ratios=[1, 1, 1, 0.04],
                wspace=0.08,
                left=0.08, right=0.95,
                top=0.82 if title else 0.92,
                bottom=0.15)
        ax_eul = fig.add_subplot(gs[0, 0])
        ax_rob = fig.add_subplot(gs[0, 1], sharey=ax_eul)
        ax_miss = fig.add_subplot(gs[0, 2], sharey=ax_eul)
        cax_shared = fig.add_subplot(gs[0, 3])
    else:
        # 3 panels + 2 individual colorbars (Eulerian, Robust)
        # Use wider figure to accommodate separate colorbar labels
        sep_figsize = (figsize[0] * 1.08, figsize[1])
        fig = plt.figure(figsize=sep_figsize)
        gs = GS(1, 7, figure=fig,
                width_ratios=[1, 0.03, 0.18, 1, 0.03, 0.18, 1],
                wspace=0.0,
                left=0.06, right=0.97,
                top=0.82 if title else 0.92,
                bottom=0.15)
        ax_eul = fig.add_subplot(gs[0, 0])
        cax_eul = fig.add_subplot(gs[0, 1])
        ax_rob = fig.add_subplot(gs[0, 3], sharey=ax_eul)
        cax_rob = fig.add_subplot(gs[0, 4])
        ax_miss = fig.add_subplot(gs[0, 6], sharey=ax_eul)

    axes = [ax_eul, ax_rob, ax_miss]

    # Axis tick label sizes
    for ax in axes:
        ax.tick_params(labelsize=7)

    # Check for NaN regions (e.g. interior of warp shells) and paint
    # them with a light gray background so they don't look like errors.
    has_nan = np.any(np.isnan(eul_2d)) or np.any(np.isnan(rob_2d))
    if has_nan:
        for ax in axes:
            ax.set_facecolor("#d9d9d9")

    # Panel 1: Eulerian margin
    im_eul = ax_eul.pcolormesh(
        x_ax, y_ax, eul_2d.T, cmap="RdBu", norm=eul_norm, shading="gouraud",
    )
    ax_eul.set_title("Eulerian", fontsize=10)
    ax_eul.set_ylabel(axis_labels[remaining[1]], fontsize=9)
    ax_eul.set_xlabel(axis_labels[remaining[0]], fontsize=9)

    # Panel 2: Robust margin
    im_rob = ax_rob.pcolormesh(
        x_ax, y_ax, rob_2d.T, cmap="RdBu", norm=rob_norm, shading="gouraud",
    )
    ax_rob.set_title("Robust", fontsize=10)
    ax_rob.set_xlabel(axis_labels[remaining[0]], fontsize=9)
    plt.setp(ax_rob.get_yticklabels(), visible=False)

    # Panel 3: Missed violations (dark red overlay on Eulerian margin)
    ax_miss.pcolormesh(
        x_ax, y_ax, eul_2d.T, cmap="RdBu", norm=eul_norm, shading="gouraud",
        alpha=0.5,
    )
    miss_masked = np.ma.masked_where(miss_2d.T < 0.5, miss_2d.T)
    from matplotlib.colors import ListedColormap
    _missed_cmap = ListedColormap(["#8B0000"])  # distinct dark red
    ax_miss.pcolormesh(
        x_ax, y_ax, miss_masked, cmap=_missed_cmap, vmin=0, vmax=1,
        shading="gouraud", alpha=0.85,
    )
    n_missed = int(np.sum(miss_2d > 0.5))
    ax_miss.set_title(f"Missed ({n_missed} pts)", fontsize=10)
    ax_miss.set_xlabel(axis_labels[remaining[0]], fontsize=9)
    plt.setp(ax_miss.get_yticklabels(), visible=False)
    # Legend patches
    _legend_handles = []
    _legend_handles.append(Patch(facecolor="#8B0000", edgecolor="black",
                                 linewidth=0.5, alpha=0.85, label="Missed"))
    if has_nan:
        _legend_handles.append(Patch(facecolor="#d9d9d9", edgecolor="black",
                                     linewidth=0.5, label="Undefined"))
    ax_miss.legend(
        handles=_legend_handles, loc="lower right", fontsize=7,
        framealpha=0.85, edgecolor="gray", handlelength=1.0,
    )

    # --- Colorbars ---
    def _fmt_cbar(cb, norm):
        """Place ticks evenly in the colorbar's *visual* space."""
        from matplotlib.ticker import FixedLocator, FuncFormatter
        cb.ax.tick_params(labelsize=6, pad=2)

        # Compute tick positions that are visually evenly spaced by
        # inverting the norm at uniform positions in [0, 1].
        n_ticks = 5
        visual_pos = np.linspace(0, 1, n_ticks)
        try:
            tick_vals = norm.inverse(visual_pos)
            # Filter out NaN / inf that can arise from SymLogNorm edges
            tick_vals = tick_vals[np.isfinite(tick_vals)]
        except Exception:
            tick_vals = np.linspace(norm.vmin, norm.vmax, n_ticks)
        cb.locator = FixedLocator(tick_vals)
        cb.update_ticks()

        # Compact number formatting
        def _compact_fmt(x, _pos):
            if x == 0:
                return "0"
            ax_val = abs(x)
            if 0.01 <= ax_val < 1000:
                return f"{x:.2g}"
            exp = int(np.floor(np.log10(ax_val)))
            coeff = x / 10**exp
            return rf"{coeff:.1f}e{exp}"
        cb.ax.yaxis.set_major_formatter(FuncFormatter(_compact_fmt))

    if share_cbar:
        cb = fig.colorbar(im_eul, cax=cax_shared)
        cb.set_label("Margin", fontsize=8)
        _fmt_cbar(cb, eul_norm)
    else:
        cb_eul = fig.colorbar(im_eul, cax=cax_eul)
        _fmt_cbar(cb_eul, eul_norm)
        cb_rob = fig.colorbar(im_rob, cax=cax_rob)
        _fmt_cbar(cb_rob, rob_norm)

    # Apply zoom limits to all panels
    if zoom is not None:
        for ax in axes:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    if title:
        fig.suptitle(title, fontsize=9)

    return _save_or_return(fig, save_path)


def plot_comparison_grid(
    npz_path: str,
    grid_bounds: list[tuple[float, float]],
    grid_shape: tuple[int, ...],
    conditions: tuple[str, ...] = ("nec", "wec", "sec", "dec"),
    slice_axis: int = 2,
    save_path: str | None = None,
) -> plt.Figure:
    """Create a 4x3 grid figure (4 conditions x 3 panels) from cached .npz.

    Parameters
    ----------
    npz_path : str
        Path to cached .npz file with per-condition margin data.
    grid_bounds : list of (float, float)
        Spatial bounds for each dimension.
    grid_shape : tuple of int
        Grid shape.
    conditions : tuple of str
        EC condition names.
    slice_axis : int
        Axis to slice.
    save_path : str or None
        If provided, save as PDF.

    Returns
    -------
    plt.Figure
    """
    data = np.load(npz_path)

    n_conds = len(conditions)

    remaining = [i for i in range(len(grid_shape)) if i != slice_axis]
    axis_labels = ["x", "y", "z"]
    extents = []
    for dim in remaining:
        lo, hi = grid_bounds[dim]
        extents.append(np.linspace(lo, hi, grid_shape[dim]))
    x_ax, y_ax = extents

    # GridSpec: 3 panels with per-panel colorbars per row
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.25 * n_conds))
    gs = fig.add_gridspec(n_conds, 7,
                          width_ratios=[1, 0.03, 0.06, 1, 0.03, 0.06, 1],
                          wspace=0.05, hspace=0.25, top=0.93, bottom=0.08)

    for row, cond in enumerate(conditions):
        eul = data[f"{cond}_eulerian"]
        rob = data[f"{cond}_robust"]
        missed = data[f"{cond}_missed"]

        eul_2d = _extract_slice(eul, slice_axis)
        rob_2d = _extract_slice(rob, slice_axis)
        miss_2d = _extract_slice(missed.astype(float), slice_axis)

        eul_norm = _diverging_norm(eul_2d)
        rob_norm = _diverging_norm(rob_2d)

        ax0 = fig.add_subplot(gs[row, 0])
        cax0 = fig.add_subplot(gs[row, 1])
        ax1 = fig.add_subplot(gs[row, 3])
        cax1 = fig.add_subplot(gs[row, 4])
        ax2 = fig.add_subplot(gs[row, 6])

        # Eulerian (own norm)
        im_eul = ax0.pcolormesh(
            x_ax, y_ax, eul_2d.T, cmap="RdBu", norm=eul_norm, shading="gouraud"
        )
        ax0.set_ylabel(f"{cond.upper()}\n{axis_labels[remaining[1]]}")
        if row == 0:
            ax0.set_title("Eulerian", fontsize=9)
        cb_eul = fig.colorbar(im_eul, cax=cax0)
        cb_eul.ax.tick_params(labelsize=6)

        # Robust (own norm)
        im_rob = ax1.pcolormesh(
            x_ax, y_ax, rob_2d.T, cmap="RdBu", norm=rob_norm, shading="gouraud"
        )
        ax1.tick_params(labelleft=False)
        if row == 0:
            ax1.set_title("Robust", fontsize=9)
        cb_rob = fig.colorbar(im_rob, cax=cax1)
        cb_rob.ax.tick_params(labelsize=6)

        # Missed (uses Eulerian norm)
        ax2.pcolormesh(
            x_ax, y_ax, eul_2d.T, cmap="RdBu", norm=eul_norm,
            shading="gouraud", alpha=0.5,
        )
        miss_masked = np.ma.masked_where(miss_2d.T < 0.5, miss_2d.T)
        from matplotlib.colors import ListedColormap as _LCM
        ax2.pcolormesh(
            x_ax, y_ax, miss_masked, cmap=_LCM(["#8B0000"]), vmin=0, vmax=1,
            shading="gouraud", alpha=0.85,
        )
        ax2.tick_params(labelleft=False)
        if row == 0:
            ax2.set_title("Missed", fontsize=9)

        # x-labels only on bottom row
        if row == n_conds - 1:
            ax1.set_xlabel(axis_labels[remaining[0]])
        else:
            ax0.tick_params(labelbottom=False)
            ax1.tick_params(labelbottom=False)
            ax2.tick_params(labelbottom=False)

    return _save_or_return(fig, save_path)


def plot_comparison_table(
    json_path: str,
    save_path: str | None = None,
) -> plt.Figure:
    """Render comparison_table.json as a formatted matplotlib table figure.

    Focuses on NEC and WEC (the paper's key conditions) with color-coded
    cells highlighting non-zero missed-violation percentages.

    Parameters
    ----------
    json_path : str
        Path to comparison_table.json.
    save_path : str or None
        If provided, save as PDF.

    Returns
    -------
    plt.Figure
    """
    with open(json_path) as f:
        rows = json.load(f)

    if not rows:
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 1))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return _save_or_return(fig, save_path)

    # Focus on the two key conditions + DEC for dramatic effect
    conditions = ("nec", "wec", "dec")
    col_labels = ["Metric", r"$v_s$"]
    for cond in conditions:
        col_labels.extend([
            f"{cond.upper()} Eul",
            f"{cond.upper()} Rob",
            f"{cond.upper()} miss",
        ])

    cell_data = []
    cell_colors = []
    for row in rows:
        cells = [row["metric"].capitalize(), f"{row['v_s']:.2f}"]
        row_colors = ["white", "white"]
        for cond in conditions:
            eul_min = row.get(f"{cond}_eulerian_min", float("nan"))
            rob_min = row.get(f"{cond}_robust_min", float("nan"))
            pct = row.get(f"{cond}_pct_missed", 0.0)
            cells.extend([f"{eul_min:.1e}", f"{rob_min:.1e}", f"{pct:.1f}"])
            # Color-code missed column (colorblind-friendly: blue-orange scheme)
            if pct > 10:
                row_colors.extend(["white", "white", "#FDAE61"])  # strong orange
            elif pct > 0.1:
                row_colors.extend(["white", "white", "#FEE08B"])  # light yellow-orange
            elif pct > 0:
                row_colors.extend(["white", "white", "#E6F5D0"])  # light green-yellow
            else:
                row_colors.extend(["white", "white", "#D9EAF7"])  # light blue
        cell_data.append(cells)
        cell_colors.append(row_colors)

    fig_height = max(2.0, 0.32 * len(cell_data) + 1.2)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.auto_set_column_width(list(range(len(col_labels))))
    table.scale(1, 1.4)

    n_cols = len(col_labels)
    # Header row: dark background, white bold text
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold", fontsize=7.5)
        cell.set_edgecolor("white")
        cell.set_linewidth(0.5)

    # Data rows: alternating background + highlight colors
    for i, row_colors in enumerate(cell_colors):
        alt_bg = "#F5F5F5" if i % 2 == 0 else "white"
        for j, color in enumerate(row_colors):
            cell = table[i + 1, j]
            cell.set_facecolor(color if color != "white" else alt_bg)
            cell.set_edgecolor("#D0D0D0")
            cell.set_linewidth(0.5)

    ax.set_title(
        "Energy Condition Violations: Eulerian vs Robust Analysis",
        fontsize=9, pad=8,
    )

    fig.tight_layout(pad=0.5)
    return _save_or_return(fig, save_path)


def plot_velocity_sweep(
    results_dir: str,
    metric_name: str,
    condition: str = "nec",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Line plot of min margin vs v_s for Eulerian and robust analysis.

    Shows how the gap between Eulerian and robust margins grows with
    warp velocity, visualizing the observer-dependence phenomenon.

    Parameters
    ----------
    results_dir : str
        Directory containing cached .npz files.
    metric_name : str
        Name of the metric (e.g. "alcubierre").
    condition : str
        EC condition to plot (default "nec").
    save_path : str or None
        If provided, save as PDF.
    ax : plt.Axes or None
        If provided, plot on this axes instead of creating a new figure.

    Returns
    -------
    plt.Figure
    """
    results_path = Path(results_dir)
    v_s_values = []
    eul_mins = []
    rob_mins = []

    # Scan for available files
    for npz_file in sorted(results_path.glob(f"{metric_name}_vs*.npz")):
        fname = npz_file.stem
        # Extract v_s from filename: "metric_vs0.5" -> 0.5
        v_s_str = fname.split("_vs")[-1]
        try:
            v_s = float(v_s_str)
        except ValueError:
            continue

        data = np.load(str(npz_file))
        eul_key = f"{condition}_eulerian"
        rob_key = f"{condition}_robust"
        if eul_key not in data or rob_key not in data:
            continue

        v_s_values.append(v_s)
        eul_mins.append(float(np.nanmin(data[eul_key])))
        rob_mins.append(float(np.nanmin(data[rob_key])))

    if not v_s_values:
        if ax is None:
            fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, f"No data for {metric_name}", ha="center", va="center")
        return _save_or_return(fig, save_path)

    # Sort by v_s
    order = np.argsort(v_s_values)
    v_s_arr = np.array(v_s_values)[order]
    eul_arr = np.array(eul_mins)[order]
    rob_arr = np.array(rob_mins)[order]

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))
    mevery = max(1, len(v_s_arr) // 8)
    ax.plot(
        v_s_arr, eul_arr, label="Eulerian",
        color=COLORS[0], **LINE_STYLES[0], markevery=mevery,
    )
    ax.plot(
        v_s_arr, rob_arr, label="Robust",
        color=COLORS[1], **LINE_STYLES[1], markevery=mevery,
    )
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel(r"$v_s$")
    ax.set_ylabel(f"Min {condition.upper()} margin")
    ax.set_title(
        rf"{metric_name.capitalize()}: {condition.upper()}"
        rf" ($50^3$ grid, $R = 1$, $\sigma = 8$)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout(pad=0.5)

    return _save_or_return(fig, save_path)
