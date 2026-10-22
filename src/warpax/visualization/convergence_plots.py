"""Uniform-grid sample plots and Richardson convergence plots.

Produces figures showing:
- Sampled values for descriptive grid studies
- Log-log convergence plot when Richardson estimates are supplied
- Convergence summary table
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from ._style import COLORS, LINE_STYLES, SINGLE_COL, apply_style, save_or_return, style_table

apply_style()


def plot_convergence(
    json_path: str,
    quantity: str = "min_margin_nec",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
    show_fit: bool = True,
) -> plt.Figure:
    """Plot a descriptive grid study or a supplied Richardson estimate.

    Data tagged ``observed_grid_spread`` show the sampled quantity against N,
    with no fitted order or extrapolated error. Otherwise:
    X-axis: grid spacing h = 1/N.
    Y-axis: |Q(h) - Q_extrapolated|.
    Plots data points, optionally with a fitted-order line (see show_fit).

    Levels with exactly zero error are dropped before plotting: a fallback
    extrapolation sets Q_ext to the finest value, so that point's error is zero
    and cannot be drawn on a log axis. The fitted line is ``error = C * h**p``
    with C fixed by the coarsest point; when the observed order p is None
    (non-monotone triplet, no order estimated) the data is drawn without it.

    Parameters
    ----------
    json_path : str
        Path to convergence_data.json.
    quantity : str
        Name of the convergence quantity to plot (default "min_margin_nec").
    save_path : str or None
        If provided, save as PDF.
    ax : plt.Axes or None
        If provided, plot on this axes instead of creating a new figure.
    show_fit : bool
        If True (default), overlay the fitted-order line; set False to show the
        data points alone (used for non-monotone sequences).

    Returns
    -------
    plt.Figure
    """
    with open(json_path) as f:
        data = json.load(f)

    resolutions = data.get("resolutions", [])
    qdata = data.get(quantity, {})

    if "error" in qdata or not qdata.get("values"):
        if ax is None:
            fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, f"No data for {quantity}", ha="center", va="center")
        ax.set_title(quantity)
        return save_or_return(fig, save_path)

    values = qdata["values"]
    if data.get("summary_method") == "observed_grid_spread":
        fig, ax = (
            (ax.figure, ax)
            if ax is not None
            else plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))
        )
        ax.plot(
            resolutions,
            values,
            color=COLORS[0],
            marker=LINE_STYLES[0]["marker"],
            markersize=5,
        )
        ax.set_xlabel("Grid points per axis N")
        ax.set_ylabel(f"Sampled {quantity}")
        ax.set_title(f"Uniform-grid samples: {quantity}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(pad=0.5)
        return save_or_return(fig, save_path)

    Q_ext = qdata["extrapolated_value"]
    p = qdata["observed_order"]

    h = np.array([1.0 / N for N in resolutions])
    errors = np.array([abs(v - Q_ext) for v in values])

    finite = errors > 0.0
    h, errors, values = (
        h[finite],
        errors[finite],
        [v for v, k in zip(values, finite, strict=True) if k],
    )

    fig, ax = (
        (ax.figure, ax) if ax is not None else plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))
    )

    ax.loglog(
        h,
        errors,
        color=COLORS[0],
        marker=LINE_STYLES[0]["marker"],
        markersize=5,
        linestyle="none",
        label="Data",
    )

    if show_fit and p is not None and errors[0] > 1e-30 and h[0] > 0:
        C = errors[0] / h[0] ** p
        h_fine = np.logspace(np.log10(h[-1] * 0.5), np.log10(h[0] * 2), 50)
        ax.loglog(
            h_fine,
            C * h_fine**p,
            color=COLORS[1],
            linestyle=LINE_STYLES[1]["linestyle"],
            linewidth=1,
            label=f"$p = {p:.2f}$",
        )

    ax.set_xlabel(r"Grid spacing $h \propto 1/N$")
    ax.set_ylabel(r"$|Q(h) - Q_{\mathrm{ext}}|$")
    ax.set_title(rf"Convergence: {quantity} ($25^3$/$50^3$/$100^3$)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(pad=0.5)

    return save_or_return(fig, save_path)


def plot_convergence_table(
    json_path: str,
    save_path: str | None = None,
) -> plt.Figure:
    """Table figure showing convergence data for all quantities.

    Parameters
    ----------
    json_path : str
        Path to convergence_data.json.
    save_path : str or None
        If provided, save as PDF.

    Returns
    -------
    plt.Figure
    """
    with open(json_path) as f:
        data = json.load(f)

    resolutions = data.get("resolutions", [])
    quantities = [k for k, q in data.items() if isinstance(q, dict) and "values" in q]
    samples_only = data.get("summary_method") == "observed_grid_spread"

    if not quantities:
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 1))
        ax.text(0.5, 0.5, "No convergence data", ha="center", va="center")
        ax.axis("off")
        return save_or_return(fig, save_path)

    col_labels = ["Quantity"]
    for N in resolutions:
        col_labels.append(f"N={N}")
    col_labels.extend(
        ["Max departure from mean"] if samples_only else ["Extrapolated", "Order p", "Error est."]
    )

    cell_data = []
    for qname in quantities:
        qdata = data[qname]
        if isinstance(qdata, dict) and "values" in qdata:
            row = [qname]
            for v in qdata["values"]:
                row.append(f"{v:.4e}")
            if samples_only:
                row.append(f"{qdata['max_abs_deviation_from_mean']:.2e}")
                cell_data.append(row)
                continue
            row.append(
                f"{qdata.get('extrapolated_value', 'N/A'):.4e}"
                if isinstance(qdata.get("extrapolated_value"), (int, float))
                else "N/A"
            )
            row.append(
                f"{qdata.get('observed_order', 'N/A'):.2f}"
                if isinstance(qdata.get("observed_order"), (int, float))
                else "N/A"
            )
            row.append(
                f"{qdata.get('error_estimate', 'N/A'):.2e}"
                if isinstance(qdata.get("error_estimate"), (int, float))
                else "N/A"
            )
            cell_data.append(row)

    n_rows = len(cell_data)
    fig_height = max(1.5, 0.38 * n_rows + 0.9)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 2, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    style_table(table, len(col_labels), 7)

    title = (
        "Uniform-grid samples and observed spread"
        if samples_only
        else "Richardson Convergence Summary"
    )
    ax.set_title(title, fontsize=9, pad=8)

    fig.tight_layout(pad=0.5)
    return save_or_return(fig, save_path)
