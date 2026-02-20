"""Richardson convergence log-log plots and summary tables.

Produces figures showing:
- Log-log convergence plot with fitted order line
- Convergence summary table
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from ._style import COLORS, LINE_STYLES, SINGLE_COL, apply_style

apply_style()


def _save_or_return(fig: plt.Figure, save_path: str | None) -> plt.Figure:
    """Save figure as PDF if save_path given, otherwise return for interactive use."""
    if save_path is not None:
        fig.savefig(save_path, format="pdf")
        plt.close(fig)
    return fig


def plot_convergence(
    json_path: str,
    quantity: str = "min_margin_nec",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Log-log convergence plot from cached convergence data.

    X-axis: grid spacing h = 1/N.
    Y-axis: |Q(h) - Q_extrapolated|.
    Plots data points plus fitted line with slope = observed order p.

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
        return _save_or_return(fig, save_path)

    values = qdata["values"]
    Q_ext = qdata["extrapolated_value"]
    p = qdata["observed_order"]

    # Grid spacing: h = 1/N
    h = np.array([1.0 / N for N in resolutions])
    errors = np.array([abs(v - Q_ext) for v in values])

    # Avoid log(0) for exact matches
    errors = np.maximum(errors, 1e-30)

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))

    # Data points
    ax.loglog(
        h, errors,
        color=COLORS[0], marker=LINE_STYLES[0]["marker"],
        markersize=5, linestyle="none", label="Data",
    )

    # Fitted line: error ~ C * h^p
    # Use the coarsest point to determine C
    if errors[0] > 1e-30 and h[0] > 0:
        C = errors[0] / h[0] ** p
        h_fine = np.logspace(np.log10(h[-1] * 0.5), np.log10(h[0] * 2), 50)
        ax.loglog(
            h_fine, C * h_fine**p,
            color=COLORS[1], linestyle=LINE_STYLES[1]["linestyle"],
            linewidth=1, label=f"$p = {p:.2f}$",
        )

    ax.set_xlabel(r"Grid spacing $h \propto 1/N$")
    ax.set_ylabel(r"$|Q(h) - Q_{\mathrm{ext}}|$")
    ax.set_title(rf"Convergence: {quantity} ($25^3$/$50^3$/$100^3$)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(pad=0.5)

    return _save_or_return(fig, save_path)


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
    quantities = [k for k in data if k not in ("metric", "resolutions")]

    if not quantities:
        fig, ax = plt.subplots(figsize=(SINGLE_COL, 1))
        ax.text(0.5, 0.5, "No convergence data", ha="center", va="center")
        ax.axis("off")
        return _save_or_return(fig, save_path)

    # Build table
    col_labels = ["Quantity"]
    for N in resolutions:
        col_labels.append(f"N={N}")
    col_labels.extend(["Extrapolated", "Order p", "Error est."])

    cell_data = []
    for qname in quantities:
        qdata = data[qname]
        if isinstance(qdata, dict) and "values" in qdata:
            row = [qname]
            for v in qdata["values"]:
                row.append(f"{v:.4e}")
            row.append(f"{qdata.get('extrapolated_value', 'N/A'):.4e}"
                       if isinstance(qdata.get("extrapolated_value"), (int, float))
                       else "N/A")
            row.append(f"{qdata.get('observed_order', 'N/A'):.2f}"
                       if isinstance(qdata.get("observed_order"), (int, float))
                       else "N/A")
            row.append(f"{qdata.get('error_estimate', 'N/A'):.2e}"
                       if isinstance(qdata.get("error_estimate"), (int, float))
                       else "N/A")
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
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(list(range(len(col_labels))))
    table.scale(1, 1.4)

    n_cols = len(col_labels)
    # Header row: dark background, white bold text
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold", fontsize=7)
        cell.set_edgecolor("white")
        cell.set_linewidth(0.5)

    # Data rows: alternating background
    for i in range(n_rows):
        alt_bg = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_facecolor(alt_bg)
            cell.set_edgecolor("#D0D0D0")
            cell.set_linewidth(0.5)

    ax.set_title("Richardson Convergence Summary", fontsize=9, pad=8)

    fig.tight_layout(pad=0.5)
    return _save_or_return(fig, save_path)
