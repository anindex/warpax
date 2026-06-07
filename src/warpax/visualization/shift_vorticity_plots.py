"""Figure for the shift-vorticity control of the Hawking-Ellis type.

Two panels:

(a) the irreducible decomposition of each drive's shift gradient into
    expansion, shear and vorticity fractions: the irrotational Rodal drive has
    no vorticity, the zero-expansion Natario drive has no expansion;

(b) the per-drive vorticity fraction against the wall Type-IV fraction across
    the velocity sweep: zero vorticity coincides with zero Type-IV (Rodal),
    nonzero vorticity with Type-IV-dominated walls.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._style import DOUBLE_COL, USE_TEX, apply_style, metric_color

apply_style()

# "%" is a comment char under usetex but literal under mathtext; pick per backend.
_PCT = r"\%" if USE_TEX else "%"

# Fixed, distinct colors for the three shift-gradient components (not metrics).
_COMPONENT_COLORS = {
    "expansion": "#BBBBBB",
    "shear": "#88CCEE",
    "vorticity": "#CC3311",
}


def plot_shift_vorticity(
    fingerprint: dict[str, dict[str, float]],
    sweep: dict[str, list[tuple[float, float, float]]],
    order: list[str],
    save_path: str | None = None,
) -> plt.Figure:
    """Render the shift-vorticity control figure.

    Parameters
    ----------
    fingerprint : dict
        ``{metric: {"expansion": f, "shear": f, "vorticity": f}}`` with the
        three velocity-independent shift-gradient fractions (summing to one).
    sweep : dict
        ``{metric: [(v_s, rotationality, type_iv_percent), ...]}``.
    order : list of str
        Metric draw order (left to right / legend order).
    save_path : str or None
        If given, save as PDF and close.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))

    # Panel (a): stacked decomposition bars.
    comps = ["expansion", "shear", "vorticity"]
    xpos = np.arange(len(order))
    bottom = np.zeros(len(order))
    for comp in comps:
        vals = np.array([fingerprint[m][comp] for m in order])
        ax_a.bar(
            xpos, vals, bottom=bottom, width=0.7,
            color=_COMPONENT_COLORS[comp], edgecolor="0.3", linewidth=0.4,
            label=comp,
        )
        bottom += vals
    ax_a.set_xticks(xpos)
    ax_a.set_xticklabels(order, rotation=20, ha="right")
    ax_a.set_ylabel("shift-gradient fraction")
    ax_a.set_ylim(0, 1.0)
    ax_a.set_title("(a) shift-gradient decomposition")
    ax_a.legend(fontsize=8, frameon=False, loc="upper center",
                ncol=3, bbox_to_anchor=(0.5, -0.28))

    # Panel (b): rotationality vs wall Type-IV fraction.
    x_max = 0.0
    for m in order:
        pts = sweep[m]
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        x_max = max([x_max, *xs])
        ax_b.scatter(xs, ys, s=22, color=metric_color(m), label=m,
                     edgecolor="0.25", linewidth=0.3, zorder=3)
    ax_b.set_xlabel(r"shift vorticity fraction $\mathcal{R}_\omega$")
    ax_b.set_ylabel(f"wall Type-IV fraction ({_PCT})")
    ax_b.set_xlim(-0.04, max(0.6, x_max * 1.1))
    ax_b.set_ylim(-5, 105)
    ax_b.set_title("(b) vorticity controls the type")
    # Data clusters sit at x >= 0.33 (plus Rodal at the origin); the
    # center-left is empty, so the legend goes there.
    ax_b.legend(fontsize=8, frameon=False, loc="center left",
                bbox_to_anchor=(0.02, 0.55))

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format="pdf")
        plt.close(fig)
    return fig
