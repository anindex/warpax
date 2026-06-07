"""Contribution 1: velocity-resolved Hawking-Ellis type & all-observer EC structure
across the luminal transition.

Frame-independently (from the eigenstructure of T^a_b only -- no Eulerian normal,
so valid at v_s >= 1), we sweep the warp speed from deep subluminal to
superluminal and record, on wall-clustered grids, the wall-restricted

  - Hawking-Ellis type fractions (I/II/III/IV), volume-weighted;
  - Type-I invariant eigenvalue margins min(rho+p_i) [NEC] and min(rho-|p_i|) [DEC];
  - the proper-volume extent of Type-IV ("no rest frame") regions.

This delivers the analysis the CQG referee (R2, pt 4) identified as missing: the
local all-observer energy-condition problem studied in a tetrad/eigenvalue
framework across subluminal, luminal, AND superluminal regimes. The Type-IV
labels here are certified physical (not numerical artefact) by the companion
gate ``validate_superluminal_classification.py`` (3-solver + 50-digit mpmath
agreement); this production sweep therefore uses the fast standard eig solver.

Outputs
-------
- results/velocity_sweep.json
- ../warpax_arxiv/tables/velocity_type_structure.tex
- ../warpax_arxiv/figures/velocity_type_structure.pdf
- ../warpax_arxiv/figures/rodal_invariant_margins.pdf
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from warpax.visualization._style import (
    COLORS,
    DOUBLE_COL,
    USE_TEX,
    apply_style,
    metric_color,
)

apply_style()

# "%" is a comment char under usetex but literal under mathtext; pick per backend.
_PCT = r"\%" if USE_TEX else "%"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions.filtering import shape_function_mask
from warpax.energy_conditions.frame_free import (
    certify_grid_frame_free,
    type_fractions,
    typeI_min_margins,
)
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.grid import build_coord_batch
from warpax.grids import wall_clustered
from warpax.metrics import NatarioMetric, RodalMetric, VanDenBroeckMetric

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")
# Figures are generated into the codebase, then copied to the paper folder
# by reproduce_all.sh (sync_figures_to_paper).
FIG_DIR = os.path.join(HERE, "..", "figures")

BOUNDS = [(-3, 3)] * 3
F_LOW, F_HIGH = 0.1, 0.9

METRICS = {
    "Alcubierre": (AlcubierreMetric, {}),
    "Natário": (NatarioMetric, {}),
    "Van den Broeck": (VanDenBroeckMetric,
                       {"R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0}),
    "Rodal": (RodalMetric, {}),
}
METRIC_ORDER = ["Alcubierre", "Natário", "Van den Broeck", "Rodal"]


def _instantiate(name, v_s):
    cls, extra = METRICS[name]
    return cls(v_s=v_s, R=1.0, sigma=8.0, **extra)


def run_point(name, v_s, N):
    shape = (N, N, N)
    metric = _instantiate(name, v_s)
    grid = wall_clustered(metric, BOUNDS, shape, a=1.2)
    curv = evaluate_curvature_grid(metric, grid, batch_size=256)
    ff = certify_grid_frame_free(
        curv.stress_energy, curv.metric, curv.metric_inv, solver="standard"
    )
    coords = build_coord_batch(grid, t=0.0)
    mask = shape_function_mask(metric, coords, shape, f_low=F_LOW, f_high=F_HIGH)
    vol_w = grid.volume_weights_array
    fr = type_fractions(ff, mask=mask, volume_weights=vol_w)
    mm = typeI_min_margins(ff, mask=np.asarray(jnp.reshape(mask, (-1,))).astype(bool))
    return {
        "metric": name, "v_s": v_s, "N": N,
        "wall_frac_type_i": fr["frac_type_i"],
        "wall_frac_type_iv": fr["frac_type_iv"],
        "wall_n": fr["n_selected"],
        "typeI_nec_min": mm["nec_min"],
        "typeI_dec_min": mm["dec_min"],
        "typeI_wec_min": mm["wec_min"],
        "n_type_i_wall": mm["n_type_i_selected"],
    }


def write_table(rows, out_path, table_vels=(0.5, 1.0, 2.0)):
    """Wall Type-I / Type-IV fractions at sub/luminal/super velocities."""
    def cell(name, v, key):
        for r in rows:
            if r["metric"] == name and abs(r["v_s"] - v) < 1e-9:
                return r[key] * 100.0
        return None
    lines = [
        r"\begin{tabular}{@{}l cc cc cc@{}}",
        r"  \toprule",
        r"  & \multicolumn{2}{c}{$v_s=0.5$ (sub)} & "
        r"\multicolumn{2}{c}{$v_s=1.0$ (luminal)} & "
        r"\multicolumn{2}{c}{$v_s=2.0$ (super)} \\",
        r"  \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"  Metric & Type~I & Type~IV & Type~I & Type~IV & Type~I & Type~IV \\",
        r"  \midrule",
    ]
    for name in METRIC_ORDER:
        vals = []
        for v in table_vels:
            vals.append(cell(name, v, "wall_frac_type_i"))
            vals.append(cell(name, v, "wall_frac_type_iv"))
        cells = " & ".join(f"{v:.1f}" if v is not None else "--" for v in vals)
        lines.append(f"  {name} & {cells} \\\\")
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")


# Per-metric marker, paired with the shared metric color.
_MARKERS = {"Alcubierre": "o", "Natário": "s",
            "Van den Broeck": "^", "Rodal": "D"}


def _panel_typeiv(ax, rows):
    """Panel: wall Type-IV fraction vs warp speed, one line per metric."""
    for name in METRIC_ORDER:
        rs = sorted([r for r in rows if r["metric"] == name], key=lambda r: r["v_s"])
        xs = [r["v_s"] for r in rs]
        ys = [r["wall_frac_type_iv"] * 100.0 for r in rs]
        ax.plot(xs, ys, marker=_MARKERS.get(name, "o"), ms=4,
                color=metric_color(name), label=name)
    ax.axvline(1.0, color="0.6", ls="--", lw=0.8)
    ax.text(1.02, 4.0, "luminal", color="0.4", fontsize=8)
    ax.set_xlabel(r"warp speed $v_s$")
    ax.set_ylabel(f"wall Type-IV fraction ({_PCT})")
    ax.set_ylim(-5, 108)
    # Legend in the empty mid band (Rodal sits at 0, the rest above ~60).
    ax.legend(loc="center right", bbox_to_anchor=(0.99, 0.30),
              fontsize=7.5, frameon=False)


def _panel_rodal_margins(ax, rows):
    """Panel: Rodal invariant Type-I NEC/DEC margins vs warp speed."""
    rs = sorted([r for r in rows if r["metric"] == "Rodal"], key=lambda r: r["v_s"])
    xs = [r["v_s"] for r in rs]
    ax.plot(xs, [r["typeI_nec_min"] for r in rs], marker="o", ms=4,
            color=COLORS[0], ls="-", label=r"$\min(\rho+p_i)$ (NEC)")
    ax.plot(xs, [r["typeI_dec_min"] for r in rs], marker="s", ms=4,
            color=COLORS[1], ls="--", label=r"$\min(\rho-|p_i|)$ (DEC)")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(1.0, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel(r"warp speed $v_s$")
    ax.set_ylabel("invariant eigenvalue margin")
    # Curves run from near 0 (top-left) to deep negative (bottom-right); the
    # top-right is empty.
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def make_figures(rows):
    os.makedirs(FIG_DIR, exist_ok=True)

    # Merged side-by-side figure used by the paper (Fig 1).
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))
    _panel_typeiv(ax_a, rows)
    _panel_rodal_margins(ax_b, rows)
    ax_a.set_title("(a) wall Type-IV fraction", fontsize=9)
    ax_b.set_title("(b) Rodal all-observer margins", fontsize=9)
    fig.tight_layout()
    pm = os.path.join(FIG_DIR, "velocity_summary.pdf")
    fig.savefig(pm)
    plt.close(fig)
    print(f"  Wrote {pm}")

    # Individual panels (kept for backward compatibility / standalone use).
    for fname, panel in (("velocity_type_structure.pdf", _panel_typeiv),
                         ("rodal_invariant_margins.pdf", _panel_rodal_margins)):
        fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, DOUBLE_COL * 0.5))
        panel(ax, rows)
        fig.tight_layout()
        p = os.path.join(FIG_DIR, fname)
        fig.savefig(p)
        plt.close(fig)
        print(f"  Wrote {p}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--velocities", type=float, nargs="+",
                   default=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0, 1.1, 1.3, 1.5, 2.0, 2.5])
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--metrics", type=str, nargs="+", default=METRIC_ORDER)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--from-cache", action="store_true",
                   help="rebuild the table and figures from results/velocity_sweep.json "
                        "without recomputing the (expensive) eigenstructure")
    args = p.parse_args()

    if args.from_cache:
        with open(os.path.join(RESULTS_DIR, "velocity_sweep.json")) as f:
            cached = json.load(f)
        rows = cached["rows"]
        write_table(rows, os.path.join(TABLES_DIR, "velocity_type_structure.tex"))
        make_figures(rows)
        return

    if args.smoke:
        args.velocities = [0.5, 1.0, 2.0]
        args.N = 24
        args.metrics = ["Alcubierre", "Rodal"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print(f"VELOCITY SWEEP (R=1, sigma=8, N={args.N}, wall-clustered)")
    print("=" * 70)
    rows = []
    for name in args.metrics:
        for v_s in args.velocities:
            r = run_point(name, v_s, args.N)
            rows.append(r)
            print(f"  {name:>15s} v_s={v_s:.2f}  "
                  f"TypeI={r['wall_frac_type_i']*100:5.1f}%  "
                  f"TypeIV={r['wall_frac_type_iv']*100:5.1f}%  "
                  f"NECmin={r['typeI_nec_min']:.3g}  DECmin={r['typeI_dec_min']:.3g}")

    with open(os.path.join(RESULTS_DIR, "velocity_sweep.json"), "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)
    print(f"\nWrote {os.path.join(RESULTS_DIR, 'velocity_sweep.json')}")

    if not args.smoke:
        write_table(rows, os.path.join(TABLES_DIR, "velocity_type_structure.tex"))
        make_figures(rows)


if __name__ == "__main__":
    main()
