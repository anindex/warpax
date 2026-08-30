"""Rodal wall-thickness sweep, wall-resolved and convergence-certified.

The superseded sweep used a uniform ``N = 50`` grid on
``[-300, 300]^3``, so ``dx = 600/49 = 12.24``. Against the exact 10-90% wall
widths that is

    sigma  = 0.01    0.03    0.1     0.3
    width  = 174.30  72.53   21.97   7.32
    cells  = 14.23    5.92    1.79    0.598

and the last two rows sit below the four-cell criterion the manuscript enforces
everywhere else, while carrying the top end of the reported trend. Reaching
four cells at ``sigma = 0.3`` on a uniform Cartesian grid needs ``N >= 329``,
i.e. 3.6e7 points, which is not affordable with an observer search at every one.

This script instead uses the *exact* axisymmetric reduction (warpax.grids.
axisymmetric_grid): the Rodal shift depends on position only through
``x - v_s t``, ``r`` and ``n_x``, so it is invariant under rotations about the
propagation axis and the ``(r, mu)`` half-plane sweeps the whole orbit space
without loss. Every sigma then clears four radial cells at every level of a
three-point ladder, for a few thousand points per run.

Outputs
-------
- results/rodal_sigma_resolved.json
- ../warpax_arxiv/tables/rodal_sigma_resolved.tex
"""

from __future__ import annotations

import argparse
import os

import jax
from _json_io import dump_json
from _json_io import write_table as write_tex_table

jax.config.update("jax_enable_x64", True)

import numpy as np

from warpax.analysis.invariant_verification import single_frame_miss
from warpax.energy_conditions.frame_free import certify_grid_frame_free
from warpax.geometry import evaluate_curvature_points
from warpax.grids import axisymmetric_grid, proper_volume_weights, wall_cells_on_axis
from warpax.metrics import RodalMetric

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
TABLES_DIR = os.path.join(HERE, "..", "..", "warpax_arxiv", "tables")

V_S = 0.5
R_B = 100.0
R_MAX = 300.0
SIGMAS = (0.01, 0.03, 0.1, 0.3)
LADDER = ((48, 48), (64, 64), (80, 80), (96, 96))
# Radial clustering strength per sigma. A thin wall inside a wide box needs more
# stretching; chosen once here so the coarsest ladder level already clears the
# four-cell floor, and reported in the output so the choice is auditable.
CLUSTER_A = {0.01: 1.0, 0.03: 1.5, 0.1: 3.0, 0.3: 4.5}
F_LOW, F_HIGH = 0.1, 0.9
MIN_WALL_CELLS = 4.0


def _wall_cells(metric, r_nodes) -> float:
    """Worst-case cells across the 10-90% wall on the radial axis."""
    return float(wall_cells_on_axis(metric, np.asarray(r_nodes)).cells)


def run_one(sigma: float, n_r: int, n_mu: int) -> dict:
    metric = RodalMetric(v_s=V_S, R=R_B, sigma=sigma)
    grid = axisymmetric_grid(R_MAX, n_r, n_mu, wall_radius=R_B, a=CLUSTER_A[sigma])
    cells = _wall_cells(metric, grid.r)

    curv = evaluate_curvature_points(metric, grid.coords, batch_size=256)
    T, g, gi = curv.stress_energy, curv.metric, curv.metric_inv

    f = np.asarray(jax.vmap(metric.shape_function_value)(grid.coords))
    wall = (f >= F_LOW) & (f <= F_HIGH)
    w = np.asarray(proper_volume_weights(grid.weights, g))

    ff = certify_grid_frame_free(T, g, gi, solver="auto")
    he = np.asarray(ff.he_types).ravel()
    sel = wall & np.isfinite(he)
    w_wall = w[sel].sum()

    # Unconditional (whole sampled ball) and wall-restricted (conditional) miss.
    grid_miss = single_frame_miss(T, g, gi, volume_weights=w)
    wall_miss = single_frame_miss(T, g, gi, mask=wall, volume_weights=w)

    return {
        "sigma": sigma,
        "n_r": n_r,
        "n_mu": n_mu,
        "cluster_a": CLUSTER_A[sigma],
        "wall_cells": cells,
        "resolved": bool(cells >= MIN_WALL_CELLS),
        "n_wall_points": int(sel.sum()),
        "wall_volume": float(w_wall),
        "frac_type_i": float(w[sel & (he == 1.0)].sum() / w_wall) if w_wall else float("nan"),
        "frac_type_iv": float(w[sel & (he == 4.0)].sum() / w_wall) if w_wall else float("nan"),
        "grid_dec_miss_pct": 100.0 * grid_miss["dec"]["miss_rate"],
        "wall_dec_miss_pct": 100.0 * wall_miss["dec"]["miss_rate"],
        "wall_wec_miss_pct": 100.0 * wall_miss["wec"]["miss_rate"],
        "wall_nec_miss_pct": 100.0 * wall_miss["nec"]["miss_rate"],
    }


def write_table(rows_by_sigma: dict, out_path: str) -> None:
    lines = [
        r"\begin{tabular}{@{}r r r r r r@{}}",
        r"  \toprule",
        r"  & \multicolumn{2}{c}{Wall cells} & Wall & Grid DEC"
        r" & Wall DEC \\",
        r"  \cmidrule(lr){2-3}",
        r"  $\sigma$ & coarsest & finest & points & miss (\%) & miss (\%) \\",
        r"  \midrule",
    ]
    for sigma in SIGMAS:
        rows = rows_by_sigma[str(sigma)]
        fine = rows[-1]
        series = [r["wall_dec_miss_pct"] for r in rows]
        # Full ladder spread, not the last-two gap: at sigma = 0.1 the top two
        # levels nearly coincide and the gap understates the drift 51x.
        spread2 = max(abs(v - series[-1]) for v in series) if len(series) > 1 else 0.0
        pts = f"{fine['n_wall_points']:,}".replace(",", r"{,}")
        lines.append(
            f"  ${sigma}$ & {rows[0]['wall_cells']:.1f} & {fine['wall_cells']:.1f}"
            f" & ${pts}$ & {fine['grid_dec_miss_pct']:.2f}"
            f" & ${fine['wall_dec_miss_pct']:.2f} \\pm {spread2:.2f}$ \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_tex_table(
        out_path,
        lines,
        script="scripts/run_rodal_sigma_resolved.py",
        sources="results/rodal_sigma_resolved.json",
    )
    print(f"  Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sigmas", type=float, nargs="+", default=list(SIGMAS))
    p.add_argument("--smoke", action="store_true", help="coarsest level only")
    args = p.parse_args()

    ladder = LADDER[:1] if args.smoke else LADDER

    print("=" * 78)
    print("RODAL SIGMA SWEEP: wall-resolved, axisymmetric reduction")
    print("=" * 78)
    print(f"  v_s={V_S}  R_b={R_B}  r_max={R_MAX}  ladder={list(ladder)}")

    out_rows: dict[str, list] = {}
    for sigma in args.sigmas:
        rows = []
        for n_r, n_mu in ladder:
            r = run_one(sigma, n_r, n_mu)
            rows.append(r)
            flag = "" if r["resolved"] else "  << UNDER-RESOLVED"
            print(
                f"  sigma={sigma:<5} ({n_r:3d}x{n_mu:3d})  cells={r['wall_cells']:6.2f}"
                f"  wallpts={r['n_wall_points']:5d}"
                f"  TypeI={r['frac_type_i'] * 100:6.2f}%"
                f"  gridDEC={r['grid_dec_miss_pct']:6.2f}%"
                f"  wallDEC={r['wall_dec_miss_pct']:6.2f}%{flag}"
            )
        series = [x["wall_dec_miss_pct"] for x in rows]
        print(
            f"      -> wall DEC miss across ladder: "
            f"{' -> '.join(f'{v:.2f}' for v in series)}"
            f"   finest-two spread {abs(series[-1] - series[-2]):.2f} pp"
            if len(series) > 1
            else ""
        )
        out_rows[str(sigma)] = rows

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "params": {
            "v_s": V_S,
            "R_b": R_B,
            "r_max": R_MAX,
            "sigmas": list(args.sigmas),
            "ladder": [list(x) for x in ladder],
            "cluster_a": CLUSTER_A,
            "wall_band": [F_LOW, F_HIGH],
            "min_wall_cells": MIN_WALL_CELLS,
            "note": "Exact axisymmetric (r, mu) reduction: the Rodal slice is "
            "invariant under rotations about the propagation axis, so "
            "the half-plane sweeps the full orbit space. Proper-volume "
            "weights 2 pi r^2 w_r w_mu; Gauss-Legendre in mu.",
        },
        "rows": out_rows,
    }
    dump_json(payload, os.path.join(RESULTS_DIR, "rodal_sigma_resolved.json"))
    print(f"\nWrote {os.path.join(RESULTS_DIR, 'rodal_sigma_resolved.json')}")
    if not args.smoke:
        write_table(out_rows, os.path.join(TABLES_DIR, "rodal_sigma_resolved.tex"))


if __name__ == "__main__":
    main()
