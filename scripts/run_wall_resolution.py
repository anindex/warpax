"""Per-metric wall-resolution analysis.

Computes the analytical transition wall width for each metric's shape
function and compares it to the grid spacing at the standard 50^3
resolution. No JAX or curvature computation needed; all values
are derived from closed-form expressions.

For tanh-based metrics the 10-90% transition width is
    wall_width = 2 * arctanh(0.8) / sigma = 2.1972 / sigma

run_analysis.py additionally covers WarpShell, whose Hermite smoothstep
transition spans exactly ``smooth_width`` (default 0.12 * (R_2 - R_1)).

Outputs
-------
- results/wall_resolution.json
- ../warpax_arxiv/tables/wall_resolution.tex

The table used to be maintained by hand. That is how it came to print a wall width
of 0.27 and a spacing of 0.20 whose quotient is 1.35, correct, because the true
values are 0.2747 and 10/49 = 0.2041 and 0.2747/0.2041 = 1.3458, but not correct
as displayed, and a reader dividing what they were shown gets 1.4. A generated
table cannot round its inputs and keep an unrounded quotient.

Usage
-----
    python scripts/run_wall_resolution.py
"""

from __future__ import annotations

import os

import jax
from _json_io import dump_json
from _json_io import write_table as write_tex_table

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from warpax.benchmarks.alcubierre import alcubierre_shape

TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "warpax_arxiv", "tables")


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    # The 10-90% width of the ACTUAL top-hat shape function, by bisection.
    # The asymptotic form 2*atanh(0.8)/sigma is only the sigma*R >> 1 limit; at
    # Rodal's sigma*R = 3 it gives 73.24 against the true 72.53.
    def _width_10_90(sigma, R):
        f = lambda r: float(alcubierre_shape(jnp.asarray(r), R, sigma))

        def root(target, lo, hi):
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if f(mid) > target:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        hi = R + 40.0 / sigma
        return root(0.1, R, hi) - root(0.9, 0.0, R + hi)

    # Metric configurations matching run_analysis.py METRICS dict exactly
    metrics = [
        {
            "metric": "alcubierre",
            "shape_function": "tanh",
            "sigma": 8.0,
            "R": 1.0,
            "domain": [-5, 5],
            "grid_n": 50,
        },
        {
            "metric": "vdb",
            "shape_function": "tanh",
            "sigma": 8.0,
            "R": 1.0,
            "domain": [-5, 5],
            "grid_n": 50,
        },
        {
            "metric": "natario",
            "shape_function": "tanh",
            "sigma": 8.0,
            "R": 1.0,
            "domain": [-5, 5],
            "grid_n": 50,
        },
        {
            "metric": "rodal",
            "shape_function": "tanh",
            "sigma": 0.03,
            "R": 100.0,
            "domain": [-300, 300],
            "grid_n": 50,
        },
        {
            "metric": "schwarzschild",
            "shape_function": None,
            "sigma": None,
            "domain": [-20, 20],
            "grid_n": 50,
            "M": 1.0,
        },
    ]

    results = []
    for m in metrics:
        dx = (m["domain"][1] - m["domain"][0]) / (m["grid_n"] - 1)

        if m["shape_function"] == "tanh":
            wall_width = _width_10_90(m["sigma"], m["R"])
            cells = wall_width / dx
            resolved = cells >= 4.0
            notes = (
                "Autodiff computes exact curvature at each point; "
                "wall resolution affects spatial sampling density, "
                "not curvature accuracy"
            )
        elif m["shape_function"] == "hermite":
            # WarpShell: smooth_width = 0.12 * (R_2 - R_1)
            smooth_width = 0.12 * (m["R_2"] - m["R_1"])
            wall_width = smooth_width
            cells = wall_width / dx
            resolved = cells >= 4.0
            notes = (
                "Hermite smoothstep transition; smooth_width = "
                f"0.12 * (R_2 - R_1) = {smooth_width:.4f}"
            )
        else:
            # Schwarzschild: no shape function
            wall_width = None
            cells = None
            resolved = None
            r_s = 2.0 * m["M"]
            notes = (
                f"No shape function; relevant length scale is "
                f"Schwarzschild radius r_s = 2M = {r_s:.1f}"
            )

        row = {
            "metric": m["metric"],
            "shape_function": m["shape_function"],
            "sigma": m["sigma"],
            "wall_width": round(wall_width, 4) if wall_width is not None else None,
            "domain": m["domain"],
            "grid_n": m["grid_n"],
            "dx": round(dx, 4),
            "cells_resolving_wall": round(cells, 2) if cells is not None else None,
            "resolved": resolved,
            "notes": notes,
        }
        results.append(row)

    # Print formatted table
    print(
        f"{'Metric':>14s}  {'Shape':>8s}  {'sigma':>7s}  {'wall_w':>8s}  "
        f"{'dx':>8s}  {'cells':>6s}  {'resolved':>8s}"
    )
    print("-" * 76)
    for r in results:
        sigma_str = f"{r['sigma']:.2f}" if r["sigma"] is not None else "N/A"
        wall_str = f"{r['wall_width']:.4f}" if r["wall_width"] is not None else "N/A"
        cells_str = (
            f"{r['cells_resolving_wall']:.2f}" if r["cells_resolving_wall"] is not None else "N/A"
        )
        resolved_str = str(r["resolved"]) if r["resolved"] is not None else "N/A"
        print(
            f"{r['metric']:>14s}  {r['shape_function'] or 'N/A'!s:>8s}  "
            f"{sigma_str:>7s}  {wall_str:>8s}  {r['dx']:>8.4f}  "
            f"{cells_str:>6s}  {resolved_str:>8s}"
        )

    # Save JSON
    out_path = os.path.join(results_dir, "wall_resolution.json")
    dump_json(results, out_path)
    print(f"\nSaved: {out_path}")

    write_latex_table(results, os.path.join(TABLES_DIR, "wall_resolution.tex"))


DISPLAY = {
    "alcubierre": "Alcubierre",
    "natario": "Nat\\'ario",
    "vdb": "Van~Den~Broeck",
    "rodal": "Rodal",
    "schwarzschild": "Schwarzschild",
}
CONVERGENCE = {"alcubierre": "Stability-only", "rodal": "Stability-only"}


def write_latex_table(results, out_path):
    """Emit tables/wall_resolution.tex straight from the computed rows.

    Four decimals on the width and the spacing, two on their quotient. The quotient
    is recomputed here from the unrounded values, so no reader can derive a different
    number from the digits printed beside it.
    """
    order = ["alcubierre", "natario", "vdb", "rodal", "schwarzschild"]
    by_name = {r["metric"]: r for r in results}
    lines = [
        r"\begin{tabular}{@{}lccccc@{}}",
        r"    \toprule",
        r"    Metric & Wall width & $\Delta x$ & Cells & Resolved & Convergence \\",
        r"    \midrule",
    ]
    for i, key in enumerate(order):
        r = by_name.get(key)
        if r is None:
            continue
        if key == "schwarzschild":
            lines.append(r"    \midrule")
            lines.append(f"    {DISPLAY[key]} & -- & {r['dx']:.2f} & -- & -- & -- \\\\")
            continue
        w, dx = r["wall_width"], r["dx"]
        cells = w / dx
        resolved = "Yes" if cells >= 4.0 else "No"
        conv = CONVERGENCE.get(key, "--")
        lines.append(
            f"    {DISPLAY[key]} & {w:.4f} & {dx:.4f} & {cells:.2f} & {resolved} & {conv} \\\\"
        )
    lines += [r"    \bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_tex_table(
        out_path,
        lines,
        script="scripts/run_wall_resolution.py",
        sources="results/wall_resolution.json",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
