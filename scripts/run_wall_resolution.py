
"""Per-metric wall-resolution analysis.

Computes the analytical transition wall width for each metric's shape
function and compares it to the grid spacing at the standard 50^3
resolution.  No JAX or curvature computation is needed all values
are derived from closed-form expressions.

For tanh-based metrics the 10-90% transition width is
    wall_width = 2 * arctanh(0.8) / sigma = 2.1972 / sigma

For WarpShell's Hermite smoothstep the transition covers exactly
``smooth_width`` (default 0.12 * (R_2 - R_1)).

Output: results/wall_resolution.json

Usage
-----
    python scripts/run_wall_resolution.py
"""
from __future__ import annotations

import json
import math
import os


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    # For f(r) = [1 - tanh(sigma*(r-R))]/2, the 10-90% width is
    # f=0.9 -> tanh=-0.8, f=0.1 -> tanh=+0.8, so Delta_r = 2*atanh(0.8)/sigma
    TANH_10_90 = 2.0 * math.atanh(0.8)

    # Metric configurations matching run_analysis.py METRICS dict exactly
    metrics = [
        {
            "metric": "alcubierre",
            "shape_function": "tanh",
            "sigma": 8.0,
            "domain": [-5, 5],
            "grid_n": 50,
        },
        {
            "metric": "vdb",
            "shape_function": "tanh",
            "sigma": 8.0,
            "domain": [-5, 5],
            "grid_n": 50,
        },
        {
            "metric": "natario",
            "shape_function": "tanh",
            "sigma": 8.0,
            "domain": [-5, 5],
            "grid_n": 50,
        },
        {
            "metric": "lentz",
            "shape_function": "tanh",
            "sigma": 8.0,
            "domain": [-300, 300],
            "grid_n": 50,
        },
        {
            "metric": "rodal",
            "shape_function": "tanh",
            "sigma": 0.03,
            "domain": [-300, 300],
            "grid_n": 50,
        },
        {
            "metric": "warpshell",
            "shape_function": "hermite",
            "sigma": None,
            "domain": [-5, 5],
            "grid_n": 50,
            "R_1": 0.5,
            "R_2": 1.0,
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
            wall_width = TANH_10_90 / m["sigma"]
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
    print(f"{'Metric':>14s}  {'Shape':>8s}  {'sigma':>7s}  {'wall_w':>8s}  "
          f"{'dx':>8s}  {'cells':>6s}  {'resolved':>8s}")
    print("-" * 76)
    for r in results:
        sigma_str = f"{r['sigma']:.2f}" if r["sigma"] is not None else "N/A"
        wall_str = f"{r['wall_width']:.4f}" if r["wall_width"] is not None else "N/A"
        cells_str = f"{r['cells_resolving_wall']:.2f}" if r["cells_resolving_wall"] is not None else "N/A"
        resolved_str = str(r["resolved"]) if r["resolved"] is not None else "N/A"
        print(f"{r['metric']:>14s}  {str(r['shape_function'] or 'N/A'):>8s}  "
              f"{sigma_str:>7s}  {wall_str:>8s}  {r['dx']:>8.4f}  "
              f"{cells_str:>6s}  {resolved_str:>8s}")

    # Save JSON
    out_path = os.path.join(results_dir, "wall_resolution.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
