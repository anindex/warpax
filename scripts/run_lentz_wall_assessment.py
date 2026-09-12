"""Compare Lentz wall width with the spacing of a specified 3D grid.

A 500-point radial cut supplements the width estimate with local curvature
diagnostics. The outputs describe sampling at the tested resolution:
results/lentz_wall_assessment.json and results/lentz_wall_report.md.

Run: python scripts/run_lentz_wall_assessment.py
"""

from __future__ import annotations

import math
import os
import time
from datetime import UTC, datetime

import matplotlib
from _json_io import dump_json

matplotlib.use("Agg")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpax.geometry import compute_curvature_chain, kretschmann_scalar
from warpax.metrics import LentzMetric

# Constants

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TANH_10_90 = 2.0 * math.atanh(0.8)  # 10-90% transition width factor

# Lentz parameters (defaults from LentzMetric)
R_LENTZ = 100.0
SIGMA_LENTZ = 8.0
GRID_N = 50  # Standard 3D grid resolution
DOMAIN_LENTZ = (-300, 300)

# Radial sampling parameters.
N_RADIAL = 500
RADIAL_RANGE = (50.0, 150.0)  # Focus on wall region around R=100


# Analytical assessment


def analytical_assessment():
    """Compute wall-width vs grid-spacing ratio analytically.

    For tanh-based shape functions, the 10-90% transition width is
        wall_width = 2 * arctanh(0.8) / sigma = TANH_10_90 / sigma

    Returns
    -------
    dict
        Analytical assessment with wall_width, dx, cells, resolved flag.
    """
    wall_width = TANH_10_90 / SIGMA_LENTZ
    dx = (DOMAIN_LENTZ[1] - DOMAIN_LENTZ[0]) / (GRID_N - 1)
    cells = wall_width / dx
    resolved = cells >= 4.0
    under_resolution_ratio = dx / wall_width

    result = {
        "wall_width": round(wall_width, 6),
        "dx": round(dx, 4),
        "cells_across_wall": round(cells, 4),
        "resolved": bool(resolved),
        "threshold_cells": 4.0,
        "under_resolution_ratio": round(under_resolution_ratio, 2),
        "sigma": SIGMA_LENTZ,
        "domain": list(DOMAIN_LENTZ),
        "grid_n": GRID_N,
        "notes": (
            "Automatic differentiation evaluates local curvature in floating-point "
            "arithmetic; spatial sampling requires a separate resolution check."
        ),
    }

    print("\n--- Analytical Wall-Width Assessment ---")
    print(f"  Wall width (10-90%): {wall_width:.6f}")
    print(f"  Grid spacing (dx):   {dx:.4f}")
    print(f"  Cells across wall:   {cells:.4f}")
    print(f"  Resolved (>= 4):     {resolved}")
    print(f"  Under-resolution:    {under_resolution_ratio:.1f}x")

    return result


# 1D radial cut assessment


def radial_cut_assessment():
    """Evaluate curvature along 1D radial cut through the Lentz wall.

    Samples N_RADIAL points along the x-axis (with y=0.01 offset to avoid
    the L1 kink at y=z=0) and computes curvature invariants at each point.

    Returns
    -------
    list[dict]
        Per-point curvature data along the radial cut.
    """
    metric = LentzMetric(v_s=0.5, R=R_LENTZ, sigma=SIGMA_LENTZ)
    r_values = np.linspace(RADIAL_RANGE[0], RADIAL_RANGE[1], N_RADIAL)

    print(f"\n--- 1D Radial Cut (N={N_RADIAL}, r=[{RADIAL_RANGE[0]}, {RADIAL_RANGE[1]}]) ---")
    print("  y=0.01 offset (L1 kink avoidance)")

    results = []
    t0 = time.time()
    for i, r in enumerate(r_values):
        coords = jnp.array([0.0, float(r), 0.01, 0.0])
        curv = compute_curvature_chain(metric, coords)

        # Coordinate-invariant Kretschmann scalar K = R_{abcd} R^{abcd}.
        K = float(kretschmann_scalar(curv.riemann, curv.metric, curv.metric_inv))
        T_frobenius = float(jnp.linalg.norm(curv.stress_energy))
        f_val = float(metric.shape_function_value(coords))

        results.append(
            {
                "r": float(r),
                "kretschmann": K,
                "T_frobenius": T_frobenius,
                "f": f_val,
            }
        )

        if (i + 1) % 100 == 0:
            print(f"  Computed {i + 1}/{N_RADIAL} points...")

    elapsed = time.time() - t0
    print(f"  Radial cut complete in {elapsed:.1f}s")

    kretschmann_vals = [p["kretschmann"] for p in results]
    T_vals = [p["T_frobenius"] for p in results]
    peak_k_idx = int(np.argmax(np.abs(kretschmann_vals)))
    peak_T_idx = int(np.argmax(T_vals))

    print(
        f"  Peak |Kretschmann|: {abs(kretschmann_vals[peak_k_idx]):.6e} "
        f"at r={results[peak_k_idx]['r']:.2f}"
    )
    print(f"  Peak T_Frobenius:   {T_vals[peak_T_idx]:.6e} at r={results[peak_T_idx]['r']:.2f}")

    return results


# JSON output


def save_json(analytical, radial_cut, start_time):
    """Save structured assessment to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output = {
        "metadata": {
            "date": start_time,
            "script": "scripts/run_lentz_wall_assessment.py",
            "lentz_params": {"R": R_LENTZ, "sigma": SIGMA_LENTZ},
            "grid_n": GRID_N,
            "domain": list(DOMAIN_LENTZ),
            "radial_n": N_RADIAL,
            "radial_range": list(RADIAL_RANGE),
        },
        "analytical": analytical,
        "radial_cut": radial_cut,
        "verdict": {
            "resolved": analytical["resolved"],
            "under_resolution_ratio": analytical["under_resolution_ratio"],
            "note": (
                "Automatic differentiation evaluates local curvature in floating-point "
                "arithmetic; spatial sampling requires a separate resolution check."
            ),
        },
    }

    outpath = os.path.join(RESULTS_DIR, "lentz_wall_assessment.json")
    dump_json(output, outpath)
    print(f"\nJSON saved to {outpath}")
    return output


# Markdown report


def save_report(analytical, radial_cut, start_time):
    """Render wall sampling scales and the retained radial diagnostics."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    peak_k = max(range(len(radial_cut)), key=lambda i: abs(radial_cut[i]["kretschmann"]))
    peak_t = max(range(len(radial_cut)), key=lambda i: radial_cut[i]["T_frobenius"])
    lines = [
        "# Lentz wall resolution",
        "",
        f"Date: {start_time}. Source: `lentz_wall_assessment.json`.",
        "",
        f"The tested {analytical['grid_n']}^3 grid on {analytical['domain']}^3 "
        f"{'passes' if analytical['resolved'] else 'fails'} the chosen "
        f"{analytical['threshold_cells']:g}-cell wall-resolution criterion at sigma={analytical['sigma']}. "
        "The conclusion applies to this grid and domain.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| 10-90% wall width, 2 atanh(0.8)/sigma | {analytical['wall_width']:.6f} |",
        f"| Grid spacing | {analytical['dx']:.4f} |",
        f"| Cells across wall | {analytical['cells_across_wall']:.4f} |",
        f"| Grid spacing / wall width | {analytical['under_resolution_ratio']:.2f} |",
        "",
        f"The {len(radial_cut)}-point cut spans r={radial_cut[0]['r']:g} to "
        f"{radial_cut[-1]['r']:g} at (t,x,y,z)=(0,r,0.01,0), v_s=0.5 and R={R_LENTZ:g}. "
        f"Its sampled maximum absolute Kretschmann scalar is "
        f"{abs(radial_cut[peak_k]['kretschmann']):.6e} at r={radial_cut[peak_k]['r']:.2f}; "
        f"the sampled maximum stress-tensor Frobenius norm is "
        f"{radial_cut[peak_t]['T_frobenius']:.6e} at r={radial_cut[peak_t]['r']:.2f}.",
        "",
        "K=R_abcd R^abcd is invariant; the Frobenius norm uses the coordinate components of T_ab. "
        "Automatic differentiation evaluates local derivatives in floating-point arithmetic. "
        "It does not bound unsampled extrema or integration errors. "
        "Selected points near the sampled peak and the endpoints follow.",
        "",
        "| r | f(r) | Absolute K | Frobenius norm of T |",
        "|---:|---:|---:|---:|",
    ]
    selected = {0, len(radial_cut) - 1, peak_t}
    selected.update(range(max(0, peak_k - 1), min(len(radial_cut), peak_k + 2)))
    for i in sorted(selected):
        p = radial_cut[i]
        lines.append(
            f"| {p['r']:.2f} | {p['f']:.6f} | {abs(p['kretschmann']):.6e} | {p['T_frobenius']:.6e} |"
        )
    lines.append("")
    path = os.path.join(RESULTS_DIR, "lentz_wall_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {path}")


# Main


def main():
    """Run Lentz wall resolution assessment."""
    start_time = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=" * 70)
    print("Lentz Wall Resolution Assessment")
    print(f"R={R_LENTZ}, sigma={SIGMA_LENTZ}")
    print(f"Started: {start_time}")
    print("=" * 70)

    # Part 1: Analytical wall-width assessment
    analytical = analytical_assessment()

    # Part 2: 1D radial cut with JAX curvature computation
    radial_cut = radial_cut_assessment()

    # Save outputs
    save_json(analytical, radial_cut, start_time)
    save_report(analytical, radial_cut, start_time)

    # Final summary
    print("\n" + "=" * 70)
    print("ASSESSMENT COMPLETE")
    print("=" * 70)
    if not analytical["resolved"]:
        print(
            f"  Verdict: UNRESOLVABLE at N={GRID_N} "
            f"({analytical['under_resolution_ratio']:.1f}x under-resolved)"
        )
    else:
        print(f"  Verdict: RESOLVED ({analytical['cells_across_wall']:.2f} cells)")


if __name__ == "__main__":
    main()
