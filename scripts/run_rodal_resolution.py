
"""Rodal missed-fraction stability across grid resolutions.

Rodal is 100% Type I, so the ground truth is algebraic eigenvalue
checks (no BFGS needed for truth). This script runs at N = {25, 50, 100}
and reports f_miss for NEC/WEC/SEC/DEC, verifying resolution stability.

Usage
-----
    python scripts/run_rodal_resolution.py
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpax.metrics import RodalMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.analysis import compare_eulerian_vs_robust


def run_resolution_study():
    """Run Rodal at multiple resolutions and report missed fractions."""
    resolutions = [25, 50, 100]
    v_s = 0.5
    R = 100.0
    sigma = 0.03

    print("=" * 70)
    print("Rodal Resolution Stability Study")
    print(f"v_s={v_s}, R={R}, sigma={sigma}")
    print("=" * 70)

    results = []
    for N in resolutions:
        grid = GridSpec(bounds=[(-300, 300)] * 3, shape=(N, N, N))
        n_total = N ** 3

        print(f"\n--- Resolution: {N}^3 = {n_total} points ---")
        metric = RodalMetric(v_s=v_s, R=R, sigma=sigma)

        t0 = time.time()
        curv = evaluate_curvature_grid(metric, grid, batch_size=256)

        # Use n_starts=8 for consistency with paper analysis
        comparison = compare_eulerian_vs_robust(
            curv.stress_energy,
            curv.metric,
            curv.metric_inv,
            grid.shape,
            n_starts=8,
            batch_size=64,
        )
        elapsed = time.time() - t0

        # Extract missed fractions
        row = {
            "N": N,
            "n_total": n_total,
        }
        for cond in ["nec", "wec", "sec", "dec"]:
            rob_margins = comparison.robust_margins[cond]
            eul_margins = comparison.eulerian_margins[cond]

            # Total violations (from robust/hybrid truth)
            violated = float(jnp.sum(rob_margins < -1e-10)) / n_total * 100
            # Missed: Eulerian says OK but robust says violated
            # Use >= 0.0 threshold to match comparison.py line 167
            missed = float(jnp.sum(
                (eul_margins >= 0.0) & (rob_margins < -1e-10)
            )) / n_total * 100
            min_margin = float(jnp.nanmin(rob_margins))

            row[f"{cond}_total"] = violated
            row[f"{cond}_missed"] = missed
            row[f"{cond}_min"] = min_margin

        row["elapsed"] = elapsed
        results.append(row)

        print(f"  NEC: {row['nec_total']:.2f}% violated, {row['nec_missed']:.2f}% missed")
        print(f"  WEC: {row['wec_total']:.2f}% violated, {row['wec_missed']:.2f}% missed")
        print(f"  SEC: {row['sec_total']:.2f}% violated, {row['sec_missed']:.2f}% missed")
        print(f"  DEC: {row['dec_total']:.2f}% violated, {row['dec_missed']:.2f}% missed")
        print(f"  Time: {elapsed:.1f}s")

        # Classification stats
        print(f"  Type I: {int(jnp.sum(comparison.he_types == 1))} / {n_total}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Missed Fraction (%) by Resolution")
    print("=" * 70)
    print(f"{'N':>5s} {'NEC_miss':>10s} {'WEC_miss':>10s} {'SEC_miss':>10s} {'DEC_miss':>10s}")
    for r in results:
        print(f"{r['N']:>5d} {r['nec_missed']:>10.2f} {r['wec_missed']:>10.2f} "
              f"{r['sec_missed']:>10.2f} {r['dec_missed']:>10.2f}")

    print(f"\n{'N':>5s} {'NEC_tot':>10s} {'WEC_tot':>10s} {'SEC_tot':>10s} {'DEC_tot':>10s}")
    for r in results:
        print(f"{r['N']:>5d} {r['nec_total']:>10.2f} {r['wec_total']:>10.2f} "
              f"{r['sec_total']:>10.2f} {r['dec_total']:>10.2f}")


if __name__ == "__main__":
    run_resolution_study()
