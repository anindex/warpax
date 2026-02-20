
"""WarpShell resolution stability sweep: 25^3, 50^3, 100^3.

Evaluates WarpShell at three resolutions to check convergence of
Eulerian NEC margins. At 25^3 and 50^3, runs full Eulerian+robust
comparison. At 100^3, runs Eulerian-only (robust optimization is
prohibitively expensive at 1M points).

Output: results/warpshell_convergence.json
"""
from __future__ import annotations

import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.analysis import compare_eulerian_vs_robust
from warpax.energy_conditions.verifier import _eulerian_ec_point
from warpax.metrics import WarpShellMetric


RESOLUTIONS = [25, 50, 100]
METRIC_PARAMS = {"R_1": 0.5, "R_2": 1.0, "v_s": 0.5}
BOUNDS = [(-5, 5)] * 3
N_STARTS = 8


def main():
    os.makedirs("results", exist_ok=True)
    metric = WarpShellMetric(**METRIC_PARAMS)

    results = {
        "resolutions": RESOLUTIONS,
        "min_nec_margin_eulerian": [],
        "integrated_nec_violation_eulerian": [],
        "min_wec_margin_eulerian": [],
        "min_nec_margin_robust": [],
        "integrated_nec_violation_robust": [],
        "min_wec_margin_robust": [],
        "n_type_i": [],
        "n_type_iv": [],
        "time_s": [],
    }

    for N in RESOLUTIONS:
        grid_spec = GridSpec(bounds=BOUNDS, shape=(N, N, N))
        n_points = N ** 3
        print(f"\n{'='*60}")
        print(f"Resolution: {N}^3 = {n_points} points")
        print(f"{'='*60}")

        # Step 1: Compute curvature grid
        print("  Computing curvature grid...")
        t0 = time.time()
        curv = evaluate_curvature_grid(metric, grid_spec, batch_size=256)
        t_curv = time.time() - t0
        print(f"  Curvature grid: {t_curv:.1f}s")

        # Step 2: Eulerian EC evaluation (always)
        print("  Computing Eulerian EC margins...")
        flat_T = curv.stress_energy.reshape(n_points, 4, 4)
        flat_g = curv.metric.reshape(n_points, 4, 4)
        flat_g_inv = curv.metric_inv.reshape(n_points, 4, 4)

        eul_results = jax.vmap(_eulerian_ec_point)(flat_T, flat_g, flat_g_inv)
        eul_nec = np.asarray(eul_results["nec"])
        eul_wec = np.asarray(eul_results["wec"])

        n_nan_nec = int(np.sum(np.isnan(eul_nec)))
        n_nan_wec = int(np.sum(np.isnan(eul_wec)))
        print(f"  NaN count: NEC={n_nan_nec}, WEC={n_nan_wec} out of {n_points}")

        min_nec_eul = float(np.nanmin(eul_nec))
        min_wec_eul = float(np.nanmin(eul_wec))
        # Use nansum to avoid NaN propagation from shell boundary points
        integrated_nec_eul = float(np.nansum(np.minimum(np.nan_to_num(eul_nec, nan=0.0), 0.0)))

        results["min_nec_margin_eulerian"].append(min_nec_eul)
        results["min_wec_margin_eulerian"].append(min_wec_eul)
        results["integrated_nec_violation_eulerian"].append(integrated_nec_eul)

        print(f"  Eulerian min NEC: {min_nec_eul:.6e}")
        print(f"  Eulerian min WEC: {min_wec_eul:.6e}")
        print(f"  Eulerian integrated NEC violation: {integrated_nec_eul:.6e}")

        # Step 3: Robust EC (only at 25^3 and 50^3)
        if N <= 50:
            print(f"  Computing robust EC (n_starts={N_STARTS})...")
            t1 = time.time()
            comparison = compare_eulerian_vs_robust(
                curv.stress_energy,
                curv.metric,
                curv.metric_inv,
                grid_spec.shape,
                n_starts=N_STARTS,
                batch_size=64,
            )
            t_rob = time.time() - t1
            print(f"  Robust EC: {t_rob:.1f}s")

            rob_nec = np.asarray(comparison.robust_margins["nec"])
            rob_wec = np.asarray(comparison.robust_margins["wec"])
            min_nec_rob = float(np.nanmin(rob_nec))
            min_wec_rob = float(np.nanmin(rob_wec))
            integrated_nec_rob = float(np.nansum(np.minimum(np.nan_to_num(rob_nec.ravel(), nan=0.0), 0.0)))

            results["min_nec_margin_robust"].append(min_nec_rob)
            results["min_wec_margin_robust"].append(min_wec_rob)
            results["integrated_nec_violation_robust"].append(integrated_nec_rob)
            results["n_type_i"].append(comparison.classification_stats["n_type_i"])
            results["n_type_iv"].append(comparison.classification_stats["n_type_iv"])
            results["time_s"].append(round(t_curv + t_rob, 1))

            print(f"  Robust min NEC: {min_nec_rob:.6e}")
            print(f"  Robust min WEC: {min_wec_rob:.6e}")
            print(f"  Robust integrated NEC violation: {integrated_nec_rob:.6e}")
            print(f"  Type I: {comparison.classification_stats['n_type_i']}")
            print(f"  Type IV: {comparison.classification_stats['n_type_iv']}")
        else:
            # 100^3: Eulerian-only (robust too expensive)
            results["min_nec_margin_robust"].append(None)
            results["min_wec_margin_robust"].append(None)
            results["integrated_nec_violation_robust"].append(None)
            results["n_type_i"].append(None)
            results["n_type_iv"].append(None)
            results["time_s"].append(round(t_curv, 1))
            print("  [Skipping robust EC at 100^3 Eulerian-only]")

    # Step 4: Richardson extrapolation on Eulerian min NEC margin
    f25, f50, f100 = results["min_nec_margin_eulerian"]
    diff1 = f25 - f50
    diff2 = f50 - f100
    if abs(diff2) > 1e-30 and abs(diff1) > 1e-30 and diff1 / diff2 > 0:
        p = math.log2(diff1 / diff2)
        # Richardson extrapolation: f_extrap = f_100 + (f_100 - f_50) / (2^p - 1)
        extrapolated = f100 + (f100 - f50) / (2**p - 1)
    else:
        p = None
        extrapolated = None

    results["richardson_order_min_nec_eulerian"] = p
    results["extrapolated_min_nec_eulerian"] = extrapolated

    # Also Richardson on integrated violation
    iv25, iv50, iv100 = results["integrated_nec_violation_eulerian"]
    diff1_iv = iv25 - iv50
    diff2_iv = iv50 - iv100
    if (not any(math.isnan(v) for v in [iv25, iv50, iv100])
            and abs(diff2_iv) > 1e-30 and abs(diff1_iv) > 1e-30
            and diff1_iv / diff2_iv > 0):
        p_iv = math.log2(diff1_iv / diff2_iv)
        extrapolated_iv = iv100 + (iv100 - iv50) / (2**p_iv - 1)
    else:
        p_iv = None
        extrapolated_iv = None

    results["richardson_order_integrated_nec_eulerian"] = p_iv
    results["extrapolated_integrated_nec_eulerian"] = extrapolated_iv

    # Save
    out_path = "results/warpshell_convergence.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Resolutions: {RESOLUTIONS}")
    print(f"Eulerian min NEC: {results['min_nec_margin_eulerian']}")
    print(f"Eulerian integrated NEC: {results['integrated_nec_violation_eulerian']}")
    print(f"Richardson order (min NEC): {p}")
    print(f"Extrapolated min NEC: {extrapolated}")
    print(f"Richardson order (integrated NEC): {p_iv}")
    print(f"Extrapolated integrated NEC: {extrapolated_iv}")


if __name__ == "__main__":
    main()
