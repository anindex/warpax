
"""Richardson extrapolation convergence validation.

Analyzes a single warp metric (default: Alcubierre at v_s=0.5) at three
grid resolutions (25^3, 50^3, 100^3) and applies Richardson extrapolation
to smooth scalar quantities to estimate convergence order and extrapolated
continuum values.

Usage
-----
Default (Alcubierre):
    python scripts/run_convergence.py

Custom metric/resolutions:
    python scripts/run_convergence.py --metric natario --resolutions 10 20 40

IMPORTANT: At 100^3 the observer optimizer is extremely expensive
(1M points x 8 starts x 4 conditions).  By default, the 100^3 resolution
uses Eulerian-only EC (skipping optimization) and computes Eulerian min
margin for convergence.  Use --full-100 to force optimization at 100^3
(may take hours).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np

import jax
import jax.numpy as jnp

from warpax.benchmarks import AlcubierreMetric
from warpax.metrics import RodalMetric, WarpShellMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.analysis import (
    compare_eulerian_vs_robust,
    richardson_extrapolation,
    compute_convergence_quantity,
)
from warpax.energy_conditions.verifier import _eulerian_ec_point

# ---------------------------------------------------------------------------
# Metric and parameter configuration
# ---------------------------------------------------------------------------

METRIC_CONFIGS = {
    "alcubierre": {
        "class": AlcubierreMetric,
        "params": {"v_s": 0.5, "R": 1.0, "sigma": 8.0},
        "bounds": [(-5, 5)] * 3,
    },
    "rodal": {
        "class": RodalMetric,
        "params": {"R": 100.0, "sigma": 0.03, "v_s": 0.5},
        "bounds": [(-300, 300)] * 3,
    },
    "warpshell": {
        "class": WarpShellMetric,
        "params": {"R_1": 0.5, "R_2": 1.0, "v_s": 0.5},
        "bounds": [(-5, 5)] * 3,
    },
}


def _cell_volume(grid_spec: GridSpec) -> float:
    """Compute volume of a single grid cell."""
    vol = 1.0
    for b, n in zip(grid_spec.bounds, grid_spec.shape):
        vol *= (b[1] - b[0]) / max(n - 1, 1)
    return vol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Richardson extrapolation convergence validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="alcubierre",
        choices=list(METRIC_CONFIGS.keys()),
        help="Metric to analyze (default: alcubierre).",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[25, 50, 100],
        help="Grid resolutions (default: 25 50 100).",
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=8,
        help="Multi-start count for BFGS optimization (default: 8).",
    )
    parser.add_argument(
        "--full-100",
        action="store_true",
        help="Run full optimization at 100^3 (very expensive).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for output (default: results).",
    )
    args = parser.parse_args()

    cfg = METRIC_CONFIGS[args.metric]
    metric_class = cfg["class"]
    metric = metric_class(**cfg["params"])
    bounds = cfg["bounds"]
    resolutions = sorted(args.resolutions)
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print(f"Convergence analysis: {args.metric}")
    print(f"Resolutions: {resolutions}")
    print("=" * 60)

    # Collect convergence quantities at each resolution
    quantities = {
        "min_margin_nec": [],
        "l2_violation_nec": [],
        "integrated_violation_nec": [],
    }
    grid_sizes = []

    for N in resolutions:
        print(f"\n--- Resolution: {N}^3 ({N**3} points) ---")
        grid_spec = GridSpec(bounds=bounds, shape=(N, N, N))
        cell_vol = _cell_volume(grid_spec)

        # Batch sizes: smaller for larger grids
        curv_batch = 256 if N <= 50 else 128
        ec_batch = 64 if N <= 50 else 16

        # Step 1: Curvature grid
        print(f"  Computing curvature grid (batch_size={curv_batch})...")
        t0 = time.time()
        curv = evaluate_curvature_grid(metric, grid_spec, batch_size=curv_batch)
        print(f"    Curvature: {time.time() - t0:.1f}s")

        # Step 2: EC analysis
        # For the largest resolution, use Eulerian-only by default to avoid
        # prohibitive optimizer cost (1M points x 8 starts x 4 conditions).
        use_full_optimizer = (N <= 50) or args.full_100

        if use_full_optimizer:
            print(f"  Running full Eulerian vs robust comparison (batch_size={ec_batch})...")
            t0 = time.time()
            comparison = compare_eulerian_vs_robust(
                curv.stress_energy,
                curv.metric,
                curv.metric_inv,
                grid_spec.shape,
                n_starts=args.n_starts,
                batch_size=ec_batch,
            )
            nec_margin = np.asarray(comparison.robust_margins["nec"])
            print(f"    Comparison: {time.time() - t0:.1f}s")
        else:
            # Eulerian-only for convergence (much cheaper)
            print(f"  Running Eulerian-only EC (skipping optimization for {N}^3)...")
            t0 = time.time()
            n_points = int(np.prod(grid_spec.shape))
            flat_T = curv.stress_energy.reshape(n_points, 4, 4)
            flat_g = curv.metric.reshape(n_points, 4, 4)
            flat_g_inv = curv.metric_inv.reshape(n_points, 4, 4)
            eul = jax.vmap(_eulerian_ec_point)(flat_T, flat_g, flat_g_inv)
            nec_margin = np.asarray(eul["nec"]).reshape(grid_spec.shape)
            print(f"    Eulerian EC: {time.time() - t0:.1f}s")
            print(f"    NOTE: Using Eulerian NEC margin for convergence at {N}^3")

        # Extract convergence quantities
        q_min = compute_convergence_quantity(nec_margin, "min_margin")
        q_l2 = compute_convergence_quantity(nec_margin, "l2_violation")
        q_int = compute_convergence_quantity(
            nec_margin, "integrated_violation", cell_volume=cell_vol
        )

        quantities["min_margin_nec"].append(q_min)
        quantities["l2_violation_nec"].append(q_l2)
        quantities["integrated_violation_nec"].append(q_int)
        grid_sizes.append(N)

        print(f"    min_margin_nec: {q_min:.6e}")
        print(f"    l2_violation_nec: {q_l2:.6e}")
        print(f"    integrated_violation_nec: {q_int:.6e}")

    # -----------------------------------------------------------------------
    # Richardson extrapolation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Richardson Extrapolation Results")
    print("=" * 60)

    convergence_data: dict = {
        "metric": args.metric,
        "resolutions": grid_sizes,
    }

    for qname, values in quantities.items():
        if len(values) < 3:
            print(f"\n  {qname}: Not enough resolutions for extrapolation ({len(values)} < 3).")
            convergence_data[qname] = {"values": values, "error": "insufficient_resolutions"}
            continue

        result = richardson_extrapolation(values, grid_sizes)
        convergence_data[qname] = {
            "values": values,
            "extrapolated_value": result["extrapolated_value"],
            "observed_order": result["observed_order"],
            "error_estimate": result["error_estimate"],
            "converged": result["converged"],
        }

        print(f"\n  {qname}:")
        for i, (N, v) in enumerate(zip(grid_sizes, values)):
            print(f"    N={N:>4d}: {v:.6e}")
        print(f"    Extrapolated: {result['extrapolated_value']:.6e}")
        print(f"    Observed order p: {result['observed_order']:.2f}")
        print(f"    Error estimate: {result['error_estimate']:.6e}")
        print(f"    Converged: {result['converged']}")

    # Save convergence data
    output_path = os.path.join(results_dir, "convergence_data.json")
    with open(output_path, "w") as f:
        json.dump(convergence_data, f, indent=2)
    print(f"\nSaved: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
