"""Uniform-grid sensitivity of one consistently defined NEC diagnostic.

Analyzes a single warp metric (default: Alcubierre at v_s=0.5) at three
grid resolutions (25^3, 49^3, 97^3). Reports the sampled minimum and two
discrete violation integrals, with their observed departures from the mean.
These statistics do not estimate a continuum error or convergence order.

Usage
-----
Default (Alcubierre):
    python scripts/run_convergence.py

Custom metric/resolutions:
    python scripts/run_convergence.py --metric rodal --resolutions 10 20 40

By default every rung uses the six Eulerian-frame null directions n +/- e_i.
The legacy --full-100 flag instead enables the full observer optimizer at
every rung; that mode is expensive and can take hours.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import math
import os
import time
from pathlib import Path

import jax
import numpy as np
from _json_io import dump_json

from warpax.analysis import compare_eulerian_vs_robust, compute_convergence_quantity
from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions.verifier import _eulerian_ec_point
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.metrics import RodalMetric, WarpShellMetric

# Metric and parameter configuration

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
    for b, n in zip(grid_spec.bounds, grid_spec.shape, strict=True):
        vol *= (b[1] - b[0]) / max(n - 1, 1)
    return vol


# Main


def main():
    parser = argparse.ArgumentParser(
        description="Uniform-grid NEC samples and observed spreads.",
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
        default=[25, 49, 97],
        help="Grid resolutions (default: 25 49 97, spacing ratio exactly 2).",
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
        help="Legacy flag: run full observer optimization at every resolution (expensive).",
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
    if len(resolutions) < 2 or len(set(resolutions)) != len(resolutions) or min(resolutions) < 2:
        parser.error("provide at least two distinct resolutions, each at least 2")
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
        if args.full_100:
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
            print("  Evaluating the six Eulerian-frame null directions...")
            t0 = time.time()
            n_points = int(np.prod(grid_spec.shape))
            flat_T = curv.stress_energy.reshape(n_points, 4, 4)
            flat_g = curv.metric.reshape(n_points, 4, 4)
            flat_g_inv = curv.metric_inv.reshape(n_points, 4, 4)
            eul = jax.vmap(_eulerian_ec_point)(flat_T, flat_g, flat_g_inv)
            nec_margin = np.asarray(eul["nec"]).reshape(grid_spec.shape)
            print(f"    Eulerian EC: {time.time() - t0:.1f}s")

        # Exclude the exact coordinate center, where the C-infinity
        # regularization guard (epsilon ~ 1e-12 inside r_s) dominates the
        # autodiff derivatives and produces a spurious O(1/epsilon) margin.
        # Only odd N samples the center; the mask is empty otherwise.
        axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, grid_spec.shape, strict=True)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        core = (X * X + Y * Y + Z * Z) < 1e-12
        if core.any():
            nec_margin = np.where(core, np.nan, nec_margin)
            print(f"    Excluded {int(core.sum())} regularized-core point(s) at r=0")

        # Extract convergence quantities
        q_min = compute_convergence_quantity(nec_margin, "min_margin")
        q_l2 = compute_convergence_quantity(nec_margin, "l2_violation", cell_volume=cell_vol)
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

    print("\n" + "=" * 60)
    print("Sampled values and observed grid spreads")
    print("=" * 60)

    source_paths = [
        Path(__file__).resolve(),
        Path(inspect.getfile(metric_class)).resolve(),
        Path(inspect.getfile(compute_convergence_quantity)).resolve(),
        Path(inspect.getfile(_eulerian_ec_point)).resolve(),
    ]
    if args.full_100:
        source_paths.append(Path(inspect.getfile(compare_eulerian_vs_robust)).resolve())
    repo_root = Path(__file__).resolve().parents[1]
    convergence_data: dict = {
        "summary_method": "observed_grid_spread",
        "metric": args.metric,
        "metric_parameters": cfg["params"],
        "resolutions": grid_sizes,
        "diagnostic": {
            "name": "robust_nec" if args.full_100 else "eulerian_six_direction_nec",
            "same_definition_at_every_resolution": True,
            "null_normalization": "k = n + s, |s| = 1, -g(n,k) = 1",
            "directions": "multistart sphere search" if args.full_100 else "s = +/- e_i, i = 1,2,3",
            "optimizer_enabled": args.full_100,
            "n_starts": args.n_starts if args.full_100 else None,
        },
        "grid": {
            "t": 0.0,
            "bounds": bounds,
            "endpoint_inclusive": True,
            "coordinate_core_exclusion_r_squared_lt": 1e-12,
            "volume_rule": "uniform coordinate cell volume times the sample sum",
        },
        "violation_roundoff_gate": "margin < -1e-10 * max(abs(negative finite margins))",
        "spread_definition": "max(abs(Q_i - mean(Q))) from unrounded samples",
        "interpretation": (
            "Observed grid sensitivity only; no fitted order, extrapolated minimum or continuum error bound."
        ),
        "source_sha256": {
            str(path.relative_to(repo_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
    }

    for qname, values in quantities.items():
        mean = math.fsum(values) / len(values)
        max_abs_deviation = max(abs(value - mean) for value in values)
        convergence_data[qname] = {
            "values": values,
            "mean": mean,
            "max_abs_deviation_from_mean": max_abs_deviation,
        }

        print(f"\n  {qname}:")
        for N, v in zip(grid_sizes, values, strict=True):
            print(f"    N={N:>4d}: {v:.6e}")
        print(f"    Mean: {mean:.6e}")
        print(f"    Max absolute departure from mean: {max_abs_deviation:.6e}")

    # Save convergence data
    output_path = os.path.join(results_dir, "convergence_data.json")
    dump_json(convergence_data, output_path)
    print(f"\nSaved: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
