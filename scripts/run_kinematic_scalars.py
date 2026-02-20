
"""Kinematic scalar field computation for warp drive metrics.

Computes expansion (theta), shear-squared, and vorticity-squared for the
Eulerian congruence across all 6 warp metrics at v_s=0.5.  Results are
cached as .npz files in results/.

Usage
-----
All metrics (default):
    python scripts/run_kinematic_scalars.py

Specific metrics:
    python scripts/run_kinematic_scalars.py --metrics alcubierre natario
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np

from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import GridSpec
from warpax.analysis import compute_kinematic_scalars_grid
from warpax.metrics import (
    LentzMetric,
    NatarioMetric,
    RodalMetric,
    VanDenBroeckMetric,
    WarpShellMetric,
)

# ---------------------------------------------------------------------------
# Metric configuration (same as run_analysis.py)
# ---------------------------------------------------------------------------

METRICS: dict[str, tuple[type, dict]] = {
    "alcubierre": (AlcubierreMetric, {"R": 1.0, "sigma": 8.0}),
    "rodal": (RodalMetric, {"R": 100.0, "sigma": 0.03}),
    "vdb": (VanDenBroeckMetric, {"R": 1.0, "sigma": 8.0, "R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0}),
    "natario": (NatarioMetric, {"R": 1.0, "sigma": 8.0}),
    "lentz": (LentzMetric, {"R": 100.0, "sigma": 8.0}),
    "warpshell": (WarpShellMetric, {"R_1": 0.5, "R_2": 1.0}),
}

GRID_STANDARD = GridSpec(bounds=[(-5, 5)] * 3, shape=(50, 50, 50))
GRID_LARGE_R = GridSpec(bounds=[(-300, 300)] * 3, shape=(50, 50, 50))

GRID_MAP: dict[str, GridSpec] = {
    "alcubierre": GRID_STANDARD,
    "rodal": GRID_LARGE_R,
    "vdb": GRID_STANDARD,
    "natario": GRID_STANDARD,
    "lentz": GRID_LARGE_R,
    "warpshell": GRID_STANDARD,
}

DEFAULT_V_S = 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute kinematic scalars for warp drive metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(METRICS.keys()),
        help=f"Metric names to analyze (default: all {len(METRICS)}).",
    )
    parser.add_argument(
        "--v-s",
        type=float,
        default=DEFAULT_V_S,
        help=f"Warp velocity (default: {DEFAULT_V_S}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for kinematic scalar computation (default: 256).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for output (default: results).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    v_s = args.v_s

    print("=" * 60)
    print(f"Kinematic scalars: v_s = {v_s}")
    print("=" * 60)

    for name in args.metrics:
        if name not in METRICS:
            print(f"  WARNING: Unknown metric '{name}', skipping.")
            continue

        metric_class, default_params = METRICS[name]
        params = dict(default_params)
        params["v_s"] = v_s
        metric = metric_class(**params)
        grid_spec = GRID_MAP[name]

        cache_path = os.path.join(results_dir, f"{name}_kinematic_vs{v_s}.npz")
        if os.path.exists(cache_path):
            print(f"\n--- {name}: [CACHED] {cache_path} ---")
            data = np.load(cache_path)
            theta = data["theta"]
        else:
            print(f"\n--- {name} (v_s={v_s}) ---")
            print(f"  Computing kinematic scalars (batch_size={args.batch_size})...")
            t0 = time.time()
            theta_grid, sigma_sq_grid, omega_sq_grid = compute_kinematic_scalars_grid(
                metric, grid_spec, batch_size=args.batch_size
            )
            elapsed = time.time() - t0
            print(f"    Kinematic scalars: {elapsed:.1f}s")

            theta = np.asarray(theta_grid)
            sigma_sq = np.asarray(sigma_sq_grid)
            omega_sq = np.asarray(omega_sq_grid)

            np.savez(
                cache_path,
                theta=theta,
                sigma_sq=sigma_sq,
                omega_sq=omega_sq,
                grid_bounds=np.array(grid_spec.bounds),
                grid_shape=np.array(grid_spec.shape),
            )
            print(f"  Saved: {cache_path}")

        # Summary
        theta_min = float(np.nanmin(theta))
        theta_max = float(np.nanmax(theta))
        print(f"  theta (expansion): min={theta_min:.4e}, max={theta_max:.4e}")

        # Natario should have theta ~ 0 (zero-expansion by construction)
        if name == "natario":
            if abs(theta_min) < 1e-6 and abs(theta_max) < 1e-6:
                print(f"  [OK] Natario: theta ~ 0 (zero-expansion confirmed)")
            else:
                print(
                    f"  [WARN] Natario: theta not near zero! "
                    f"min={theta_min:.4e}, max={theta_max:.4e}"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
