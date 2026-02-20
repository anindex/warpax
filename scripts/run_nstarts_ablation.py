
"""N_starts ablation study: sensitivity of robust margins to restart count.

Sweeps N_starts in {1, 2, 4, 8, 16} for three representative metrics at
v_s = 0.5:
  - Alcubierre  (missed=0, severity-only)
  - Rodal       (largest missed%)
  - WarpShell   (small missed%)

The curvature grid is computed ONCE per metric (expensive), then
compare_eulerian_vs_robust is called with different n_starts values
(only the BFGS optimization re-runs).

Output: results/nstarts_ablation.json

Usage
-----
    python scripts/run_nstarts_ablation.py
    python scripts/run_nstarts_ablation.py --n-starts 1 2 4 8 16 32
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
jax.config.update("jax_enable_x64", True)

from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.analysis import compare_eulerian_vs_robust
from warpax.metrics import RodalMetric, WarpShellMetric

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METRICS: dict[str, tuple[type, dict, GridSpec]] = {
    "alcubierre": (
        AlcubierreMetric,
        {"R": 1.0, "sigma": 8.0, "v_s": 0.5},
        GridSpec(bounds=[(-5, 5)] * 3, shape=(50, 50, 50)),
    ),
    "rodal": (
        RodalMetric,
        {"R": 100.0, "sigma": 0.03, "v_s": 0.5},
        GridSpec(bounds=[(-300, 300)] * 3, shape=(50, 50, 50)),
    ),
    "warpshell": (
        WarpShellMetric,
        {"R_1": 0.5, "R_2": 1.0, "v_s": 0.5},
        GridSpec(bounds=[(-5, 5)] * 3, shape=(50, 50, 50)),
    ),
}

DEFAULT_N_STARTS = [1, 2, 4, 8, 16]


def main():
    parser = argparse.ArgumentParser(
        description="N_starts ablation study for observer optimization stability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n-starts",
        nargs="+",
        type=int,
        default=DEFAULT_N_STARTS,
        help="N_starts values to sweep (default: 1 2 4 8 16).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for verify_grid (default: 64).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for output (default: results).",
    )
    args = parser.parse_args()

    n_starts_values = sorted(args.n_starts)
    batch_size = args.batch_size
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    results: dict[str, dict] = {}

    for name, (metric_cls, params, grid_spec) in METRICS.items():
        print("=" * 60)
        print(f"Metric: {name} (v_s = {params.get('v_s', '?')})")
        print("=" * 60)

        # Build metric and evaluate curvature grid ONCE
        metric = metric_cls(**params)
        print("  Computing curvature grid (one-time cost)...")
        t0 = time.time()
        curv = evaluate_curvature_grid(metric, grid_spec, batch_size=256)
        t_curv = time.time() - t0
        print(f"  Curvature grid: {t_curv:.1f}s")

        metric_results: dict[str, list] = {
            "n_starts_values": n_starts_values,
            "min_wec_margin": [],
            "min_wec_algebraic": [],
            "min_wec_opt": [],
            "min_nec_margin": [],
            "pct_missed_wec": [],
            "pct_missed_dec": [],
            "time_s": [],
            "n_type_i": [],
            "n_type_iv": [],
            "max_imag_eigenvalue": [],
        }

        for ns in n_starts_values:
            print(f"\n  --- N_starts = {ns} ---")
            t0 = time.time()
            comparison = compare_eulerian_vs_robust(
                curv.stress_energy,
                curv.metric,
                curv.metric_inv,
                grid_spec.shape,
                n_starts=ns,
                batch_size=batch_size,
            )
            elapsed = time.time() - t0

            min_wec = float(np.nanmin(np.asarray(comparison.robust_margins["wec"])))
            min_nec = float(np.nanmin(np.asarray(comparison.robust_margins["nec"])))
            min_wec_opt = float(np.nanmin(np.asarray(comparison.opt_margins["wec"])))
            pct_wec = comparison.pct_missed["wec"]
            pct_dec = comparison.pct_missed["dec"]
            cls_stats = comparison.classification_stats

            # Algebraic min: min of merged margin over Type-I points only
            is_type_i = np.asarray(comparison.he_types) == 1.0
            min_wec_alg = float(np.nanmin(np.where(
                is_type_i, np.asarray(comparison.robust_margins["wec"]), np.inf
            )))

            metric_results["min_wec_margin"].append(min_wec)
            metric_results["min_wec_algebraic"].append(min_wec_alg)
            metric_results["min_wec_opt"].append(min_wec_opt)
            metric_results["min_nec_margin"].append(min_nec)
            metric_results["pct_missed_wec"].append(pct_wec)
            metric_results["pct_missed_dec"].append(pct_dec)
            metric_results["time_s"].append(round(elapsed, 1))
            metric_results["n_type_i"].append(cls_stats["n_type_i"])
            metric_results["n_type_iv"].append(cls_stats["n_type_iv"])
            metric_results["max_imag_eigenvalue"].append(cls_stats["max_imag_eigenvalue"])

            print(f"    min WEC margin (algebraic, Type I): {min_wec_alg:.6e}")
            print(f"    min WEC margin (optimizer): {min_wec_opt:.6e}")
            print(f"    min NEC margin:  {min_nec:.6e}")
            print(f"    % missed WEC:    {pct_wec:.2f}%")
            print(f"    % missed DEC:    {pct_dec:.2f}%")
            print(f"    Type I: {cls_stats['n_type_i']}, Type IV: {cls_stats['n_type_iv']}")
            print(f"    max |Im λ|:      {cls_stats['max_imag_eigenvalue']:.2e}")
            print(f"    Time:            {elapsed:.1f}s")

        results[name] = metric_results

    # Save results
    out_path = os.path.join(results_dir, "nstarts_ablation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = f"{'Metric':<14} {'N_starts':>8} {'min WEC':>12} {'min NEC':>12} {'%miss WEC':>10} {'%miss DEC':>10}"
    print(header)
    print("-" * len(header))
    for name, mr in results.items():
        for i, ns in enumerate(mr["n_starts_values"]):
            print(
                f"{name:<14} {ns:>8} "
                f"{mr['min_wec_margin'][i]:>12.4e} "
                f"{mr['min_nec_margin'][i]:>12.4e} "
                f"{mr['pct_missed_wec'][i]:>9.2f}% "
                f"{mr['pct_missed_dec'][i]:>9.2f}%"
            )
        print()


if __name__ == "__main__":
    main()
