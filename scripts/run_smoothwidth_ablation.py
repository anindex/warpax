
"""WarpShell smooth_width sensitivity ablation.

Sweeps smooth_width in [0.03, 0.06, 0.12, 0.24] for WarpShell at v_s=0.5
using a 25^3 grid. Demonstrates that qualitative conclusions (violation
fractions, Type I percentages) are robust to smoothing parameter choice.

Output: results/smoothwidth_ablation.json

Usage
-----
    python scripts/run_smoothwidth_ablation.py
    python scripts/run_smoothwidth_ablation.py --smooth-widths 0.03 0.06 0.12
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

from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.analysis import compare_eulerian_vs_robust
from warpax.metrics import WarpShellMetric

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SMOOTH_WIDTHS = [0.03, 0.06, 0.12, 0.24]
GRID_SPEC = GridSpec(bounds=[(-5, 5)] * 3, shape=(25, 25, 25))
V_S = 0.5
R_1 = 0.5
R_2 = 1.0
N_STARTS = 8
BATCH_SIZE = 64


def main():
    parser = argparse.ArgumentParser(
        description="WarpShell smooth_width sensitivity ablation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--smooth-widths",
        nargs="+",
        type=float,
        default=DEFAULT_SMOOTH_WIDTHS,
        help="smooth_width values to sweep (default: 0.03 0.06 0.12 0.24).",
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=N_STARTS,
        help="Multi-start count for BFGS optimization (default: 8).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for optimization (default: 64).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for output (default: results).",
    )
    args = parser.parse_args()

    smooth_widths = sorted(args.smooth_widths)
    n_starts = args.n_starts
    batch_size = args.batch_size
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    default_sw = 0.12 * (R_2 - R_1)

    metadata = {
        "metric": "warpshell",
        "v_s": V_S,
        "R_1": R_1,
        "R_2": R_2,
        "grid_shape": list(GRID_SPEC.shape),
        "bounds": [list(b) for b in GRID_SPEC.bounds],
        "n_starts": n_starts,
        "default_smooth_width": default_sw,
    }

    ablation_results = []

    for sw in smooth_widths:
        print("=" * 60)
        print(f"smooth_width = {sw}  ({sw / default_sw:.1f}x default)")
        print("=" * 60)

        # Build metric with explicit smooth_width
        metric = WarpShellMetric(R_1=R_1, R_2=R_2, v_s=V_S, smooth_width=sw)

        # Compute curvature grid (must recompute metric changes)
        print("  Computing curvature grid...")
        t0 = time.time()
        curv = evaluate_curvature_grid(metric, GRID_SPEC, batch_size=256)
        t_curv = time.time() - t0
        print(f"    Curvature grid: {t_curv:.1f}s")

        # Run comparison
        print(f"  Running comparison (n_starts={n_starts})...")
        t0 = time.time()
        comparison = compare_eulerian_vs_robust(
            curv.stress_energy,
            curv.metric,
            curv.metric_inv,
            GRID_SPEC.shape,
            n_starts=n_starts,
            batch_size=batch_size,
        )
        t_comp = time.time() - t0
        print(f"    Comparison: {t_comp:.1f}s")

        elapsed = t_curv + t_comp

        # Extract results
        cls_stats = comparison.classification_stats
        total_points = int(
            cls_stats["n_type_i"]
            + cls_stats["n_type_ii"]
            + cls_stats["n_type_iii"]
            + cls_stats["n_type_iv"]
        )
        type_i_frac = cls_stats["n_type_i"] / total_points if total_points > 0 else 0.0

        row = {
            "smooth_width": sw,
            "type_i_frac": round(type_i_frac, 4),
            "nec_pct_missed": round(comparison.pct_missed["nec"], 4),
            "wec_pct_missed": round(comparison.pct_missed["wec"], 4),
            "sec_pct_missed": round(comparison.pct_missed["sec"], 4),
            "dec_pct_missed": round(comparison.pct_missed["dec"], 4),
            "nec_pct_violated_robust": round(comparison.pct_violated_robust["nec"], 4),
            "wec_pct_violated_robust": round(comparison.pct_violated_robust["wec"], 4),
            "sec_pct_violated_robust": round(comparison.pct_violated_robust["sec"], 4),
            "dec_pct_violated_robust": round(comparison.pct_violated_robust["dec"], 4),
            "nec_min_robust": round(float(np.nanmin(np.asarray(comparison.robust_margins["nec"]))), 6),
            "wec_min_robust": round(float(np.nanmin(np.asarray(comparison.robust_margins["wec"]))), 6),
            "n_type_i": cls_stats["n_type_i"],
            "n_type_ii": cls_stats["n_type_ii"],
            "n_type_iii": cls_stats["n_type_iii"],
            "n_type_iv": cls_stats["n_type_iv"],
            "time_s": round(elapsed, 1),
        }
        ablation_results.append(row)

        # Print summary for this width
        print(f"    Type I fraction: {type_i_frac:.4f}")
        print(f"    NEC missed: {comparison.pct_missed['nec']:.2f}%")
        print(f"    WEC missed: {comparison.pct_missed['wec']:.2f}%")
        print(f"    DEC missed: {comparison.pct_missed['dec']:.2f}%")
        print(f"    NEC violated (robust): {comparison.pct_violated_robust['nec']:.2f}%")
        print(f"    WEC violated (robust): {comparison.pct_violated_robust['wec']:.2f}%")
        print(f"    Time: {elapsed:.1f}s")
        print()

    # Save results
    output = {"metadata": metadata, "results": ablation_results}
    out_path = os.path.join(results_dir, "smoothwidth_ablation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = (
        f"{'smooth_w':>9s}  {'Type I%':>7s}  {'NEC miss%':>9s}  "
        f"{'WEC miss%':>9s}  {'NEC viol%':>9s}  {'WEC viol%':>9s}  {'time':>6s}"
    )
    print(header)
    print("-" * len(header))
    for r in ablation_results:
        print(
            f"{r['smooth_width']:>9.4f}  "
            f"{r['type_i_frac'] * 100:>7.2f}  "
            f"{r['nec_pct_missed']:>9.2f}  "
            f"{r['wec_pct_missed']:>9.2f}  "
            f"{r['nec_pct_violated_robust']:>9.2f}  "
            f"{r['wec_pct_violated_robust']:>9.2f}  "
            f"{r['time_s']:>5.0f}s"
        )

    # Qualitative robustness check
    type_i_fracs = [r["type_i_frac"] for r in ablation_results]
    max_variation = max(type_i_fracs) - min(type_i_fracs)
    print(f"\nType I fraction range: [{min(type_i_fracs):.4f}, {max(type_i_fracs):.4f}]")
    print(f"Maximum variation: {max_variation:.4f}")
    if max_variation < 0.10:
        print("ROBUST: Type I fraction varies less than 10 percentage points across smooth_widths.")
    else:
        print("WARNING: large variation in Type I fraction across smooth_widths.")


if __name__ == "__main__":
    main()
