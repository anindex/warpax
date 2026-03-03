"""Generate Van Den Broeck NEC/WEC and SEC/DEC comparison figures.

- VdB NEC comparison (NEC missed = 0.1% at v_s=0.5)
- VdB WEC comparison (WEC missed = 0.4% at v_s=0.5)
- VdB SEC comparison (SEC missed = 1.2%)
- VdB DEC comparison (DEC missed = 0.3%)

Usage:
    python scripts/generate_vdb_comparison_figures.py
"""
from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from warpax.metrics import VanDenBroeckMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.analysis import compare_eulerian_vs_robust
from warpax.visualization.comparison_plots import plot_comparison_panel

# VdB parameters matching run_analysis.py and paper Table 1
VDB_PARAMS = {"R": 1.0, "sigma": 8.0, "R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0}
GRID = GridSpec(bounds=[(-5, 5)] * 3, shape=(50, 50, 50))
V_S = 0.5

# Output directory
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("Van Den Broeck EC Comparison Figure Generator")
    print("=" * 70)

    metric = VanDenBroeckMetric(v_s=V_S, **VDB_PARAMS)
    print(f"\nMetric: {metric.name()}, v_s={V_S}")
    print(f"Grid: {GRID.shape}, bounds={GRID.bounds}")

    print("\nComputing curvature grid...")
    t0 = time.time()
    curv = evaluate_curvature_grid(metric, GRID, batch_size=256)
    t_curv = time.time() - t0
    print(f"  Curvature grid: {t_curv:.1f}s")

    print("\nRunning Eulerian vs robust comparison (n_starts=8)...")
    t0 = time.time()
    comparison = compare_eulerian_vs_robust(
        curv.stress_energy,
        curv.metric,
        curv.metric_inv,
        GRID.shape,
        n_starts=8,
        zeta_max=5.0,
        batch_size=64,
    )
    t_comp = time.time() - t0
    print(f"  Comparison: {t_comp:.1f}s")

    print("\nResults summary:")
    for cond in ("nec", "wec", "sec", "dec"):
        pct_v = comparison.pct_violated_robust[cond]
        pct_m = comparison.pct_missed[cond]
        cond_miss = comparison.conditional_miss_rate[cond]
        print(f"  {cond.upper()}: Total violated={pct_v:.1f}%, "
              f"Missed={pct_m:.1f}%, Conditional miss={cond_miss:.1f}%")

    grid_bounds = GRID.bounds

    for cond, label in [("nec", "NEC"), ("wec", "WEC"), ("sec", "SEC"), ("dec", "DEC")]:
        print(f"\nGenerating VdB {label} comparison figure...")
        path = os.path.join(FIGURES_DIR, f"vdb_{cond}_comparison.pdf")
        plot_comparison_panel(
            eulerian_margin=np.asarray(comparison.eulerian_margins[cond]),
            robust_margin=np.asarray(comparison.robust_margins[cond]),
            missed=np.asarray(comparison.missed[cond]),
            grid_bounds=grid_bounds,
            grid_shape=GRID.shape,
            title=rf"Van Den Broeck {label}: Eulerian vs Robust ($v_s = 0.5$, $50^3$ grid)",
            save_path=path,
        )
        print(f"  Saved: {path}")

    # Save cached results for reproducibility
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    cache_path = os.path.join(results_dir, f"vdb_vs{V_S}.npz")
    save_dict = {}
    for cond in ("nec", "wec", "sec", "dec"):
        save_dict[f"{cond}_eulerian"] = np.asarray(comparison.eulerian_margins[cond])
        save_dict[f"{cond}_robust"] = np.asarray(comparison.robust_margins[cond])
        save_dict[f"{cond}_missed"] = np.asarray(comparison.missed[cond])
    save_dict["grid_bounds"] = np.array(GRID.bounds)
    save_dict["grid_shape"] = np.array(GRID.shape)
    np.savez(cache_path, **save_dict)
    print(f"\n  Cached results: {cache_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
