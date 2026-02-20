
"""Full metric analysis sweep: Eulerian vs robust EC comparison.

Runs compare_eulerian_vs_robust for all 6 warp metrics across 4 warp
velocities (v_s = 0.1, 0.5, 0.9, 0.99), plus Schwarzschild as a
velocity-independent baseline.  Results are cached as .npz files in
results/ and a comparison table is saved to results/comparison_table.json.

Usage
-----
Full sweep (default):
    python scripts/run_analysis.py

Quick validation (Alcubierre only at v_s=0.5):
    python scripts/run_analysis.py --phase 1

Core results (all metrics at v_s=0.5 + Schwarzschild):
    python scripts/run_analysis.py --phase 2

Specific metrics/velocities:
    python scripts/run_analysis.py --metrics alcubierre lentz --velocities 0.1 0.5
"""
from __future__ import annotations

import argparse
import os
import time

# Non-interactive backend (before any other matplotlib import)
import matplotlib
matplotlib.use("Agg")

import numpy as np

from warpax.benchmarks import AlcubierreMetric, SchwarzschildMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.analysis import build_comparison_table, compare_eulerian_vs_robust
from warpax.energy_conditions.verifier import verify_grid
from warpax.metrics import (
    LentzMetric,
    NatarioMetric,
    RodalMetric,
    VanDenBroeckMetric,
    WarpShellMetric,
)

# ---------------------------------------------------------------------------
# Metric configuration
# ---------------------------------------------------------------------------

# Metric class + default parameter overrides (excluding v_s which varies)
METRICS: dict[str, tuple[type, dict]] = {
    "alcubierre": (AlcubierreMetric, {"R": 1.0, "sigma": 8.0}),
    "rodal": (RodalMetric, {"R": 100.0, "sigma": 0.03}),
    "vdb": (VanDenBroeckMetric, {"R": 1.0, "sigma": 8.0, "R_tilde": 1.0, "alpha_vdb": 0.5, "sigma_B": 8.0}),
    "natario": (NatarioMetric, {"R": 1.0, "sigma": 8.0}),
    "lentz": (LentzMetric, {"R": 100.0, "sigma": 8.0}),
    "warpshell": (WarpShellMetric, {"R_1": 0.5, "R_2": 1.0}),
    "schwarzschild": (SchwarzschildMetric, {"M": 1.0}),
}

# Warp metrics: these participate in the v_s sweep
WARP_METRICS = ["alcubierre", "rodal", "vdb", "natario", "lentz", "warpshell"]

# Velocity sweep values
V_S_VALUES = [0.1, 0.5, 0.9, 0.99]

# Grid specs per metric category
# Standard metrics (R ~ 1): spatial domain [-5, 5]^3
GRID_STANDARD = GridSpec(bounds=[(-5, 5)] * 3, shape=(50, 50, 50))
# Large-R metrics (Rodal R=100, Lentz R=100): domain [-300, 300]^3
GRID_LARGE_R = GridSpec(bounds=[(-300, 300)] * 3, shape=(50, 50, 50))
# Schwarzschild: domain [-20, 20]^3
GRID_SCHWARZSCHILD = GridSpec(bounds=[(-20, 20)] * 3, shape=(50, 50, 50))

GRID_MAP: dict[str, GridSpec] = {
    "alcubierre": GRID_STANDARD,
    "rodal": GRID_LARGE_R,
    "vdb": GRID_STANDARD,
    "natario": GRID_STANDARD,
    "lentz": GRID_LARGE_R,
    "warpshell": GRID_STANDARD,
    "schwarzschild": GRID_SCHWARZSCHILD,
}


# ---------------------------------------------------------------------------
# Core analysis helper
# ---------------------------------------------------------------------------


def analyze_single(
    name: str,
    metric,
    grid_spec: GridSpec,
    n_starts: int,
    batch_size: int,
    cache_path: str,
) -> None:
    """Run full Eulerian vs robust EC comparison for a single metric.

    Steps:
    1. Check cache: skip if results already exist.
    2. Evaluate curvature grid.
    3. Run compare_eulerian_vs_robust for per-point comparison.
    4. Run verify_grid (compute_eulerian=False) for worst_params/worst_observers.
    5. Save all data to .npz cache.
    6. Print summary.
    """
    if os.path.exists(cache_path):
        print(f"  [CACHED] {cache_path} exists, skipping.")
        return

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"  Computing curvature grid for {name}...")
    t0 = time.time()
    curv = evaluate_curvature_grid(metric, grid_spec, batch_size=256)
    t_curv = time.time() - t0
    print(f"    Curvature grid: {t_curv:.1f}s")

    # Run comparison (Eulerian + robust, separate)
    print(f"  Running Eulerian vs robust comparison (n_starts={n_starts})...")
    t0 = time.time()
    comparison = compare_eulerian_vs_robust(
        curv.stress_energy,
        curv.metric,
        curv.metric_inv,
        grid_spec.shape,
        n_starts=n_starts,
        batch_size=batch_size,
    )
    t_comp = time.time() - t0
    print(f"    Comparison: {t_comp:.1f}s")

    # Run verify_grid for worst_params and worst_observers
    print(f"  Running verify_grid for worst-observer data...")
    t0 = time.time()
    ec_grid = verify_grid(
        curv.stress_energy,
        curv.metric,
        curv.metric_inv,
        n_starts=n_starts,
        batch_size=batch_size,
        compute_eulerian=False,
    )
    t_vg = time.time() - t0
    print(f"    verify_grid: {t_vg:.1f}s")

    # Save all data
    save_dict: dict[str, np.ndarray] = {}
    for cond in ("nec", "wec", "sec", "dec"):
        save_dict[f"{cond}_eulerian"] = np.asarray(comparison.eulerian_margins[cond])
        save_dict[f"{cond}_robust"] = np.asarray(comparison.robust_margins[cond])
        save_dict[f"{cond}_missed"] = np.asarray(comparison.missed[cond])
        save_dict[f"{cond}_severity"] = np.asarray(comparison.severity[cond])
        if comparison.opt_margins[cond] is not None:
            save_dict[f"{cond}_opt"] = np.asarray(comparison.opt_margins[cond])
        save_dict[f"{cond}_conditional_miss_rate"] = np.array(
            comparison.conditional_miss_rate[cond]
        )

    # Worst-observer data from ECGridResult
    save_dict["worst_params"] = np.asarray(ec_grid.worst_params)
    save_dict["worst_observers"] = np.asarray(ec_grid.worst_observers)
    save_dict["he_types"] = np.asarray(ec_grid.he_types)

    # Classification statistics
    save_dict["n_type_i"] = np.array(ec_grid.n_type_i)
    save_dict["n_type_ii"] = np.array(ec_grid.n_type_ii)
    save_dict["n_type_iii"] = np.array(ec_grid.n_type_iii)
    save_dict["n_type_iv"] = np.array(ec_grid.n_type_iv)
    save_dict["max_imag_eigenvalue"] = np.array(ec_grid.max_imag_eigenvalue)

    # Optimizer convergence diagnostics
    for cond in ("nec", "wec", "sec", "dec"):
        conv_field = getattr(ec_grid, f"{cond}_opt_converged", None)
        nsteps_field = getattr(ec_grid, f"{cond}_opt_n_steps", None)
        if conv_field is not None:
            save_dict[f"{cond}_opt_converged"] = np.asarray(conv_field)
        if nsteps_field is not None:
            save_dict[f"{cond}_opt_n_steps"] = np.asarray(nsteps_field)

    # Grid metadata
    save_dict["grid_bounds"] = np.array(grid_spec.bounds)
    save_dict["grid_shape"] = np.array(grid_spec.shape)

    np.savez(cache_path, **save_dict)
    print(f"  Saved: {cache_path}")

    # Summary
    for cond in ("nec", "wec", "sec", "dec"):
        eul_min = float(np.nanmin(save_dict[f"{cond}_eulerian"]))
        rob_min = float(np.nanmin(save_dict[f"{cond}_robust"]))
        pct_m = comparison.pct_missed[cond]
        print(
            f"    {cond.upper()}: Eulerian min={eul_min:.4e}, "
            f"Robust min={rob_min:.4e}, Missed={pct_m:.2f}%"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run Eulerian vs robust EC analysis across warp metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Subset of metric names to analyze (default: all 6 warp + schwarzschild).",
    )
    parser.add_argument(
        "--velocities",
        nargs="+",
        type=float,
        default=None,
        help="Subset of v_s values (default: 0.1 0.5 0.9 0.99).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for verify_grid (default: 64).",
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=8,
        help="Multi-start count for BFGS optimization (default: 8).",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help=(
            "Execution phase for incremental runs: "
            "1=quick (Alcubierre v_s=0.5), "
            "2=core (all metrics v_s=0.5 + Schwarzschild), "
            "3=full sweep (default)."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for cached results (default: results).",
    )
    args = parser.parse_args()

    # Determine which metrics and velocities to run based on --phase
    if args.metrics is not None:
        run_metrics = args.metrics
    elif args.phase == 1:
        run_metrics = ["alcubierre"]
    elif args.phase == 2:
        run_metrics = WARP_METRICS + ["schwarzschild"]
    else:
        run_metrics = WARP_METRICS + ["schwarzschild"]

    if args.velocities is not None:
        run_velocities = args.velocities
    elif args.phase == 1:
        run_velocities = [0.5]
    elif args.phase == 2:
        run_velocities = [0.5]
    else:
        run_velocities = V_S_VALUES

    n_starts = args.n_starts
    batch_size = args.batch_size
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Schwarzschild baseline: velocity-independent, outside v_s loop
    # -----------------------------------------------------------------------
    if "schwarzschild" in run_metrics:
        print("=" * 60)
        print("Schwarzschild baseline (v_s=0.0)")
        print("=" * 60)
        sch_class, sch_params = METRICS["schwarzschild"]
        sch_metric = sch_class(**sch_params)
        sch_grid = GRID_MAP["schwarzschild"]
        analyze_single(
            "schwarzschild",
            sch_metric,
            sch_grid,
            n_starts=n_starts,
            batch_size=batch_size,
            cache_path=os.path.join(results_dir, "schwarzschild_vs0.0.npz"),
        )

    # -----------------------------------------------------------------------
    # Main v_s sweep: warp metrics only
    # -----------------------------------------------------------------------
    warp_to_run = [m for m in run_metrics if m != "schwarzschild"]

    for v_s in run_velocities:
        print()
        print("=" * 60)
        print(f"v_s = {v_s}")
        print("=" * 60)

        for name in warp_to_run:
            if name not in METRICS:
                print(f"  WARNING: Unknown metric '{name}', skipping.")
                continue

            metric_class, default_params = METRICS[name]
            params = dict(default_params)
            params["v_s"] = v_s
            metric = metric_class(**params)

            grid_spec = GRID_MAP[name]
            cache_path = os.path.join(results_dir, f"{name}_vs{v_s}.npz")

            print(f"\n--- {name} (v_s={v_s}) ---")
            analyze_single(
                name, metric, grid_spec, n_starts, batch_size, cache_path
            )

    # -----------------------------------------------------------------------
    # Build comparison table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Building comparison table...")
    print("=" * 60)

    all_metrics_for_table = warp_to_run + (
        ["schwarzschild"] if "schwarzschild" in run_metrics else []
    )
    all_vs_for_table = sorted(set(run_velocities + ([0.0] if "schwarzschild" in run_metrics else [])))

    rows = build_comparison_table(results_dir, all_metrics_for_table, all_vs_for_table)
    print(f"  Table: {len(rows)} rows written to {results_dir}/comparison_table.json")

    # Print summary
    for row in rows:
        metric = row["metric"]
        v_s = row["v_s"]
        nec_missed = row.get("nec_pct_missed", 0.0)
        wec_missed = row.get("wec_pct_missed", 0.0)
        print(f"  {metric:>12s} v_s={v_s:.2f}: NEC missed={nec_missed:.2f}%, WEC missed={wec_missed:.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
