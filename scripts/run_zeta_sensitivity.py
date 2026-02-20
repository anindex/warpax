
"""Rapidity-cap sensitivity experiment for the Alcubierre metric.

Measures how the minimum energy-condition margins (Eulerian and robust)
vary with the maximum rapidity zeta_max used in the observer-space
optimization.  A larger zeta_max explores more highly boosted observers,
potentially uncovering deeper violations that the Eulerian frame misses.

Metric configuration
--------------------
  Alcubierre warp drive at v_s = 0.5, R = 1.0, sigma = 8.0
  Grid: 25^3 on (-5, 5)^3

Sweep
-----
  zeta_max in {1, 3, 5, 7}

Outputs
-------
  - Console table of WEC / NEC margins (Eulerian + robust) and missed fractions
  - results/zeta_sensitivity.npz with full numerical data

Usage
-----
    python scripts/run_zeta_sensitivity.py
"""
from __future__ import annotations

import os
import time

# Non-interactive backend (before any other matplotlib import)
import matplotlib
matplotlib.use("Agg")

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.analysis import compare_eulerian_vs_robust

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

V_S = 0.5
R = 1.0
SIGMA = 8.0
GRID_SHAPE = (25, 25, 25)
BOUNDS = [(-5.0, 5.0)] * 3
ZETA_MAX_VALUES = [1.0, 3.0, 5.0, 7.0]
N_STARTS = 8
BATCH_SIZE = 64
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "zeta_sensitivity.npz")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("Rapidity-cap (zeta_max) sensitivity experiment")
    print(f"  Metric  : Alcubierre  v_s={V_S}, R={R}, sigma={SIGMA}")
    print(f"  Grid    : {GRID_SHAPE} on {BOUNDS[0]}")
    print(f"  n_starts: {N_STARTS}")
    print(f"  zeta_max: {ZETA_MAX_VALUES}")
    print("=" * 72)

    # --- Step 1: build metric and curvature grid (shared across all zeta_max) ---
    metric = AlcubierreMetric(v_s=V_S, R=R, sigma=SIGMA)
    grid_spec = GridSpec(bounds=BOUNDS, shape=GRID_SHAPE)

    print("\nComputing curvature grid (shared for all zeta_max values)...")
    t0 = time.time()
    curv = evaluate_curvature_grid(metric, grid_spec, batch_size=256)
    t_curv = time.time() - t0
    print(f"  Curvature grid computed in {t_curv:.1f}s")

    T_field = curv.stress_energy   # (*grid_shape, 4, 4)
    g_field = curv.metric          # (*grid_shape, 4, 4)
    g_inv_field = curv.metric_inv  # (*grid_shape, 4, 4)

    # --- Step 2: sweep over zeta_max values ---
    # Storage for results
    results = {
        "zeta_max_values": np.array(ZETA_MAX_VALUES),
        "v_s": V_S,
        "R": R,
        "sigma": SIGMA,
        "grid_shape": np.array(GRID_SHAPE),
    }

    wec_eul_min_list = []
    wec_rob_min_list = []
    wec_alg_min_list = []
    wec_opt_min_list = []
    wec_missed_pct_list = []
    nec_eul_min_list = []
    nec_rob_min_list = []
    nec_missed_pct_list = []
    n_type_i_list = []
    n_type_iv_list = []
    max_imag_list = []

    for zeta_max in ZETA_MAX_VALUES:
        print(f"\n--- zeta_max = {zeta_max} ---")
        t0 = time.time()

        comparison = compare_eulerian_vs_robust(
            T_field,
            g_field,
            g_inv_field,
            grid_shape=GRID_SHAPE,
            n_starts=N_STARTS,
            zeta_max=zeta_max,
            batch_size=BATCH_SIZE,
            key=jax.random.PRNGKey(42),
        )

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        # Extract WEC results
        wec_eul = float(jnp.min(comparison.eulerian_margins["wec"]))
        wec_rob = float(jnp.min(comparison.robust_margins["wec"]))
        wec_opt = float(jnp.min(comparison.opt_margins["wec"]))
        wec_miss = comparison.pct_missed["wec"]

        # Algebraic min: min of merged margin over Type-I points only
        # (eigenvalue margins are cap-independent at Type-I points)
        is_type_i = comparison.he_types == 1.0
        wec_alg = float(jnp.min(jnp.where(
            is_type_i, comparison.robust_margins["wec"], jnp.inf
        )))

        # Extract NEC results
        nec_eul = float(jnp.min(comparison.eulerian_margins["nec"]))
        nec_rob = float(jnp.min(comparison.robust_margins["nec"]))
        nec_miss = comparison.pct_missed["nec"]

        cls_stats = comparison.classification_stats

        wec_eul_min_list.append(wec_eul)
        wec_rob_min_list.append(wec_rob)
        wec_alg_min_list.append(wec_alg)
        wec_opt_min_list.append(wec_opt)
        wec_missed_pct_list.append(wec_miss)
        nec_eul_min_list.append(nec_eul)
        nec_rob_min_list.append(nec_rob)
        nec_missed_pct_list.append(nec_miss)
        n_type_i_list.append(cls_stats["n_type_i"])
        n_type_iv_list.append(cls_stats["n_type_iv"])
        max_imag_list.append(cls_stats["max_imag_eigenvalue"])

        print(f"  WEC: Eulerian min={wec_eul:.6e}, Alg.(TypeI) min={wec_alg:.6e}, "
              f"Capped min={wec_opt:.6e}, Missed={wec_miss:.2f}%")
        print(f"  NEC: Eulerian min={nec_eul:.6e}, Robust min={nec_rob:.6e}, "
              f"Missed={nec_miss:.2f}%")
        print(f"  Type I: {cls_stats['n_type_i']}, Type IV: {cls_stats['n_type_iv']}, "
              f"max |Im λ|: {cls_stats['max_imag_eigenvalue']:.2e}")

    # Convert to arrays
    wec_eul_min = np.array(wec_eul_min_list)
    wec_rob_min = np.array(wec_rob_min_list)
    wec_alg_min = np.array(wec_alg_min_list)
    wec_opt_min = np.array(wec_opt_min_list)
    wec_missed_pct = np.array(wec_missed_pct_list)
    nec_eul_min = np.array(nec_eul_min_list)
    nec_rob_min = np.array(nec_rob_min_list)
    nec_missed_pct = np.array(nec_missed_pct_list)

    # --- Step 3: save results ---
    results.update({
        "wec_eulerian_min": wec_eul_min,
        "wec_robust_min": wec_rob_min,
        "wec_algebraic_min": wec_alg_min,
        "wec_opt_min": wec_opt_min,
        "wec_missed_pct": wec_missed_pct,
        "nec_eulerian_min": nec_eul_min,
        "nec_robust_min": nec_rob_min,
        "nec_missed_pct": nec_missed_pct,
        "n_type_i": np.array(n_type_i_list),
        "n_type_iv": np.array(n_type_iv_list),
        "max_imag_eigenvalue": np.array(max_imag_list),
    })
    np.savez(OUTPUT_PATH, **results)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # --- Step 4: print summary table ---
    print("\n" + "=" * 72)
    print("SUMMARY TABLE: zeta_max sensitivity (Alcubierre, v_s=0.5)")
    print("=" * 72)

    header = (
        f"{'zeta_max':>8s} | "
        f"{'WEC Eul min':>14s}  {'WEC Alg min':>14s}  {'WEC Opt min':>14s}  {'WEC f_miss':>10s} | "
        f"{'NEC Eul min':>14s}  {'NEC Rob min':>14s}  {'NEC f_miss':>10s}"
    )
    print(header)
    print("-" * len(header))

    for i, zeta_max in enumerate(ZETA_MAX_VALUES):
        row = (
            f"{zeta_max:>8.1f} | "
            f"{wec_eul_min[i]:>14.6e}  {wec_alg_min[i]:>14.6e}  {wec_opt_min[i]:>14.6e}  {wec_missed_pct[i]:>9.2f}% | "
            f"{nec_eul_min[i]:>14.6e}  {nec_rob_min[i]:>14.6e}  {nec_missed_pct[i]:>9.2f}%"
        )
        print(row)

    print("-" * len(header))
    print("\nDone.")


if __name__ == "__main__":
    main()
