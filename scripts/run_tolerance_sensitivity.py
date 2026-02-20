
"""Classifier tolerance sensitivity analysis.

Varies the classification tolerance `tol` in {1e-12, 1e-10, 1e-8, 1e-6}
and the imaginary relative tolerance `imag_rtol` in {1e-4, 3e-3, 1e-2}.
Reports % Type I and violation counts for each combination.

Usage
-----
    python scripts/run_tolerance_sensitivity.py
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
from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.energy_conditions.classification import classify_hawking_ellis
from warpax.energy_conditions.eigenvalue_checks import check_all


def run_sensitivity():
    """Run tolerance sensitivity across metrics."""
    tol_values = [1e-12, 1e-10, 1e-8, 1e-6]
    imag_rtol_values = [1e-4, 3e-3, 1e-2]

    metrics_config = {
        "alcubierre": {
            "cls": AlcubierreMetric,
            "params": {"v_s": 0.5, "R": 1.0, "sigma": 8.0},
            "grid": GridSpec(bounds=[(-5, 5)] * 3, shape=(25, 25, 25)),
        },
        "rodal": {
            "cls": RodalMetric,
            "params": {"v_s": 0.5, "R": 100.0, "sigma": 0.03},
            "grid": GridSpec(bounds=[(-300, 300)] * 3, shape=(25, 25, 25)),
        },
    }

    print("=" * 70)
    print("Classifier Tolerance Sensitivity")
    print("=" * 70)

    for name, cfg in metrics_config.items():
        metric = cfg["cls"](**cfg["params"])
        grid = cfg["grid"]
        n_total = int(np.prod(grid.shape))

        print(f"\n{'='*50}")
        print(f"Metric: {name} ({grid.shape[0]}^3 = {n_total} points)")
        print(f"{'='*50}")

        # Compute curvature once
        curv = evaluate_curvature_grid(metric, grid, batch_size=256)
        flat_T = curv.stress_energy.reshape(-1, 4, 4)
        flat_g = curv.metric.reshape(-1, 4, 4)
        flat_g_inv = curv.metric_inv.reshape(-1, 4, 4)
        flat_T_mixed = jax.vmap(jnp.matmul)(flat_g_inv, flat_T)

        print(f"\n{'tol':>10s} {'imag_rtol':>12s} {'%TypeI':>8s} {'%TypeII':>8s} "
              f"{'%TypeIV':>8s} {'NEC_viol':>10s} {'WEC_viol':>10s}")
        print("-" * 72)

        for tol in tol_values:
            for imag_rtol in imag_rtol_values:
                # Classify with custom tolerances
                # We need to pass custom tol/imag_rtol to classify_hawking_ellis
                # The function signature uses defaults; we'll call it point-by-point
                # with overridden module-level constants

                # classify_hawking_ellis takes (T_mixed, g_ab, tol, imag_rtol)
                cls_fn = lambda T, g: classify_hawking_ellis(
                    T, g, tol=tol, imag_rtol=imag_rtol
                )
                cls_results = jax.vmap(cls_fn)(flat_T_mixed, flat_g)

                he_types = np.array(cls_results.he_type)
                n_type_i = int(np.sum(he_types == 1))
                n_type_ii = int(np.sum(he_types == 2))
                n_type_iv = int(np.sum(he_types == 4))
                pct_i = 100 * n_type_i / n_total
                pct_ii = 100 * n_type_ii / n_total
                pct_iv = 100 * n_type_iv / n_total

                # Violation counts for Type I points only (algebraic truth)
                rho = cls_results.rho
                pressures = cls_results.pressures
                nec_m, wec_m, sec_m, dec_m = jax.vmap(check_all)(rho, pressures)

                # Only count violations at Type I points
                type_i_mask = he_types == 1
                nec_viol = int(np.sum(
                    (np.array(nec_m) < -1e-10) & type_i_mask
                ))
                wec_viol = int(np.sum(
                    (np.array(wec_m) < -1e-10) & type_i_mask
                ))

                print(f"{tol:>10.0e} {imag_rtol:>12.0e} {pct_i:>7.1f}% "
                      f"{pct_ii:>7.1f}% {pct_iv:>7.1f}% "
                      f"{nec_viol:>10d} {wec_viol:>10d}")


if __name__ == "__main__":
    run_sensitivity()
