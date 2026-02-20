
"""Regularization sensitivity analysis for the Rodal metric.

Runs the Rodal metric at v_s=0.5 with varying regularization parameter
eps^2 in {1e-24, 1e-18, 1e-12, 1e-6} to quantify the effect on
energy condition violation counts.

The Rodal metric uses sqrt(r^2 + eps^2) to regularize the angular
profile G(r) at r=0. This script verifies that the chosen default
eps^2 = 1e-24 does not significantly affect results.

Usage
-----
    python scripts/run_regularization_sensitivity.py
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpax.metrics.rodal import RodalMetric, _rodal_g_paper
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.energy_conditions.verifier import verify_grid


def run_sensitivity():
    """Run Rodal at multiple regularization values and compare."""
    eps_sq_values = [1e-24, 1e-18, 1e-12, 1e-6]
    v_s = 0.5
    R = 100.0
    sigma = 0.03
    grid = GridSpec(bounds=[(-300, 300)] * 3, shape=(25, 25, 25))

    print("=" * 70)
    print("Rodal Regularization Sensitivity Analysis")
    print(f"v_s={v_s}, R={R}, sigma={sigma}, grid={grid.shape}")
    print("=" * 70)

    results = []
    for eps_sq in eps_sq_values:
        print(f"\n--- eps^2 = {eps_sq:.0e} ---")

        # Monkey-patch the regularization constant
        import warpax.metrics.rodal as rodal_module
        original_fn = rodal_module._rodal_g_paper

        def patched_g_paper(r, R, sigma, _eps_sq=eps_sq):
            r_safe = jnp.sqrt(r**2 + _eps_sq)
            a = sigma * (r_safe - R)
            b = sigma * (r_safe + R)
            from warpax.metrics.rodal import _stable_logcosh
            log_ratio = _stable_logcosh(a) - _stable_logcosh(b)
            sinh_R_sigma = jnp.sinh(R * sigma)
            cosh_R_sigma = jnp.cosh(R * sigma)
            numerator = 2.0 * r_safe * sigma * sinh_R_sigma + cosh_R_sigma * log_ratio
            denominator = 2.0 * r_safe * sigma * sinh_R_sigma
            return numerator / jnp.maximum(denominator, 1e-30)

        rodal_module._rodal_g_paper = patched_g_paper

        try:
            metric = RodalMetric(v_s=v_s, R=R, sigma=sigma)
            t0 = time.time()
            curv = evaluate_curvature_grid(metric, grid, batch_size=256)
            ec = verify_grid(
                curv.stress_energy, curv.metric, curv.metric_inv,
                n_starts=8, batch_size=64,
            )
            elapsed = time.time() - t0

            nec_viol = int(jnp.sum(ec.nec_margins < -1e-10))
            wec_viol = int(jnp.sum(ec.wec_margins < -1e-10))
            sec_viol = int(jnp.sum(ec.sec_margins < -1e-10))
            dec_viol = int(jnp.sum(ec.dec_margins < -1e-10))
            n_total = int(np.prod(grid.shape))

            row = {
                "eps_sq": eps_sq,
                "nec_viol": nec_viol,
                "wec_viol": wec_viol,
                "sec_viol": sec_viol,
                "dec_viol": dec_viol,
                "n_total": n_total,
                "nec_min": float(jnp.nanmin(ec.nec_margins)),
                "wec_min": float(jnp.nanmin(ec.wec_margins)),
                "elapsed": elapsed,
            }
            results.append(row)

            print(f"  NEC violations: {nec_viol}/{n_total} ({100*nec_viol/n_total:.2f}%)")
            print(f"  WEC violations: {wec_viol}/{n_total} ({100*wec_viol/n_total:.2f}%)")
            print(f"  SEC violations: {sec_viol}/{n_total} ({100*sec_viol/n_total:.2f}%)")
            print(f"  DEC violations: {dec_viol}/{n_total} ({100*dec_viol/n_total:.2f}%)")
            print(f"  min NEC margin: {row['nec_min']:.6e}")
            print(f"  min WEC margin: {row['wec_min']:.6e}")
            print(f"  Time: {elapsed:.1f}s")

        finally:
            rodal_module._rodal_g_paper = original_fn

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'eps^2':>10s} {'NEC':>6s} {'WEC':>6s} {'SEC':>6s} {'DEC':>6s} {'min_NEC':>12s} {'min_WEC':>12s}")
    for r in results:
        print(f"{r['eps_sq']:>10.0e} {r['nec_viol']:>6d} {r['wec_viol']:>6d} "
              f"{r['sec_viol']:>6d} {r['dec_viol']:>6d} "
              f"{r['nec_min']:>12.4e} {r['wec_min']:>12.4e}")

    # Check stability
    ref = results[0]  # eps^2 = 1e-24 is our default
    stable = True
    for r in results[1:]:
        for cond in ["nec_viol", "wec_viol", "sec_viol", "dec_viol"]:
            if r[cond] != ref[cond]:
                print(f"\nWARNING: {cond} changed from {ref[cond]} to {r[cond]} "
                      f"at eps^2={r['eps_sq']:.0e}")
                stable = False

    if stable:
        print("\nAll violation counts are STABLE across regularization values.")
    else:
        print("\nSome violation counts CHANGED review sensitivity.")


if __name__ == "__main__":
    run_sensitivity()
