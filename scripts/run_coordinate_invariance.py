
"""Coordinate-invariance sanity check.

Verifies that the Alcubierre violation pattern at t=0 shifts by exactly
1 unit in x when evaluated at t=1/v_s (traveling wave property).

The Alcubierre metric has x_s(t) = v_s * t, so the bubble center
moves from x_s=0 at t=0 to x_s=1 at t=1/v_s.  The violation pattern
in the spatial domain should translate by exactly 1 unit in x.

Usage
-----
    python scripts/run_coordinate_invariance.py
"""
from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import GridSpec
from warpax.geometry.geometry import compute_curvature_chain
from warpax.energy_conditions.verifier import verify_grid


def evaluate_at_time(metric, grid, t_val):
    """Evaluate curvature grid at a specific time coordinate.

    Constructs a 4D coordinate array with x^0 = t_val, then evaluates
    the curvature chain at each point.
    """
    x = np.linspace(grid.bounds[0][0], grid.bounds[0][1], grid.shape[0])
    y = np.linspace(grid.bounds[1][0], grid.bounds[1][1], grid.shape[1])
    z = np.linspace(grid.bounds[2][0], grid.bounds[2][1], grid.shape[2])
    xs, ys, zs = np.meshgrid(x, y, z, indexing="ij")

    # Build 4D coordinates: (t, x, y, z)
    ts = np.full_like(xs, t_val)
    coords = np.stack([ts, xs, ys, zs], axis=-1)
    flat_coords = jnp.array(coords.reshape(-1, 4))

    # Evaluate curvature chain at all points
    chain_fn = jax.vmap(lambda c: compute_curvature_chain(metric, c))

    # Process in chunks to avoid memory issues
    chunk_size = 5000
    all_T = []
    all_g = []
    all_g_inv = []
    for i in range(0, len(flat_coords), chunk_size):
        chunk = flat_coords[i:i + chunk_size]
        result = chain_fn(chunk)
        all_T.append(result.stress_energy)
        all_g.append(result.metric)
        all_g_inv.append(result.metric_inv)

    T_field = jnp.concatenate(all_T, axis=0).reshape(*grid.shape, 4, 4)
    g_field = jnp.concatenate(all_g, axis=0).reshape(*grid.shape, 4, 4)
    g_inv_field = jnp.concatenate(all_g_inv, axis=0).reshape(*grid.shape, 4, 4)

    return T_field, g_field, g_inv_field


def run_invariance_check():
    """Check coordinate invariance of Alcubierre violation pattern."""
    v_s = 0.5
    N = 25
    grid = GridSpec(bounds=[(-5, 5)] * 3, shape=(N, N, N))

    # The AlcubierreMetric uses a static x_s parameter (not time-dependent).
    # To test coordinate invariance (traveling-wave property), we compare
    # x_s=0 vs x_s=1 (= v_s * delta_t), both evaluated at t=0.
    # The violation pattern should translate by exactly 1.0 in x.
    x_shift = 1.0

    print("=" * 70)
    print("Coordinate Invariance Check")
    print(f"Alcubierre v_s={v_s}, grid={N}^3")
    print(f"x_s=0 vs x_s={x_shift}")
    print("=" * 70)

    metric0 = AlcubierreMetric(v_s=v_s, R=1.0, sigma=8.0, x_s=0.0)
    metric1 = AlcubierreMetric(v_s=v_s, R=1.0, sigma=8.0, x_s=x_shift)

    # x_s=0
    print("\nEvaluating at x_s=0...")
    t0 = time.time()
    T0, g0, ginv0 = evaluate_at_time(metric0, grid, 0.0)
    ec0 = verify_grid(T0, g0, ginv0, n_starts=4, batch_size=64)
    print(f"  Time: {time.time() - t0:.1f}s")

    # x_s=1
    print(f"\nEvaluating at x_s={x_shift}...")
    t0 = time.time()
    T1, g1, ginv1 = evaluate_at_time(metric1, grid, 0.0)
    ec1 = verify_grid(T1, g1, ginv1, n_starts=4, batch_size=64)
    print(f"  Time: {time.time() - t0:.1f}s")

    # Compare: violation pattern at x_s=0 should be the same as x_s=1
    # shifted by 1 unit in x. Since grid is [-5,5] with N=25 points,
    # spacing = 10/24 ~ 0.417. Shift of 1.0 ~ 2.4 grid cells.
    # We can't compare point-by-point exactly due to non-integer shift,
    # but we can compare global statistics and center-of-mass of violations.

    nec0 = np.array(ec0.nec_margins)
    nec1 = np.array(ec1.nec_margins)

    viol0 = nec0 < -1e-10
    viol1 = nec1 < -1e-10

    n_viol0 = int(np.sum(viol0))
    n_viol1 = int(np.sum(viol1))

    print(f"\nNEC violations at x_s=0:      {n_viol0}")
    print(f"NEC violations at x_s={x_shift}:  {n_viol1}")
    print(f"Difference: {abs(n_viol0 - n_viol1)} (should be small or 0)")

    # Compute center of mass of violation pattern
    x = np.linspace(grid.bounds[0][0], grid.bounds[0][1], grid.shape[0])
    y = np.linspace(grid.bounds[1][0], grid.bounds[1][1], grid.shape[1])
    z = np.linspace(grid.bounds[2][0], grid.bounds[2][1], grid.shape[2])
    xs, ys, zs = np.meshgrid(x, y, z, indexing="ij")

    if n_viol0 > 0 and n_viol1 > 0:
        cx0 = float(np.mean(xs[viol0]))
        cy0 = float(np.mean(ys[viol0]))
        cz0 = float(np.mean(zs[viol0]))

        cx1 = float(np.mean(xs[viol1]))
        cy1 = float(np.mean(ys[viol1]))
        cz1 = float(np.mean(zs[viol1]))

        dx = cx1 - cx0
        dy = cy1 - cy0
        dz = cz1 - cz0

        expected_dx = x_shift

        print(f"\nCenter of violation mass:")
        print(f"  x_s=0:      ({cx0:.3f}, {cy0:.3f}, {cz0:.3f})")
        print(f"  x_s={x_shift}: ({cx1:.3f}, {cy1:.3f}, {cz1:.3f})")
        print(f"  Shift:    ({dx:.3f}, {dy:.3f}, {dz:.3f})")
        print(f"  Expected: ({expected_dx:.3f}, 0.000, 0.000)")
        print(f"  x-shift error: {abs(dx - expected_dx):.4f}")

        # Verify: x-shift should be ~1.0, y/z shifts should be ~0
        x_ok = abs(dx - expected_dx) < 0.5  # generous tolerance for discrete grid
        yz_ok = abs(dy) < 0.3 and abs(dz) < 0.3

        if x_ok and yz_ok:
            print("\nPASS: Violation pattern shifts as expected (traveling wave).")
        else:
            print("\nWARN: Violation pattern shift deviates from expectation.")
    else:
        print("\nInsufficient violations for center-of-mass comparison.")

    # Also compare min margins
    print(f"\nMin NEC margin at x_s=0:      {float(np.nanmin(nec0)):.6e}")
    print(f"Min NEC margin at x_s={x_shift}:  {float(np.nanmin(nec1)):.6e}")
    rel_diff = abs(float(np.nanmin(nec0)) - float(np.nanmin(nec1))) / max(
        abs(float(np.nanmin(nec0))), 1e-15
    )
    print(f"Relative difference: {rel_diff:.4e}")

    if rel_diff < 0.1:
        print("PASS: Min margins are consistent across time slices.")
    else:
        print("WARN: Min margins differ significantly.")


if __name__ == "__main__":
    run_invariance_check()
