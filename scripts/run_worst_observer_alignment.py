
"""Worst-observer alignment analysis at multiple velocities.

For the Rodal metric at v_s = [0.1, 0.5, 0.9, 0.99], computes the angle
between the BFGS worst-case observer direction and the eigenvector associated
with the minimum eigenvalue margin min(rho - |p_i|) at DEC-violation points.

Saves all results to results/alignment_rodal.npz with per-velocity arrays.

Usage
-----
    python scripts/run_worst_observer_alignment.py
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

from warpax.metrics import RodalMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.energy_conditions.verifier import verify_grid
from warpax.energy_conditions.classification import classify_hawking_ellis


# Primary metric configuration
R = 100.0
SIGMA = 0.03
GRID_BOUNDS = [(-300, 300)] * 3
GRID_SHAPE = (50, 50, 50)
VELOCITIES = [0.1, 0.5, 0.9, 0.99]


def _compute_alignment_angles(
    metric,
    grid: GridSpec,
    n_starts: int = 8,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute alignment angles between BFGS boost direction and eigenvector prediction.

    Parameters
    ----------
    metric : RodalMetric
        The metric to analyze.
    grid : GridSpec
        Grid specification.
    n_starts : int
        Multi-start count for optimization.
    batch_size : int
        Batch size for memory-safe processing.

    Returns
    -------
    np.ndarray
        Array of alignment angles in degrees at DEC-violation Type I points.
    """
    curv = evaluate_curvature_grid(metric, grid, batch_size=256)

    # Run full verification
    ec = verify_grid(
        curv.stress_energy, curv.metric, curv.metric_inv,
        n_starts=n_starts, batch_size=batch_size,
    )

    # Flatten
    flat_T = curv.stress_energy.reshape(-1, 4, 4)
    flat_g = curv.metric.reshape(-1, 4, 4)
    flat_g_inv = curv.metric_inv.reshape(-1, 4, 4)

    # Classification
    flat_T_mixed = jax.vmap(jnp.matmul)(flat_g_inv, flat_T)
    cls = jax.vmap(classify_hawking_ellis)(flat_T_mixed, flat_g)

    # Find DEC violation points (Type I only)
    dec_margins = np.array(ec.dec_margins.reshape(-1))
    he_types = np.array(ec.he_types.reshape(-1))
    worst_params = np.array(ec.worst_params.reshape(-1, 3))

    viol_mask = (dec_margins < -1e-10) & (he_types == 1)
    viol_indices = np.where(viol_mask)[0]

    if len(viol_indices) == 0:
        return np.array([])

    rho = np.array(cls.rho)
    pressures = np.array(cls.pressures)
    eigenvectors = np.array(cls.eigenvectors)

    angles = []
    for idx in viol_indices:
        p = pressures[idx]
        margins_i = rho[idx] - np.abs(p)
        worst_i = int(np.argmin(margins_i))

        evec = eigenvectors[idx, :, worst_i + 1]

        zeta, theta, phi = worst_params[idx]
        if zeta < 1e-6:
            continue

        boost_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])

        evec_spatial = evec[1:4]
        evec_norm = np.linalg.norm(evec_spatial)
        if evec_norm < 1e-10:
            continue
        evec_hat = evec_spatial / evec_norm

        cos_angle = np.clip(np.abs(np.dot(boost_dir, evec_hat)), 0, 1)
        angle_deg = np.degrees(np.arccos(cos_angle))
        angles.append(angle_deg)

    return np.array(angles) if angles else np.array([])


def run_alignment():
    """Analyze worst-observer alignment at multiple velocities."""
    grid = GridSpec(bounds=GRID_BOUNDS, shape=GRID_SHAPE)
    n_total = int(np.prod(grid.shape))

    print("=" * 70)
    print("Worst-Observer Alignment Analysis (Multi-Velocity)")
    print(f"Rodal R={R}, sigma={SIGMA}, grid={GRID_SHAPE}")
    print(f"Velocities: {VELOCITIES}")
    print("=" * 70)

    save_data = {"velocities": np.array(VELOCITIES)}

    for v_s in VELOCITIES:
        print(f"\n--- v_s = {v_s} ---")
        metric = RodalMetric(v_s=v_s, R=R, sigma=SIGMA)

        t0 = time.time()
        angles = _compute_alignment_angles(metric, grid, n_starts=8, batch_size=64)
        elapsed = time.time() - t0

        n_viol = len(angles)
        save_data[f"angles_vs{v_s}"] = angles
        save_data[f"n_violations_vs{v_s}"] = n_viol

        if n_viol > 0:
            median_angle = float(np.median(angles))
            save_data[f"median_angle_vs{v_s}"] = median_angle
            print(f"  DEC violations (Type I): {n_viol}/{n_total}")
            print(f"  Alignment angles ({n_viol} points):")
            print(f"    Mean:   {np.mean(angles):.1f} deg")
            print(f"    Median: {median_angle:.1f} deg")
            print(f"    < 10 deg: {100 * np.mean(angles < 10):.1f}%")
            print(f"    < 30 deg: {100 * np.mean(angles < 30):.1f}%")
            print(f"    < 45 deg: {100 * np.mean(angles < 45):.1f}%")
        else:
            save_data[f"median_angle_vs{v_s}"] = float("nan")
            print(f"  No DEC violations found at v_s={v_s}")

        print(f"  Time: {elapsed:.1f}s")

    # Save all results
    os.makedirs("results", exist_ok=True)
    out_path = "results/alignment_rodal.npz"
    np.savez(out_path, **save_data)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    run_alignment()
