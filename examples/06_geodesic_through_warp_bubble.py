
"""Example 06 - Timelike geodesic through an Alcubierre warp bubble.

Demonstrates the geodesic integration pipeline:
1. Build timelike initial conditions for a test particle.
2. Integrate the geodesic trajectory through the warp bubble.
3. Monitor 4-velocity norm conservation (should stay near -1).
4. Compute tidal eigenvalues along the trajectory.
5. Visualise tidal evolution with ``plot_tidal_evolution``.

The particle starts far ahead of the bubble and the bubble sweeps past it.
Tidal forces should spike at the bubble wall and vanish inside and outside.
"""
from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# warpax imports
from warpax.benchmarks import AlcubierreMetric
from warpax.geodesics import (
    integrate_geodesic,
    timelike_ic,
    monitor_conservation,
    tidal_eigenvalues,
)
from warpax.visualization import plot_tidal_evolution

# 1. Metric setup
# Alcubierre bubble with v_s = 0.1 (gentle) so tidal forces are moderate.
metric = AlcubierreMetric(v_s=0.1, R=1.0, sigma=8.0)
print(f"Metric : Alcubierre  v_s = {metric.v_s}")
print(f"Bubble : R = {metric.R}, sigma = {metric.sigma}")

# 2. Initial conditions
# Place the particle at x = 1.0 - right at the bubble wall (R = 1)
# with zero spatial velocity - a static observer.
x0_pos = jnp.array([0.0, 1.0, 0.0, 0.0])
v_spatial = jnp.array([0.0, 0.0, 0.0])  # initially at rest
x0, v0 = timelike_ic(metric, x0_pos, v_spatial)

# Ensure future-directed: v^t > 0  (the quadratic solver may return
# the past-directed root for g_00 < 0).
v0 = v0.at[0].set(jnp.abs(v0[0]))

norm0 = jnp.einsum("ab,a,b", metric(x0), v0, v0)
print(f"\nInitial position  : {x0}")
print(f"Initial 4-velocity: {v0}")
print(f"Initial norm g_ab v^a v^b = {norm0:.6e}  (should be -1)")

# 3. Geodesic integration
tau_span = (0.0, 40.0)
num_pts = 500
print(f"\nIntegrating τ ∈ {tau_span} with {num_pts} save points ...")

t0 = time.perf_counter()
sol = integrate_geodesic(
    metric, x0, v0, tau_span,
    num_points=num_pts,
    dt0=0.005,
    rtol=1e-10,
    atol=1e-10,
    max_steps=65536,
)
elapsed = time.perf_counter() - t0
print(f"Integration completed in {elapsed:.2f} s  (result code = {sol.result})")

# 4. Conservation check
norms = monitor_conservation(metric, sol)
norms_np = np.asarray(norms)
max_drift = np.max(np.abs(norms_np + 1.0))  # deviation from -1
print(f"\n4-velocity norm conservation:")
print(f"  mean(g_ab v^a v^b) = {norms_np.mean():.10f}")
print(f"  max |norm + 1|     = {max_drift:.2e}")
assert max_drift < 1e-4, f"Norm conservation violated: max_drift = {max_drift}"

# 5. Tidal eigenvalues along trajectory
print("\nComputing tidal eigenvalues at each saved point ...")
t0 = time.perf_counter()
# vmap tidal_eigenvalues over saved trajectory points
tidal_eigs = jax.vmap(
    lambda x, v: tidal_eigenvalues(metric, x, v)
)(sol.positions, sol.velocities)
tidal_time = time.perf_counter() - t0
print(f"Tidal eigenvalues computed in {tidal_time:.2f} s  shape = {tidal_eigs.shape}")

tidal_np = np.asarray(tidal_eigs)
tau_np = np.asarray(sol.ts)

# Summary statistics
peak_idx = np.argmax(np.sum(np.abs(tidal_np), axis=1))
print(f"\nPeak tidal force at τ = {tau_np[peak_idx]:.2f}")
print(f"  x-position at peak: {float(sol.positions[peak_idx, 1]):.3f}")
print(f"  eigenvalues: {tidal_np[peak_idx]}")

# 6. Visualise tidal evolution
os.makedirs("examples/output", exist_ok=True)
save_path = "examples/output/tidal_evolution_alcubierre.pdf"
print(f"\nSaving tidal evolution plot -> {save_path}")
plot_tidal_evolution(
    tidal_np,
    tau_np,
    title="Tidal eigenvalues - Alcubierre $v_s = 0.1$",
    save_path=save_path,
)

# 7. Trajectory summary
x_start = float(sol.positions[0, 1])
x_end = float(sol.positions[-1, 1])
print(f"\nTrajectory x-range: {x_start:.3f} -> {x_end:.3f}")
print(f"Coordinate time elapsed: {float(sol.positions[-1, 0]):.2f}")
print("\n✓ Example 06 complete - geodesic + tidal analysis through warp bubble.")
