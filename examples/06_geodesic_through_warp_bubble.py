"""Geodesics through an Alcubierre warp bubble.

Demonstrates the geodesic integration pipeline:
1. Build timelike initial conditions for a test particle at rest.
2. Integrate the geodesic as the bubble sweeps past it.
3. Monitor 4-velocity norm conservation (should stay at -1).
4. Compute tidal eigenvalues along the trajectory, jitted over the saved
   points, and plot them with ``plot_tidal_evolution``.
5. Launch a null ray through the bubble and check that its tangent stays on
   the light cone (norm 0).

The particle starts at the leading bubble wall (x = R = 1) of a gentle
bubble (v_s = 0.1). Tidal forces spike while each wall passes over it and
vanish in the flat interior and exterior. ``timelike_ic`` returns the
future-directed root, which matters here because the bubble is not
symmetric under ``t -> -t``.

A null tangent has no intrinsic scale, so the affine parameter is pinned by
``null_ic_eulerian_normalized``: ``-g(k, n) = 1`` against the Eulerian normal
``n`` at the launch point, so ``lambda`` measures Eulerian proper time there.
"""

import os
import time

import matplotlib

matplotlib.use("Agg")

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from warpax.benchmarks import AlcubierreMetric
from warpax.geodesics import (
    integrate_geodesic,
    monitor_conservation,
    null_ic_eulerian_normalized,
    tidal_eigenvalues,
    timelike_ic,
)
from warpax.visualization import plot_tidal_evolution

metric = AlcubierreMetric(v_s=0.1, R=1.0, sigma=8.0)
print(f"Metric : Alcubierre  v_s = {metric.v_s}")
print(f"Bubble : R = {metric.R}, sigma = {metric.sigma}")

x0, v0 = timelike_ic(metric, jnp.array([0.0, 1.0, 0.0, 0.0]), jnp.zeros(3))

norm0 = jnp.einsum("ab,a,b", metric(x0), v0, v0)
print(f"\nInitial position  : {x0}")
print(f"Initial 4-velocity: {v0}")
print(f"Initial norm g_ab v^a v^b = {norm0:.6e}  (expected -1)")

tau_span = (0.0, 40.0)
num_pts = 500
print(f"\nIntegrating tau in {tau_span} with {num_pts} save points ...")

t0 = time.perf_counter()
sol = integrate_geodesic(
    metric,
    x0,
    v0,
    tau_span,
    num_points=num_pts,
    dt0=0.005,
    rtol=1e-10,
    atol=1e-10,
    max_steps=65536,
)
elapsed = time.perf_counter() - t0
ok = bool(sol.result == diffrax.RESULTS.successful)
print(f"Integration completed in {elapsed:.2f} s  (solver succeeded: {ok})")

norms_np = np.asarray(monitor_conservation(metric, sol))
max_drift = np.max(np.abs(norms_np + 1.0))
print("\n4-velocity norm conservation:")
print(f"  mean(g_ab v^a v^b) = {norms_np.mean():.10f}  (expected -1)")
print(f"  max |norm + 1|     = {max_drift:.2e}")
assert max_drift < 1e-4, f"Norm conservation violated: max_drift = {max_drift}"

print("\nComputing tidal eigenvalues at each saved point ...")
tidal_batch = jax.jit(jax.vmap(lambda x, v: tidal_eigenvalues(metric, x, v)))
t0 = time.perf_counter()
tidal_np = np.asarray(tidal_batch(sol.positions, sol.velocities))
print(f"Tidal eigenvalues computed in {time.perf_counter() - t0:.2f} s  shape = {tidal_np.shape}")

tau_np = np.asarray(sol.ts)
tidal_mag = np.max(np.abs(tidal_np), axis=1)
peak_idx = int(np.argmax(tidal_mag))
mid_idx = int(np.argmin(np.abs(tau_np - 10.0)))
print(f"\nPeak tidal force at tau = {tau_np[peak_idx]:.2f}")
print(f"  x-position at peak: {float(sol.positions[peak_idx, 1]):.3f}")
print(f"  eigenvalues: {tidal_np[peak_idx]}")
print(
    f"Max |eigenvalue| at tau = {tau_np[mid_idx]:.1f} (inside):  {tidal_mag[mid_idx]:.2e}  (expected ~0)"
)
print(f"Max |eigenvalue| at tau = {tau_np[-1]:.1f} (outside): {tidal_mag[-1]:.2e}  (expected ~0)")

output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "tidal_evolution_alcubierre.pdf")
print(f"\nSaving tidal evolution plot -> {save_path}")
plot_tidal_evolution(
    tidal_np,
    tau_np,
    title="Tidal eigenvalues - Alcubierre $v_s = 0.1$",
    save_path=save_path,
)

print(
    f"\nTrajectory x-range: {float(sol.positions[0, 1]):.3f} -> {float(sol.positions[-1, 1]):.3f}"
)
print(f"Coordinate time elapsed: {float(sol.positions[-1, 0]):.2f}")

print("\nNull ray through the bubble")
print("=" * 40)
xk0, k0 = null_ic_eulerian_normalized(
    metric, jnp.array([0.0, -2.0, 0.3, 0.0]), jnp.array([1.0, 0.0, 0.0])
)
print(f"Initial tangent: {k0}  (affine scale: -g(k, n) = 1 at launch)")
print(f"Initial norm g_ab k^a k^b = {jnp.einsum('ab,a,b', metric(xk0), k0, k0):.2e}  (expected 0)")
ray = integrate_geodesic(metric, xk0, k0, (0.0, 6.0), num_points=200, dt0=0.005, max_steps=65536)
null_drift = float(np.max(np.abs(np.asarray(monitor_conservation(metric, ray)))))
print(f"max |g_ab k^a k^b| along the ray: {null_drift:.2e}  (expected 0)")
print(f"Ray end point (t, x, y, z): {np.asarray(ray.positions[-1])}")
assert null_drift < 1e-6, f"Null tangent left the light cone: {null_drift}"

print("\nGeodesic and tidal analysis through the warp bubble complete.")
