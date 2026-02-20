
"""Alcubierre warp drive analysis.

Demonstrates observer-robust energy condition verification on the
Alcubierre (1994) warp drive metric. Shows that NEC and WEC are violated,
matching the known analytical result for Eulerian energy density.

Key outputs:
- Curvature tensors and Kretschner scalar
- Energy condition margins (negative = violated)
- Comparison of Eulerian vs observer-robust NEC margins
"""

import jax.numpy as jnp

from warpax.benchmarks import AlcubierreMetric
from warpax.benchmarks.alcubierre import eulerian_energy_density
from warpax.geometry import compute_curvature_chain, kretschner_scalar
from warpax.energy_conditions import verify_point, compute_eulerian_ec

v_s = 0.5
R = 1.0
sigma = 8.0

metric = AlcubierreMetric(v_s=v_s, R=R, sigma=sigma)

# Evaluate at a point on the bubble wall (where violations are strongest)
coords = jnp.array([0.0, 1.0, 0.5, 0.0])  # (t, x, y, z)

print("Alcubierre Warp Drive Analysis")
print("=" * 40)
print(f"Parameters: v_s={v_s}, R={R}, sigma={sigma}")

r_s = jnp.sqrt(coords[1]**2 + coords[2]**2 + coords[3]**2)
print(f"Point: (t, x, y, z) = ({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})")
print(f"Distance from bubble center: r_s = {r_s:.4f}")

# Compute curvature
result = compute_curvature_chain(metric, coords)
K = kretschner_scalar(result.riemann, result.metric, result.metric_inv)

print(f"\nRicci scalar: {result.ricci_scalar:.6e}")
print(f"Kretschner scalar: {K:.6e}")
print(f"Max |T_ab|: {jnp.max(jnp.abs(result.stress_energy)):.6e}")

# Observer-robust energy condition verification
ec = verify_point(result.stress_energy, result.metric, result.metric_inv)

print(f"\nObserver-robust EC margins (negative = violated):")
print(f"  NEC: {ec.nec_margin:.6e}")
print(f"  WEC: {ec.wec_margin:.6e}")
print(f"  SEC: {ec.sec_margin:.6e}")
print(f"  DEC: {ec.dec_margin:.6e}")
print(f"  Worst observer 4-velocity: {ec.worst_observer}")
print(f"  Worst observer params (zeta, theta, phi): {ec.worst_params}")

# Eulerian-frame EC for comparison
eulerian_ec = compute_eulerian_ec(result.stress_energy, result.metric, result.metric_inv)
print(f"\nEulerian-frame EC margins:")
print(f"  NEC: {eulerian_ec['nec']:.6e}")
print(f"  WEC: {eulerian_ec['wec']:.6e}")

# Analytical Eulerian energy density for validation
rho_analytical = eulerian_energy_density(
    jnp.array(coords[1]), jnp.array(coords[2]), jnp.array(coords[3]),
    v_s=v_s, R=R, sigma=sigma,
)
print(f"\nAnalytical Eulerian energy density: {rho_analytical:.6e}")
print(f"(Negative confirms WEC/NEC violation)")

# Verify that violations are detected
assert ec.nec_margin < 0, "NEC should be violated for the Alcubierre metric!"
assert ec.wec_margin < 0, "WEC should be violated for the Alcubierre metric!"

print("\nNEC/WEC violation confirmed!")
print("The Alcubierre warp drive requires exotic matter (negative energy density).")
