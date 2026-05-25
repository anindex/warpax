"""Schwarzschild verification (isotropic coordinates).

Checks the curvature pipeline against a vacuum black hole solution:
``T_ab`` should vanish and the Kretschmann scalar should match the
analytical ``K = 48 M^2 / r_schw^6``.
"""

import jax.numpy as jnp

from warpax.benchmarks import SchwarzschildMetric
from warpax.benchmarks.schwarzschild import kretschmann_isotropic
from warpax.energy_conditions import verify_point
from warpax.geometry import compute_curvature_chain, kretschmann_scalar

M = 1.0

metric = SchwarzschildMetric(M=M)
coords = jnp.array([0.0, 5.0, 3.0, 0.0])

result = compute_curvature_chain(metric, coords)

print("Schwarzschild verification")
print("=" * 40)
print(f"Point: (t, x, y, z) = ({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})")

r_iso = jnp.sqrt(coords[1]**2 + coords[2]**2 + coords[3]**2)
print(f"Isotropic radius: {r_iso:.4f}")

T_max = jnp.max(jnp.abs(result.stress_energy))
print(f"\nMax |T_ab|: {T_max:.4e} (vacuum should be ~0)")

K_numerical = kretschmann_scalar(result.riemann, result.metric, result.metric_inv)
K_analytical = kretschmann_isotropic(
    jnp.array(coords[1]), jnp.array(coords[2]), jnp.array(coords[3]), M=M
)
rel_error = jnp.abs(K_numerical - K_analytical) / jnp.abs(K_analytical)
print(f"Kretschmann (numerical):  {K_numerical:.8e}")
print(f"Kretschmann (analytical): {K_analytical:.8e}")
print(f"Relative error:           {rel_error:.4e}")

ec = verify_point(result.stress_energy, result.metric, result.metric_inv)
print("\nEnergy-condition margins:")
print(f"  NEC: {ec.nec_margin:.4e}")
print(f"  WEC: {ec.wec_margin:.4e}")
print(f"  SEC: {ec.sec_margin:.4e}")
print(f"  DEC: {ec.dec_margin:.4e}")

assert T_max < 1e-8, f"Vacuum violation: max |T| = {T_max}"
assert rel_error < 1e-8, f"Kretschmann mismatch: relative error = {rel_error}"

print("\nVerification complete.")
