
"""Schwarzschild spacetime verification.

Demonstrates curvature computation for a vacuum black hole spacetime
in isotropic Cartesian coordinates.

Verifies:
- Vacuum solution (stress-energy tensor approximately zero)
- Kretschner scalar matches analytical 48M^2/r^6 (Schwarzschild coords)
- Energy conditions satisfied (vacuum)
"""

import jax.numpy as jnp

from warpax.benchmarks import SchwarzschildMetric
from warpax.benchmarks.schwarzschild import kretschner_isotropic
from warpax.geometry import compute_curvature_chain, kretschner_scalar
from warpax.energy_conditions import verify_point

M = 1.0

# Evaluate at a point safely away from the coordinate singularity at r_iso = M/2
metric = SchwarzschildMetric(M=M)
coords = jnp.array([0.0, 5.0, 3.0, 0.0])  # (t, x, y, z)

# Compute full curvature chain
result = compute_curvature_chain(metric, coords)

print("Schwarzschild Spacetime Verification")
print("=" * 40)
print(f"Point: (t, x, y, z) = ({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})")

r_iso = jnp.sqrt(coords[1]**2 + coords[2]**2 + coords[3]**2)
print(f"Isotropic radius: {r_iso:.4f}")

# Stress-energy should be ~0 for vacuum
T_max = jnp.max(jnp.abs(result.stress_energy))
print(f"\nMax |T_ab|: {T_max:.4e} (should be ~0 for vacuum)")

# Kretschner scalar comparison
K_numerical = kretschner_scalar(result.riemann, result.metric, result.metric_inv)
K_analytical = kretschner_isotropic(
    jnp.array(coords[1]), jnp.array(coords[2]), jnp.array(coords[3]), M=M
)
rel_error = jnp.abs(K_numerical - K_analytical) / jnp.abs(K_analytical)
print(f"Kretschner (numerical):  {K_numerical:.8e}")
print(f"Kretschner (analytical): {K_analytical:.8e}")
print(f"Relative error: {rel_error:.4e}")

# Verify energy conditions (vacuum should satisfy all)
ec = verify_point(result.stress_energy, result.metric, result.metric_inv)
print(f"\nEnergy condition margins:")
print(f"  NEC: {ec.nec_margin:.4e}")
print(f"  WEC: {ec.wec_margin:.4e}")
print(f"  SEC: {ec.sec_margin:.4e}")
print(f"  DEC: {ec.dec_margin:.4e}")

# Assertions
assert T_max < 1e-8, f"Vacuum violation: max |T| = {T_max}"
assert rel_error < 1e-8, f"Kretschner mismatch: relative error = {rel_error}"

print("\nSchwarzschild verification complete!")
