
"""Minkowski spacetime sanity check.

Demonstrates warpax basics: metric evaluation, curvature computation, and
energy condition verification on flat spacetime where all tensors must vanish.

Verifies:
- All curvature tensors are zero
- All energy conditions are trivially satisfied
- Kretschner scalar is zero
"""

import jax.numpy as jnp

from warpax.benchmarks import MinkowskiMetric
from warpax.geometry import compute_curvature_chain, kretschner_scalar
from warpax.energy_conditions import verify_point

# Create Minkowski metric and evaluate at the origin
metric = MinkowskiMetric()
coords = jnp.array([0.0, 1.0, 1.0, 0.0])  # (t, x, y, z)

# Compute full curvature chain via autodiff
result = compute_curvature_chain(metric, coords)

print("Minkowski Spacetime Sanity Check")
print("=" * 40)
print(f"Metric tensor:\n{result.metric}")
print(f"Ricci scalar: {result.ricci_scalar:.2e}")
print(f"Max |Riemann|: {jnp.max(jnp.abs(result.riemann)):.2e}")
print(f"Max |Einstein|: {jnp.max(jnp.abs(result.einstein)):.2e}")
print(f"Max |stress-energy|: {jnp.max(jnp.abs(result.stress_energy)):.2e}")

# Kretschner scalar
K = kretschner_scalar(result.riemann, result.metric, result.metric_inv)
print(f"Kretschner scalar: {K:.2e}")

# Verify energy conditions
ec = verify_point(result.stress_energy, result.metric, result.metric_inv)
print(f"\nEnergy condition margins (positive = satisfied):")
print(f"  NEC: {ec.nec_margin:.2e}")
print(f"  WEC: {ec.wec_margin:.2e}")
print(f"  SEC: {ec.sec_margin:.2e}")
print(f"  DEC: {ec.dec_margin:.2e}")

# Verify ground truth
assert jnp.allclose(result.stress_energy, 0, atol=1e-10), "T_ab should be zero!"
assert jnp.abs(K) < 1e-10, "Kretschner should be zero!"
assert ec.nec_margin >= -1e-10, "NEC should be satisfied!"
assert ec.wec_margin >= -1e-10, "WEC should be satisfied!"

print("\nAll sanity checks passed!")
