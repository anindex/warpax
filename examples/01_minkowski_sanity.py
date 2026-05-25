"""Minkowski spacetime sanity check.

Walks through the basic warpax pipeline: metric -> curvature chain ->
energy-condition verification. On flat spacetime every curvature tensor
must vanish and every energy condition must be (trivially) satisfied.
"""

import jax.numpy as jnp

from warpax.benchmarks import MinkowskiMetric
from warpax.energy_conditions import verify_point
from warpax.geometry import compute_curvature_chain, kretschmann_scalar

metric = MinkowskiMetric()
coords = jnp.array([0.0, 1.0, 1.0, 0.0])

result = compute_curvature_chain(metric, coords)

print("Minkowski sanity check")
print("=" * 40)
print(f"Metric tensor:\n{result.metric}")
print(f"Ricci scalar:        {result.ricci_scalar:.2e}")
print(f"Max |Riemann|:       {jnp.max(jnp.abs(result.riemann)):.2e}")
print(f"Max |Einstein|:      {jnp.max(jnp.abs(result.einstein)):.2e}")
print(f"Max |stress-energy|: {jnp.max(jnp.abs(result.stress_energy)):.2e}")

K = kretschmann_scalar(result.riemann, result.metric, result.metric_inv)
print(f"Kretschmann scalar:  {K:.2e}")

ec = verify_point(result.stress_energy, result.metric, result.metric_inv)
print("\nEnergy-condition margins (positive = satisfied):")
print(f"  NEC: {ec.nec_margin:.2e}")
print(f"  WEC: {ec.wec_margin:.2e}")
print(f"  SEC: {ec.sec_margin:.2e}")
print(f"  DEC: {ec.dec_margin:.2e}")

assert jnp.allclose(result.stress_energy, 0, atol=1e-10), "T_ab should vanish"
assert jnp.abs(K) < 1e-10, "Kretschmann scalar should vanish"
assert ec.nec_margin >= -1e-10, "NEC should be satisfied"
assert ec.wec_margin >= -1e-10, "WEC should be satisfied"

print("\nAll sanity checks passed.")
