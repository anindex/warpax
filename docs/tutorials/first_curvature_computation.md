# Your first curvature computation

Evaluate flat spacetime, where all curvature tensors vanish.

## Prerequisites

- Python 3.12+
- `warpax` installed: `pip install warpax`
- Runnable counterpart: `examples/01_minkowski_sanity.py`

## Evaluate the curvature

```python
import jax.numpy as jnp
from warpax.benchmarks import MinkowskiMetric
from warpax.geometry import compute_curvature_chain

metric = MinkowskiMetric()
coords = jnp.array([0.0, 0.0, 0.0, 0.0])

result = compute_curvature_chain(metric, coords)

print(f"Metric g_ab: {result.metric[0, 0]:+.3e}")
print(f"Riemann R^a_bcd max norm: {float(jnp.max(jnp.abs(result.riemann))):.3e}")
print(f"Ricci R_ab max norm: {float(jnp.max(jnp.abs(result.ricci))):.3e}")
print(f"Ricci scalar R: {float(result.ricci_scalar):+.3e}")
print(f"Einstein G_ab max norm: {float(jnp.max(jnp.abs(result.einstein))):.3e}")
print(f"Stress-energy T_ab max: {float(jnp.max(jnp.abs(result.stress_energy))):.3e}")
```

The first output is `-1`; the curvature and stress outputs are zero.

## Curved vacuum: Schwarzschild

For zero cosmological constant, vacuum has zero Ricci tensor, but its Riemann
tensor can be nonzero. Schwarzschild provides an analytical check:

```python
from warpax.benchmarks import SchwarzschildMetric
from warpax.geometry import kretschmann_scalar

M = 1.0
coords = jnp.array([0.0, 5.0, 3.0, 0.0])
result = compute_curvature_chain(SchwarzschildMetric(M=M), coords)
K = kretschmann_scalar(result.riemann, result.metric, result.metric_inv)

r_iso = jnp.linalg.norm(coords[1:])
r_areal = r_iso * (1 + M / (2 * r_iso)) ** 2
K_exact = 48 * M**2 / r_areal**6
print("Maximum Ricci component:", jnp.max(jnp.abs(result.ricci)))
print("Relative Kretschmann error:", jnp.abs(K / K_exact - 1))
assert jnp.max(jnp.abs(result.ricci)) < 1e-12
assert jnp.abs(K / K_exact - 1) < 1e-12
```

This implementation uses isotropic Cartesian coordinates. The radial coordinate
`r_iso` differs from the areal radius in $K=48M^2/r_{\rm areal}^6$.
Use `r_iso > M/2` for this exterior example; the isotropic chart is singular
at the horizon. Numerical vacuum residuals should be compared with a tolerance.
The full calculation is in `examples/02_schwarzschild_verification.py`.

## How it works

1. `MinkowskiMetric` returns $g_{ab}=\mathrm{diag}(-1,1,1,1)$ at every point.
   Its ADM lapse is 1, shift is zero, and spatial metric is Euclidean.
2. `compute_curvature_chain` applies `jax.jacfwd` at two differentiation
   stages: first on the metric to obtain the Christoffel symbols, then a
   nested `jax.jacfwd` on the Christoffel map for the Riemann tensor
   $R^\alpha{}_{\beta\gamma\delta}$; the Ricci and Einstein tensors follow
   by pure contraction, not autodiff.
3. All tensors are JAX arrays with jaxtyping shape annotations (e.g.
   `Float[Array, "4 4 4 4"]`).

## Next steps

- [Tensor conventions and units](../explanation/theory.md#tensor-conventions-and-units).
- [Quickstart](quickstart.md) - observer-robust EC on a warp metric in one file.
- [How-To: custom metrics](../how-to/custom_metric_tutorial.md) - subclass
  `ADMMetric` to plug in your own spacetime.
- [API reference](../reference/index.md) - generated documentation for the public modules.
