# Your first curvature computation

A walk-through of the full curvature chain on a Minkowski spacetime.
Useful as a sanity check: Minkowski is flat, so every curvature tensor should
be numerically zero to machine precision. If this works, your environment is
ready for the more interesting warp metrics.

## Prerequisites

- Python 3.12+
- `warpax` installed editable: `pip install -e ".[dev]"` inside the repo
- Runnable counterpart: `examples/01_minkowski_sanity.py`

## The full pipeline, end to end

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

Expected output: every non-metric quantity is zero to $\le 10^{-10}$.

## What just happened

1. `MinkowskiMetric` returns the metric directly rather than through an ADM
   split: its pointwise call $g_{ab}(t,x,y,z)$ is
   $\eta_{ab} = \mathrm{diag}(-1,1,1,1)$ at every point. In ADM terms that is
   lapse $\alpha=1$, shift $\beta^i=0$, spatial metric
   $\gamma_{ij}=\delta_{ij}$.
2. `compute_curvature_chain` applies `jax.jacfwd` at two differentiation
   stages: first on the metric to obtain the Christoffel symbols, then a
   nested `jax.jacfwd` on the Christoffel map for the Riemann tensor
   $R^\alpha{}_{\beta\gamma\delta}$; the Ricci and Einstein tensors follow
   by pure contraction, not autodiff.
3. All tensors are JAX arrays with jaxtyping shape annotations (e.g.
   `Float[Array, "4 4 4 4"]`).

## Next steps

- [Quickstart](quickstart.md) - observer-robust EC on a warp metric in one file.
- [How-To: custom metrics](../how-to/custom_metric_tutorial.md) - subclass
  `ADMMetric` to plug in your own spacetime.
- [API reference](../reference/index.md) - autodoc of every public symbol.
