# Quickstart

Use Python 3.12 or later. From a repository checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,viz,design,solver]"
python examples/03_alcubierre_analysis.py
```

CPU is sufficient. `warpax` enables float64 at import. The `design` extra adds
spline tools, `solver` adds the generalized eigensolver, and `viz` adds figure
helpers; the separate `manim` extra requires system rendering dependencies.

The example evaluates the Alcubierre metric at `v_s=0.5, R=1, sigma=8` and
`(t,x,y,z)=(0,1,0.5,0)`. Representative values are:

| Quantity | Value |
|---|---:|
| Ricci scalar | `-5.924607` |
| Kretschmann scalar | `-8.004342` |
| Eulerian WEC contraction | `-1.658823e-3` |
| Best-found WEC contraction, rapidity cap 5 | `-4.561904e2` |
| Best-found normalized NEC contraction | `-8.283962e-2` |

A Lorentzian curvature contraction such as the Kretschmann scalar can be negative.
The negative WEC and NEC contractions witness their respective violations. The
large boosted WEC magnitude depends on the rapidity cap and frame; it is not
an invariant measure of severity. A nonnegative observer-search result would
not establish satisfaction for every observer.

For a pointwise all-observer test, use the LMI API:

```python
import jax.numpy as jnp
from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import compute_curvature_chain
from warpax.energy_conditions.slemma import certify_point, noise_floor

cur = compute_curvature_chain(
    AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0),
    jnp.array([0.0, 1.0, 0.5, 0.0]),
)
verdict = certify_point(cur.stress_energy, cur.metric)
print(verdict)
print("Decision tolerance:", noise_floor(cur.stress_energy, cur.metric))
```

Compare the numerical margins with both signs of the noise floor. See
[Interpreting results](../how-to/interpreting_ec_results.md) for their meaning.

To define a metric and plot a grid comparison:

```bash
python examples/07_custom_warp_metric.py
```

This writes `examples/output/gaussian_warp_comparison.pdf` on a default `16^3`
grid. Follow the [custom metric tutorial](../how-to/custom_metric_tutorial.md)
or the [examples tour](examples_tour.md) for larger calculations.

All lengths and times share one unit because $G=c=1$; `v_s` is dimensionless.
