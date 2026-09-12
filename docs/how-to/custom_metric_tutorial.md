# Define a custom warp metric

Define a Gaussian shift profile, evaluate its energy conditions, and summarize
the bubble wall. The complete script is `examples/07_custom_warp_metric.py`.
Use a coarse grid for exploration and check refinement before reporting results.

## Step 1: Subclass `ADMMetric`

Subclass `ADMMetric` and implement six methods:

- `lapse(coords) -> alpha(t, x, y, z)`, ADM lapse function
- `shift(coords) -> beta^i(t, x, y, z)`, 3-vector shift
- `spatial_metric(coords) -> gamma_{ij}(t, x, y, z)`, 3x3 spatial metric
- `symbolic() -> SymbolicMetric`, SymPy form for cross-validation against
  the JAX autodiff pipeline
- `name() -> str`, registry key (used for logging and result JSON keys)
- `shape_function_value(coords) -> f(t, x, y, z)`, a wall indicator.
  Return a value in `[0, 1]`. Consumed by `shape_function_mask` to
  build wall-restricted diagnostics (see Step 4).

A compact implementation:

```python
import jax.numpy as jnp
import sympy as sp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from warpax.geometry.metric import ADMMetric, SymbolicMetric


class GaussianWarpMetric(ADMMetric):
    v_s: float = 0.5
    w: float = 1.0

    @jaxtyped(typechecker=beartype)
    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        return jnp.array(1.0)

    @jaxtyped(typechecker=beartype)
    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        f = self.shape_function_value(coords)
        # Shift acts only along x (warp propagation axis), like Alcubierre.
        return jnp.array([-self.v_s * f, 0.0, 0.0])

    @jaxtyped(typechecker=beartype)
    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        return jnp.eye(3)

    @jaxtyped(typechecker=beartype)
    def shape_function_value(
        self, coords: Float[Array, "4"]
    ) -> Float[Array, ""]:
        t, x, y, z = coords
        dx = x - self.v_s * t
        r_squared = dx * dx + y * y + z * z
        return jnp.exp(-r_squared / (2.0 * self.w * self.w))

    def symbolic(self) -> SymbolicMetric:
        """SymPy form for comparison with the JAX implementation."""
        t, x, y, z = sp.symbols("t x y z")
        v_s = sp.Symbol("v_s", positive=True)
        w = sp.Symbol("w", positive=True)
        r_s = sp.sqrt((x - v_s * t) ** 2 + y**2 + z**2)
        beta_x = -v_s * sp.exp(-(r_s**2) / (2 * w**2))
        g = sp.Matrix(
            [
                [-(1 - beta_x**2), beta_x, 0, 0],
                [beta_x, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        return SymbolicMetric([t, x, y, z], g)

    def name(self) -> str:
        return "GaussianWarp"
```

All six methods are abstract. Runtime shape checking with `jaxtyped` is optional.
Use the squared radius directly for this Gaussian so its Cartesian derivatives
remain regular at the center. Reusing `shape_function_value` keeps the mask and
shift consistent. This example uses unit lapse and a flat spatial metric;
custom metrics may supply other ADM fields.

## Step 2: Verify at a single point

Probe at `r_s ~ w`, the bubble wall, where the gradient of the shape
function is steep and violations concentrate.

```python
import jax.numpy as jnp
from warpax.energy_conditions import verify_point, compute_eulerian_ec
from warpax.geometry import compute_curvature_chain

metric = GaussianWarpMetric(v_s=0.5, w=1.0)
coords = jnp.array([0.0, 1.0, 0.5, 0.0])

result = compute_curvature_chain(metric, coords)
ec = verify_point(result.stress_energy, result.metric, result.metric_inv)

print(f"NEC margin (robust): {float(ec.nec_margin):+.6e}")
print(f"Hawking-Ellis type: {int(ec.he_type)}")
```

`verify_point` returns Type-I eigenvalue slacks when applicable and capped BFGS
diagnostics. At other types, its margins come from the observer search. For an
all-observer LMI test, use `energy_conditions.slemma.certify_point`.

## Step 3: Grid-level comparison

Evaluate a spatial grid and compare the reported margins:

```python
import numpy as np
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.analysis import compare_eulerian_vs_robust

grid = GridSpec(
    bounds=[(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
    shape=(16, 16, 16),
)
grid_result = evaluate_curvature_grid(metric, grid)

comparison = compare_eulerian_vs_robust(
    grid_result.stress_energy,
    grid_result.metric,
    grid_result.metric_inv,
    grid_shape=grid.shape,
    n_starts=8,
    zeta_max=5.0,
    batch_size=256,
)

for cond in ("nec", "wec", "sec", "dec"):
    eul_min = float(np.min(comparison.eulerian_margins[cond]))
    rob_min = float(np.min(comparison.robust_margins[cond]))
    print(
        f"{cond.upper()}: eul_min={eul_min:+.3e} rob_min={rob_min:+.3e}"
        f" cond_miss={comparison.conditional_miss_rate[cond]:.1f}%"
    )
```

`ComparisonResult` holds both margin arrays and masks of violations found by
the reference method but missed by the Eulerian test. Its rates are percentages
on `[0,100]`; format with `:.1f}%`. Wall rates below are fractions on `[0,1]`
and return `None` when no violations are found.

## Step 4: Wall-restricted statistics

Select the transition region with the shape function:

```python
from warpax.energy_conditions import (
    shape_function_mask,
    compute_wall_restricted_stats,
    verify_grid,
)
from warpax.geometry import build_coord_batch

coords_batch = build_coord_batch(grid, t=0.0)
wall_mask = shape_function_mask(
    metric,
    coords_batch,
    grid.shape,
    f_low=0.1,
    f_high=0.9,
)

ec_grid = verify_grid(
    grid_result.stress_energy,
    grid_result.metric,
    grid_result.metric_inv,
    n_starts=8,
    zeta_max=5.0,
    batch_size=256,
)

stats = compute_wall_restricted_stats(
    ec_grid,
    wall_mask,
    eulerian_margins=comparison.eulerian_margins,
)

print(f"Type IV fraction in wall: {stats.frac_type_iv:.1%}")
nec_rate = stats.nec_miss_rate
print(f"NEC miss rate in wall: {'n/a' if nec_rate is None else format(nec_rate, '.1%')}")
```

The mask selects `0.1 <= f <= 0.9`. `compute_wall_restricted_stats` requires
an `ECGridResult`, so the example calls `verify_grid` separately from the
comparison. These are sampled statistics; see
[Interpreting results](interpreting_ec_results.md) for denominator and resolution limits.

## Step 5: Save the figure

```python
import numpy as np
from warpax.visualization import plot_comparison_panel

fig = plot_comparison_panel(
    eulerian_margin=np.asarray(comparison.eulerian_margins["nec"]),
    robust_margin=np.asarray(comparison.robust_margins["nec"]),
    missed=np.asarray(comparison.missed["nec"]),
    grid_bounds=grid.bounds,
    grid_shape=grid.shape,
    title="Gaussian Warp NEC: Eulerian vs Robust",
    save_path="examples/output/gaussian_warp_comparison.pdf",
)
```

The panels show Eulerian NEC, the reference NEC margin, and missed violations.

```bash
python examples/07_custom_warp_metric.py
```

The script defaults to `grid_n=16`; larger grids increase both compilation and
execution cost.

## Common pitfalls

- **Forgetting to enable float64.** `warpax/__init__.py` enables float64 at
  import, but import warpax before constructing JAX arrays in standalone scripts.
- **Returning a shape function outside `[0, 1]`.** `shape_function_mask`
  filters on the interval `[f_low, f_high]` (default `[0.1, 0.9]`); a
  shape function that saturates above 1 or dips below 0 produces an
  unexpected empty or truncated mask.
- **Geometric units.** `v_s` is a dimensionless fraction of `c`. Positions
  and times share a single unit because `G = c = 1`.

## See also

- [`interpreting_ec_results.md`](interpreting_ec_results.md), margin sign
  convention, Hawking-Ellis Type I-IV semantics, miss-rate definitions
- [`quickstart.md`](../tutorials/quickstart.md), faster install-to-first-result path
- [`ARCHITECTURE.md`](../explanation/ARCHITECTURE.md), curvature pipeline internals
