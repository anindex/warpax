# Custom Warp Metric Tutorial

Defining a new warp spacetime and running the full energy condition (EC)
pipeline with wall-restricted diagnostics. Full runnable script:
`examples/07_custom_warp_metric.py` in the repo.

## What you will build

A Gaussian warp bubble -- a minimal custom warp metric where the shift acts
as a Gaussian envelope instead of the Alcubierre `tanh` top-hat. You will:

1. Define a metric by subclassing `ADMMetric`
2. Verify energy conditions at a single bubble-wall point
3. Run a grid-level Eulerian vs observer-robust comparison
4. Apply a wall-restricted filter and compute focused statistics
5. Save a 3-panel NEC comparison figure

Runtime target: under 30 seconds on a laptop CPU at `grid_n=16` (the default
in the script). Scaling up to `grid_n=50` or higher takes minutes rather
than seconds and is the right choice for publication-density figures.

## Step 1: Subclass `ADMMetric`

Every warp metric in warpax is an `ADMMetric` subclass. You implement six
methods:

- `lapse(coords) -> alpha(t, x, y, z)` -- ADM lapse function
- `shift(coords) -> beta^i(t, x, y, z)` -- 3-vector shift
- `spatial_metric(coords) -> gamma_{ij}(t, x, y, z)` -- 3x3 spatial metric
- `symbolic -> SymbolicMetric` -- SymPy form for cross-validation against
  the JAX autodiff pipeline
- `name -> str` -- registry key (used for logging and result JSON keys)
- `shape_function_value(coords) -> f(t, x, y, z)` -- the addition.
  Must return a value in `[0, 1]`. Consumed by `shape_function_mask` to
  build wall-restricted diagnostics (see Step 4).

Excerpt from `examples/07_custom_warp_metric.py`:

```python
from warpax.geometry.metric import ADMMetric, SymbolicMetric
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype
import jax.numpy as jnp
import sympy as sp


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
        r_s = jnp.sqrt(dx * dx + y * y + z * z)
        return jnp.exp(-(r_s * r_s) / (2.0 * self.w * self.w))
```

The `@jaxtyped(typechecker=beartype)` decorator is project convention --
shapes are validated at runtime. It is not required for the metric to
function.

Two patterns worth flagging:

- **Reuse `shape_function_value` from inside `shift`.** The Gaussian
  envelope only lives in one place, which keeps the shape function and the
  shift aligned when parameters change.
- **Unit lapse + flat spatial metric.** All of the warp geometry is carried
  by the shift. This is the Alcubierre-style ADM shape; WarpShell is the
  only built-in metric that departs from it.

## Step 2: Verify at a single point

Probe at `r_s ~ w` -- the bubble wall, where the gradient of the shape
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

`verify_point` runs the Optimistix BFGS optimizer across the full bounded
timelike observer manifold and returns the worst-case observer 4-velocity
alongside the four margins. `compute_eulerian_ec` runs the same point
through the ADM-normal observer only and returns a dict of margins for
the single-frame comparison.

## Step 3: Grid-level comparison

To see which regions of the bubble are violated and which are not, build a
3D grid, evaluate the full curvature chain on it, and compare Eulerian vs
robust margins at every point.

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
        f"{cond.upper}: eul_min={eul_min:+.3e} rob_min={rob_min:+.3e}"
        f" cond_miss={comparison.conditional_miss_rate[cond]:.1%}"
    )
```

The returned `ComparisonResult` carries per-condition margins for both
frames plus the missed-violation mask (points where Eulerian reports
"satisfied" but the robust optimizer reports "violated").

## Step 4: Wall-restricted statistics

Key insight: full-grid statistics dilute warp-wall violations across vacuum
regions. Wall-restricted filtering uses the shape function to isolate the
active region. This is where the `shape_function_value` override in Step 1
pays off.

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
print(f"NEC miss rate in wall: {stats.nec_miss_rate}")
```

The default interval `[f_low=0.1, f_high=0.9]` captures the transition
region where the shape function is neither fully interior (`f = 1`) nor
fully exterior (`f = 0`). See
[`interpreting_ec_results.md`](interpreting_ec_results.md) for the
rationale and a worked numeric example.

`compute_wall_restricted_stats` consumes an `ECGridResult` (the output of
`verify_grid`). Because `compare_eulerian_vs_robust` returns a
`ComparisonResult` of a different shape, the script calls `verify_grid`
explicitly to get the right input. Passing the comparison's
`eulerian_margins` lets the stats object carry both Type breakdown **and**
conditional miss rates in a single call.

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

The PDF is 3 panels: Eulerian NEC (left), robust NEC (center), and the
missed mask (right) highlighting grid points where the Eulerian frame
misreports the violation.

## Running it

```bash
python examples/07_custom_warp_metric.py
```

Default runtime ~22 seconds on a laptop CPU at `grid_n=16`. Edit the
`grid_n=16` argument inside `main` to scale up; at `grid_n=50` you get
~125k points and the job runs in minutes.

## Common pitfalls

- **Forgetting to enable float64.** `warpax/__init__.py` enables float64 at
  import, but standalone scripts should re-enable defensively with
  `jax.config.update("jax_enable_x64", True)` before any JAX operation.
- **Returning a shape function outside `[0, 1]`.** `shape_function_mask`
  filters on the interval `[f_low, f_high]` (default `[0.1, 0.9]`); a
  shape function that saturates above 1 or dips below 0 produces an
  unexpected empty or truncated mask.
- **Geometric units.** `v_s` is a dimensionless fraction of `c`. Positions
  and times share a single unit because `G = c = 1`.

## See also

- [`interpreting_ec_results.md`](interpreting_ec_results.md) -- margin sign
  convention, Hawking-Ellis Type I-IV semantics, miss-rate definitions
- [`quickstart.md`](../tutorials/quickstart.md) -- faster install-to-first-result path
- [`ARCHITECTURE.md`](../explanation/ARCHITECTURE.md) -- curvature pipeline internals
