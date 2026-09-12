# Load an external metric

Three loaders return `InterpolatedADMMetric` objects with the standard metric
interface. Curvature accuracy depends on interpolation regularity and resolution;
loading data alone does not validate its derivatives.

| Source | Loader | Fixture |
|--------|--------|---------|
| WarpFactory MATLAB `.mat` | `warpax.io.load_warpfactory` | `tests/fixtures/warpfactory/` |
| EinFields Flax/Orbax | `warpax.io.load_einfield` | `tests/fixtures/einfields/` |
| Cactus / Einstein Toolkit HDF5 | `warpax.io.load_cactus_slice` | `tests/fixtures/cactus/` |

Install the `interop` extra for HDF5/MATLAB data and `einfields` for Orbax:

```bash
python -m pip install -e ".[interop,solver]"
```

## WarpFactory `.mat` exports

`warpax.io.load_warpfactory(path)` parses a MATLAB export from
[WarpFactory](https://github.com/NerdsWithAttitudes/WarpFactory) into
an `InterpolatedADMMetric`. Schema-tolerant: v7.3 HDF5-backed `.mat`
via `mat73`; older v7 / v6 / v4 via `scipy.io.loadmat`.

```python
from warpax.io import load_warpfactory

metric = load_warpfactory("path/to/alcubierre.mat")
import jax.numpy as jnp
coords = jnp.array([0.0, 0.0, 0.0, 0.0])
g = metric(coords) # 4x4 covariant metric
alpha = metric.lapse(coords) # scalar lapse
```

Expected schema (after `metricGet_Alcubierre` + `save('...', '-v7.3')` in MATLAB):

- `metric.tensor` - float64 array shape `(4, 4, Nt, Nx, Ny, Nz)`
- `metric.coords` - struct of 1D `t`, `x`, `y`, `z` arrays
- `metric.type` - `str` (e.g., `"Alcubierre"`)

## EinFields Flax/Orbax checkpoints

`warpax.io.load_einfield(checkpoint_path)` restores a trained
[EinFields](https://arxiv.org/abs/2507.11589) Flax NNX network via
Orbax, samples it on a regular 4D grid, and returns an
`InterpolatedADMMetric`. Install the optional extra first:

```bash
pip install 'warpax[einfields]'
```

```python
from warpax.io import load_einfield

metric = load_einfield(
    "path/to/checkpoint.ckpt",
    sample_bounds=((-1.0, 1.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)),
    sample_shape=(2, 8, 8, 8),
    interp_method="linear",
)
```

For the synthetic Minkowski fixture, see
`tests/fixtures/einfields/generate_minkowski_ckpt.py`.

## Cactus / Einstein Toolkit HDF5 slices

`warpax.io.load_cactus_slice(path, iteration=0, timelevel=0)` reads an
ET-compatible HDF5 single-slice export into an `InterpolatedADMMetric`.

```python
from warpax.io import load_cactus_slice

metric = load_cactus_slice(
    "path/to/simulation.h5",
    iteration=0,
    timelevel=0,
    interp_method="linear",
)
```

Expected HDF5 schema:

```
/ITERATION={i}/TIMELEVEL={t}/
    alp : float64 (nz, ny, nx) lapse
    betax/y/z: float64 (nz, ny, nx) contravariant shift beta^i
    gxx/.../gzz: float64 (nz, ny, nx) spatial metric (symmetric)

Group attributes:
    time, x0, y0, z0, dx, dy, dz
```

Orientation convention: ET ASC output is C-order `(nz, ny, nx)`; the
loader transposes to `(nx, ny, nz)` for the canonical warpax
`(t, x, y, z)` ordering.

The loader supports one iteration and time level; it does not combine AMR
components. A single slice cannot determine the metric's physical time derivatives.

## Interpolation limits

The returned object supplies `lapse`, `shift`, `spatial_metric`, and
`__call__(coords)`. Its `symbolic` and `shape_function_value` methods raise
`NotImplementedError`, so provide an explicit wall mask for sampled data.

The current interpolator is multilinear. `interp_method="cubic"` warns and
falls back to linear, and out-of-domain values clamp to the nearest boundary.
Piecewise multilinear fields are generally not $C^2$ across cell boundaries;
AD curvature of that interpolant is not automatically a converged curvature
estimate for the underlying simulation. Check resolution and derivative
regularity before interpreting energy-condition results.

See the [API reference](../reference/index.md) for loader options.
