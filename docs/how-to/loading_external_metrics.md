# Load an external metric

warpax ships three loaders for third-party numerical-metric data formats.
They wrap the data in an ``InterpolatedADMMetric`` so the standard curvature
chain + EC verification pipeline works identically to an analytic metric.

| Source | Loader | Fixture |
|--------|--------|---------|
| WarpFactory MATLAB `.mat` | `warpax.io.load_warpfactory` | `tests/fixtures/warpfactory/` |
| EinFields Flax/Orbax | `warpax.io.load_einfield` | `tests/fixtures/einfields/` |
| Cactus / Einstein Toolkit HDF5 | `warpax.io.load_cactus_slice` | `tests/fixtures/cactus/` |

## WarpFactory `.mat` exports

`warpax.io.load_warpfactory(path)` parses a MATLAB export from
[WarpFactory](https://github.com/NerdsWithAttitudes/WarpFactory) into
an `InterpolatedADMMetric`. Schema-tolerant: v7.3 HDF5-backed `.mat`
via `mat73`; older v7 / v6 / v4 via `scipy.io.loadmat`.

```python
from warpax.io import load_warpfactory

metric = load_warpfactory("path/to/alcubierre.mat")
# metric is an InterpolatedADMMetric - plug into the curvature chain
# and EC verifier exactly like any analytic metric.

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
    interp_method="cubic",
)
```

For the hand-synth Minkowski fixture, see
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
    interp_method="cubic",
)
```

Expected HDF5 schema:

```
/ITERATION={i}/TIMELEVEL={t}/
    alp : float64 (nz, ny, nx) lapse
    betax/y/z: float64 (nz, ny, nx) shift (lower index)
    gxx/.../gzz: float64 (nz, ny, nx) spatial metric (symmetric)

Group attributes:
    time, x0, y0, z0, dx, dy, dz
```

Orientation convention: ET ASC output is C-order `(nz, ny, nx)`; the
loader transposes to `(nx, ny, nz)` for the canonical warpax
`(t, x, y, z)` ordering.

Scope: single-iteration, single-timelevel only. Multi-iteration + AMR
component groups are deferred to a future release.

## Common contract

Every loader returns an ``InterpolatedADMMetric`` instance. The returned object
exposes the same `lapse`, `shift`, `spatial_metric`, and `__call__(coords)`
methods as any analytic metric - the difference is that values at arbitrary
coordinates are produced by interpolation over the stored grid rather than
evaluated from a closed-form expression.

See [`InterpolatedADMMetric`](../reference/index.md)
for the shared base class contract.

## Why this exists

The WarpFactory / EinFields / Cactus communities publish numerical warp
spacetimes and neutron-star simulations in domain-specific formats. warpax
keeps the curvature + EC pipeline agnostic to the data source so external
practitioners can apply observer-robust verification without porting their
metrics back into SymPy / JAX by hand.
