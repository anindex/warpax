# WarpFactory fixture

`alcubierre.mat` - MATLAB v7 export of an Alcubierre warp drive. Used
by `tests/test_io_warpfactory.py` to exercise the `load_warpfactory`
reader (, ).

## Schema

```
metric.tensor : float64 array (4, 4, Nt, Nx, Ny, Nz)
metric.coords : struct
    .t : float64 (Nt,)
    .x : float64 (Nx,)
    .y : float64 (Ny,)
    .z : float64 (Nz,)
metric.type : char "Alcubierre"
```

Current fixture: `Nt=2, Nx=4, Ny=4, Nz=4`. File size: ~17 KB. Format: v7
(written via `scipy.io.savemat(format="5")`); dispatched through the
`_load_v6_v7` scipy path in `load_warpfactory`.

## Canonical generation recipe (upstream WarpFactory MATLAB)

```matlab
addpath('path/to/WarpFactory');
metric = metricGet_Alcubierre(0.5, 2.0, 8.0);
save('alcubierre.mat', '-v7.3', '-struct', 'metric');
```

The v7.3 HDF5 variant dispatches through the `_load_v7_3` mat73 path;
schema keys and `metric.tensor` shape are identical.

## Hand-synth provenance (this fixture)

Because the the CI/CD environment has no MATLAB, this fixture
was generated from the warpax v0.1.x `AlcubierreMetric(v_s=0.5, R=2.0,
sigma=8.0)` - the same parameters as the canonical WarpFactory recipe
- by evaluating the metric pointwise on the 4D grid `(Nt=2, Nx=4,
Ny=4, Nz=4)` over bounds `(-3, 3)` on each spatial axis. The result
is a schema-valid v7 `.mat` file that exercises the full
`load_warpfactory` code path.

If a true upstream WarpFactory export becomes available, replace this
file and regenerate with the canonical MATLAB recipe.
