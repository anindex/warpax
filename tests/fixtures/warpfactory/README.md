# WarpFactory fixture

`alcubierre.mat` is a synthetic MATLAB v7 Alcubierre fixture for
`tests/test_io.py` and `load_warpfactory`. It is not an upstream MATLAB export.

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

## Fixture source

The fixture was generated with warpax v0.1.x
`AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)` on a `(2,4,4,4)` spacetime grid,
with spatial bounds `(-3,3)` on each axis. It tests schema compatibility.
An independently generated upstream export would provide a stronger
cross-implementation comparison.
