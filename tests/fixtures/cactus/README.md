# Cactus / Einstein Toolkit fixture

`minkowski_slice.h5` - hand-synth ET-compatible HDF5 single-slice
Minkowski fixture. Used by `tests/test_io_cactus.py` to exercise
`warpax.io.load_cactus_slice`.

## Schema

```
/ITERATION=0/TIMELEVEL=0/
    alp : (nz=8, ny=8, nx=8) float64 lapse (all 1.0)
    betax : (nz=8, ny=8, nx=8) float64 shift (all 0.0)
    betay : (nz=8, ny=8, nx=8) float64
    betaz : (nz=8, ny=8, nx=8) float64
    gxx : (nz=8, ny=8, nx=8) float64 spatial metric (eye-3 at every point)
    gxy : (nz=8, ny=8, nx=8) float64
    gxz : (nz=8, ny=8, nx=8) float64
    gyy : (nz=8, ny=8, nx=8) float64
    gyz : (nz=8, ny=8, nx=8) float64
    gzz : (nz=8, ny=8, nx=8) float64

Attributes on the TIMELEVEL group:
    time : float coordinate time (0.0)
    x0, y0, z0 : float lower-bound of grid ((-1, -1, -1))
    dx, dy, dz : float grid spacing (2/7 on each axis)
```

## ARCH-3 orientation pin

Arrays are written in C-order with shape `(nz, ny, nx)` - matches the
ET ASC output convention. `load_cactus_slice` transposes to
`(nx, ny, nz)` on read so downstream warpax code sees the canonical
`(t, x, y, z)` ordering.

## Regeneration

```bash
python tests/fixtures/cactus/generate_minkowski_slice.py
```

No network dependency. Deterministic output. Safe to regenerate on any
`h5py>=3.16.0` install.

## Scope pin

Single iteration + single timelevel ONLY. Multi-iteration and AMR
component-group support is deferred to a future release.
