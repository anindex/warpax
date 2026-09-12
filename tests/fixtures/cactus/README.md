# Cactus / Einstein Toolkit fixture

`minkowski_slice.h5` is a synthetic Einstein Toolkit-compatible Minkowski
slice used by `tests/test_io.py` to test `load_cactus_slice`.

## Schema

```
/ITERATION=0/TIMELEVEL=0/
    alp : (nz=8, ny=8, nx=8) float64 lapse (all 1.0)
    betax : (nz=8, ny=8, nx=8) float64 contravariant shift beta^x (all 0.0)
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

## Orientation

Arrays are written in C-order with shape `(nz, ny, nx)` - matches the
ET ASC output convention. `load_cactus_slice` transposes to
`(nx, ny, nz)` on read so downstream warpax code sees the canonical
`(t, x, y, z)` ordering.

## Regeneration

```bash
python tests/fixtures/cactus/generate_minkowski_slice.py
```

Requires `h5py>=3.16.0`; no network data is needed. The fixture covers one
iteration and time level, without AMR component groups.
