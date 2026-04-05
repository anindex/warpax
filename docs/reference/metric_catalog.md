# Metric catalog

warpax ships eight metrics: six warp drive variants under
``warpax.metrics``, and two reference spacetimes (Minkowski, Schwarzschild)
plus an Alcubierre benchmark under ``warpax.benchmarks``.

## Warp drive metrics - ``warpax.metrics``

### `AlcubierreMetric`

The original Alcubierre (1994) warp drive: flat lapse, shift along the
propagation direction with a `tanh`-based top-hat shape function.
Parameters: `v_s` (warp velocity), `R` (bubble radius), `sigma`
(wall-sharpness), `x_s` (center).

::: warpax.benchmarks.AlcubierreMetric

### `RodalMetric`

Rodal (2025) construction, globally Type I Hawking-Ellis. Serves as the
positive-control for warp-wall EC verification: all grid points are Type I,
unlike Alcubierre which is dominated by Type IV at the wall.

::: warpax.metrics.RodalMetric

### `WarpShellMetric`

Spherical-shell geometry with $C^1$ / $C^2$ smooth transitions.
Curvature magnitudes are extreme near the shell boundary; the metric is
useful as a stress-test of the curvature chain at large Kretschmann values.

::: warpax.metrics.WarpShellMetric

### `LentzMetric`

Lentz (2020) shift-only, positive-energy candidate. See the paper for
under-resolution caveats - the wall is thinly sampled at low grid
resolutions.

::: warpax.metrics.LentzMetric

### `NatarioMetric`

Natario (2001) zero-expansion variant. The spatial-metric form is
non-trivial (unlike most warp metrics which keep a flat spatial metric).

::: warpax.metrics.NatarioMetric

### `VanDenBroeckMetric`

Van den Broeck (1999) volume-expansion variant. Two-parameter
nested-warp envelope with an exterior radius `R` and interior radius
`R_tilde`.

::: warpax.metrics.VanDenBroeckMetric

## Reference spacetimes - ``warpax.benchmarks``

### `MinkowskiMetric`

Pure Minkowski: $g_{ab}=\\eta_{ab}=\\mathrm{diag}(-1,1,1,1)$. The
ground-truth sanity check - all curvature tensors vanish.

::: warpax.benchmarks.MinkowskiMetric

### `SchwarzschildMetric`

Schwarzschild exterior in standard coordinates, parameterized by the mass
`M`. Ground-truth non-trivial Ricci-flat curvature; used for cross-validation.

::: warpax.benchmarks.SchwarzschildMetric
