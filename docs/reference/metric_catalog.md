# Metric catalog

warpax ships eleven metrics: nine warp drive variants under
``warpax.metrics``, and two reference spacetimes (Minkowski, Schwarzschild)
under ``warpax.benchmarks``.

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
useful as a stress-test of the curvature chain at large Kretschner values.

::: warpax.metrics.WarpShellMetric

### `LentzMetric`

Lentz (2020) shift-only, positive-energy candidate. See the paper for
under-resolution caveats -- the wall is thinly sampled at low grid
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

### `FuchsMetric`

Fuchs et al. (2024) constant-velocity physical warp shell
(arXiv:2405.02709). Iterative Gaussian-kernel smoothing of an
isotropic-pressure TOV intermediate, with metric functions $a(r)$ and
$b(r)$ recovered from Carroll Eqs. 5.143 / 5.152 on a uniform radial
grid. Default factory ``fuchs_default()`` matches the paper parameters
($R_1 = 10$, $R_2 = 20$, $v_s = 0.02$, $r_s = 5$).

The pre-smoothing analytical intermediate (constant-density shell + TOV
pressure, steps 1-2 only) is retained in ``warpax.metrics._fuchs_legacy``
as ``_FuchsAnalytical`` for diagnostic comparison.

::: warpax.metrics.FuchsMetric

### `SShellMetric`

Source-first Class I shell (S-shell). Flow-orthogonal matter ($u^a =
n^a$), non-flat spatial metric, non-unit lapse, isotropic pressure. Metric
potentials derived from the Hamiltonian constraint and anisotropic TOV
equilibrium. Zero shift (no transport utility); serves as a clean baseline
for constraint satisfaction.

::: warpax.metrics.SShellMetric

### `TShellMetric`

Source-first Class II shell (T-shell). Tilted matter flow with nonzero
Eulerian momentum density $S_i$. Shift $\beta^x$ derived from the momentum
constraint (not prescribed). Addresses the Barzegar et al.
source-consistency critique. Achieves $\epsilon_{\mathcal{H}} \approx 5
\times 10^{-3}$ with positive EC margins in the deep shell interior.

::: warpax.metrics.TShellMetric

## Reference spacetimes - ``warpax.benchmarks``

### `MinkowskiMetric`

Pure Minkowski: $g_{ab}=\eta_{ab}=\mathrm{diag}(-1,1,1,1)$. The
ground-truth sanity check -- all curvature tensors vanish.

::: warpax.benchmarks.MinkowskiMetric

### `SchwarzschildMetric`

Schwarzschild exterior in standard coordinates, parameterized by the mass
`M`. Ground-truth non-trivial Ricci-flat curvature; used for cross-validation.

::: warpax.benchmarks.SchwarzschildMetric
