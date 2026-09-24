# Metric catalog

warpax provides ten warp/shell metrics, plus Minkowski and Schwarzschild
references. Alcubierre is in `warpax.benchmarks`; the other warp metrics are in
`warpax.metrics`. Parameters and signatures appear below each description.

## `AlcubierreMetric`

Unit lapse, Euclidean spatial metric, and an axial shift with a `tanh` wall.
Parameters are `v_s`, radius `R`, wall sharpness `sigma`, and center `x_s`.

::: warpax.benchmarks.AlcubierreMetric

## `RodalMetric`

Rodal's ideal irrotational construction admits a zero-momentum Type-I reduction.
An even series near the origin keeps Cartesian derivatives finite. Numerical
roundoff and series error still matter when enclosing normalized null deficits.

::: warpax.metrics.RodalMetric

## `WarpShellMetric`

Spherical shell with $C^1$ or $C^2$ transitions. Large curvature near the shell
boundary makes it a numerical stress test; results depend on the regularization.

::: warpax.metrics.WarpShellMetric

## `LentzMetric`

A shift-only profile based on the Lentz proposal. Its thin wall needs dedicated resolution;
coarse-grid fractions do not establish continuum energy-condition satisfaction.

::: warpax.metrics.LentzMetric

## `NatarioMetric`

Natário's zero-expansion drive has unit lapse and a Euclidean spatial metric.
The laboratory shift is $\beta_{\rm lab}=-X(x-v_st e_x)-v_se_x$, including the
transverse components of the divergence-free field $X$.

::: warpax.metrics.NatarioMetric

## `VanDenBroeckMetric`

Nested bubble with a conformal spatial factor, exterior radius `R`, and
interior radius `R_tilde`.

::: warpax.metrics.VanDenBroeckMetric

## `FuchsMetric`

The [Fuchs et al. construction](https://arxiv.org/abs/2405.02709) smooths a shell
source and reconstructs radial metric functions. `fuchs_default()` uses the
published boxcar kernel by default; `kernel_type="gaussian"` selects a
variance-matched alternative. Defaults include $R_1=10$, $R_2=20$, $R_b=1$,
$v_s=0.02$, and Schwarzschild-radius parameter `r_s_param=6.668692`.

Motion modifies covariant $g_{01}=g_{10}$ only; lapse and contravariant shift
are recovered from the full metric. The natural-endpoint sigmoid, cubic radial
spline, and $C^2$ joins are the implementation's regularization. The private
`_fuchs_legacy._FuchsAnalytical` retains the unsmoothed intermediate for comparison.

::: warpax.metrics.FuchsMetric

## `SShellMetric`

Legacy shift-free shell generated from a prescribed density and an inward TOV
integration. Its nonnegative isotropic pressure cannot match a regular empty
inner cavity without surface stress. It is a metric diagnostic, not the
Einstein–elastic equilibrium of the elastic-shell paper. See the
[shell guide](../how-to/reproduce_warpshell_paper.md).

::: warpax.metrics.SShellMetric

## `TShellMetric`

Legacy tilted-source prescription. Its scalar radial shift reduction does not
solve the full Cartesian-vector momentum constraint; the spatial Einstein
equations and constitutive stress also require separate agreement. A Type-IV
metric-derived stress cannot represent its prescribed perfect fluid. See the
[shell guide](../how-to/reproduce_warpshell_paper.md).

::: warpax.metrics.TShellMetric

## `GarattiniMetric`

Garattini-Zatrimaylov bubble matched to a de Sitter flow:
$x_s(t)=(v_s/H)e^{Ht}$ and $v(t)=Hx_s(t)$ for nonzero $H$. This matched
construction is irrotational and exactly Type I. Report $H$, initial position
$r_0=v_s/H$, time, shape parameters, and units when comparing it with other metrics.
The returned stress includes the effective cosmological contribution.

`matched(H=..., r0=...)` sets `v_s=H*r0`. At `H=0`, the implementation has a
separate Alcubierre branch; this is not a continuous fixed-`v_s` matched limit.

::: warpax.metrics.GarattiniMetric

## `MinkowskiMetric`

$g_{ab}=\mathrm{diag}(-1,1,1,1)$; all curvature vanishes.

::: warpax.benchmarks.MinkowskiMetric

## `SchwarzschildMetric`

Schwarzschild exterior in isotropic Cartesian coordinates, parameterized by mass `M`.
Ricci curvature vanishes while Riemann curvature remains nonzero.

::: warpax.benchmarks.SchwarzschildMetric
