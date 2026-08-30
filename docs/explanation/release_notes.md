# Release notes

Pre-1.0 history. Every 1.x release is in
[`CHANGELOG.md`](https://github.com/anindex/warpax/blob/main/CHANGELOG.md).

## v0.4: source-first shells and design sweeps

The Bernstein-parameterized S-shell and T-shell ansatze, constraint-derived
metric potentials, and the 2D sweep over compactness and thickness. The Fuchs
construction split into the Gaussian-smoothed pipeline and a pre-smoothing
analytical intermediate so the two paths can be compared. Lentz-family curvature
gained on-axis floors that removed silent NaN gradients.

## v0.3: ADM split and source consistency

`adm_split`, the source-consistency residual `stress_energy_residual`, autodiff
TOV residuals, and the two-sided Israel junction formulation. ADM mass became a
surface integral with Gauss-Legendre angular quadrature, and transport
diagnostics moved to a geodesic basis throughout.

## v0.2: toolkit maturation

The Hawking-Ellis classifier gained a generalized-eigenvalue fallback, the BFGS
optimizer learned warm starts and projected-gradient bounds, and the design API
(`design_metric`, `ShapeFunction`, objective and constraint registries) landed
alongside the ANEC, AWEC and Ford-Roman evaluators. WarpFactory, EinsteinFields
and Cactus readers arrived in `warpax.io`.

## v0.1: initial release

Accompanied [arXiv:2602.18023](https://arxiv.org/abs/2602.18023): the autodiff
curvature chain, six warp-drive metrics (Alcubierre, Lentz, Van den Broeck,
Natario, Rodal, WarpShell), the classification and optimization pipeline, and
the Diffrax geodesic integrator that later work builds on.
