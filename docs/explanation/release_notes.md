# Release notes

How `warpax` got here. Per-release line items live in
[`CHANGELOG.md`](https://github.com/anindex/warpax/blob/main/CHANGELOG.md).

## v1.2.0 (2026-06)

Adds `warpax.bondi`: the Bondi four-momentum radiated to null infinity and the
Newman--Penrose peeling structure, read directly from the curvature
(`radiated_momentum_flux`, `peeling_slopes`, `weyl_scalars`, `psi4_at`), with a
gravitational-news ($\Psi_4$) proxy. The existing API is unchanged.

## v1.1.0 (2026-06)

The frame-independent certification core: `warpax.certify` decides the
all-observer energy conditions from the eigenstructure of `T^a_b` at all warp
speeds (including superluminal `v_s >= 1`), with explicit, physically certified
Type-IV detection and a closed-form Type-I worst observer. New physics: the shift
vorticity controls the Hawking-Ellis type (`f = κ ω`); a rigorous
geodesic-integrated ANEC via a symplectic null integrator with an on-cone witness;
a cross-construction all-observer verification panel; a boost-invariant exoticity
ranking; and universal `v_s` scaling laws for the wall NEC deficit (saturating the
Santiago-Schuster-Visser bound) and the wall curvature invariants. The
Garattini-Zatrimaylov de Sitter bubble joins the metric set. See `CHANGELOG.md`
for the full line items.

## v1.0.0 (2026-05)

Version 1.0 ships the energy-condition stack, source-first shell
constructions, autodiff curvature chain, IO layer, and metric-design API
together. Classification matches the paper benchmarks; the optimizer
includes smooth-cap and projected-gradient BFGS; constraint solvers derive
shells from Hamiltonian and momentum constraints. Docs build under
`mkdocs --strict`, the fast pytest suite runs with `pytest -n auto`, and
the public API is frozen for the 1.x line.

## v0.4: source-first shells and design sweeps

0.4 added the Bernstein-parameterized S-shell and T-shell ansatze,
constraint-derived metric potentials, and the 2D sweep over compactness
and thickness. The Fuchs construction split into the Gaussian-smoothed
pipeline and a pre-smoothing analytical intermediate so the two paths
can be compared. Lentz-family curvature gained on-axis floors that
killed silent NaN gradients.

## v0.3: ADM split and source consistency

0.3 added `adm_split`, the source-consistency residual
`stress_energy_residual`, autodiff TOV residuals, and the two-sided
Israel junction formulation. ADM mass became a surface integral with
Gauss-Legendre angular quadrature; transport diagnostics moved to
geodesic-based throughout.

## v0.2: toolkit maturation

The Hawking-Ellis classifier gained a generalized-eigenvalue fallback,
the BFGS optimizer learned warm-starts and projected-gradient bounds,
and the design API (`design_metric`, `ShapeFunction`, objective and
constraint registries) landed alongside the ANEC, AWEC, and Ford-Roman
evaluators. WarpFactory, EinsteinFields, and Cactus readers arrived in
`warpax.io`.

## v0.1: initial release

Accompanied [arXiv:2602.18023](https://arxiv.org/abs/2602.18023): autodiff
curvature chain, six warp-drive metrics (Alcubierre, Lentz, Van den Broeck,
Natario, Rodal, WarpShell), classification and optimization pipeline, and
the Diffrax geodesic integrator that later work builds on.
