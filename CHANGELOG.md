# Changelog

All notable changes to `warpax`, following
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Pre-1.0 history is
in `docs/explanation/release_notes.md`.

## [1.4.0] - 2026-08-31

Type-free certification. The energy-condition verdict comes from a $4\times4$
linear matrix inequality rather than from an eigendecomposition, so it holds at
every Hawking-Ellis type, and each verdict carries an exact rational
certificate.

### Added

- `energy_conditions.slemma.certify_point_lmi`: NEC/WEC/SEC/DEC over every
  timelike and null observer from $\hat T + \sigma\eta \succeq 0$, at Types I
  through IV alike, with no rapidity cap and no classification tolerance.
  `witness_observer` returns the violating boost.
- `energy_conditions.certificate`: exact rational certificates checked over
  `Fraction`. A multiplier $\sigma$ when the condition holds, a boost $w$ with
  $|w| \le 1$ and $q(w) < 0$ when it fails.
- `energy_conditions.enclosure`: verified continuum enclosures of the worst-case
  violation by interval branch-and-bound (Moore-Skelboe, `mpmath.iv`, outward
  rounding), bracketing the extremum over the domain rather than over a grid.
- `energy_conditions.interval_lmi`: the same certificate in interval arithmetic,
  from the metric rather than from a sampled stress-energy tensor.
- `grids.proper_volume_weights`: the slice Jacobian
  $\mathrm{d}V = \sqrt{\det\gamma}\,\mathrm{d}^3x$. Every volume-weighted
  fraction on a conformally scaled slice moves.
- `analysis.extrema.refine_extremum`: polishes a grid extremum off the grid.

### Fixed

- `averaged.anec`: the on-cone witness is measured on the trajectory tangent
  $\mathrm{d}x/\mathrm{d}\lambda$, not on the projected vector, which is null by
  construction. `_project_to_null` uses the Citardauq form.
- `averaged.awec`: a spacelike curve is reported rather than normalised as
  timelike, and the line integral carries $\mathrm{d}\tau/\mathrm{d}\lambda$.
- `analysis.convergence`: grid spacing is $1/(N-1)$ on endpoint-inclusive grids.
  Non-geometric ladders are refused rather than fitted.
- `energy_conditions.enclosure.tail_bound`: the minimum of $|x|$ over the slab,
  not of $|$endpoint$|$, which excluded a slab straddling the origin.
- `junction.darmois`: one-sided limits at $\Sigma$, so a smooth metric gives a
  vanishing jump. A $\Sigma$ of changing causal character gives no verdict.
- Every absolute floor is relative to the tensor or metric scale, so a verdict
  survives a coordinate rescaling.
- Both classifiers report no verdict on non-finite input, and agree with each
  other.
- `frame_free`: `lmi_substituted` marks margins taken from the LMI rather than
  from the eigenvalue inequalities. Only their sign is comparable.
- ANEC affine windows come from the ray's horizon crossing, not from a
  stationarity search that walked into integrator drift.

### Performance

- Interval branch-and-bound 3.3x faster, with bit-identical enclosures.
- Metric parameters are array leaves, so a velocity sweep no longer recompiles
  the curvature chain per value.
- `import warpax` drops from 2.5 s to 0.75 s; `design` and sympy load on first
  use.

### Removed

- `exceptions`, `grids._refined`, `benchmarks.registry`,
  `visualization.common._themes` and `_jit_cache`, none of which had a caller.

## [1.3.0] - 2026-06-22

Manim scenes renamed to physically accurate names, colormaps matched to each
field's sign, and a headless 3D render path through OpenGL/EGL.

## [1.2.0] - 2026-06-21

New `warpax.bondi`: Bondi four-momentum radiated flux, Newman-Penrose Weyl
scalars, and peeling falloff $\Psi_n \sim r^{-(5-n)}$ at null infinity.

## [1.1.1] - 2026-06-20

Reproduction artifacts and docs for the source-shell paper
(arXiv:2605.25417). No API changes.

## [1.1.0] - 2026-06-10

Frame-independent certification.

- `warpax.certify(metric)`: one-call all-observer certification from the
  Hawking-Ellis eigenvalue test of $T^a{}_b$, valid at all warp speeds
  including superluminal $v_s \ge 1$.
- Closed-form Type-I worst observer, cross-checked against the BFGS optimizer.
- Rigorous geodesic-integrated ANEC through a symplectic null integrator that
  conserves $g_{ab}k^ak^b$ where the adaptive integrator drifts off the cone.
- Shift vorticity controls the wall type: the Type-IV imaginary pair is linear
  in it, and shear amplifies but does not open it.
- Wall NEC deficit and curvature invariants as power laws in $v_s$, splitting
  by vorticity ($v_s^2$ vortical, $v_s^4$ irrotational).
- New `metrics.GarattiniMetric`, and cross-construction verification of the
  Fuchs, WarpShell and Garattini-Zatrimaylov constructions.

## [1.0.0] - 2026-05-26

First stable release: Hawking-Ellis classification with exact Type-I margins,
multistart BFGS observer optimization, source-first S-/T-shell construction from
the Einstein constraints, a pure-JAX autodiff curvature chain, WarpFactory /
Cactus / EinsteinFields IO, and a differentiable metric-design API.

[1.4.0]: https://github.com/anindex/warpax/releases/tag/v1.4.0
[1.3.0]: https://github.com/anindex/warpax/releases/tag/v1.3.0
[1.2.0]: https://github.com/anindex/warpax/releases/tag/v1.2.0
[1.1.1]: https://github.com/anindex/warpax/releases/tag/v1.1.1
[1.1.0]: https://github.com/anindex/warpax/releases/tag/v1.1.0
[1.0.0]: https://github.com/anindex/warpax/releases/tag/v1.0.0
