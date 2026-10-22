# Changelog

## [Unreleased]

Regenerate `results/` and the figures with `./reproduce_all.sh --stage core`.

### Added

- `warpax.numerics` cubic Hermite interpolation (`cubic_interp`,
  `hermite_cubic`, `local_cubic_slopes`, `not_a_knot_slopes`) with knot slopes
  computed once per profile.
- `eulerian_normal`, the one Eulerian-frame construction; it returns NaN where
  `g^{00} >= 0`.
- `weyl_tensor` in `warpax.geometry`.
- `sources(r)` on `SShellMetric` and `TShellMetric`, used by the constraint
  residual, the design loss, and the parameter sweep.
- `GridSpec.volume_weights_array`, computed on demand.
- `QIResult.kernel_weight`, `n_undecided` in `single_frame_miss`, and
  `uncertain_reason` from the mpmath classifier.
- `probe_shape` and `probe_bounds` on `design_metric`.
- Benchmarks for `certify`, the frame-free grid, `verify_grid`, the clustered
  grid builder, and a retrace probe; `make bench-compare` fails on a
  regression.

### Changed

- Dependencies move to JAX 0.12, beartype 0.23, jaxtyping 0.3.11, NumPy 2.5,
  SciPy 1.18, Matplotlib 3.11, Manim 0.22, ruff 0.17, and uv 0.13. mpmath
  stays below 1.4 until SymPy lifts its cap; Pillow stays below 13 until Manim
  supports it. interpax is no longer a dependency and the `design` extra is
  empty.
- The Eulerian null-energy margin is the exact minimum of `T(k,k)` over null
  `k` with `-g(k,n) = 1`; `verify_grid` and `verify_point` fold it into the
  optimizer's NEC margin, and single-frame miss rates count every
  Hawking-Ellis type.
- `WarpShellPhysical` evaluates its Schwarzschild factors at
  `max(r, R_1 - smooth_width)`, is C2 across the inner blend, and requires
  `r_s_param < R_1 - smooth_width`. Its ground-truth labels record the negative
  energy the flat exterior forces in the outer blend.
- The `hard_bound` observer strategy runs BFGS on the objective composed with
  the radial projection, so the full rapidity cap is reachable.
- `weyl_squared(riemann, ricci, ricci_scalar, metric, metric_inv)` contracts
  the Weyl tensor directly.
- `asymptotic_flatness_report` samples the six axis directions and reports
  every component under `components`; `diagonal` and `shift` are views of it.
- `GridSpec.bounds` is an immutable tuple; curvature grids run through a
  coordinate kernel, so a new time slice does not recompile, and `certify`
  evaluates only the stress-energy and metric fields.
- The classifier, the frame-free margins, the kinematic grids, and the margin
  sweeps run as compiled kernels; the shell metrics store scalars as array
  leaves; the Bobrick-Martire taxonomy scan is batched.
- `WARPAX_JIT_CACHE=1` also lowers JAX's one-second minimum compile time.
- `ConstraintResult.satisfied` is an array.
- `interp_method="cubic"` is a C2 tensor-product B-spline, and importers
  reject non-positive lapses and non-positive-definite spatial metrics.
- Ford-Roman integrates in proper time out to `1000 tau0`, reports the captured
  kernel weight, and states its units (hbar = 1).
- `scripts/render_all_scenes.py` follows the Manim 0.22 command line; scripts
  share `_paths.py` and `_json_io.py`.

### Fixed

- `design_metric` raises the objective margin instead of lowering it, and
  constrained runs take optimizer steps.
- `WarpShellPhysical` had a derivative jump at `R_1`.
- The Eulerian NEC sampled six null directions, so a finite search could
  report a weaker NEC than the closed form.
- `hard_bound` applied the rapidity cap twice and reached only `0.76 zeta_max`.
- Eulerian-normal constructions fabricated observers on non-spacelike slices;
  the interval chain now refuses such boxes.
- The split-null admission in the classifier mixed stress and metric units; the
  50-digit check now flags Jordan structure already split by float64 rounding.
- Bondi: the null tetrad comes from the metric, both `Psi_4` paths use the Weyl
  tensor, the angular RMS uses the Gauss-Legendre weights, and the matter flux
  is labeled as such.
- Junction: `surface_stress_energy` extrapolates both sides to the shell,
  checks first-form compatibility and orientation, inverts the induced metric
  exactly, and returns NaN for null boundaries.
- Ford-Roman requires a timelike worldline and centers the kernel at the
  requested event.
- The TOV scan takes exact midpoint samples (fourth order for S-shell profiles)
  and returns NaN for trapped inputs.
- ADM Richardson extrapolation solves the order equation for arbitrary radii
  with a scale-aware gate.
- `weyl_squared` canceled catastrophically in Ricci-dominated regions.
- Null projection handles the double root at `g_00 = 0`, initial data use a
  scale-invariant degeneracy test, analytic worst observers are unit and
  future-directed, and each metric class supplies its own Killing vector
  (`d_t` for the stationary Fuchs and T-shell constructions, none for
  Garattini-Zatrimaylov).
- Loading a single time slice divided by zero.
- `certify` ran the frame-free certification twice, skipped proper-volume
  weighting on uniform grids, and passed boolean masks through float
  annotations; non-finite probes report as infeasible.
- The shell constraint residual used vacuum sources inside the matter shell,
  and the Bernstein basis had NaN derivatives at the interval endpoints.
- Transport diagnostics return NaN for static observers inside an ergoregion
  and for round trips that never reach the receiver.
- Sweeps record NaN for failed points and write checkpoints atomically; broad
  `except Exception` handlers are narrowed.
- Manim color stops were non-monotone for fields of one sign, and the
  alignment plot leaked a figure.
- `sanitize_nonfinite` handles NumPy float32 values.

### Removed

- `warpax.metrics._fuchs_legacy` (`FuchsShellProfiles`, `fuchs_shell_profiles`,
  `fuchs_input_stress_energy`).
- The aliases `WarpShellStressTest`, `warpax.optimization.OptimizationResult`,
  `null_round_trip_asymmetry`, and `benchmarks.alcubierre._shape_function`;
  the unused `n_ec_starts` argument of `sweep_transport`.
- `ProjectedBFGSSolver`, `optimize_wec_adaptive`, `_natario_d2n_dr2`, and
  `_standard_solver_unreliable_scalar`.
- `GridSpec.volume_weights`; use `volume_weights_array`.
- Unused visualization helpers (`freeze_ec`, `scene_velocity_rampdown`,
  `add_text_overlay`, `add_watermark`, `linear_ramp`, `sigmoid_ramp`,
  `constant_velocity`, `plot_comparison_grid`, `plot_alignment_histogram`,
  `plot_kinematic_comparison`, the `_color` and `alignment_plots` modules) and
  the per-scene copies of the `dark_diverge` colormap.
- The `smoke` pytest marker, three unused conftest fixtures, and eleven stale
  GPU xfail entries.

## [1.5.1] - 2026-09-24

- Corrected Alcubierre and Rodal curvature near the bubble center and
  weak-energy-condition boost thresholds.
- Added elastic-shell equilibrium, redshift, frame dragging, and leading
  centrifugal tides and central clock shifts.

## [1.5.0] - 2026-09-12

- Corrected the Natário laboratory shift and Fuchs covariant metric, with smooth
  profile joins and updated numerical results.
- Fixed Garattini–Zatrimaylov violation counts, NEC tolerances, and interval
  bounds. Selected finite-ray runs now resume after interruption and report
  their convergence limits.
- Shortened the documentation, clarified numerical guarantees, and tightened
  CI and release checks.

## [1.4.0] - 2026-08-31

Added LMI energy-condition tests, rational-certificate checks, interval
enclosures, proper-volume weights, and local extremum refinement. Fixed null
projection, proper-time weighting, convergence spacing, and junction limits.
Reduced repeated compilation and import work.

## [1.3.0] - 2026-06-22

Renamed Manim scenes, matched colormaps to field signs, and added headless
OpenGL rendering.

## [1.2.0] - 2026-06-21

Added Bondi four-momentum flux, Newman–Penrose scalars, and peeling diagnostics.

## [1.1.1] - 2026-06-20

Added reproduction artifacts and documentation for the source-shell paper.

## [1.1.0] - 2026-06-10

Added `certify`, Type-I observer formulas, symplectic null integration, speed
sweeps, and the Garattini–Zatrimaylov metric.

## [1.0.0] - 2026-05-26

First stable release: autodiff curvature, Hawking–Ellis classification, observer
optimization, source-first shells, external metric loaders, and metric design.

## Pre-1.0 history

| Version | Changes |
|---|---|
| 0.4 | Bernstein S-/T-shell profiles, constraint-derived potentials, compactness/thickness sweeps, separate analytical and smoothed Fuchs constructions, Lentz axis handling. |
| 0.3 | ADM split, source-consistency and TOV residuals, two-sided Israel junctions, ADM surface integrals, geodesic transport diagnostics. |
| 0.2 | Generalized eigenvalue fallback, bounded observer optimization, metric-design API, averaged-energy and quantum-inequality diagnostics, external metric readers. |
| 0.1 | Autodiff curvature, six warp metrics, classification and observer optimization, Diffrax geodesics. Accompanied [arXiv:2602.18023](https://arxiv.org/abs/2602.18023). |

[1.5.1]: https://github.com/anindex/warpax/releases/tag/v1.5.1
[1.5.0]: https://github.com/anindex/warpax/releases/tag/v1.5.0
[1.4.0]: https://github.com/anindex/warpax/releases/tag/v1.4.0
[1.3.0]: https://github.com/anindex/warpax/releases/tag/v1.3.0
[1.2.0]: https://github.com/anindex/warpax/releases/tag/v1.2.0
[1.1.1]: https://github.com/anindex/warpax/releases/tag/v1.1.1
[1.1.0]: https://github.com/anindex/warpax/releases/tag/v1.1.0
[1.0.0]: https://github.com/anindex/warpax/releases/tag/v1.0.0
