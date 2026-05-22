# Changelog

All notable changes to `warpax` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.3]

Wall-restricted energy-condition statistics. The full-domain Type-I
fraction is dominated by near-vacuum points outside the bubble wall, so
the classifier now tags those points and the analysis reports counts
restricted to the active wall region.

### Added
- `is_vacuum` field on `ClassificationResult` and `n_vacuum` on
  `ECGridResult`: flag near-vacuum points (max|Re lambda| < tol) so
  statistics can exclude them.
- Wall-restricted classification and miss-rate stats in
  `run_analysis.py`, conditioned on the active wall (f in [0.1, 0.9]).
- `--strategy` flag on `run_analysis.py` to select the hard-bound
  projected-gradient observer optimizer.
- `scripts/run_clustered_convergence.py`: wall-clustered convergence
  study on uniform vs cosh-stretched grids.
- `scripts/summarize_results.py`: compact summary of the result files.
- `tests/test_adm_superluminal.py`, `tests/test_classifier_vacuum_class.py`.

### Changed
- `run_analysis.py` benchmark table excludes Lentz (wall sub-grid at
  default parameters); Lentz is still evaluated as a qualitative check.
- CI runs the new ADM and vacuum-classifier tests.

---

## [0.4.2]

Docs housekeeping: warp-shell paper reproduction guide moved out of the
README into its own how-to page.

### Added
- `docs/how-to/reproduce_warpshell_paper.md`: self-contained per-figure,
  per-claim reproduction guide.
- `scripts/verify_fuchs.py`, `scripts/verify_proposals.py`: standalone
  entry points for Fuchs and cross-proposal verification reports.
- `results/fuchs_verification_report.json`,
  `results/proposals_verification_report.json`: cached verification outputs.

### Changed
- `mkdocs.yml`: added `reproduce_warpshell_paper.md` to the nav.

---

## [0.4.1]

Fuchs construction split, Lentz on-axis NaN fix, gauge-invariant
transport in the sweep, and a small suite of per-metric verification
scripts.

### Added
- `warpax.metrics.fuchs_construction`: canonical Fuchs metric with
  iterative Gaussian-kernel smoothing (steps 1-5 of the construction).
  `fuchs_default()` returns this canonical form.
- `warpax.metrics._fuchs_legacy`: the pre-smoothing analytical
  intermediate (constant-density + TOV pressure, steps 1-2) retained
  for diagnostic comparison.
- `transport_invariant` (delta_tau) field on `SweepPoint`, populated
  during T-shell sweeps when `max|beta^x| > 1e-6`.
- `scripts/_radial_sweep.py`: shared single-point evaluator and
  aggregator for the verification scripts below.
- `scripts/run_fuchs_canonical.py`, `run_lentz.py`, `run_landscape.py`
  (Alcubierre, Natario, Van den Broeck), `run_sshell_sweep.py`,
  `run_v0_ablation.py`, `run_tshell_convergence.py`,
  `run_delta_tau_scan.py`, `run_anec_profiles.py`: per-metric
  verification entry points.

### Fixed
- `LentzMetric.shift` and `shape_function_value` now floor the
  perpendicular radius with `+ 1e-60` inside the `sqrt`. The autodiff
  derivative of `sqrt(y**2 + z**2)` at `y = z = 0` was returning NaN
  and silently invalidating Lentz curvature-chain results.
- `sweep_transport` no longer swallows arbitrary exceptions from
  `null_round_trip_asymmetry`; narrows to `(ValueError, RuntimeError,
  FloatingPointError)` and emits a `RuntimeWarning` so failures surface
  rather than appearing as `NaN` transport.

### Changed
- `warpax.metrics.FuchsMetric` now refers to the canonical
  Gaussian-smoothed construction; legacy users should explicitly
  import `_FuchsAnalytical` from `warpax.metrics._fuchs_legacy`.
- Phase-diagram colorbar/title relabelled to "Shift proxy
  $\\max|\\beta^x|$" / "Coordinate shift magnitude" to make the
  gauge-dependent character explicit.

### Removed
- `warpax/src/warpax/metrics/fuchs.py`, superseded by
  `fuchs_construction.py` and `_fuchs_legacy.py`.

## [0.4.0]

Source-first warp shell construction, parameter sweep, and phase-diagram
visualization.  Introduces two new shell ansatze (S-shell Class I, T-shell
Class II) with Bernstein-parameterized source profiles and constraint-derived
metric potentials, a 2D design-space sweep over compactness and thickness, and
publication-quality phase diagram plotting.

### Added
- `warpax.metrics.SShellMetric` + `sshell_default()` + `sshell_from_profiles()` -
  source-first Class I shell with flow-orthogonal matter, non-unit lapse, and
  isotropic pressure.  Metric potentials derived from the Hamiltonian constraint
  and anisotropic TOV equilibrium.
- `warpax.metrics.TShellMetric` + `tshell_default()` + `tshell_from_profiles()` -
  source-first Class II shell with tilted matter flow.  Shift `beta^x` derived
  from the momentum constraint (not prescribed), addressing the Barzegar et al.
  (arXiv:2602.16495) source-consistency critique.
- `warpax.metrics.sshell_profiles` - S-shell source profile factories:
  `constant_density_profiles`, `parabolic_density_profiles`,
  `bernstein_density_profiles` with `SShellSourceProfiles` container.
- `warpax.metrics.tshell_profiles` - T-shell velocity profile factories:
  `constant_velocity_profiles`, `parabolic_velocity_profiles`,
  `bernstein_velocity_profiles` with `TShellSourceProfiles` container.
- `warpax.constraints.constraint_solver` - pure-JAX S-shell constraint solver
  using `jax.lax.linalg.tridiagonal_solve`.
- `warpax.constraints.tshell_solver` - pure-JAX T-shell constraint solver
  with momentum-constraint-derived shift.
- `warpax.optimization.basis` - Bernstein polynomial basis with compact support
  for profile parameterization: `bernstein_basis`, `bernstein_eval`,
  `pack_theta`, `unpack_theta`, `ShellCoeffs`.
- `warpax.optimization.loss` - multi-objective loss combining constraint
  residuals, EC penalty, tidal force, transport proxy, and ADM mass.
- `warpax.optimization.ec_constraints` - EC soft penalty (softplus) and hard
  feasibility check with per-condition margin arrays.
- `warpax.optimization.optimizer` - Nelder-Mead optimizer driver for shell
  profile optimization.
- `warpax.optimization.sweep` - `sweep_transport(...)` sweeps (compactness,
  thickness_ratio) for T-shell or S-shell, certifying EC admissibility at
  every point.  Returns a `SweepResult` with `.to_grids()`, `.save()`,
  `.load()` helpers.
- `warpax.visualization.plot_phase_diagram` and `plot_phase_summary`:
  single-panel transport heatmap with EC boundary + hatching, and a 2x2
  summary (transport / EC margin / constraint residual / tidal).
- `examples/10_phase_diagram.py` end-to-end demo (`--full` for
  paper-resolution 20x15 sweep).

### Fixed
- Sweep failures used to swallow the exception.  They now warn with the
  failing `(compactness, thickness)` point and the exception.
- Constraint-residual probe grid used hard-coded `+/- 0.5` offsets, which
  inverted for thin shells.  Now uses 2% of the shell width.
- `_save_or_return` returned a closed figure handle when `save_path` was
  given.  Now returns `None`.
- `pcolormesh` used `shading="gouraud"` on cell-centred sweep data,
  smearing the EC boundary.  Now uses `shading="nearest"`.

### Removed
- `compute_invariant_transport()` built profiles via a different factory
  than the rest of the sweep code, so the metrics it produced did not
  match what `sweep_transport` evaluated.  No external callers.
- Unused `weights` parameter from `sweep_transport()`.

## [0.3.1]

Cleanup release. Removes dead test scaffolding, fixes a spot bug in
`optimize_sec`, tightens a loose tolerance, and merges duplicated fuchs
test modules. No public-API changes.

### Fixed
- `optimize_sec(...)` crashed with `NameError: _resolve_backend is not
  defined`. The helper had been deleted but two call-sites survived.

### Removed
- v1 API-surface tests and their golden fixtures (`test_v1_api_surface.py`,
  `v1_api_surface_v1_0.json`, `v1_api_defaults_v1_0.json`,
  `docs/reference/api_defaults.md`).
- Tests that exercised infrastructure rather than physics:
  `test_t_ab_symmetry`, `test_jit_cache`, `test_beartype_optin`,
  `test_reproduce_cpu_pin`, `test_gpu_baseline_harness`,
  `test_precision_parity`.
- The `precision='fp32_screen+fp64_verify'` path from
  `evaluate_curvature_grid` and its benchmark. fp32 screening did not save
  wall time at production grid sizes.

### Changed
- Merged `test_fuchs.py` + `test_fuchs_hardened.py` into `test_fuchs_metric.py`
  (33 tests; dropped 4 that only checked `isfinite`).
- Renamed `test_integration_audit.py` to `test_admissibility_smoke.py` (later
  removed in v0.4.0 as a duplicate of `test_adm.py` + `test_adm_constraints.py`).
- Schwarzschild tidal-eigenvalue tolerance `rtol=0.10` → `1e-6` (the
  analytical answer is exact).
- Pruned the sm_120 GPU xfail registry to entries that still fail.
- Trimmed revision-history comments from public docstrings.

## [0.3.0]

Source-consistent warp shell infrastructure: 3+1 ADM decomposition, hardened
constraint residuals, Fuchs metric, surface-integral ADM mass, Israel junction
upgrade, geodesic-based transport diagnostics, and source-consistency checks.

### Added
- `warpax.geometry.adm_split(metric, coords)` - extract (alpha, beta^i, gamma_{ij},
  K_{ij}) from any 4D metric via JAX autodiff. Returns an `ADMSplit` named
  tuple.
- `warpax.constraints.source_consistency.stress_energy_residual(metric, coords,
  T_input)` - compute DeltaT = T_input - G/(8pi) for source-consistency validation.
- `warpax.metrics.FuchsMetric` + `fuchs_default()` - Fuchs et al. (CQG 2024)
  constant-velocity subluminal warp shell with paper-matched parameters
  (v_s=0.01, R_1=10, R_2=20, r_s=5).
- `warpax.exceptions` - domain-specific exception hierarchy:
  `ConstraintViolationError`, `TOVInconsistencyError`,
  `JunctionDiscontinuityError`, `AsymptoticFalloffError`,
  `TransportUndefinedError`.
- `examples/09_admissibility_diagnostics.py` - end-to-end admissibility
  report on the Fuchs warp shell exercising all new modules.
- `tests/test_adm_constraints.py` - 9 ADM decomposition and constraint residual
  tests against Minkowski, Schwarzschild, and WarpShell analytical solutions.

### Changed
- `constraints.residuals` - full 3+1 decomposition via `adm_split()` (was
  assuming K=0); covariant momentum divergence D_j(K^j_i - delta^j_i K) via
  autodiff; scale-invariant normalization with floor=1.0 for vacuum stability.
- `tov.residuals` - `jax.grad` replaces finite differences for dp_r/dr;
  new `tov_residual_from_metric()` auto-extracts Phi' from g_{tt}.
- `adm.mass` - proper surface integral with Gauss-Legendre angular quadrature
  (was single-point trace estimate); new `adm_mass_richardson()` for convergence
  verification at multiple radii.
- `transport.diagnostics` - all three observables are now geodesic-based:
  `null_round_trip_asymmetry` uses the Diffrax integrator,
  `geodesic_deviation_diagnostic` uses tidal eigenvalues,
  `blueshift_hazard` fires null geodesics in 6 directions (were returning 0).
- `junction.darmois.surface_stress_energy` - two-sided Israel jump formulation
  [K_{ab}] = K+_{ab} - K-_{ab} with averaged induced metric (was single-sided).
- Bumped version to 0.3.0.

### Fixed
- Normalized constraint residuals returning eps ~1 for vacuum Schwarzschild due
  to 0/eps division; fixed with floor=1.0 normalization denominator.

## [0.2.0]

Major revision addressing the CQG-115130 reviewer report and extending the
toolkit with new analysis capabilities, classifier hardening, observer-coverage
improvements, a metric design API, and additional quantum/averaged modules.

### Added
- `warpax.classify.bobrick_martire(metric)` - Bobrick-Martire Class I/II/III
  taxonomy classifier.
- `warpax.junction.darmois(metric, boundary_fn)` - Darmois / Israel
  junction-condition checker.
- `warpax.quantum.ford_roman(metric, worldline, tau0, sampling='lorentzian')` -
  Ford-Roman quantum inequality evaluator with constant `C = 3/(32 pi^2)` pinned
  to Fewster 2012.
- `warpax.averaged.anec(metric, geodesic, tangent_norm='renormalized')` - ANEC
  line integral along null geodesics with `geodesic_complete: bool` flag.
- `warpax.averaged.awec(metric, geodesic, tangent_norm='renormalized')` - AWEC
  line integral along timelike geodesics.
- `warpax.design.design_metric(shape, objective, constraints, strategy, n_starts)` +
  `OptimizationReport` - constrained-BFGS shape-function optimizer with sigmoid
  reparameterization. Multistart n=16 with per-start `jax.random.fold_in`.
- `warpax.design.{ec_margin_objective, averaged_objective, quantum_objective}` +
  `OBJECTIVE_REGISTRY` - EC-margin objective library for the optimizer.
- `warpax.design.{bubble_size_constraint, velocity_constraint, boundedness_constraint}`
  + `ConstraintResult` + `CONSTRAINT_REGISTRY` - signed-margin constraint library.
- `warpax.design.ShapeFunctionMetric(ADMMetric)` - ADMMetric subclass wrapping a
  `ShapeFunction` with construction-time `verify_physical` gate.
- `warpax.design.ShapeFunction.{spline,bernstein,gmm}(params)` - differentiable
  shape-function basis library (cubic B-spline + Bernstein polynomial + GMM).
- `warpax/examples/08_metric_design.py` - reproduces Alcubierre tanh with
  24-knot cubic B-spline at relative error `< 1e-4`.
- `classify_hawking_ellis(..., backend_eig='scipy_callback')` kwarg + `_std_eig_callback`
  (scipy.linalg.eig wrapped in `jax.pure_callback(..., vmap_method='sequential')`)
  for sm_120 GPU workaround; 3-tier precedence via `_resolve_backend_eig`.
- Genuine-fp32 Pass-1 compute in
  `evaluate_curvature_grid(..., precision='fp32_screen+fp64_verify')`.
- `optimize_{nec,wec,sec,dec}(..., pool_composition='blended')` kwarg for
  blended pool-composition (1-of-16 neighbor seed + 15 diverse starts).
- `optimize_{nec,wec,sec,dec}(..., lattice_generator='fibonacci')` kwarg +
  `_fibonacci_lattice.py` helper implementing Fibonacci+Gaussian hybrid lattice.
- `optimize_{nec,wec,sec,dec}(..., warm_start='spatial_neighbor',
  neighbor_fraction=1/16)` - blended pool (1-of-16 starts seeded from grid
  neighbor's worst-observer).
- `optimize_{nec,wec,sec,dec}(..., starts='fibonacci+bfgs_top_k')` - Fibonacci
  lattice + BFGS-top-k starter pool.
- `optimize_{nec,wec,sec,dec}(..., strategy='hard_bound')` - projected-gradient
  BFGS solver with radial-projection step override.
- `optimize_dec(..., mode='per_subcondition_min')` - evaluates DEC as min over
  three independent BFGS multistarts.
- `evaluate_curvature_grid(..., auto_chunk_threshold=N)` caps grid-eval memory
  via `lax.map` dispatch.
- Persistent JIT compilation cache via `WARPAX_JIT_CACHE` env var;
  version-salted path prevents cross-version/cross-backend poisoning.
- `pytest --gpu-baseline` harness + sm_120 xfail registry for CPU-GPU
  delta-report.
- `make bench-gpu` Makefile target + asv CUDA matrix row.
- `warpax.energy_conditions.classification_mpmath` module for post-hoc
  arbitrary-precision (50-digit) Hawking-Ellis classification.
- `cond_V` and `uncertain` keys in `classify_hawking_ellis_mpmath` for
  Bauer-Fike sensitivity diagnostic.
- Opt-in strict runtime type checking via `WARPAX_BEARTYPE=1` environment variable.
- `pytest-xdist>=3.6.0` added for `pytest -n auto` parallel test runs.
- `requirements-lock.txt` and `requirements-dev-lock.txt` for pip-only users.
- `solver='generalized'` kwarg to `classify_hawking_ellis` and
  `classify_mixed_tensor`; the generalized path solves `(T - lambda*g)v = 0` directly
  via `scipy.linalg.eig(T_ab, g_ab)` wrapped in `jax.pure_callback`.

### Changed
- Symmetrize stress-energy tensor at the pipeline exit (`T = 0.5 * (T + T.T)`
  in `geometry.stress_energy_tensor`).
- `classify_hawking_ellis` causal-character test now uses a relative-sign
  threshold normalized against max |v^T g v| instead of an absolute `tol`.
- Refreshed `tests/_gpu_xfail_registry.py` to current GPU baseline on Blackwell
  sm_120 (RTX 5090, CUDA 13.2, JAX 0.10.0; measured 2026-04-24).
- Bumped dependency floor pins: `jax>=0.10.0,<0.11.0`, `equinox>=0.13.6,<0.14.0`,
  `numpy>=2.2.0,<3.0.0`, `matplotlib>=3.10.0,<4.0.0`, `beartype>=0.22.9,<0.23.0`,
  `scipy>=1.17.0`.
- Bumped dev-extras pins: `pytest>=9.0.0,<10.0.0`, `pytest-cov>=7.0.0,<8.0.0`,
  `ruff>=0.15.0,<1.0.0`.
- Added `Programming Language :: Python :: 3.14` classifier.

### Fixed
- GPU-speedup scope caveat added to manuscript Sec. Scope-and-limitations.
- Close pre-v0.1.1 WarpShell cache non-determinism. Add `--deterministic` CLI flag
  to `scripts/run_analysis.py` that pins `JAX_PLATFORMS=cpu` +
  `XLA_FLAGS=--xla_cpu_enable_fast_math=false` + classifier `solver='standard'`.
  Regenerated `results/warpshell_vs0.5.npz` under the recipe.
  New `results/MANIFEST.txt` (41 entries, sha256sum-compatible) for provenance
  tracking.

## [0.1.1] - 2026-04-18

Revision addressing the CQG-115130 major-revision reviewer report.

### Added
- `WallRestrictedStats` type and `filtering` module for wall-restricted
  diagnostics (shape-function masks, determinant guards, Frobenius-norm masks).
- `shape_function_value(coords)` abstract method on every `ADMMetric` subclass;
  all six warp-drive metrics plus the Alcubierre benchmark implement it.
- `docs/` with `quickstart.md`, `custom_metric_tutorial.md`,
  `interpreting_ec_results.md`, and `ARCHITECTURE.md`.
- `examples/07_custom_warp_metric.py` rewritten against the filtering API.
- `scripts/generate_paper_tables.py` emits canonical `metric_metadata.json` and
  wall-restricted Table 8, Table 3 family.
- Additional ablation and investigation scripts under `scripts/`.
- `reproduce_all.sh` wires core computation, ablations, and figure generation into a single entry point.

### Changed
- Accompanying paper revised against CQG reviewer report:
  terminology, framing, Lentz/WarpShell segregation, wall-restricted narrative.

## [0.1.0] - 2026-03 (initial public release)

Initial public release accompanying arXiv:2602.18023. JAX-based observer-robust
energy-condition verification for warp-drive spacetimes: autodiff curvature
chain, Hawking-Ellis classification, BFGS observer optimization, geodesic
integration, six warp-drive metrics (Alcubierre, Lentz, Van den Broeck,
Natario, Rodal, WarpShell). See paper Sec. 3 for full method description.

---

[Unreleased]: https://github.com/anindex/warpax/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/anindex/warpax/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/anindex/warpax/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/anindex/warpax/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/anindex/warpax/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/anindex/warpax/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/anindex/warpax/releases/tag/v0.1.0
