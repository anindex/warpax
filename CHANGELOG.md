# Changelog

All notable changes to `warpax` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- `pytest --gpu-baseline` harness + sm_120 xfail registry for CPU↔GPU
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
  `classify_mixed_tensor`; the generalized path solves `(T − λg)v = 0` directly
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
- GPU-speedup scope caveat added to manuscript §Scope-and-limitations.
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
- `reproduce_all.sh` wires core computation, ablations, figure generation, and
  the paper LaTeX build into a single entry point.

### Changed
- Paper text (`warpax_arxiv/main.tex`) revised against CQG reviewer report:
  terminology, framing, Lentz/WarpShell segregation, wall-restricted narrative.

## [0.1.0] - 2026-03 (initial public release)

Initial public release accompanying arXiv:2602.18023. JAX-based observer-robust
energy-condition verification for warp-drive spacetimes: autodiff curvature
chain, Hawking-Ellis classification, BFGS observer optimization, geodesic
integration, six warp-drive metrics (Alcubierre, Lentz, Van den Broeck,
Natário, Rodal, WarpShell). See paper §3 for full method description.

---

[Unreleased]: https://github.com/anindex/warpax/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/anindex/warpax/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/anindex/warpax/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/anindex/warpax/releases/tag/v0.1.0
