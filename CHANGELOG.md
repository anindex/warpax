# Changelog

All notable changes to `warpax` are recorded here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Pre-1.0
history is summarized in `docs/explanation/release_notes.md`.

---

## [1.0.0] - 2026-05-26

Stable release: observer-robust energy-condition verification, source-first
shell construction, autodiff curvature analysis, and a metric-design API.

### Energy conditions

- Full Hawking-Ellis classification (Types I-IV); exact Type-I eigenvalue margins.
- Multistart BFGS observer optimization for NEC, WEC, SEC, and DEC.
- `verify_grid` ships both `vmap` and `lax.map`-chunked paths for memory safety.
- Optimizer supports `axis+gaussian` and Fibonacci-lattice starts.
- Smooth `tanh` rapidity caps plus a projected-gradient hard bound.
- Spatial-neighbor warm starts for grid sweeps.

### Source-first shells

- S-shell and T-shell derive metric potentials from the Hamiltonian and
  momentum constraints rather than prescribing the shift.
- Bernstein-parameterized source profiles via the standalone solvers in
  `warpax.constraints`.
- Fuchs construction ([CQG 2024, arXiv:2405.02709](https://arxiv.org/abs/2405.02709))
  with iterative Gaussian-kernel smoothing and a moving-average variant for
  exact-pipeline reproduction.

### Curvature, IO, design

- `compute_curvature_chain` and `evaluate_curvature_grid` are pure JAX with
  `equinox.filter_jit` caching.
- Source-consistency diagnostics, ADM mass via Gauss-Legendre angular
  quadrature, Israel junction checks (covariant $\nabla n$ and $h^{ab}$ trace),
  Ford-Roman quantum-inequality evaluator.
- IO layer: WarpFactory, Cactus, and EinsteinFields exports through a shared
  interpolated-base layer.
- Metric design: `design_metric`, `ShapeFunction`, `OBJECTIVE_REGISTRY`,
  `CONSTRAINT_REGISTRY`: define a differentiable shape function and search
  parameters that satisfy chosen energy conditions.

### Tooling

- Test fixtures, example scripts, and golden artifacts deterministic under
  `JAX_PLATFORMS=cpu` and the classifier's standard solver.
- Fast suite parallelizes with `pytest -n auto -m "not slow"` (732 passing, 11 skipped).
- Docs build under `mkdocs build --strict`; library targets Python 3.12+.

---

[1.0.0]: https://github.com/anindex/warpax/releases/tag/v1.0.0
