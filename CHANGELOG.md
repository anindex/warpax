# Changelog

All notable changes to `warpax` are recorded here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Pre-1.0
history is summarized in `docs/explanation/release_notes.md`.

---

## [Unreleased]

### Frame-independent certification

- `warpax.certify(metric)`: one-call all-observer, all-velocity energy-condition
  certifier built on the Hawking-Ellis eigenvalue test of $T^a{}_b$ — never
  constructs the Eulerian normal, so it is valid at all warp speeds including
  superluminal $v_s \ge 1$ (`energy_conditions.frame_free`).
- Type-IV physical-certification gate: three-solver agreement (`eig`, LAPACK
  `zggev` generalized pencil) plus a 50-digit `mpmath` recomputation, with a
  refinement-stability and tolerance-insensitivity check
  (`validate_superluminal_classification.py`).
- Closed-form Type-I worst observer $\sinh^2\zeta_{\rm th} = \rho/|\rho+p_i|$
  with principal-eigenvector boost direction, validated against the BFGS
  optimizer (`energy_conditions.worst_observer_analytic`).

### New physics analyses

- Velocity-resolved Hawking-Ellis type map across the luminal transition
  ($v_s \in [0.1, 2.5]$) and a matched-parameter, wall-resolved invariant
  benchmark with per-metric wall-clustered convergence (`run_velocity_sweep.py`,
  `run_invariant_verification.py`, `run_matched_benchmark.py`).
- Shift-vorticity analysis: the vorticity of the ADM shift controls the
  Hawking-Ellis type of the bubble wall (`analysis.shift_kinematics`,
  `run_shift_vorticity.py`).
- Averaged null energy (ANEC) line integrals for the retained metrics and a
  Ford-Roman quantum-inequality diagnostic (`run_anec_retained.py`,
  `run_quantum_inequality.py`).
- **Rigorous geodesic-integrated ANEC**: a structure-preserving symplectic
  null-geodesic integrator (`geodesics.symplectic`, Tao-2016 extended phase
  space with JAX autodiff, Yoshida-4) that conserves $g_{ab}k^a k^b$ to ~machine
  precision where the adaptive RK integrator drifts off the null cone by $O(0.1)$
  on long bubble crossings. `averaged.anec.anec_rigorous` returns the ANEC with
  an on-cone rigor witness $\max|g(k,k)|$ and a projection-corrected fallback
  for the strongest-shift walls (`run_anec_symplectic.py`). `ANECResult` now
  carries `max_abs_g_kk` and `null_preserved`.
- **Cross-construction all-observer verification**: a uniform adapter
  (`analysis.construction_adapter`) flows the Fuchs constant-velocity shell, the
  Bobrick-Martire/Fell-Heisenberg WarpShell, and the source-first S-/T-shells
  (alongside Alcubierre and Rodal) through the same frame-independent certifier
  with a wall-resolution gate (`run_construction_verification.py`).
- **Garattini-Zatrimaylov de Sitter bubble** (`metrics.GarattiniMetric`): a
  warp bubble on a de Sitter background with a faithful closed-form `.symbolic()`
  that reduces exactly to Alcubierre at $H=0$.
- **Vorticity -> Type-IV mechanism**: the imaginary part of the Hawking-Ellis
  Type-IV eigenvalue pair is shown to be linear in the shift vorticity,
  $f = \kappa\,\omega$ (`analysis.vorticity_type_analytic`,
  `derive_vorticity_type.py`; controlled pure-rotation fit $R^2 = 1$).
- **Invariant exoticity ranking and $v_s$ scaling laws**: a boost-invariant
  multi-axis figure of merit (NEC severity, Type-IV fraction, rigorous ANEC
  minimum) plus power-law fits of the NEC severity, recovering the universal
  $v_s^2$ scaling (`run_exoticity_ranking.py`).
- **Universal $v_s$ scaling of the wall curvature invariants**: the wall-peak
  Kretschmann, Weyl-squared and Ricci-squared invariants follow clean power
  laws in the warp speed, split by shift vorticity — the vortical walls grow as
  $v_s^2$, the irrotational Rodal wall as $v_s^4$ ($R^2 = 1$;
  `run_curvature_scaling.py`).
- **Quantitative Santiago-Schuster-Visser bound saturation**: the wall NEC
  deficit saturates the necessarily-quadratic $\min(\rho+p_i) = -C\,v_s^2$ form
  with a geometry-fixed coefficient $C>0$, making the no-go theorem a measured
  rather than asserted statement (`run_ssv_bound.py`).

### Performance & correctness

- De-serialized the two host-side Python loops in the grid energy-condition
  verifier (`energy_conditions.verifier`): the generalized-pencil fallback and
  the skip-Type-I optimizer scatter are now single fused JAX operations
  (bit-identical output, no per-point host callbacks/loops).
- Removed a redundant Christoffel autodiff from the curvature chain by threading
  the precomputed symbols into `riemann_tensor` (halves the per-point autodiff in
  the most-called kernel; bit-identical). Threaded $g^{-1}$ through the SEC/DEC
  optimizers and vmapped the multi-direction blueshift diagnostic.
- Fixed a scale-dependent timelike-eigenvector tiebreak in the Hawking-Ellis
  classifier that could mis-select the rest frame (and flip $\rho$'s sign) at
  very large $\|T\|$; the selection is now scale-invariant.
- Fixed a closest-approach selection in the null round-trip transport diagnostic
  that could pick a spurious re-approach for bubble-crossing geodesics (changes
  `delta_t_coord` for such geodesics by design; not an energy-condition number).
- Hardened NaN handling in the imaginary-eigenvalue summary (`nanmax`).

### Tooling

- New test modules cover the symplectic integrator and rigorous ANEC, the
  Garattini metric, the construction adapter, the vorticity->Type-IV mechanism,
  the curvature-invariant scaling, and the SSV bound saturation, plus a
  certified-output parity harness (`tests/test_parity_golden.py`) that pins
  every paper-feeding number against a golden snapshot. The fast suite now
  exceeds 950 tests across 34 modules.
- Paper numbers are regenerated from `results/*.json` by
  `scripts/emit_paper_numbers.py` and guarded in CI by
  `scripts/audit_paper_numbers.py` (every cited number traces to a cached
  result). New `make` targets: `lint`, `reproduce`, `numbers`, `audit-numbers`.
- Repaired the GPU expected-failure registry (stale node IDs remapped to the
  current test tree) with a staleness guard, and pinned `mpmath` as a direct
  dependency.

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
