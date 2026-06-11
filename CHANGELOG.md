# Changelog

All notable changes to `warpax` are recorded here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Pre-1.0
history is summarized in `docs/explanation/release_notes.md`.

---

## [1.1.0] - 2026-06-10

### Frame-independent certification

- `warpax.certify(metric)`: one-call all-observer, all-velocity energy-condition
  certifier built on the Hawking-Ellis eigenvalue test of $T^a{}_b$; it never
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
  Bobrick-Martire/Fell-Heisenberg WarpShell, and the Garattini-Zatrimaylov de
  Sitter bubble (alongside Alcubierre and Rodal)
  through the same frame-independent certifier with a wall-resolution gate
  (`run_construction_verification.py`). The source-first S-/T-shells stay in the
  construction registry as a toolkit but are now certified in the companion note
  (arXiv:2605.25417), not here, keeping the two contributions disjoint.
- **Garattini-Zatrimaylov de Sitter bubble** (`metrics.GarattiniMetric`): a
  warp bubble on a de Sitter background with a faithful closed-form `.symbolic()`
  that reduces exactly to Alcubierre at $H=0$. Certified at its matched
  $v_s = H R$ averaged-condition regime, the wall is Hawking-Ellis Type IV and
  the Eulerian frame misses ~63% of the wall weak-energy violations even though
  the de Sitter background renders the Eulerian density non-negative.
- **Vorticity -> Type-IV mechanism**: the imaginary part of the Hawking-Ellis
  Type-IV eigenvalue pair is shown to be linear in the shift vorticity,
  $f = \kappa\,\omega$ with slope $\kappa \approx 0.06$ in the controlled
  construction (`analysis.vorticity_type_analytic`,
  `derive_vorticity_type.py`; controlled pure-rotation fit $R^2 = 1$).
- **Shear amplification of the Type-IV pair quantified**: the cross-metric
  validation now records the shift expansion, shear, shear-to-vorticity ratio,
  and the excess of the measured $f$ over the pure-rotation prediction
  $\kappa\,\omega$ at the matched wall sample (`excess_over_pure_rotation` in
  `analysis.vorticity_type_analytic`). The excess (x2.1 Van den Broeck, x3.7
  Alcubierre, x31.8 Natário) grows with $\sigma/\omega$; the zero-expansion
  Natário wall makes the symmetric gradient pure shear, and the irrotational
  Rodal shift carries same-order shear with $f = 0$: shear amplifies the
  imaginary pair that vorticity opens, but does not open one itself.
- **Invariant exoticity ranking and $v_s$ scaling laws**: a boost-invariant
  multi-axis figure of merit (NEC severity, Type-IV fraction, rigorous ANEC
  minimum) plus power-law fits of the NEC severity, recovering the universal
  $v_s^2$ scaling (`run_exoticity_ranking.py`).
- **Universal $v_s$ scaling of the wall curvature invariants**: the wall-peak
  Kretschmann, Weyl-squared and Ricci-squared invariants follow clean power
  laws in the warp speed, split by shift vorticity: the vortical walls grow as
  $v_s^2$, the irrotational Rodal wall as $v_s^4$ ($R^2 \ge 0.996$;
  `run_curvature_scaling.py`).
- **Santiago-Schuster-Visser no-go made quantitative**: the wall NEC
  deficit follows the necessarily-quadratic $\min(\rho+p_i) = -C\,v_s^2$ form
  with a geometry-fixed coefficient $C>0$, making the no-go theorem a measured
  rather than asserted statement (`run_ssv_bound.py`).

Various bug fixes and improvements were applied in this version.

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

[1.1.0]: https://github.com/anindex/warpax/releases/tag/v1.1.0
[1.0.0]: https://github.com/anindex/warpax/releases/tag/v1.0.0
