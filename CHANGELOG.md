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
  Rodal shift carries same-order shear with $f = 0$ — shear amplifies the
  imaginary pair that vorticity opens, but does not open one itself.
- **Invariant exoticity ranking and $v_s$ scaling laws**: a boost-invariant
  multi-axis figure of merit (NEC severity, Type-IV fraction, rigorous ANEC
  minimum) plus power-law fits of the NEC severity, recovering the universal
  $v_s^2$ scaling (`run_exoticity_ranking.py`).
- **Universal $v_s$ scaling of the wall curvature invariants**: the wall-peak
  Kretschmann, Weyl-squared and Ricci-squared invariants follow clean power
  laws in the warp speed, split by shift vorticity — the vortical walls grow as
  $v_s^2$, the irrotational Rodal wall as $v_s^4$ ($R^2 \ge 0.996$;
  `run_curvature_scaling.py`).
- **Santiago-Schuster-Visser no-go made quantitative**: the wall NEC
  deficit follows the necessarily-quadratic $\min(\rho+p_i) = -C\,v_s^2$ form
  with a geometry-fixed coefficient $C>0$, making the no-go theorem a measured
  rather than asserted statement (`run_ssv_bound.py`).

### Adversarial audit fixes (June 2026)

A multi-agent adversarial audit (five independent finders, every claim
re-verified by three skeptics against 50-digit or analytic oracles) confirmed
fourteen defects; all are fixed, each with a regression test:

- **Near-vacuum gate ignored imaginary parts**: a stress tensor with a purely
  imaginary spectrum (pure momentum flux, eigenvalues $\pm iq$ — genuine
  Type IV at any $q$) was absorbed as vacuum Type I. The gate now tests the
  eigenvalue modulus, in both the float64 and the 50-digit classifier.
- **Relative imaginary tolerance now has a scale floor**: the $3\times10^{-3}$
  split-degenerate tier exists to absorb eigensolver noise at large $\|T\|$
  (WarpShell), but at small $\|T\|$ it was absorbing genuine weak Type-IV
  physics (the Alcubierre far-field tail, $|{\rm Im}|\sim10^{-8}$, certified
  complex at 50 digits). The tier engages only above $\|T\|\sim10^6$; below it
  the absolute tier governs. Full-grid type fractions shift accordingly;
  wall-restricted fractions are unchanged.
- **Scale-aware violation gate**: the violated/satisfied threshold now carries
  a relative term ($10^{-12}\,\max|\lambda|$) so float64 roundoff at
  $\|T\|\gtrsim10^6$ cannot mint violations — about a quarter of the previous
  WarpShell "violated" counts were noise, not physics.
- **Geodesic initial conditions were past-directed**: `timelike_ic`/`null_ic`
  returned the $u^0<0$ root, so the published tidal/blueshift geodesics ran
  backwards and never met the bubble (peak tidal exactly 0). Both now select
  the future-directed root; the regenerated geodesics cross the wall
  (min $r_s\approx0.05$ at $v_s=0.5$), and the round-trip asymmetry and
  blueshift diagnostics are recomputed on the future cone.
- **Null projection picked the reflected root** where $g_{00}>0$ (superluminal
  interiors): `_project_to_null` now picks the root closest to identity, so an
  exactly-null tangent is returned unchanged.
- **diffrax result codes were silently mapped to success**: `int()` on a
  diffrax 0.7.2 `EnumerationItem` raises, and the fallback declared every
  failed integration complete. A shared converter now resolves codes robustly
  (unknown maps to *not* complete) and the termination-reason table matches
  diffrax 0.7.2; `awec()` accepts real `GeodesicResult`s again.
- **ANEC default is now `tangent_norm='null_projected'`** (the rigorous path);
  `'renormalized'` remains available. Paper scripts already passed it
  explicitly, so published ANEC numbers are unaffected.
- **Shell `total_mass` was a static jit leaf**, recompiling the curvature chain
  for every candidate in `optimize_shell`; it is now an array leaf.
- **Design optimizer was a silent no-op**: `ShapeFunctionMetric` validation
  raised under tracing and degraded BFGS to its random seeds; validation is
  now trace-safe and the optimizer actually descends.
- **Sweep checkpoints scrambled cells in parallel mode**: checkpoints now keep
  full positional lists, so an interrupted parallel sweep reloads correctly.
- **Einstein Toolkit loader used the wrong shift convention** (ADMBase
  `betax/betay/betaz` are contravariant); fixed with the correct
  $g_{0i}=\gamma_{ij}\beta^j$ assembly.
- **Multistart PRNG keys are now distinct** per probe point and condition in
  the EC constraint helpers (previously every point reused `PRNGKey(42)`).
- **`results/*.json` are now valid RFC 8259 JSON**: all scripts serialize
  through a shared helper that maps non-finite floats to `null`.

The full pipeline was regenerated after these fixes. The headline numbers
survive untouched: the invariant benchmark (Rodal 72/73% Eulerian misses,
Type-IV wall fractions, peak deficits) is bit-for-bit identical, and all 15
audited paper macros are unchanged. What did move: the rigorous ANEC values
(the geodesics now actually cross the bubble) — Rodal $-0.0041$, Van den
Broeck $-0.045$, Natário $-0.051$ (now symplectically certified, no fallback),
Alcubierre $-0.159$ — so the exoticity index becomes Rodal 0.014 / Natário
0.661 / VdB 0.629 / Alcubierre 0.997 (Rodal ~70x below baseline, mildest by
one to two orders); the tidal/blueshift validation figures show genuine
bubble crossings (frequency ratio matches $1/\gamma(v_s)$ to six figures);
and the matched-benchmark wall Type-I percentages drop a few points as weak
Type-IV points are no longer absorbed.

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
- Corrected the goodness-of-fit metric for the through-origin (no-intercept)
  fits — the $f=\kappa\,\omega$ vorticity fit and the SSV $-C\,v_s^2$ fit — to
  use the uncentered total sum of squares. The previous mean-centered form
  understated $R^2$ and produced a spurious negative value for the poorly-fit
  Van den Broeck branch; fitted coefficients and every cited number are
  unchanged.

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
- Added analytic physics-fidelity sentinels independent of the code's own
  output: Schwarzschild $C^2 = K$ and Chern-Pontryagin $= 0$, a conformally-flat
  $C^2 = 0$ check that exercises the three-term Gauss-Bonnet cancellation, the
  Type-IV imaginary-tolerance threshold boundary, tetrad orthonormality at
  superluminal speeds, and a negative-ANEC wall sign check.
- Repaired the GPU expected-failure registry (stale node IDs remapped to the
  current test tree) with a staleness guard, and pinned `mpmath` as a direct
  dependency.
- The reproduction pipeline enables the persistent JIT cache by default
  (`WARPAX_JIT_CACHE=1` with a 0.05 s min-compile-time floor); warm stages
  skip recompilation entirely (~60% less CPU on the velocity-sweep smoke).
  Dependency floors aligned with current releases (jaxtyping 0.3.10,
  matplotlib 3.10.9, ruff 0.15.16); `mpmath` stays `<1.4` because sympy 1.14
  (latest) requires it.
- New regression net from the audit: `tests/test_bug_hunt_regressions.py`
  (near-vacuum modulus gate, scale-aware violation gate, imag-tolerance
  sentinel with a 50-digit agreement check on a superluminal slab, timelike
  tiebreak across 30 decades of $\|T\|$, DEC bound-vs-optimizer parity), plus
  future-directed IC, null-projection, result-code, checkpoint, and
  ET-convention tests in their home modules.
- Test-suite hardening sweep: every test module reviewed against its source for
  vacuous or circular assertions, stale baselines, and duplicate coverage.
  Highlights: stress-energy symmetry tests now check the pre-symmetrization
  Einstein tensor (the old T-symmetry assertion was made vacuous by the
  explicit `0.5*(T+T.T)` projection in the source); WarpShell density
  baselines re-pinned against the live pipeline (the stored values had
  drifted, passing only because `atol` exceeded the values); the C1-vs-C2
  regression now pins lapse/Riemann values that genuinely distinguish the
  two smoothness classes; the fibonacci starter-pool golden is regenerated
  and actually compared (the old snapshot was stored but never asserted
  against); the parity-golden fixture gains a per-group key-count sentinel
  and `capture_goldens.py` fails on any skipped block (no more silent
  shrink); the mpmath imag-tolerance sentinel re-targeted above the 1e6
  scale floor where the float64 tier genuinely absorbs the split; sentinel
  ANEC/AWEC/transport values pinned to measured numbers instead of
  isfinite checks; `evaluate_loss` threads distinct PRNG keys per probe
  point (same fix the EC-constraint path already had) with regression
  coverage; TOV residual tests now exercise the radial pressure-gradient
  term, and `tov_residual_from_metric` gains first coverage; stale
  "necessary but not sufficient" comments about the Type-I DEC eigenvalue
  bound corrected (it is necessary and sufficient at Type I); ~45 duplicate
  or tautological tests merged or removed with their assertions preserved
  in the surviving tests; `test_curvature_scaling.py` and
  `test_ssv_bound.py` are now standalone-runnable (no hidden sys.path
  dependency on collection order).
- CI restructured: the full fast suite runs on the primary Python with the
  beartype smoke; the other supported versions run the critical core tier
  (classification, frame-free path, verifier, regression nets, parity
  goldens, benchmarks) for cross-version compatibility.
- Diagnostic/ablation tables are now script-emitted from cached results
  (`scripts/emit_diagnostic_tables.py`, wired into the figures stage of
  `reproduce_all.sh`): the uniform-grid missed-violation table, the
  Hawking-Ellis type breakdown, the `N_starts` ablation, the C1-vs-C2
  WarpShell smoothness comparison, and the Richardson convergence table all
  trace to `results/*.json` and cannot drift.
- The wall-restricted analysis records the full-grid `max|Im lambda|` per
  metric, and the Richardson convergence study excludes the exact coordinate
  center, where the C-infinity regularization guard (not physics) dominates
  the autodiff derivatives at odd grid sizes; the convergence JSON also keeps
  the non-monotone-fallback flag so assumed-order entries are labelled.

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
