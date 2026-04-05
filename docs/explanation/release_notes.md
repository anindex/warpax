# Release Notes

This page summarises each warpax release with a narrative overview. For full
line-item details, see the canonical
[`CHANGELOG.md`](https://github.com/anindex/warpax/blob/main/CHANGELOG.md) in the
repository root.

## v0.2.0 (2026-04-25) - Major Revision + Toolkit Maturation

warpax v0.2.0 is a major revision addressing the CQG-115130 reviewer report
and extending the toolkit with new analysis capabilities, classifier hardening,
observer-coverage improvements, a metric design API, and additional
quantum/averaged modules. All v0.1.x defaults are preserved.

### Highlights

- **Correctness hardening.**
    - 50-digit mpmath verification infrastructure for post-hoc arbitrary-precision
      verification of Type-IV Hawking-Ellis classifications.
    - Projected-gradient BFGS optimiser via `optimize_*(strategy='hard_bound')`.
    - Per-subcondition DEC evaluator via `optimize_dec(mode='per_subcondition_min')`.
    - Relative-sign causal-character threshold in `classify_hawking_ellis`.
    - Bauer-Fike `cond(V)` diagnostic with `uncertain=True` flag.
    - Generalized eigenvalue classifier - `classify_hawking_ellis(...,
      solver='generalized')` solves `(T − λg)v = 0` directly via LAPACK QZ.
    - Deterministic reproduction - `scripts/run_analysis.py --deterministic` pins
      JAX backend and XLA flags for bit-exact cache regeneration.

- **Performance.**
    - Persistent JIT cache (`WARPAX_JIT_CACHE=1`); version-salted path.
    - `evaluate_curvature_grid(..., auto_chunk_threshold=N)` memory cap.
    - HLO fusion trace verifying the paper's §2.1 single-pass curvature claim.
    - 2-pass `fp32_screen+fp64_verify` screening pipeline.
    - `optimize_*(..., warm_start='spatial_neighbor')` grid warm-starting.
    - `optimize_*(..., starts='fibonacci+bfgs_top_k')` deterministic pool.
    - `pytest --gpu-baseline` harness with sm_120 xfail registry.
    - Blackwell sm_120 workaround - `warpax.config.sm_120_workaround` and
      `backend_eig='scipy_callback'` bypass cuBLAS and cuSolver failures.
    - GPU speedup measurement - honest Blackwell RTX 5090 benchmarks (geomean
      0.600x at observer-optimization inner-loop scale).

- **Community interop.**
    - WarpFactory `.mat` reader (`warpax.io.load_warpfactory`).
    - EinFields Orbax checkpoint reader (`warpax.io.load_einfield`).
    - Cactus HDF5 slice reader (`warpax.io.load_cactus_slice`).
    - Wall-clustered + 2-level-AMR grid families.
    - MkDocs-Material documentation site with Diataxis layout.
    - asv regression benchmark harness.

- **Research extensions.**
    - Bobrick-Martire Class I/II/III classifier (`warpax.classify.bobrick_martire`).
    - Darmois junction-condition checker (`warpax.junction.darmois`).
    - Ford-Roman quantum inequality evaluator (`warpax.quantum.ford_roman`).
    - ANEC / AWEC line integrals along timelike and null geodesics.
    - Differentiable shape-function metric design driver (`warpax.design`).

### Breaking changes

None. v0.2.0 is fully additive relative to v0.1.x.

## v0.1.1 (2026-04-18) - CQG Major-Revision Codebase Prep

Revision addressing the CQG-115130 major-revision reviewer report.
Introduced the `WallRestrictedStats` filtering API, the abstract
`shape_function_value(coords)` method on every `ADMMetric` subclass, the MkDocs
`docs/` documentation structure, and the `reproduce_all.sh` entry point.

## v0.1.0 (2026-03) - Initial Public Release

Initial public release accompanying arXiv:2602.18023. JAX-based observer-robust
energy-condition verification for warp-drive spacetimes.
