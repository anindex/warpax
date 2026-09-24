# Architecture

warpax evaluates curvature from a metric, tests pointwise energy conditions,
and reports observer searches, interval bounds, and geodesic diagnostics.

## Curvature pipeline

```text
metric g_ab -> Christoffel -> Riemann -> Ricci -> Einstein -> T_ab = G_ab/(8 pi)
                                  \-> curvature scalars
```

`jax.jacfwd` differentiates the metric and Christoffel map; contractions give
Ricci and Einstein tensors. Automatic differentiation avoids finite-difference
truncation error, but floating-point error and metric regularity still matter.
Metrics are Equinox modules, so JIT and `vmap` can act on their parameters.

## Energy conditions

- `slemma.py` tests quadratic forms on the unit ball or sphere through
  `T_hat + sigma eta >= 0`. NEC uses a free multiplier; WEC uses a nonnegative
  multiplier; SEC applies WEC to the trace-reversed tensor. DEC requires WEC
  and a second ball inequality for `-T g^{-1} T`.
- These equivalences cover all Hawking-Ellis types without a rapidity cap.
  The numerical search uses binary64 and a reported noise floor. Its Boolean
  condition is frame independent; its margin magnitude uses a fixed tetrad.
- `certificate.py` checks sufficient rational certificates for supplied
  tensors. Certificate construction can be inconclusive, and exact arithmetic
  on rounded tensor entries does not bound curvature error.
- `interval_lmi.py` encloses pointwise tensor evaluations; `enclosure.py`
  bounds extrema over spatial domains by interval branch and bound. A point
  certificate alone does not establish a uniform multiplier on a box.
- `classification.py` labels the stress tensor. The optional generalized
  pencil solver and `classification_mpmath.py` help check ill-conditioned
  cases. Type-I eigenvalue slacks are available when a reliable rest frame exists.
- `optimization.py` searches a specified rapidity range with multistart BFGS.
  A negative contraction witnesses a violation; a nonnegative search result
  does not certify the whole observer cone. Rest-frame Type-I formulas and
  Eulerian rapidity caps are distinct diagnostics.

## Package map

| Package | Purpose |
|---|---|
| `geometry` | Metric interfaces, curvature, invariants, grid evaluation |
| `metrics`, `benchmarks` | Warp/shell metrics and reference spacetimes; see the [catalog](../reference/metric_catalog.md) |
| `energy_conditions` | Algebraic tests, classification, observer searches, exact certificates, interval bounds |
| `geodesics` | Diffrax geodesics, Jacobi deviation, symplectic null integration |
| `averaged`, `quantum` | Finite null-energy integrals and flat-space quantum-inequality diagnostics |
| `grids`, `numerics` | Wall-clustered grids, resolution checks, proper-volume weights, numerical helpers |
| `analysis`, `classify` | Comparisons, convergence, extrema, shift kinematics, construction adapters |
| `constraints`, `tov` | Einstein constraints, source residuals, legacy shell prescriptions, anisotropic TOV equations |
| `adm`, `junction`, `bondi` | Asymptotic mass, surface stresses, asymptotic diagnostics |
| `transport` | Geodesic deviation, observer-dependent blueshift, coordinate-time asymmetry |
| `optimization`, `design` | Shell parameter sweeps and metric-profile optimization |
| `io` | Interpolated external metrics |
| `visualization` | Matplotlib figures and optional Manim scenes |

Null integrators report the affine interval, numerical refinement, and
`max|g(k,k)|`. These are finite-segment diagnostics; small null-norm drift does
not bound omitted tails or prove complete-geodesic ANEC. Likewise, fitted
speed exponents and type/vorticity associations describe the sampled families,
not universal laws.

## Runtime conventions

- Python uses a `src/` layout; import through an installed package.
- JAX runs in float64. CPU is the reference backend for recorded numerical data.
- `WARPAX_JIT_CACHE=1` enables JAX's persistent compilation cache; tune it with
  `JAX_PERSISTENT_CACHE_*` settings.
- Optimistix BFGS receives tolerances through its solver constructor.
- Diffrax uses `Tsit5`, `PIDController`, and `throw=False`; callers must inspect
  the returned integration status. The default adjoint is
  `RecursiveCheckpointAdjoint`.
- Chunked grid evaluation controls memory use. Proper-volume summaries use
  `sqrt(det gamma) d^3x`; point-count summaries must be identified separately.

The [API reference](../reference/index.md) documents signatures and result fields.
