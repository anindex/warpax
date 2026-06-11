# Architecture

## Overview

`warpax` verifies NEC, WEC, SEC, and DEC for warp-drive spacetimes using JAX
autodiff and continuous observer optimization. Eulerian-frame checks miss
violations that boosted observers detect; `warpax` searches the full timelike
observer manifold via BFGS to find them.

## Autodiff curvature pipeline

All computation flows from analytic metric functions through a single JAX AD chain:

```
MetricFunction g_{μν}(x)
      │
      ▼ (JAX jacfwd)
Christoffel Γ^α_{βγ}
      │
      ▼ (JAX jacfwd)
Riemann R^α_{βγδ}
      │
      ├──▶ Ricci tensor R_{μν}
      │ │
      │ ├──▶ Ricci scalar R
      │ │
      │ ▼
      │ Einstein tensor G_{μν}
      │ │
      │ ▼ (Einstein field equations)
      │ Stress-energy T_{μν}
      │
      ▼
Curvature invariants (Kretschmann, Ricci², Weyl²)
```

No symbolic algebra or finite-difference stencils are used. JAX's forward-mode AD
(`jacfwd`) computes exact derivatives at machine precision.

## Sub-packages

### `geometry`
Core differential geometry: Christoffel symbols, curvature tensors, stress-energy,
invariants, grid evaluation. All tensors are JAX arrays with jaxtyping annotations.

### `energy_conditions`
Frame-independent, all-velocity certification from the eigenstructure of `T^a_b`
(`frame_free.py`, public `warpax.certify`); it never builds the Eulerian normal, so
it is valid at all warp speeds including `v_s >= 1`. Two-tier verification strategy:

1. **Hawking-Ellis classification** (`solver='auto'` by default) determines the
   algebraic type (I/II/III/IV). Ill-conditioned points fall back to the
   generalized pencil solve when `warpax[solver]` is installed; Type-IV labels are
   cross-checked against a 50-digit `mpmath` recomputation (`classification_mpmath.py`).
2. **BFGS optimization** over the observer 4-velocity (rapidity-bounded), retained
   as a one-sided diagnostic. Type I points use exact eigenvalue margins (and the
   closed-form worst observer `sinh^2 zeta_th = rho/|rho+p_i|`,
   `worst_observer_analytic.py`); `verify_grid` skips BFGS on Type I by default
   (`skip_type_i_optimization=True`).

### `metrics`
Nine warp/shell drives in `warpax.metrics`, each implementing
`MetricFunction: (4,) -> (4,4)` (Alcubierre lives in `benchmarks`):

- **Natario** - zero-expansion variant (2001)
- **Lentz** - shift-only positive-energy candidate (2020)
- **Rodal** - globally Type I algebraic solution (2025)
- **Van den Broeck** - volume-expansion variant (1999)
- **WarpShell** - spherical shell geometry with C^1/C^2 transitions
- **Fuchs** - constant-velocity physical warp shell (2024)
- **S-shell** - source-first Class I (flow-orthogonal, constraint-derived potentials)
- **T-shell** - source-first Class II (tilted matter flow, momentum-constraint-derived shift)
- **Garattini-Zatrimaylov** - warp bubble on a de Sitter background (2025)

### `geodesics`
Geodesic integration via Diffrax ODE solvers. Computes:
- Timelike and null geodesics through warp bubbles
- Jacobi deviation (tidal forces) via the geodesic deviation equation
- Blueshift factors, energy conservation, proper time
- Structure-preserving symplectic null integrator (`symplectic.py`, Tao-2016
  extended phase space) that conserves `g(k,k)` to ~machine precision for the
  rigorous geodesic-integrated ANEC

### `analysis`
Higher-level analysis utilities:
- Eulerian vs robust EC comparison tables
- Richardson extrapolation convergence analysis
- Kinematic scalars: expansion θ, shear σ², vorticity ω² (`shift_kinematics.py`)
- Vorticity -> Type-IV mechanism `f = κ ω` (`vorticity_type_analytic.py`)
- Cross-construction all-observer verification with a wall-resolution gate
  (`construction_adapter.py`)
- Curvature-invariant and NEC-severity `v_s` scaling laws and the boost-invariant
  exoticity ranking

### `constraints`
Source-consistency modules: Hamiltonian and momentum constraint residuals with
scale-invariant normalization, source-consistency T_ab residual comparison,
S-shell constraint solver (tridiagonal), T-shell constraint solver with
momentum-constraint-derived shift. All solvers use pure JAX (no scipy).

### `tov`
Anisotropic TOV equilibrium checker for spherical shells.

### `adm`
ADM mass via surface integral with Gauss-Legendre angular quadrature.
Asymptotic flatness report with multi-radius falloff verification.

### `junction`
Israel/Darmois junction conditions with two-sided extrinsic curvature jump.
Surface stress-energy for thin-shell constructions.

### `transport`
Transport diagnostics: geodesic deviation (`A_geo`, gauge-invariant),
blueshift hazard (gauge-invariant for a chosen observer worldline), and
null round-trip *coordinate-time* asymmetry (`null_coord_time_asymmetry`,
gauge-dependent; invariant only under constant time shifts). All
geodesic-based via Diffrax.

### `optimization`
Source-first shell optimization: Bernstein polynomial basis for profile
parameterization, multi-objective loss (constraints + EC penalty + tidal +
transport + ADM mass), EC soft/hard constraint enforcement, 2D parameter
sweep over (compactness, thickness) with `SweepResult` serialization.

### `visualization`
- **Matplotlib**: static figures (heatmaps, convergence plots,
  comparison grids, kinematic scalar maps, geodesic observables)
- **Phase diagrams**: transport heatmap with EC boundary hatching, 2x2
  summary panel (transport / EC margin / constraint residual / tidal)
- **Manim CE**: Animated 3D scenes (bubble collapse, velocity ramp, EC heatmap
  contours, observer sweep)

## Key design decisions

- **`src/` layout** - avoids accidental imports from the project root.
- **Equinox modules** - metrics and geometry types are `eqx.Module` instances,
  enabling clean pytree handling for JIT compilation across parameter sweeps.
- **Bounded rapidity** - observer optimization uses sigmoid-bounded rapidity to
  stay within the physical timelike cone, avoiding divergences at the light cone.
- **Grid-then-optimize** - coarse Fibonacci sphere sampling provides initial
  candidates; BFGS refines each to the true worst-case observer.

## Dependency pins

- **JAX 0.10.x** - `pmap` is in maintenance mode upstream and is not used
  by warpax. Multi-device fan-out, when needed, will go through
  `jax.sharding.NamedSharding` / `shard_map`. The compile cache is opt-in
  via `WARPAX_JIT_CACHE=1` and respects
  `WARPAX_JIT_CACHE_MIN_ENTRY_SIZE_BYTES` and
  `WARPAX_JIT_CACHE_MIN_COMPILE_TIME_SECS`.
- **Optimistix 0.1** - `optx.minimise(fn, solver, y0, args=..., options=...,
  max_steps=...)` matches the projected-BFGS observer pipeline; tolerances
  are passed through the solver constructor (`rtol`, `atol`, `norm`) and
  documented at each call site.
- **Diffrax 0.7** - all geodesic integrators use `diffrax.Tsit5()` with
  `PIDController(rtol, atol)`; `PIController` is not used. `throw=False`
  is set so non-success result codes are returned and surfaced by callers
  (e.g. `GeodesicResult.result`). Differentiating geodesics relies on the
  default `RecursiveCheckpointAdjoint`; callers that need a different
  adjoint should pass it explicitly to `diffeqsolve`.
- **Equinox 0.13** - public entry points use `eqx.filter_jit` so
  `eqx.field(static=True)` parameters key the compile cache automatically.
