# Architecture

## Overview

warpax verifies energy conditions (NEC, WEC, SEC, DEC) for warp drive spacetimes
using JAX automatic differentiation and continuous observer optimization. The key
insight is that Eulerian-frame checks can miss violations detectable by boosted
observers - warpax searches the full timelike observer manifold via BFGS.

## Autodiff curvature pipeline

All computation flows from analytic metric functions through a single JAX AD chain:

```
MetricFunction  g_{μν}(x)
      │
      ▼  (JAX jacfwd)
Christoffel  Γ^α_{βγ}
      │
      ▼  (JAX jacfwd)
Riemann  R^α_{βγδ}
      │
      ├──▶ Ricci tensor  R_{μν}
      │         │
      │         ├──▶ Ricci scalar  R
      │         │
      │         ▼
      │    Einstein tensor  G_{μν}
      │         │
      │         ▼  (Einstein field equations)
      │    Stress-energy  T_{μν}
      │
      ▼
Curvature invariants  (Kretschner, Ricci², Weyl²)
```

No symbolic algebra or finite-difference stencils are used. JAX's forward-mode AD
(`jacfwd`) computes exact derivatives at machine precision.

## Sub-packages

### `geometry`
Core differential geometry: Christoffel symbols, curvature tensors, stress-energy,
invariants, grid evaluation. All tensors are JAX arrays with jaxtyping annotations.

### `energy_conditions`
Two-tier verification strategy:

1. **Hawking–Ellis classification** determines the algebraic type (I/II/III/IV) of
   the stress-energy tensor. Type I admits closed-form eigenvalue checks.
2. **BFGS optimization** over the observer 4-velocity parameterized by rapidity
   ζ ∈ [0, ζ_max] and direction via stereographic projection. Finds the worst-case
   T_{ab} u^a u^b across the bounded timelike observer manifold.

### `metrics`
Six warp drive metrics, each implementing `MetricFunction: (4,) -> (4,4)`:

- **Alcubierre** - the original warp drive (1994)
- **Natário** - zero-expansion variant (2001)
- **Lentz** - shift-only positive-energy candidate (2020)
- **Rodal** - globally Type I algebraic solution (2025)
- **Van den Broeck** - volume-expansion variant (1999)
- **WarpShell** - spherical shell geometry with C¹/C² transitions

### `geodesics`
Geodesic integration via Diffrax ODE solvers. Computes:
- Timelike and null geodesics through warp bubbles
- Jacobi deviation (tidal forces) via the geodesic deviation equation
- Blueshift factors, energy conservation, proper time

### `analysis`
Higher-level analysis utilities:
- Eulerian vs robust EC comparison tables
- Richardson extrapolation convergence analysis
- Kinematic scalars: expansion θ, shear σ², vorticity ω²

### `visualization`
- **Matplotlib**:  static figures (heatmaps, convergence plots,
  comparison grids, kinematic scalar maps, geodesic observables)
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
