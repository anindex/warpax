# Architecture

## Overview

warpax verifies energy conditions (NEC, WEC, SEC, DEC) for warp drive spacetimes
using JAX automatic differentiation and continuous observer optimization. The key
insight is that Eulerian-frame checks can miss violations detectable by boosted
observers - warpax searches the full timelike observer manifold via BFGS.

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
Curvature invariants (Kretschner, Ricci², Weyl²)
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
Nine warp drive metrics, each implementing `MetricFunction: (4,) -> (4,4)`:

- **Alcubierre** (WarpShell) - the original warp drive (1994)
- **Natario** - zero-expansion variant (2001)
- **Lentz** - shift-only positive-energy candidate (2020)
- **Rodal** - globally Type I algebraic solution (2025)
- **Van den Broeck** - volume-expansion variant (1999)
- **WarpShell** - spherical shell geometry with C^1/C^2 transitions
- **Fuchs** - constant-velocity physical warp shell (2024)
- **S-shell** - source-first Class I (flow-orthogonal, constraint-derived potentials)
- **T-shell** - source-first Class II (tilted matter flow, momentum-constraint-derived shift)

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
Invariant transport diagnostics: geodesic deviation, null round-trip asymmetry,
blueshift hazard. All geodesic-based via Diffrax.

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
