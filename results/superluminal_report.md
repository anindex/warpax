# Superluminal Characterization Report

**Date:** 2026-04-18T14:04:11Z
**Script:** `scripts/run_superluminal_investigation.py`
**Metrics tested:** Alcubierre (tanh), Lentz (L1/diamond)
**Velocities:** 1.0, 1.5, 2.0

## Key Finding

The superluminal failure mode is NOT metric degeneracy (det(g) = 0). For all unit-lapse, flat-spatial ADM warp metrics, det(g) = -1 at ALL velocities. The actual failure mode is a g_00 sign flip: g_00 = -(1 - v_s^2 * f(r)^2) becomes positive when v_s * f(r) > 1, creating a region with no static observers.

The metric signature (-,+,+,+) is preserved: one eigenvalue of g_ab remains negative at all tested velocities. The curvature chain (Christoffel -> Riemann -> Ricci -> Einstein -> stress-energy) produces no NaN. The EC pipeline continues to work, but the physical interpretation of EC margins in the g_00 > 0 region is questionable because coordinate-stationary observers become spacelike.

## v_s = 1.0 (Luminal Threshold)

| Metric | g_00 = 0 location | det(g) | NaN found | HE types | NEC margin range | WEC margin range |
|--------|-------------------|--------|-----------|----------|-----------------|-----------------|
| Alcubierre | N/A | det(g) = -1 (max dev: 0.0e+00) | No | 1, 2 | [-6.37e-01, 0.00e+00] | [-3.51e+03, -0.00e+00] |
| Lentz | N/A | det(g) = -1 (max dev: 0.0e+00) | No | 1, 4 | [-1.63e+01, 0.00e+00] | [-8.96e+04, -0.00e+00] |

At the luminal threshold (v_s = 1.0), g_00 = 0 exactly at points where f(r) = 1 (bubble center). This is a coordinate degeneracy of the zero-shift surface, not a metric degeneracy.

## v_s = 1.5

| Metric | g_00 = 0 location | det(g) | NaN found | HE types | NEC margin range | WEC margin range |
|--------|-------------------|--------|-----------|----------|-----------------|-----------------|
| Alcubierre | r ~ 100.3 | det(g) = -1 (max dev: 0.0e+00) | No | 1, 2 | [-1.43e+00, 0.00e+00] | [-7.89e+03, -0.00e+00] |
| Lentz | r ~ 100.3 | det(g) = -1 (max dev: 0.0e+00) | No | 1, 4 | [-2.49e+01, 0.00e+00] | [-1.37e+05, -0.00e+00] |

## v_s = 2.0

| Metric | g_00 = 0 location | det(g) | NaN found | HE types | NEC margin range | WEC margin range |
|--------|-------------------|--------|-----------|----------|-----------------|-----------------|
| Alcubierre | r ~ 101.1 | det(g) = -1 (max dev: 6.7e-16) | No | 1, 2 | [-2.55e+00, 0.00e+00] | [-1.40e+04, -0.00e+00] |
| Lentz | r ~ 101.1 | det(g) = -1 (max dev: 2.2e-16) | No | 1, 4 | [-3.40e+01, 0.00e+00] | [-1.87e+05, -0.00e+00] |

## det(g) = -1 Confirmation

For unit-lapse (alpha = 1) and flat-spatial (gamma_ij = delta_ij) ADM warp metrics:

    det(g) = -alpha^2 * det(gamma) = -1 * 1 = -1

This holds at ALL velocities because neither the lapse nor the spatial metric depend on the shift vector magnitude. The shift only enters g_0i components but does not affect the determinant of the spatial block.

**Numerical verification:** det(g) = -1 within tolerance at all 300 sampled points: CONFIRMED

## Scope Claim Evidence

The restriction to subluminal velocities (v_s < 1) is justified on both physical and computational grounds:

1. **Physical:** When v_s * f(r) > 1, the g_00 component flips sign. Coordinate-stationary observers become spacelike in this region. While the Eulerian normal n^a remains timelike (alpha = 1), the physical interpretation of energy conditions measured by these observers becomes ambiguous in a region where the metric signature locally resembles Euclidean space in the (t, x) sector.

2. **Computational:** The curvature chain and EC pipeline remain numerically functional at all tested velocities (no NaN, finite margins). However, the EC margins in the g_00 > 0 region reflect mathematical quantities without clear physical meaning, making observer-robust verification unreliable as a diagnostic tool.

3. **Scope boundary:** warpax restricts to v_s < 1 where Lorentzian signature is globally preserved and EC verification has unambiguous physical interpretation. Superluminal analysis requires fundamentally different mathematical tools (causal structure analysis, horizon formation criteria) that are beyond warpax's current scope.
