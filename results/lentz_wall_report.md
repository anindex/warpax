# Lentz Wall Resolution Assessment

**Date:** 2026-04-18T14:11:32Z

**Script:** `scripts/run_lentz_wall_assessment.py`

## Verdict

The Lentz wall is **UNRESOLVABLE** at practical 3D resolution. At sigma=8.0 on a [-300,300]^3 grid with N=50, the wall width (0.274653) is spanned by only 0.0224 grid cells (threshold: 4.0). The grid spacing is 44.6x larger than the wall width.

## Analytical Assessment

| Parameter | Value |
|-----------|------:|
| Wall width (10-90%) | 0.274653 |
| Grid spacing (dx) | 12.2449 |
| Cells across wall | 0.0224 |
| Resolved (>= 4 cells) | False |
| Under-resolution ratio | 44.6x |

## 1D Radial Cut (N=500, r=[50.0, 150.0])

Curvature peaks sharply at the wall (r ~ R=100.0). The Kretschner scalar peaks at |K|=4.143598e+04 (r=99.90) and the stress-energy Frobenius norm peaks at ||T||=4.384302e+00 (r=99.90). The sharp curvature concentration near the wall confirms that standard 3D grids cannot adequately sample the wall structure.

### Selected Radial Cut Data Points

| r | f(r) | |Kretschner| | ||T|| |
|--:|-----:|------------:|------:|
| 50.00 | 1.000000 | 0.000000e+00 | 0.000000e+00 |
| 60.02 | 1.000000 | 0.000000e+00 | 0.000000e+00 |
| 70.04 | 1.000000 | 0.000000e+00 | 0.000000e+00 |
| 80.06 | 1.000000 | 0.000000e+00 | 0.000000e+00 |
| 90.08 | 1.000000 | 0.000000e+00 | 0.000000e+00 |
| 100.10 | 0.146389 | 2.079927e+04 | 2.533619e+00 |
| 110.12 | 0.000000 | 0.000000e+00 | 0.000000e+00 |
| 120.14 | 0.000000 | 0.000000e+00 | 0.000000e+00 |
| 130.16 | 0.000000 | 0.000000e+00 | 0.000000e+00 |
| 140.18 | 0.000000 | 0.000000e+00 | 0.000000e+00 |

## Note for Paper

Lentz diagnostics should be presented as lower-bound estimates. The autodiff approach computes exact curvature at each sampled point, but the spatial sampling density at practical 3D grid resolution (N=50 on [-300,300]^3) is insufficient to capture the wall structure. Lentz results should be segregated in a separate table or footnoted with a resolution caveat.
