# Lentz wall resolution

Date: 2026-09-12T09:36:23Z. Source: `lentz_wall_assessment.json`.

The tested 50^3 grid on [-300, 300]^3 fails the chosen 4-cell wall-resolution criterion at sigma=8.0. The conclusion applies to this grid and domain.

| Quantity | Value |
|---|---:|
| 10-90% wall width, 2 atanh(0.8)/sigma | 0.274653 |
| Grid spacing | 12.2449 |
| Cells across wall | 0.0224 |
| Grid spacing / wall width | 44.58 |

The 500-point cut spans r=50 to 150 at (t,x,y,z)=(0,r,0.01,0), v_s=0.5 and R=100. Its sampled maximum absolute Kretschmann scalar is 3.110818e+04 at r=99.90; the sampled maximum stress-tensor Frobenius norm is 4.384302e+00 at r=99.90.

K=R_abcd R^abcd is invariant; the Frobenius norm uses the coordinate components of T_ab. Automatic differentiation evaluates local derivatives in floating-point arithmetic. It does not bound unsampled extrema or integration errors. Selected points near the sampled peak and the endpoints follow.

| r | f(r) | Absolute K | Frobenius norm of T |
|---:|---:|---:|---:|
| 50.00 | 1.000000 | 0.000000e+00 | 0.000000e+00 |
| 99.70 | 0.990525 | 1.182909e+02 | 3.000338e-01 |
| 99.90 | 0.808951 | 3.110818e+04 | 4.384302e+00 |
| 100.10 | 0.146389 | 2.041381e+04 | 2.533619e+00 |
| 150.00 | 0.000000 | 0.000000e+00 | 0.000000e+00 |
