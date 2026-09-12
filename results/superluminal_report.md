# Superluminal characterization

Date: 2026-09-12T09:35:08Z. Source: `superluminal_characterization.json`.

Alcubierre (tanh) and Lentz (L1/diamond), R=100 and sigma=8. For unit-lapse, flat-spatial ADM metrics, det(g)=-1 and the signature is (-,+,+,+) at every speed. With the convention ds^2=-alpha^2 dt^2+gamma_ij(dx^i+beta^i dt)(dx^j+beta^j dt), the Eulerian normal is n^a=(1,-beta^i)/alpha and satisfies n^a n_a=-1. Its energy-condition measurements remain meaningful when g_00>0.

For these shifts, g_00=-1+v_s^2 f^2. Coordinate-stationary worldlines become null at g_00=0 and spacelike at g_00>0. This surface alone does not identify an event horizon. At v_s=1, g_00=0 wherever f=1; the metric remains invertible.

| Metric | v_s | Estimated g_00 crossing r | Max det(g) error | NaN in T | HE types | NEC range | WEC range |
|---|---:|---:|---:|---|---|---|---|
| Alcubierre | 1.0 | none bracketed | 0.0e+00 | no | 1, 4 | [-6.37e-01, 0.00e+00] | [-3.51e+03, -0.00e+00] |
| Lentz | 1.0 | none bracketed | 0.0e+00 | no | 1, 4 | [-1.63e+01, 0.00e+00] | [-8.96e+04, -0.00e+00] |
| Alcubierre | 1.5 | 100.3 | 0.0e+00 | no | 1, 4 | [-1.43e+00, 0.00e+00] | [-7.89e+03, -0.00e+00] |
| Lentz | 1.5 | 100.3 | 0.0e+00 | no | 1, 4 | [-2.49e+01, 0.00e+00] | [-1.37e+05, -0.00e+00] |
| Alcubierre | 2.0 | 101.1 | 6.7e-16 | no | 1, 4 | [-2.55e+00, 0.00e+00] | [-1.40e+04, -0.00e+00] |
| Lentz | 2.0 | 101.1 | 2.2e-16 | no | 1, 4 | [-3.40e+01, 0.00e+00] | [-1.87e+05, -0.00e+00] |

Determinant checks cover 300 sampled points; EC checks cover 60. All sampled EC margins are finite: yes. The EC search uses eight starts and rapidity cap 5. Ranges are sampled extrema, and crossing locations are interpolated from a coarse radial scan at (t,x,y,z)=(0,r,0.01,0). Neither provides a continuum error bound.
