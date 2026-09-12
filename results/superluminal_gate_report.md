# Superluminal classification checks

Source: `superluminal_gate.json`. R=1, sigma=8, domain=[-3,3]^3; wall fractions use proper-volume weights where 0.1 <= f <= 0.9.

Pass requires an observed 50-digit mpmath flip fraction <= 0.01, tolerance spread <= 0.5 percentage points (pp) for tol=1e-12, 1e-10, 1e-8, and tested refinement stability. Refinement passes if the maximum deviation from the mean wall percentage is <= 0.5 pp or <= 5% of that mean. Incomplete means a required check was not run; fail means a measured criterion failed.

Standard/generalized agreement and mpmath flips are fractions of sampled wall points, with Type IV prioritized. Im/Re is the median ratio of maximum absolute imaginary to real eigenvalue parts among wall Type-IV points (denominator floor 1e-30). `--` means unavailable. These are numerical consistency checks, not continuum proofs.

| Metric | v_s | Wall Type IV % | Tolerance spread (pp) | Std/gen agreement | mpmath flip | Refinement stable | Im/Re | Checks |
|---|---:|---:|---:|---:|---:|---|---:|---|
| Alcubierre | 0.50 | 99.02 | 0.00 | 1.000 | 0.000 | -- | 2.946 | incomplete |
| Alcubierre | 0.90 | 96.27 | 0.00 | 1.000 | 0.000 | -- | 1.567 | incomplete |
| Alcubierre | 0.99 | 93.91 | 0.00 | 1.000 | 0.000 | yes | 1.409 | pass |
| Alcubierre | 1.00 | 92.71 | 0.00 | 1.000 | 0.000 | yes | 1.391 | pass |
| Alcubierre | 1.10 | 92.30 | 0.00 | 1.000 | 0.000 | -- | 1.189 | incomplete |
| Alcubierre | 1.50 | 90.16 | 0.00 | 1.000 | 0.000 | yes | 0.643 | pass |
| Alcubierre | 2.00 | 80.54 | 0.00 | 1.000 | 0.000 | -- | 0.429 | incomplete |
| Alcubierre | 2.50 | 79.05 | 0.00 | 1.000 | 0.000 | -- | 0.250 | incomplete |
| Natario | 0.50 | 84.97 | 0.00 | 1.000 | 0.000 | -- | 3.721 | incomplete |
| Natario | 0.90 | 76.99 | 0.00 | 1.000 | 0.000 | -- | 1.586 | incomplete |
| Natario | 0.99 | 75.16 | 0.00 | 1.000 | 0.000 | yes | 1.373 | pass |
| Natario | 1.00 | 75.16 | 0.00 | 1.000 | 0.000 | yes | 1.335 | pass |
| Natario | 1.10 | 70.63 | 0.00 | 1.000 | 0.000 | -- | 1.164 | incomplete |
| Natario | 1.50 | 62.50 | 0.00 | 1.000 | 0.000 | yes | 0.596 | pass |
| Natario | 2.00 | 57.46 | 0.00 | 1.000 | 0.000 | -- | 0.375 | incomplete |
| Natario | 2.50 | 54.02 | 0.00 | 1.000 | 0.000 | -- | 0.286 | incomplete |
| VanDenBroeck | 0.50 | 79.98 | 0.00 | 1.000 | 0.000 | -- | 0.334 | incomplete |
| VanDenBroeck | 0.90 | 79.46 | 0.00 | 1.000 | 0.000 | -- | 0.682 | incomplete |
| VanDenBroeck | 0.99 | 78.81 | 0.00 | 1.000 | 0.000 | yes | 0.743 | pass |
| VanDenBroeck | 1.00 | 78.81 | 0.00 | 1.000 | 0.000 | yes | 0.751 | pass |
| VanDenBroeck | 1.10 | 77.32 | 0.00 | 1.000 | 0.000 | -- | 0.820 | incomplete |
| VanDenBroeck | 1.50 | 76.17 | 0.00 | 1.000 | 0.000 | NO | 0.714 | fail |
| VanDenBroeck | 2.00 | 78.33 | 0.00 | 1.000 | 0.000 | -- | 0.411 | incomplete |
| VanDenBroeck | 2.50 | 72.14 | 0.00 | 1.000 | 0.000 | -- | 0.239 | incomplete |
| Rodal | 0.50 | 0.00 | 0.00 | 1.000 | 0.000 | -- | -- | incomplete |
| Rodal | 0.90 | 0.00 | 0.00 | 1.000 | 0.000 | -- | -- | incomplete |
| Rodal | 0.99 | 0.00 | 0.00 | 1.000 | 0.000 | yes | -- | pass |
| Rodal | 1.00 | 0.00 | 0.00 | 1.000 | 0.000 | yes | -- | pass |
| Rodal | 1.10 | 0.00 | 0.00 | 1.000 | 0.000 | -- | -- | incomplete |
| Rodal | 1.50 | 0.00 | 0.00 | 1.000 | 0.000 | yes | -- | pass |
| Rodal | 2.00 | 0.00 | 0.00 | 1.000 | 0.000 | -- | -- | incomplete |
| Rodal | 2.50 | 0.00 | 0.00 | 1.000 | 0.000 | -- | -- | incomplete |

Main grid N=[50]; tested refinement grids N=[30, 50, 70]. Each row applies only to its tested speed; these samples do not define a velocity ceiling. A passing census with zero wall Type IV does not establish the presence of Type IV.
