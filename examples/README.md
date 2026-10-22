# Examples

Run every script from the repository root with `JAX_PLATFORMS=cpu python
examples/<script>.py`. Scripts 01-07 write figures to `examples/output/`, which
git ignores; 08, 10 and 11 write under `tests/fixtures/` and `results/` as their
headers state. The [examples guide](../docs/tutorials/examples_tour.md) covers
installation extras, runtimes and the animations.

| # | Script | What it does |
|---|--------|--------------|
| 01 | `01_minkowski_sanity.py` | Curvature chain on flat space: every tensor vanishes and `verify_point` reports zero margins |
| 02 | `02_schwarzschild_verification.py` | Vacuum check, Kretschmann against `48 M^2 / r^6`, Richardson-extrapolated ADM mass against `M` |
| 03 | `03_alcubierre_analysis.py` | Type IV wall point: capped observer search, exact Eulerian NEC, energy density against the closed form |
| 04 | `04_warp_drive_comparison.py` | Six warp metrics at their walls, Hawking-Ellis types, Eulerian vs robust margins, `v_s^2` density scaling |
| 05 | `05_grid_analysis.py` | Grid workflow through `compare_eulerian_vs_robust` and `certify`, type counts cross-checked, SEC miss figure |
| 06 | `06_geodesic_through_warp_bubble.py` | Diffrax timelike geodesic, norm conservation, jitted tidal eigenvalues, null ray staying on the cone |
| 07 | `07_custom_warp_metric.py` | Subclass `ADMMetric` and run the full point, grid and wall-restricted pipeline on it |
| 08 | `08_metric_design.py` | `design_metric` reproducing the Alcubierre shape function with a cubic B-spline |
| 09 | `09_admissibility_diagnostics.py` | Fuchs shell: ADM split, constraints, ADM mass, source consistency, junction, tides |
| 10 | `10_phase_diagram.py` | Sampled energy-condition sweep over T-shell compactness and thickness (`--full` enlarges the grid) |
| 11 | `11_elastic_shell.py` | Einstein-elastic shell equilibrium, redshift, frame dragging, centrifugal tides and clock shifts |
