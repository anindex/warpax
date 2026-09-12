# Scripts catalog

Run scripts from the repository root. See the
[paper reproduction guide](../docs/how-to/reproduce_observer_robust_paper.md)
for stage order and scientific limits.

## Start here

The public API provides cap-free pointwise energy-condition tests:

```python
from warpax import certify
from warpax.metrics import RodalMetric
r = certify(RodalMetric(v_s=2.0, R=1.0, sigma=8.0))
```

`results/` is local. Table paths refer to the sibling `warpax_arxiv/tables/`;
figures are generated locally or in that manuscript tree, depending on the script.

| Script | Produces | Paper artifact |
|--------|----------|----------------|
| `run_velocity_sweep.py` | `tables/velocity_type_structure.tex`, `figures/velocity_type_structure.pdf`, `figures/rodal_invariant_margins.pdf` | Type/EC structure across the luminal transition |
| `run_invariant_verification.py` | `tables/invariant_benchmark.tex` | Invariant all-observer verification (single-frame miss, E_-) |
| `validate_superluminal_classification.py` | `results/superluminal_gate*` | Type-IV trustworthiness check (3-solver + 50-digit) |
| `run_matched_benchmark.py` | `tables/missed_wall_restricted.tex`, `tables/convergence_per_metric.tex` | Matched wall-resolved benchmark + per-metric convergence |
| `run_shift_vorticity.py` | `tables/shift_vorticity.tex`, `figures/shift_vorticity.pdf`, `results/shift_vorticity.json` | Shift-vorticity decomposition and sampled type association; reads `velocity_sweep.json` |

For cross-metric refinement, start with `run_matched_benchmark.py` and
`run_velocity_sweep.py`; the other scripts below cover specific diagnostics.

## Pipeline (`reproduce_all.sh`)

### Core computation

| Script | Output |
|--------|--------|
| `run_anec_retained.py` | `results/anec/retained.json` (finite coordinate-path null-energy integrals) |
| `run_anec_symplectic.py` | `results/anec/retained_symplectic.json`, `tables/anec_symplectic.tex` (finite geodesic integrals, selected-ray step refinement and null-norm drift) |
| `run_quantum_inequality.py` | `results/quantum/ford_roman.json`, `tables/averaged_quantum.tex`, `figures/averaged_quantum.pdf` (Ford-Roman quantum-inequality diagnostic, reads `run_anec_retained.py`) |
| `run_construction_verification.py` | `results/construction_verification.json`, `tables/construction_matched.tex`, `tables/construction_native.tex` (cross-construction all-observer verification; under-resolved walls carry cell counts only) |
| `run_rodal_sigma_resolved.py` | `results/rodal_sigma_resolved.json`, `tables/rodal_sigma_resolved.tex` (wall-resolved Rodal sigma sweep on the exact axisymmetric reduction) |
| `run_enclosures.py` | `results/enclosures.json`, `tables/enclosures.tex` (certified global interval enclosures of the wall null deficit; hours, opt-in stage) |
| `run_classifier_error_rate.py` | `results/classifier_error_rate.json` (Jordan displacement exponents and LMI/classifier comparison) |
| `run_exoticity_ranking.py` | `results/exoticity_ranking.json`, `tables/exoticity_ranking.tex`, `tables/scaling_laws.tex` (specified-slice composite and empirical speed fits; reads velocity and finite-ray data) |
| `derive_vorticity_type.py` | `results/vorticity_type_analytic.json` (restricted vorticity model and empirical cross-metric comparisons) |
| `run_curvature_scaling.py` | `results/curvature_scaling.json`, `tables/curvature_scaling.tex`, `figures/curvature_scaling.pdf` (empirical speed fits of wall curvature invariants) |
| `run_ssv_bound.py` | `results/ssv_bound.json`, `tables/ssv_bound.tex` (pointwise wall deficits and speed fits; reads `run_velocity_sweep.py`) |
| `run_analysis.py` | `results/comparison_table.json` |
| `run_convergence.py` | `results/convergence_data.json` |
| `run_kinematic_scalars.py` | kinematic scalar NPZ/JSON under `results/` |
| `run_geodesics.py` | `results/geodesic_scaling.json` |
| `run_clustered_convergence.py` | `results/clustered_convergence_*.json` |
| `run_diagnostic_convergence.py` | `tables/diagnostic_convergence.tex`, `results/diagnostic_convergence.json` (N=80/100/120 sampled fractions, miss-rate spreads, and polished extrema) |
| `run_curvature_convergence.py` | `tables/curvature_convergence.tex`, `results/curvature_convergence.json` (N=80/100/120 stability of fitted curvature exponents) |

### Ablations

| Script | Output |
|--------|--------|
| `run_c1_vs_c2_comparison.py` | `results/c1_vs_c2_comparison.json` |
| `run_nstarts_ablation.py` | `results/nstarts_ablation.json` |
| `run_zeta_sensitivity.py` | zeta sensitivity JSON under `results/` |
| `rodal_dec_ablation.py` | Rodal DEC ablation under `results/` |
| `run_warpshell_convergence.py` | `results/warpshell_convergence.json` |
| `run_wall_resolution.py` | `results/wall_resolution.json` |
| `run_sampling_comparison.py` | `results/sampling_comparison.json` |
| `run_smoothwidth_ablation.py` | `results/smoothwidth_ablation.json` |
| `run_worst_observer_alignment.py` | alignment JSON under `results/` |
| `run_missed_detection_comparison.py` | `results/missed_detection_comparison.json` |
| `run_superluminal_investigation.py` | `results/superluminal_characterization.json` |
| `run_rodal_matched_resolution.py` | `results/rodal_matched_resolution.json` |
| `run_rodal_native_resolution.py` | `results/rodal_native_resolution.json`, `tables/rodal_resolution.tex` |
| `run_lentz_wall_assessment.py` | `results/lentz_wall_assessment.json` |
| `run_wall_restricted_analysis.py` | `results/wall_restricted_analysis.json` |

### Figures and emitted tables

| Script | Output |
|--------|--------|
| `reproduce_figures.py` | `figures/*.pdf` |
| `generate_vdb_comparison_figures.py` | Van den Broeck comparison figures |
| `emit_diagnostic_tables.py` | `tables/{missed_uniform,nstarts,convergence_richardson}.tex` from cached `results/*.json` |

## Other analyses and companion calculations

See the [companion guide](../docs/how-to/reproduce_warpshell_paper.md) for
shell results and current calculation commands.

| Script | Output |
|--------|--------|
| `verify_fuchs.py` | `results/fuchs_verification_report.json` |
| `verify_proposals.py` | `results/proposals_verification_report.json` |
| `run_sshell_sweep.py` | S-shell sweep under `results/` |
| `run_integrated_negative_energy.py` | `tables/integrated_volume.tex` (slice-integrated negative-energy volume vs `v_s`) |
| `run_delta_crosscheck.py` | `results/delta_crosscheck.json` (algebraic `Delta < 0` label vs the eigensolver Type-IV label) |
| `run_exoticity_anec_convergence.py` | `tables/extra_convergence.tex` (composite and selected finite-integral convergence) |
| `run_error_budget.py` | `results/error_budget.json` (sign robustness of the T-shell boundary DEC deficit across resolution, velocity and source-profile family) |
| `run_criterion_e_verification.py` | Criterion E (global) verification |
| `run_tshell_convergence.py` | T-shell convergence study |
| `run_tshell_kterm_angular.py` | T-shell angular k-term |
| `run_v0_ablation.py` | T-shell matter-tilt ablation |

The cross-metric comparison covers Alcubierre, Natario, Van den Broeck, and Rodal.
WarpShell and Lentz remain implemented as metrics but are not part of the paper's
matched quantitative comparison (their thin walls are not resolved at common parameters).

## Rendering

| Script | Notes |
|--------|-------|
| `render_all_scenes.py` | Manim scene batch (see README) |
| `render_manim_scenes.sh` | Shell wrapper for Manim |
| `generate_showcase.py` | Delegates to render pipeline |

## Shared helpers

| Module | Provides |
|--------|----------|
| `_json_io.py` | RFC 8259 JSON output without bare `NaN` or `Infinity` |
| `_paper_metrics.py` | The four swept constructions and their instantiation |
| `_benchmark_grid.py` | The matched wall-resolved benchmark grid |
| `_anec_window.py` | Prescribed coordinate/affine windows for finite null-energy diagnostics |
| `_paper_numbers_map.py` | `paper_numbers.tex` macros to `results/*.json`, for `emit_paper_numbers.py` |
