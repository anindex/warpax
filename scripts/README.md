# Scripts catalog

Run scripts from the repository root. See the
[paper reproduction guide](../docs/how-to/reproduce_observer_robust_paper.md)
for stage order and scientific limits.

Scripts write to `results/`. Table paths refer to the sibling `warpax_arxiv/tables/`;
figures are generated locally or in that manuscript tree, depending on the script.
The reproduction guide owns the [generated table and figure mappings](../docs/how-to/reproduce_observer_robust_paper.md#generated-table-catalog).

## Pipeline (`reproduce_all.sh`)

### Core computation

The rows follow `run_core` in [reproduce_all.sh](../reproduce_all.sh).
Labels match its console output.

| Stage | Script | Output and purpose |
|---|---|---|
| K1 | `run_velocity_sweep.py` | `results/velocity_sweep.json`, `tables/velocity_type_structure.tex`, three velocity-summary/type/margin PDFs (type and energy-condition structure across the luminal transition) |
| K2 | `run_invariant_verification.py` | `results/invariant_verification.json`, `tables/invariant_benchmark.tex` (all-observer verification, single-frame miss, and E_-) |
| K3 | `validate_superluminal_classification.py` | `results/superluminal_gate*` (Type-IV comparison using three solvers and 50-digit arithmetic) |
| K4 | `run_matched_benchmark.py` | `tables/missed_wall_restricted.tex`, `tables/convergence_per_metric.tex` (matched wall-resolved benchmark and per-metric convergence) |
| K5 | `run_shift_vorticity.py` | `tables/shift_vorticity.tex`, `figures/shift_vorticity.pdf`, `results/shift_vorticity.json` (shift-vorticity decomposition and sampled type association; reads `velocity_sweep.json`) |
| K6 | `run_anec_retained.py` | `results/anec/retained.json` (finite coordinate-path null-energy integrals) |
| K6b | `run_anec_symplectic.py` | `results/anec/retained_symplectic.json`, `tables/anec_symplectic.tex` (finite geodesic integrals, selected-ray step refinement and null-norm drift) |
| K7 | `run_quantum_inequality.py` | `results/quantum/ford_roman.json`, `tables/averaged_quantum.tex`, `figures/averaged_quantum.pdf` (Ford-Roman quantum-inequality diagnostic, reads `run_anec_retained.py`) |
| K8 | `run_construction_verification.py` | `results/construction_verification.json`, `tables/construction_matched.tex`, `tables/construction_native.tex` (cross-construction all-observer verification; under-resolved walls carry cell counts only) |
| K9 | `run_exoticity_ranking.py` | `results/exoticity_ranking.json`, `tables/exoticity_ranking.tex`, `tables/scaling_laws.tex` (specified-slice composite and empirical speed fits; reads velocity and finite-ray data) |
| K10 | `derive_vorticity_type.py` | `results/vorticity_type_analytic.json` (restricted vorticity model and empirical cross-metric comparisons) |
| K11 | `run_curvature_scaling.py` | `results/curvature_scaling.json`, `tables/curvature_scaling.tex`, `figures/curvature_scaling.pdf` (empirical speed fits of wall curvature invariants) |
| K12 | `run_ssv_bound.py` | `results/ssv_bound.json`, `tables/ssv_bound.tex` (pointwise wall deficits and speed fits; reads `run_velocity_sweep.py`) |
| K13 | `run_delta_crosscheck.py` | `results/delta_crosscheck.json` (algebraic `Delta < 0` label vs the eigensolver Type-IV label) |
| K14 | `run_integrated_negative_energy.py` | `results/integrated_negative_energy.json`, `tables/integrated_volume.tex` (specified-slice negative-energy volume) |
| K15 | `run_rodal_sigma_resolved.py` | `results/rodal_sigma_resolved.json`, `tables/rodal_sigma_resolved.tex` (wall-resolved Rodal sigma sweep on the exact axisymmetric reduction) |
| K16 | `run_classifier_error_rate.py` | `results/classifier_error_rate.json` (Jordan displacement exponents and LMI/classifier comparison) |
| K16b | `run_type_transitions.py` | `results/type_transitions.json`, `tables/type_transition.tex` (analytic families across the Type-II locus) |
| K16c | `run_lmi_agreement.py` | `results/lmi_agreement.json`, `tables/lmi_typefree.tex` (type-free LMI versus type-based decisions) |
| K16d | `run_closing_speed.py` | `results/closing_speed.json`, `tables/closing_speed.tex` (momentum-channel closing speed; reads `velocity_sweep.json`) |
| K16e | `run_interval_lmi_spotcheck.py` | `results/interval_lmi_spotcheck.json`, `tables/interval_lmi_spotcheck.tex` (12 point verdicts enclosed from the metric) |
| K16f | `run_interval_lmi_census.py` | `results/interval_lmi_census.json`, `tables/interval_lmi_census.tex` (all four conditions on sampled wall points) |
| K17 | `emit_paper_numbers.py` | `../warpax_arxiv/paper_numbers.tex` (macros from cached results) |
| 1/8 | `run_analysis.py` | `results/comparison_table.json` |
| 2/8 | `run_convergence.py` | `results/convergence_data.json` |
| 3/8 | `run_kinematic_scalars.py` | kinematic scalar NPZ/JSON under `results/` |
| 4/8 | `run_geodesics.py` | `results/geodesic_scaling.json` |
| 5/8 | `run_clustered_convergence.py` | `results/clustered_convergence_*.json` |
| 6/8 | `run_diagnostic_convergence.py` | `tables/diagnostic_convergence.tex`, `results/diagnostic_convergence.json` (N=80/100/120 sampled fractions, miss-rate spreads, and polished extrema) |
| 7/8 | `run_exoticity_anec_convergence.py` | `tables/extra_convergence.tex` (composite and selected finite-integral convergence) |
| 8/8 | `run_curvature_convergence.py` | `tables/curvature_convergence.tex`, `results/curvature_convergence.json` (N=80/100/120 stability of fitted curvature exponents) |

### Ablations

| Script | Output |
|--------|--------|
| `run_nstarts_ablation.py` | `results/nstarts_ablation.json` |
| `run_zeta_sensitivity.py` | zeta sensitivity JSON under `results/` |
| `rodal_dec_ablation.py` | Rodal DEC ablation under `results/` |
| `run_warpshell_convergence.py` | `results/warpshell_convergence.json` |
| `run_wall_resolution.py` | `results/wall_resolution.json` |
| `run_sampling_comparison.py` | `results/sampling_comparison.json` |
| `run_smoothwidth_ablation.py` | `results/smoothwidth_ablation.json` |
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
| `emit_diagnostic_tables.py` | Diagnostic tables from cached `results/*.json`; see the reproduction guide for filenames |
| `write_manifest.py` | `results/MANIFEST.txt` (cached-grid SHA-256 hashes and sizes) |

### Optional stages

| Stage | Script | Output and purpose |
|---|---|---|
| `enclosures` (E1) | `run_enclosures.py` | `results/enclosures.json`, `tables/enclosures.tex` (global wall-null-deficit brackets; can take hours) |
| `gate` | `check_paper_numbers.py` | Manuscript number and source checks; requires matching manuscript inputs |

## Elastic shells

[Example 11](../examples/11_elastic_shell.py) reproduces the elastic-shell
numerics. The [shell guide](../docs/how-to/reproduce_warpshell_paper.md)
gives the commands and the [data definitions](../results/elastic_shell/README.md)
specify units and approximation limits.

The cross-metric comparison covers Alcubierre, Natario, Van den Broeck, and Rodal.
WarpShell and Lentz remain implemented as metrics but are not part of the paper's
matched quantitative comparison (their thin walls are not resolved at common parameters).

## Rendering

| Script | Notes |
|--------|-------|
| `render_all_scenes.py` | Manim scene batch; see the [animation instructions](../docs/tutorials/examples_tour.md#animations) |

## Shared helpers

| Module | Provides |
|--------|----------|
| `_json_io.py` | RFC 8259 JSON output without bare `NaN` or `Infinity` |
| `_paper_metrics.py` | The four swept constructions and their instantiation |
| `_benchmark_grid.py` | The matched wall-resolved benchmark grid |
| `_anec_window.py` | Prescribed coordinate/affine windows for finite null-energy diagnostics |
| `_paper_numbers_map.py` | `paper_numbers.tex` macros to `results/*.json`, for `emit_paper_numbers.py` |
