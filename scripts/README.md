# Scripts catalog

All scripts assume the repository root as working directory (as `reproduce_all.sh` does).

## Canonical certification entrypoints (start here)

The frame-independent, all-velocity energy-condition certification is exposed
as a one-call public API:

```python
from warpax import certify
from warpax.metrics import RodalMetric
r = certify(RodalMetric(v_s=2.0, R=1.0, sigma=8.0))   # works at any v_s, incl. >= 1
```

The paper's three contribution results are produced by these scripts (the others
below are supporting ablations / convergence studies, retained for
reproducibility):

| Script | Produces | Paper artifact |
|--------|----------|----------------|
| `run_velocity_sweep.py` | `tables/velocity_type_structure.tex`, `figures/velocity_type_structure.pdf`, `figures/rodal_invariant_margins.pdf` | K1: type/EC structure across the luminal transition |
| `run_invariant_verification.py` | `tables/invariant_benchmark.tex` | K2: invariant all-observer verification (single-frame miss, E_-) |
| `validate_superluminal_classification.py` | `results/superluminal_gate*` | K3: Type-IV trustworthiness gate (3-solver + 50-digit) |
| `run_matched_benchmark.py` | `tables/missed_wall_restricted.tex`, `tables/convergence_per_metric.tex` | K4: matched wall-resolved benchmark + per-metric convergence |
| `run_shift_vorticity.py` | `tables/shift_vorticity.tex`, `figures/shift_vorticity.pdf`, `results/shift_vorticity.json` | K5: shift vorticity controls the Hawking-Ellis type (reads cached `velocity_sweep.json`) |

Several convergence scripts overlap (`run_convergence.py`,
`run_clustered_convergence.py`, `run_rodal_matched_resolution.py`,
`run_warpshell_convergence.py`, `run_tshell_convergence.py`); for new work
prefer `run_matched_benchmark.py` (cross-metric) and `run_velocity_sweep.py`.

## Pipeline (`reproduce_all.sh`)

### Core computation

| Script | Output |
|--------|--------|
| `run_anec_retained.py` | `results/anec/retained.json` (K6: ANEC line integrals along null rays) |
| `run_anec_symplectic.py` | `results/anec/retained_symplectic.json`, `tables/anec_symplectic.tex` (K6b: rigorous geodesic-integrated ANEC, symplectic + on-cone witness) |
| `run_quantum_inequality.py` | `results/quantum/ford_roman.json`, `tables/averaged_quantum.tex`, `figures/averaged_quantum.pdf` (K7: Ford-Roman quantum-inequality diagnostic, reads K6) |
| `run_construction_verification.py` | `results/construction_verification.json`, `tables/construction_verification.tex` (K8: cross-construction all-observer verification) |
| `run_exoticity_ranking.py` | `results/exoticity_ranking.json`, `tables/exoticity_ranking.tex`, `tables/scaling_laws.tex` (K9: boost-invariant exoticity ranking + v_s scaling laws, reads K1/K6b) |
| `derive_vorticity_type.py` | `results/vorticity_type_analytic.json`, `figures/vorticity_type_mechanism.pdf` (K10: vorticity -> Type-IV mechanism f = kappa*omega; cross-metric entries record theta, sigma, sigma/omega, and the excess Im/(kappa*omega)) |
| `run_curvature_scaling.py` | `results/curvature_scaling.json`, `tables/curvature_scaling.tex`, `figures/curvature_scaling.pdf` (K11: universal v_s scaling of wall curvature invariants) |
| `run_ssv_bound.py` | `results/ssv_bound.json`, `tables/ssv_bound.tex` (K12: SSV NEC lower-bound saturation, reads K1) |
| `run_analysis.py` | `results/comparison_table.json` |
| `run_convergence.py` | `results/convergence_data.json` |
| `run_kinematic_scalars.py` | kinematic scalar NPZ/JSON under `results/` |
| `run_geodesics.py` | `results/geodesic_scaling.json` |
| `run_clustered_convergence.py` | `results/clustered_convergence_*.json` |

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
| `emit_diagnostic_tables.py` | `tables/{missed_uniform,type_breakdown,nstarts,c1_vs_c2,convergence_richardson}.tex` from cached `results/*.json` |

## Paper verification

Documented in [docs/how-to/reproduce_warpshell_paper.md](../docs/how-to/reproduce_warpshell_paper.md):

| Script | Output |
|--------|--------|
| `verify_fuchs.py` | `results/fuchs_verification_report.json` |
| `verify_proposals.py` | `results/proposals_verification_report.json` |
| `run_sshell_sweep.py` | S-shell sweep under `results/` |
| `run_anec_profiles.py` | ANEC profile data |
| `run_anec_geodesic_check.py` | geodesic ANEC checks |

## Rendering / showcase

| Script | Notes |
|--------|-------|
| `render_all_scenes.py` | Manim scene batch (see README) |
| `render_manim_scenes.sh` | Shell wrapper for Manim |
| `generate_showcase.py` | Delegates to render pipeline |

## Ad-hoc research (not in `reproduce_all.sh`)

| Script | Purpose |
|--------|---------|
| `run_delta_tau_scan.py` | Delta-tau parameter scan |
| `run_tshell_convergence.py` | T-shell convergence study |
| `run_tshell_kterm_angular.py` | T-shell angular k-term |
| `run_v0_ablation.py` | v0 ablation |
| `run_fuchs_canonical.py` | Fuchs canonical radial sweep |
| `run_landscape.py` | EC landscape exploration |
| `run_lentz.py` | Lentz metric radial sweep |
| `run_lentz_distance_comparison.py` | Lentz L1 vs L2 distance |
| `run_error_budget.py` | Numerical error budget |
| `run_criterion_e_verification.py` | Criterion E verification |
| `constraint_residual_verification.py` | ADM constraint residuals → `results/` |
| `fuchs_kernel_comparison.py` | Fuchs kernel comparison |
| `verify_rodal.py` | Standalone Rodal verification |
| `summarize_results.py` | Manual results inspection |
| `dump_hlo_curvature.py` / `.sh` | XLA HLO profiling for curvature |

## Shared helpers

| Module | Used by |
|--------|---------|
| `_radial_sweep.py` | `run_fuchs_canonical.py`, `run_lentz.py`, `run_landscape.py`, `fuchs_kernel_comparison.py`, `run_lentz_distance_comparison.py` |
