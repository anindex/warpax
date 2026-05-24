# Scripts catalog

All scripts assume the repository root as working directory (as `reproduce_all.sh` does).

## Pipeline (`reproduce_all.sh`)

### Core computation

| Script | Output |
|--------|--------|
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
| `run_lentz_wall_assessment.py` | `results/lentz_wall_assessment.json` |
| `run_wall_restricted_analysis.py` | `results/wall_restricted_analysis.json` |

### Figures

| Script | Output |
|--------|--------|
| `reproduce_figures.py` | `figures/*.pdf` |
| `generate_vdb_comparison_figures.py` | Van den Broeck comparison figures |

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
