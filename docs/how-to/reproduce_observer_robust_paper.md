# Reproducing "Observer-robust energy condition verification for warp drive spacetimes"

This guide maps the analysis scripts and generated artifacts for
[the paper](https://arxiv.org/abs/2602.18023) to **warpax v1.5.0**. The manuscript
is a separate sibling checkout, `../warpax_arxiv`; the input metrics are generated
from their stated parameters.
The [software archive](https://doi.org/10.5281/zenodo.18715933) records releases.

## Run the pipeline

From the repository root, install the numerical extras and select the interpreter:

```bash
uv sync --locked --extra dev --extra viz --extra design --extra solver
export PYTHON="$PWD/.venv/bin/python"
uv run python -m pytest
```

To regenerate numerical artifacts, run these stages in order:

```bash
bash reproduce_all.sh --stage core
bash reproduce_all.sh --stage ablation
bash reproduce_all.sh --stage figures
bash reproduce_all.sh --stage enclosures
```

`core` clears cached JSON/NPZ results and figures unless `--keep-cache` is set.
That flag preserves files; each script controls whether it reuses or recomputes
its inputs. The default full run executes core, ablation, and figures.
Enclosures are opt-in and can take hours. Use CPU for comparison with the shipped
numerical data. GPU curvature timings do not validate GPU eigensolver results.
The optional `gate` stage requires the corresponding manuscript sources and
matching numerical inputs.

## Scope of the numerical results

- Eight selected null rays are finite-segment diagnostics. Natário excludes
  9 of 50 fan rays, and its selected impact parameter remains unbracketed.
  No omitted-tail bound or complete-geodesic ANEC ranking is supplied.
- Four global wall enclosures have negative upper bounds. Only Natário <
  Alcubierre and Van den Broeck < Alcubierre are supported strict comparisons.
  The search budgets are 120,000 evaluations for Alcubierre and 4,096 for each
  other metric; none reaches the chosen width target of `1e-4`.
- Composite scores fix a slice, mask, rapidity cap, numerical floor, and affine
  normalization. Their magnitudes are not boost invariant.
- Garattini-Zatrimaylov uses its exact Type-I reduction throughout construction
  masks and miss-rate denominators. The comparison standardizes kinematics only.
- Exact pointwise quadratic speed laws require zero Eulerian momentum density
  and the stated fixed-profile/domain hypotheses. Other exponents are fits.

## Core stages

| Stage | Script | Produces |
|---|---|---|
| K1 | `run_velocity_sweep.py` | velocity-resolved Hawking-Ellis type map |
| K2 | `run_invariant_verification.py` | invariant all-observer benchmark |
| K3 | `validate_superluminal_classification.py` | Type-IV three-solver and 50-digit gate |
| K4 | `run_matched_benchmark.py` | matched wall-resolved benchmark, convergence |
| K5 | `run_shift_vorticity.py` | shift-vorticity decomposition (reads K1) |
| K6 | `run_anec_retained.py` | finite coordinate-path null-energy integrals |
| K6b | `run_anec_symplectic.py` | finite null-geodesic integrals, refinement and null-norm diagnostics |
| K7 | `run_quantum_inequality.py` | Ford-Roman comparison (reads K6) |
| K8 | `run_construction_verification.py` | cross-construction verification panel |
| K9 | `run_exoticity_ranking.py` | specified-slice composite and empirical speed fits (reads K1, K6b) |
| K10 | `derive_vorticity_type.py` | restricted vorticity/type model and empirical comparisons |
| K11 | `run_curvature_scaling.py` | empirical wall-curvature speed fits |
| K12 | `run_ssv_bound.py` | wall deficit and speed fit (reads K1) |
| K13 | `run_delta_crosscheck.py` | momentum discriminant against the eigensolver |
| K14 | `run_integrated_negative_energy.py` | slice-integrated negative energy |
| K15 | `run_rodal_sigma_resolved.py` | wall-resolved Rodal sigma sweep |
| K16 | `run_classifier_error_rate.py` | Jordan displacement limit, label check |
| K16b | `run_type_transitions.py` | LMI across the Type-II locus, analytic families |
| K16c | `run_lmi_agreement.py` | type-free LMI against the type-based route, every grid point |
| K16d | `run_closing_speed.py` | momentum-channel closing speed (reads K1) |
| K16e | `run_interval_lmi_spotcheck.py` | verdicts certified from the metric, 12 points |
| K16f | `run_interval_lmi_census.py` | sampled wall points, all four conditions |
| K17 | `emit_paper_numbers.py` | the `\newcommand` macros the manuscript uses |
| E1 | `run_enclosures.py` | certified global enclosures (opt-in stage) |

## Generated table catalog

Tables are written to the sibling manuscript directory. The catalog includes
supporting diagnostics beyond the manuscript tables.

| Diagnostic | `tables/` file | Script |
|---|---|---|
| Per-metric parameters | `params_caption_suffix` | hand-written, pinned by the gate |
| Wall resolution | `wall_resolution` | `run_wall_resolution.py` |
| Invariant benchmark | `invariant_benchmark` | `run_invariant_verification.py` |
| Velocity type structure | `velocity_type_structure` | `run_velocity_sweep.py` |
| Shift vorticity | `shift_vorticity` | `run_shift_vorticity.py` |
| Restricted vorticity model | `vorticity_mechanism` | `derive_vorticity_type.py` |
| Closing speed | `closing_speed` | `run_closing_speed.py` |
| Missed (uniform grid) | `missed_uniform` | `emit_diagnostic_tables.py` |
| Missed (wall-restricted) | `missed_wall_restricted` | `run_matched_benchmark.py` |
| Per-metric convergence | `convergence_per_metric` | `run_matched_benchmark.py` |
| Finite geodesic null-energy integral | `anec_symplectic` | `run_anec_symplectic.py` |
| Averaged and quantum | `averaged_quantum` | `run_quantum_inequality.py` |
| Cross-construction, matched | `construction_matched` | `run_construction_verification.py` |
| Cross-construction, native | `construction_native` | `run_construction_verification.py` |
| Specified-slice composite | `exoticity_ranking` | `run_exoticity_ranking.py` |
| Scaling laws | `scaling_laws` | `run_exoticity_ranking.py` |
| Single-term deficit | `ssv_bound` | `run_ssv_bound.py` |
| Integrated negative energy | `integrated_volume` | `run_integrated_negative_energy.py` |
| Curvature scaling | `curvature_scaling` | `run_curvature_scaling.py` |
| Curvature convergence | `curvature_convergence` | `run_curvature_convergence.py` |
| Per-diagnostic convergence | `diagnostic_convergence` | `run_diagnostic_convergence.py` |
| Extra convergence | `extra_convergence` | `run_exoticity_anec_convergence.py` |
| Clustered convergence | `clustered_convergence` | `run_clustered_convergence.py` |
| Richardson triplet | `convergence_richardson` | `emit_diagnostic_tables.py` |
| `N_starts` ablation | `nstarts` | `emit_diagnostic_tables.py` |
| Type breakdown | `type_breakdown` | `emit_diagnostic_tables.py` |
| `C1`/`C2` comparison | `c1_vs_c2` | `emit_diagnostic_tables.py` |
| Rodal native resolution | `rodal_resolution` | `run_rodal_native_resolution.py` |
| Rodal sigma sweep | `rodal_sigma_resolved` | `run_rodal_sigma_resolved.py` |
| LMI label check | `lmi_typefree` | `run_lmi_agreement.py` |
| Type transition | `type_transition` | `run_type_transitions.py` |
| Interval spot check | `interval_lmi_spotcheck` | `run_interval_lmi_spotcheck.py` |
| Interval census | `interval_lmi_census` | `run_interval_lmi_census.py` |
| Global enclosures | `enclosures` | `run_enclosures.py` (stage `enclosures`) |

## Generated figure catalog

Core scripts produce their own figures; the `figures` stage assembles the
comparison panels and emitted tables.

| Figure | Script |
|---|---|
| `velocity_summary.pdf` | `run_velocity_sweep.py` |
| `shift_vorticity.pdf` | `run_shift_vorticity.py` |
| `curvature_scaling.pdf` | `run_curvature_scaling.py` |
| `averaged_quantum.pdf` | `run_quantum_inequality.py` |
| `velocity_convergence_merged.pdf` | `reproduce_figures.py` |
| `alcubierre_nec_comparison.pdf`, `alcubierre_wec_comparison.pdf` | `reproduce_figures.py` |
| `vdb_nec_comparison.pdf`, `vdb_sec_comparison.pdf` | `reproduce_figures.py` |
| `missed_violations_vs_velocity.pdf` | `reproduce_figures.py` |
| `fibonacci_vs_bfgs_dec.pdf` | `reproduce_figures.py` |
| `alcubierre_tidal_forces.pdf`, `alcubierre_blueshift.pdf` | `reproduce_figures.py` |
| `rodal_dec_ablation.pdf` | `rodal_dec_ablation.py` |

## Consistency checks

`scripts/check_paper_numbers.py` compares manuscript numbers with their
source tables and JSON, checks required artifacts, and rejects stale inputs.
Recorded source/artifact hashes can establish compatibility when code changes do
not affect a retained result. Timestamps alone do not establish compatibility.
Missing required artifacts fail the check.

```bash
.venv/bin/python scripts/check_paper_numbers.py
```

Keep local interval error, global search brackets, and observed refinement
spreads separate when interpreting the tables.
