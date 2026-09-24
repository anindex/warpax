# Reproducing "Observer-robust energy condition verification for warp drive spacetimes"

This guide describes the **warpax v1.5.1** analysis workflow associated with
[the paper](https://arxiv.org/abs/2602.18023). The manuscript is a separate sibling
checkout, `../warpax_arxiv`; the input metrics are generated from their stated
parameters.
The [software archive](https://doi.org/10.5281/zenodo.18715933) records releases.

The CQG manuscript received provisional acceptance on September 21, 2026.
The current local manuscript includes corrections beyond the public arXiv
version. Compare calculations with the equations, parameters and numerical
precision of the manuscript version being cited.

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

`core` clears cached JSON/NPZ results and figures unless `--keep-cache` is set;
`enclosures.json` and `results/elastic_shell/` are preserved. The flag skips
deletion; each script controls whether it reuses or recomputes its inputs.
The default full run executes core, ablation, and figures.
Enclosures are opt-in and can take hours. Use CPU for comparison with the shipped
numerical data. GPU curvature timings do not validate GPU eigensolver results.
The optional `gate` stage requires the corresponding manuscript sources and
matching numerical inputs.

## Scope of the numerical results

Use the [interpretation guide](interpreting_ec_results.md) for margins,
frames, normalization and finite-search limits, and the
[theory guide](../explanation/theory.md#warp-wall-conclusions-and-limits) for
the hypotheses behind speed-scaling and algebraic-type statements.

Report each calculation's grid, observer normalization, tolerances and
excluded rays. Stored results are numerical datasets, not evidence that
an arbitrary later checkout reproduces them. Changed computational paths
require fresh checks; unchanged data can be reused after checking inputs
and dependencies.

## Core stages

The [script catalog](https://github.com/anindex/warpax/blob/main/scripts/README.md)
lists every core command in runner order, its stage label, and its output,
followed by ablations and optional stages. In a local checkout, use
`scripts/README.md` and `reproduce_all.sh` from the same version. The artifact
catalogs below map manuscript tables and figures to their generators.

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
| Uniform-grid sampled values and maximum departure from mean | `convergence_richardson` | `emit_diagnostic_tables.py` |
| `N_starts` ablation | `nstarts` | `emit_diagnostic_tables.py` |
| Type breakdown | `type_breakdown` | `emit_diagnostic_tables.py` |
| Rodal native resolution | `rodal_resolution` | `run_rodal_native_resolution.py` |
| Rodal sigma sweep | `rodal_sigma_resolved` | `run_rodal_sigma_resolved.py` |
| LMI label check | `lmi_typefree` | `run_lmi_agreement.py` |
| Type transition | `type_transition` | `run_type_transitions.py` |
| Interval spot check | `interval_lmi_spotcheck` | `run_interval_lmi_spotcheck.py` |
| Interval census | `interval_lmi_census` | `run_interval_lmi_census.py` |
| Global enclosures | `enclosures` | `run_enclosures.py` (stage `enclosures`) |

The `convergence_richardson` table reports
observed grid spread, with no fitted order, extrapolation, or continuum error
bound.

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
