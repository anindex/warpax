# Examples

Runnable examples for curvature, energy conditions, geodesics, and shell design.

## Install for examples

From the repository root:

```bash
pip install -e ".[dev,viz,design,solver]"
```

| Extra | Needed for |
|-------|------------|
| `dev` | Test tools; runtime dependencies are installed by the base package |
| `viz` | Additional figure helpers; Matplotlib is a base dependency |
| `design` | Example 08 (interpax B-splines) |
| `solver` | Ill-conditioned metrics (WarpShell); optional for the numbered examples |

CPU is sufficient. Set `JAX_PLATFORMS=cpu` to select the reference backend.

## Suggested path for new users

Start with 01 and 03, then choose a grid, geodesic, or shell example. Runtimes
are approximate and include environment-dependent JAX compilation.

| # | Script | ~Runtime | What you learn |
|---|--------|----------|----------------|
| 01 | `01_minkowski_sanity.py` | 10 s | Curvature chain on flat space; `verify_point` on vacuum |
| 02 | `02_schwarzschild_verification.py` | 10 s | Non-trivial curvature; analytical Kretschmann cross-check |
| 03 | `03_alcubierre_analysis.py` | 10 s | Capped observer search and Eulerian EC at a wall point |
| 04 | `04_warp_drive_comparison.py` | 20 s | Six shipped warp metrics + Hawking-Ellis types + velocity scaling |
| 05 | `05_grid_analysis.py` | 25 s | Grid workflow and sampled SEC miss rates |
| 06 | `06_geodesic_through_warp_bubble.py` | 15 s | Diffrax geodesics, norm conservation, tidal eigenvalues |
| 07 | `07_custom_warp_metric.py` | 40 s | Subclass `ADMMetric`; wall-restricted diagnostics |
| 08 | `08_metric_design.py` | 7 s | Shape-function design via `design_metric` (B-spline reproduction) |
| 09 | `09_admissibility_diagnostics.py` | 30 s | Fuchs shell: constraints, ADM mass, junction, transport |
| 10 | `10_phase_diagram.py` | 2 min demo | T-shell parameter sweep + phase diagram (`--full` for paper quality) |

## Quick commands

```bash
# Fastest smoke check
python examples/01_minkowski_sanity.py

# Same entry point as docs/tutorials/quickstart.md
python examples/03_alcubierre_analysis.py

# Custom metric + PDF comparison figure
python examples/07_custom_warp_metric.py

# Metric design (writes tests/fixtures/alcubierre_optimal_parameters.npy)
python examples/08_metric_design.py

# Phase diagram demo (8x6 grid)
python examples/10_phase_diagram.py
python examples/10_phase_diagram.py --full   # 20x15, ~30 min on GPU
```

## Where outputs land

Most figures and arrays go to gitignored `examples/output/`. Example 08 writes
`tests/fixtures/alcubierre_optimal_parameters.npy`; example 10 writes
`results/phase_diagram/`. Check each script header before running it.

## Further reading

- [Quickstart](../docs/tutorials/quickstart.md): install plus a walkthrough of example 03.
- [Examples tour](../docs/tutorials/examples_tour.md): the MkDocs mirror of this page.
- [Custom metric tutorial](../docs/how-to/custom_metric_tutorial.md): pairs with example 07.
- [Interpreting EC results](../docs/how-to/interpreting_ec_results.md): how to read margins and Hawking-Ellis types.
- [Architecture](../docs/explanation/ARCHITECTURE.md): package map and design decisions.
