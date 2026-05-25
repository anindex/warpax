# Examples

Runnable scripts that tour warpax from flat-space sanity checks through
observer-robust energy conditions, geodesics, custom metrics, shell
admissibility, and design-space sweeps.

## Install for examples

From the repository root:

```bash
pip install -e ".[dev,viz,design,solver]"
```

| Extra | Needed for |
|-------|------------|
| `dev` | All examples (JAX, pytest) |
| `viz` | Examples 05-07, 10 (Matplotlib figures) |
| `design` | Example 08 (interpax B-splines) |
| `solver` | Ill-conditioned metrics (WarpShell); optional for the numbered examples |

CPU is enough for every script below. Set `JAX_PLATFORMS=cpu` if a GPU
backend is installed but you want reproducible CPU runs.

## Suggested path for new users

Work through the numbered scripts in order the first time. Each example
introduces one layer of the stack before the next combines them.

| # | Script | ~Runtime | What you learn |
|---|--------|----------|----------------|
| 01 | `01_minkowski_sanity.py` | 5 s | Curvature chain on flat space; `verify_point` on vacuum |
| 02 | `02_schwarzschild_verification.py` | 5 s | Non-trivial curvature; analytical Kretschmann cross-check |
| 03 | `03_alcubierre_analysis.py` | 5 s | **Core story**: robust vs Eulerian EC at a bubble-wall point |
| 04 | `04_warp_drive_comparison.py` | 30 s | Six shipped warp metrics + Hawking--Ellis types + velocity scaling |
| 05 | `05_grid_analysis.py` | 1-2 min | Grid workflow; 3-panel Eulerian vs robust figure |
| 06 | `06_geodesic_through_warp_bubble.py` | 30 s | Diffrax geodesics, norm conservation, tidal eigenvalues |
| 07 | `07_custom_warp_metric.py` | 20-30 s | Subclass `ADMMetric`; wall-restricted diagnostics |
| 08 | `08_metric_design.py` | 10 s | Shape-function design via `design_metric` (B-spline reproduction) |
| 09 | `09_admissibility_diagnostics.py` | 15 s | Fuchs shell: constraints, ADM mass, junction, transport |
| 10 | `10_phase_diagram.py` | 5 min demo | T-shell parameter sweep + phase diagram (`--full` for paper quality) |

After **03**, you have seen the main research result (observer-robust EC).
**05** and **07** are the best next steps for publication-style figures.
**09** and **10** cover the source-consistency and shell-design stack.

## Quick commands

```bash
# Fastest “does it work?” check
python examples/01_minkowski_sanity.py

# Same entry point as docs/tutorials/quickstart.md
python examples/03_alcubierre_analysis.py

# Custom metric + PDF comparison figure
python examples/07_custom_warp_metric.py

# Metric design golden path (writes tests/fixtures/alcubierre_optimal_parameters.npy)
python examples/08_metric_design.py

# Phase diagram demo (8×6 grid)
python examples/10_phase_diagram.py
python examples/10_phase_diagram.py --full   # 20×15, ~30 min on GPU
```

## Where outputs land

Each script writes its figures and arrays under `examples/output/`
(gitignored). Example 10 writes its phase diagram to `output/` at the
repository root. Check the header of any script you run for the exact path.

## Further reading

- [Quickstart](../docs/tutorials/quickstart.md): install plus a walkthrough of example 03.
- [Examples tour](../docs/tutorials/examples_tour.md): the MkDocs mirror of this page.
- [Custom metric tutorial](../docs/how-to/custom_metric_tutorial.md): pairs with example 07.
- [Interpreting EC results](../docs/how-to/interpreting_ec_results.md): how to read margins and Hawking--Ellis types.
- [Architecture](../docs/explanation/ARCHITECTURE.md): package map and design decisions.
