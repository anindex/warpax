# Examples tour

The `examples/` directory ships ten numbered scripts that walk through the
toolkit, from flat-space sanity checks to a full design-space phase diagram.
Each example is self-contained and runnable on CPU; `examples/README.md` lists
runtime estimates and which optional extras you need.

This page is a guided walkthrough: read the script header, run it, look at
the output, then move to the next one.

## Install once

```bash
pip install -e ".[dev,viz,design,solver]"
```

If a GPU is installed but you want bit-identical CPU runs, prefix any command
with `JAX_PLATFORMS=cpu`.

## First three runs

```bash
python examples/01_minkowski_sanity.py      # ~5 s
python examples/03_alcubierre_analysis.py   # ~5 s, same as the quickstart
python examples/07_custom_warp_metric.py    # ~30 s, custom metric + figure
```

After **01** you have verified the curvature chain returns exact zero on flat
space. **03** is the core result: at a bubble-wall point of the Alcubierre
metric, the Eulerian observer says WEC is satisfied while the worst-case
boosted observer finds a violation. **07** lifts the same machinery onto a
custom `ADMMetric` subclass and produces a publication-style comparison figure.

## What the rest cover

| # | Script | Purpose |
|---|--------|---------|
| 02 | `02_schwarzschild_verification.py` | Non-trivial curvature; analytical Kretschmann cross-check |
| 04 | `04_warp_drive_comparison.py` | Six warp drives side-by-side with Hawking--Ellis types |
| 05 | `05_grid_analysis.py` | Grid workflow; three-panel Eulerian vs robust figure |
| 06 | `06_geodesic_through_warp_bubble.py` | Diffrax geodesics, norm conservation, tidal eigenvalues |
| 08 | `08_metric_design.py` | Shape-function design via `design_metric` (B-spline) |
| 09 | `09_admissibility_diagnostics.py` | Fuchs shell: constraints, ADM mass, junction, transport |
| 10 | `10_phase_diagram.py` | T-shell parameter sweep + phase diagram (`--full` for paper-quality) |

## Where to go next

- After **03**: [Interpreting EC results](../how-to/interpreting_ec_results.md)
  for what margin signs and Hawking--Ellis types mean.
- After **07**: [Define a custom warp metric](../how-to/custom_metric_tutorial.md)
  for the full subclassing recipe.
- After **09** or **10**: [Reproducing the warp-shell admissibility paper](../how-to/reproduce_warpshell_paper.md)
  for the figure-by-figure mapping back to the published results.

The full canonical list (runtime estimates, install extras, suggested order)
lives in `examples/README.md` at the repository root.
