# Examples tour

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
| `solver` | Example 11's SciPy boundary-value solvers and ill-conditioned metric diagnostics |

CPU is sufficient. Set `JAX_PLATFORMS=cpu` to select the reference backend.

## Numbered examples

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
| 10 | `10_phase_diagram.py` | 2 min demo | Legacy T-shell diagnostic sweep (`--full` enlarges the sampled grid) |
| 11 | `11_elastic_shell.py` | Under 1 min | Einstein–elastic equilibrium, redshift, dragging, centrifugal tides and clock shifts |

Example 10 explores a prescription with unresolved source closure; enlarging
the grid does not establish a physical equilibrium. See the
[shell guide](../how-to/reproduce_warpshell_paper.md).

## Run an example

```bash
python examples/01_minkowski_sanity.py
python examples/03_alcubierre_analysis.py
python examples/07_custom_warp_metric.py
python examples/08_metric_design.py
python examples/10_phase_diagram.py         # 8x6 grid
python examples/10_phase_diagram.py --full  # 20x15 grid
python examples/11_elastic_shell.py        # Revised elastic-shell paper
```

## Outputs

Most figures and arrays go to `examples/output/`. Example 08 writes
`tests/fixtures/alcubierre_optimal_parameters.npy`; example 10 writes
`results/phase_diagram/`. Example 11 writes profiles and numerical comparisons
to `results/elastic_shell/` and reads the exact response coefficients supplied
there. It uses NumPy, SciPy and SymPy, with no JAX compilation or sibling
manuscript dependency. The [shell guide](../how-to/reproduce_warpshell_paper.md)
describes the datasets and the limits of the rotational expansion.
Check each script header before running it.

## Animations

Install the `manim` extra for animated scenes. The renderer uses FFmpeg for
GIF conversion, LaTeX for mathematical labels, Cairo for 2D scenes and an
OpenGL/EGL context for the 3D scenes.

```bash
pip install -e ".[manim]"
python scripts/render_all_scenes.py --scene NECMargin2D --quality l
python scripts/render_all_scenes.py --quality h --skip-gif
```

`--output-dir` selects the media directory. The
[render script](https://github.com/anindex/warpax/blob/main/scripts/render_all_scenes.py)
lists scene names and options. The numerical fields use JAX on CPU by default;
3D rendering still requires an OpenGL context.

## Further reading

- [Quickstart](quickstart.md): walkthrough of example 03.
- [Custom metrics](../how-to/custom_metric_tutorial.md): the recipe behind example 07.
- [Interpret results](../how-to/interpreting_ec_results.md): margins, types and numerical limits.
