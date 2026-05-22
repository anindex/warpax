# warpax

[![arXiv](https://img.shields.io/badge/arXiv-2602.18023-brown)](https://arxiv.org/abs/2602.18023)
[![DOI](https://zenodo.org/badge/1162355401.svg)](https://doi.org/10.5281/zenodo.18715933)
[![CI](https://github.com/anindex/warpax/actions/workflows/ci.yml/badge.svg)](https://github.com/anindex/warpax/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-888_passed-brightgreen)](#running-tests)

[**Observer-robust energy condition verification for warp drive spacetimes.**](https://arxiv.org/abs/2602.18023)

warpax uses JAX automatic differentiation to compute exact curvature tensors from
analytic warp drive metrics and performs continuous BFGS optimization over the full
timelike observer manifold to find worst-case energy condition violations.  This goes
beyond the standard Eulerian-observer approach (as in WarpFactory) by searching over
all physically admissible observers, parameterized by bounded rapidity in stereographic
coordinates, to detect violations that axis-aligned sampling can miss.

![Alcubierre Bubble Collapse](./figures/bubble_collapse.gif)

<p align="center"><em>Geodesic paths through a collapsing Alcubierre warp bubble, computed via warpax's autodiff curvature pipeline.</em></p>

## Highlights

- **Observer-robust EC verification**: multi-start BFGS over the full timelike observer manifold with Hawking--Ellis algebraic classification (Type I--IV)
- **Exact curvature**: forward-mode JAX autodiff from metric to stress-energy; no finite differences
- **9 warp metrics**: Alcubierre, Natario, Lentz, Rodal, Van den Broeck, WarpShell, Fuchs, S-shell, T-shell
- **Source-consistency modules**: Hamiltonian/momentum constraint residuals, anisotropic TOV equilibrium, ADM mass with falloff verification, Israel junction conditions, invariant transport diagnostics
- **Source-first shell construction**: Bernstein-parameterized source profiles with constraint-derived metric potentials (S-shell Class I, T-shell Class II)
- **Parameter sweep**: 2D design-space sweep over (compactness, thickness) with EC certification and phase-diagram visualization

## Quick start

```bash
# Create environment and install
conda create -n warpax python=3.12 -y && conda activate warpax
pip install -e ".[dev]"

# Run a quick example
python examples/01_minkowski_sanity.py
```

For a 5--10 minute walkthrough from install to seeing an energy condition violation,
see the [Quickstart tutorial](docs/tutorials/quickstart.md).

## Key results

### Observer-robust vs. Eulerian analysis

Eulerian-only energy condition analysis can miss significant violations.  For six warp drive metrics, warpax finds that 15--28% of DEC-violating grid points are invisible to the Eulerian observer.  The Fuchs constant-velocity shell shows 92% of shell-interior EC violations missed by Eulerian analysis.

### Custom metric design with robust EC validation

Users can define their own warp manifold by subclassing `ADMMetric` and run the
full verification pipeline.  Below, a Gaussian warp bubble is validated on a 24x24x4
grid.  The three panels compare SEC margins seen by the Eulerian observer (left) vs.
the worst-case boosted observer found by BFGS (center).  The right panel highlights
356 grid points where the Eulerian frame incorrectly reports SEC as satisfied ---
violations only visible to non-Eulerian observers.

![Gaussian Warp Grid Comparison](./figures/gaussian_warp_grid_comparison.png)

<p align="center"><em>SEC comparison for a custom Gaussian warp bubble (v<sub>s</sub> = 0.5).  Red regions are violations missed by Eulerian-only analysis.</em></p>

See [`examples/07_custom_warp_metric.py`](examples/07_custom_warp_metric.py) for the full worked example.

### Source-consistency audit

warpax v0.3+ provides a five-criterion admissibility standard for warp shells:

| Criterion | What it checks |
|-----------|---------------|
| A. Regularity | Metric $C^2$ continuity (thick-shell) or Israel junction conditions (thin-shell) |
| B. Constraints | Hamiltonian + momentum constraint residuals $\epsilon_{\mathcal{H}}$, $\epsilon_{\mathcal{M}}$ |
| C. Matter model | Interpretable source (anisotropic fluid, elastic shell, etc.) |
| D. EC margins | Observer-robust NEC/WEC/DEC via Hawking--Ellis + BFGS |
| E. Global diagnostics | Positive ADM mass, asymptotic falloff, tidal forces, invariant transport |

Applied to the Fuchs constant-velocity shell: $\epsilon_{\mathcal{H}} = 0.165$, 12/13 shell-interior points violate ECs under observer-robust certification.  The source-first T-shell achieves $\epsilon_{\mathcal{H}} \approx 5\times10^{-3}$ (33$\times$ improvement) with positive EC margins in the deep shell interior.

## Examples

| Script | Description |
|--------|-------------|
| `01_minkowski_sanity.py` | Flat-space sanity check (all ECs satisfied) |
| `02_schwarzschild_verification.py` | Schwarzschild ground-truth validation |
| `03_alcubierre_analysis.py` | Alcubierre warp drive EC analysis |
| `04_warp_drive_comparison.py` | Multi-metric comparison (6 warp drives) |
| `05_grid_analysis.py` | Grid-based EC verification |
| `06_geodesic_through_warp_bubble.py` | Geodesic integration with tidal forces and blueshift |
| `07_custom_warp_metric.py` | Custom warp manifold design with robust EC validation |
| `08_metric_design.py` | Shape-function metric design with constrained optimization |
| `09_admissibility_diagnostics.py` | Admissibility diagnostics on the Fuchs warp shell |
| `10_phase_diagram.py` | Parameter-space sweep and EC-admissible transport phase diagram |

```bash
python examples/01_minkowski_sanity.py
python examples/10_phase_diagram.py          # 8x6 demo (~5 min)
python examples/10_phase_diagram.py --full   # 20x15 sweep (~30 min GPU)
```

## Architecture

```
metrics -> geometry -> energy_conditions -> analysis
              |              |
          geodesics    classification (Hawking--Ellis)
              |
         transport / tidal / blueshift
```

| Package | Description |
|---------|-------------|
| `geometry` | JAX autodiff pipeline: metric $\to$ Christoffel $\to$ Riemann $\to$ Ricci $\to$ Einstein $\to$ $T_{\mu\nu}$; ADM 3+1 split; $C^2$ regularity diagnostics |
| `energy_conditions` | NEC/WEC/SEC/DEC via Hawking--Ellis classification, eigenvalue algebra, multi-start BFGS observer optimization |
| `metrics` | Nine warp drive metrics: Alcubierre (WarpShell), Natario, Lentz, Rodal, Van den Broeck, Fuchs, S-shell, T-shell |
| `constraints` | Hamiltonian + momentum constraint residuals; S-shell and T-shell constraint solvers (pure JAX) |
| `tov` | Anisotropic TOV equilibrium checker |
| `adm` | ADM mass with surface integral and asymptotic falloff verification |
| `junction` | Israel/Darmois junction conditions and surface stress-energy |
| `transport` | Invariant diagnostics: geodesic deviation, null round-trip asymmetry, blueshift hazard |
| `optimization` | Bernstein basis, multi-objective loss, EC soft/hard constraints, parameter sweep |
| `geodesics` | Timelike/null geodesic integration via Diffrax, tidal deviation, blueshift extraction |
| `design` | Differentiable shape-function parametrization with constrained BFGS optimizer |
| `analysis` | Eulerian vs. robust comparison, Richardson convergence, kinematic scalars |
| `io` | External metric loaders: WarpFactory (.mat), EinFields (checkpoint), Cactus (HDF5) |
| `visualization` | Matplotlib publication figures, Manim animations, phase diagram plots |
| `classify` | Bobrick--Martire subluminal/superluminal taxonomy |
| `averaged` | ANEC/AWEC line integrals along geodesics |
| `quantum` | Ford--Roman quantum inequality evaluator |

All metrics implement a common `MetricFunction` interface: a callable `(4,) -> (4,4)` mapping
coordinates $x^\mu$ to the covariant metric tensor $g_{\mu\nu}$.

## Running tests

```bash
pytest                      # Full suite (935 tests, 70 files)
pytest -m "not slow"        # Skip expensive grid tests
pytest -n auto              # Parallel execution
```

## Reproducing results

To pin the exact Python environment used to produce the published results:

```bash
export PYTHON=$(uv run which python)
bash reproduce_all.sh
```

Phases can be run individually:

```bash
bash reproduce_all.sh --phase 1   # Core computation
bash reproduce_all.sh --phase 2   # Ablation studies
bash reproduce_all.sh --phase 3   # Figure generation
```

Use `--keep-cache` to skip cache deletion and only recompute missing results.

For the per-figure, per-claim mapping that backs the warp-shell admissibility
paper (*Positive-energy warp shells and the boundary cost of source
consistency*), see the dedicated how-to guide:
[**Reproducing the warp-shell admissibility paper**](docs/how-to/reproduce_warpshell_paper.md).

## Documentation

warpax ships comprehensive documentation in [`docs/`](docs/), organized following the
[Diataxis](https://diataxis.fr/) framework:

### Tutorials

- [**Quickstart**](docs/tutorials/quickstart.md) -- 5--10 minutes from install to seeing an energy condition violation
- [**First curvature computation**](docs/tutorials/first_curvature_computation.md) -- full curvature chain on Minkowski as a warm-up

### How-to guides

- [**Define a custom warp metric**](docs/how-to/custom_metric_tutorial.md) -- subclass `ADMMetric` and run the verification pipeline
- [**Interpret EC results**](docs/how-to/interpreting_ec_results.md) -- read margin signs, Hawking--Ellis types, and worst-case observers
- [**Load an external metric**](docs/how-to/loading_external_metrics.md) -- use WarpFactory, EinFields, or Cactus data
- [**Reproduce the warp-shell admissibility paper**](docs/how-to/reproduce_warpshell_paper.md) -- per-figure, per-claim mapping to scripts and outputs

### Reference

- [**API reference**](docs/reference/index.md) -- autodoc of the public API
- [**Metric catalog**](docs/reference/metric_catalog.md) -- all nine shipped metrics
- [**Benchmarks**](docs/reference/benchmarks.md) -- asv regression harness

### Explanation

- [**Architecture**](docs/explanation/ARCHITECTURE.md) -- package structure and design decisions
- [**Theory: ADM 3+1 and Hawking--Ellis types**](docs/explanation/theory.md) -- mathematical background
- [**Release notes**](docs/explanation/release_notes.md) -- changelog with narrative context

## Manim visualizations

Animated 3D visualizations of warp bubble dynamics require the optional `manim`
dependencies:

```bash
# System dependencies (Ubuntu/Debian)
sudo apt install texlive-latex-extra texlive-fonts-recommended dvipng cm-super ffmpeg

# Python dependencies
pip install -e ".[manim]"

# Render all scenes
python scripts/render_all_scenes.py
```

Rendered videos and images are written to `media/` (not tracked by git).

## Citation

If you found this work useful, please consider citing:

```bibtex
@article{le2026warpax,
    title   = {Observer-robust energy condition verification for warp drive spacetimes},
    author  = {An T. Le},
    year    = {2026},
    eprint  = {arXiv:2602.18023},
    doi     = {10.5281/zenodo.18715933}
}
```
