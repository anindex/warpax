# warpax

[![arXiv](https://img.shields.io/badge/arXiv-2602.18023-brown)](https://arxiv.org/abs/2602.18023)
[![DOI](https://zenodo.org/badge/1162355401.svg)](https://doi.org/10.5281/zenodo.18715933)
[![CI](https://github.com/anindex/warpax/actions/workflows/ci.yml/badge.svg)](https://github.com/anindex/warpax/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**Observer-robust energy condition verification for warp drive spacetimes.**](https://arxiv.org/abs/2602.18023)

`warpax` certifies the all-observer energy-condition structure of warp-drive
spacetimes *frame-independently*, from the eigenstructure of the mixed
stress-energy tensor $T^a{}_b$, with exact curvature from JAX forward-mode
autodiff. The Hawking--Ellis eigenvalue test it is built on never constructs the
Eulerian normal, so it is well-defined at *all* warp speeds — including
$v_s \ge 1$, through and beyond the luminal transition, where coordinate-stationary
(Eulerian) tools break down. At a Type-I point an eigenvalue inequality decides
each energy condition exactly and for every observer; a Type-IV point has no rest
frame and violates every condition unconditionally. A multistart BFGS observer
optimizer over the timelike manifold is retained as a one-sided diagnostic at the
residual non-Type-I points. Unlike single-frame Eulerian checks (as used in
WarpFactory), it returns the observer-independent truth.

![Alcubierre Bubble Collapse](./figures/bubble_collapse.gif)

<p align="center"><em>Geodesic paths through a collapsing Alcubierre warp bubble, computed via warpax's autodiff curvature pipeline.</em></p>

## Highlights

- Frame-independent, all-observer, all-velocity energy-condition certification (including superluminal $v_s \ge 1$) from the eigenstructure of $T^a{}_b$ — the eigenvalue test never builds the Eulerian normal.
- Hawking--Ellis classification (Type I--IV) with explicit Type-IV detection, certified physical by a three-solver (`eig`, LAPACK `zggev` pencil) and 50-digit `mpmath` cross-check.
- Closed-form Type-I worst observer ($\sinh^2\zeta_{\rm th} = \rho/|\rho+p_i|$), validated against the BFGS optimizer.
- Shift-vorticity analysis: the vorticity of the ADM shift controls the Hawking--Ellis type of the bubble wall — the imaginary part of the Type-IV eigenvalue pair is linear in the vorticity, $f = \kappa\,\omega$ with slope $\kappa \approx 0.06$ (established in a controlled pure-rotation limit), so the Type-I $\to$ Type-IV transition itself is linear in the shift vorticity.
- Rigorous geodesic-integrated ANEC via a structure-preserving symplectic null integrator (conserves $g(k,k)$ to ~machine precision where adaptive RK drifts off the cone), reported with an on-cone witness $\max|g(k,k)|$; plus a Ford--Roman quantum-inequality diagnostic.
- Cross-construction all-observer verification (Fuchs, WarpShell, Garattini--Zatrimaylov de Sitter, Rodal, Alcubierre) with a wall-resolution gate, and a boost-invariant exoticity ranking with universal $v_s$ scaling laws. Even the de Sitter bubble — certified at its matched $v_s = H R$ averaged-condition regime — is Type-IV walled, with the Eulerian frame missing ~63% of the wall weak-energy violations.
- Universal $v_s$ scaling of the wall curvature invariants (Kretschmann, Weyl-squared, Ricci-squared), split by shift vorticity — the vortical walls grow as $v_s^2$, the irrotational Rodal wall as $v_s^4$; and the Santiago--Schuster--Visser no-go made quantitative, $\min(\rho+p_i) = -C\,v_s^2$.
- Exact curvature via forward-mode JAX autodiff (no finite-difference stencils); multistart BFGS retained as a one-sided diagnostic at non-Type-I points.
- Ten warp/shell metrics: Alcubierre (in `warpax.benchmarks`, with Minkowski and Schwarzschild), plus Natario, Lentz, Rodal, Van den Broeck, WarpShell, Fuchs, S-shell, T-shell, and Garattini--Zatrimaylov (de Sitter) in `warpax.metrics`.
- Hamiltonian + momentum constraint residuals, anisotropic TOV, ADM mass with falloff, Israel junctions, invariant transport diagnostics.
- Source-first shells: Bernstein-parameterized profiles with constraint-derived metric potentials (S-shell, T-shell); 2D design sweep over (compactness, thickness) with EC certification and phase-diagram plots. These constructions and their boundary-cost analysis are the subject of the companion note (arXiv:2605.25417); warpax ships them as toolkit and does not certify them in the main paper.

## Quick start

```bash
# Create environment and install
conda create -n warpax python=3.12 -y && conda activate warpax
pip install -e ".[dev,viz,design,solver]"

# Run a quick example
python examples/01_minkowski_sanity.py
```

See [`examples/README.md`](examples/README.md) for a numbered learning path (01-10)
and which optional extras each script needs.

For a 5--10 minute walkthrough from install to seeing an energy condition violation,
see the [Quickstart tutorial](docs/tutorials/quickstart.md).

## Key results

### Frame-independent type map across the luminal transition

On matched, wall-resolved grids, the Rodal irrotational geometry is globally
Hawking--Ellis Type I at every speed from $v_s = 0.1$ to $2.5$, while the
Alcubierre/Natario/Van den Broeck bubble walls are Type-IV dominated (no rest
frame, no invariant energy density) at every speed. The split is controlled by a
single geometric quantity — the vorticity of the ADM shift: the irrotational drive
is the unique globally Type-I geometry, while a rotational shift drives the wall to
Type IV. For Rodal's globally Type-I drive the Eulerian frame does not register
~72% of the wall weak-energy and ~73% of the wall dominant-energy violations seen
by boosted observers — an exact
eigenvalue statement, not an optimizer artefact. A rigorous geodesic-integrated
ANEC (symplectic integrator with an on-cone witness) and a Ford--Roman comparison
preserve the ordering: every drive violates, and the irrotational Rodal geometry is
the mildest by one to three orders of magnitude.

### Invariant exoticity ranking and scaling laws

A boost-invariant ranking (NEC severity, Type-IV fraction, rigorous ANEC minimum)
places the irrotational Rodal drive about a factor of sixty below the bubble-wall
drives — driven by its vanishing Type-IV fraction and tiny averaged-null energy,
not by a milder pointwise NEC. Two universal $v_s$ laws follow: the wall NEC deficit
$\min(\rho+p_i) = -C\,v_s^2$ makes the Santiago--Schuster--Visser no-go quantitative
(measured, not asserted), and the wall curvature splits by the same vorticity
that sets the type — the vortical walls grow as $v_s^2$, the irrotational Rodal wall
as $v_s^4$ ($R^2 \ge 0.996$).

### Observer-robust vs Eulerian

Across six warp drives, 15--28% of DEC-violating grid points are invisible to the
Eulerian observer. The Fuchs constant-velocity shell hides 92% of its
shell-interior violations from an Eulerian-only check.

### Custom metrics

Subclass `ADMMetric` and run the full pipeline. The figure below validates a
Gaussian warp bubble on a 24x24x4 grid: SEC margins from the Eulerian observer
(left), from the worst-case boosted observer found by BFGS (center), and 356 grid
points the Eulerian frame reports as SEC-satisfied while the boosted observer
sees them violated (right).

![Gaussian Warp Grid Comparison](./figures/gaussian_warp_grid_comparison.png)

<p align="center"><em>SEC comparison for a custom Gaussian warp bubble (v<sub>s</sub> = 0.5). Red marks violations the Eulerian frame misses.</em></p>

See [`examples/07_custom_warp_metric.py`](examples/07_custom_warp_metric.py).

### Shell admissibility

`warpax` ships a five-criterion admissibility standard for warp shells:

| Criterion | Checks |
|-----------|--------|
| A. Regularity | $C^2$ metric continuity (thick) or Israel conditions (thin) |
| B. Constraints | Hamiltonian + momentum residuals $\epsilon_{\mathcal{H}}$, $\epsilon_{\mathcal{M}}$ |
| C. Matter model | Identifiable source (anisotropic fluid, elastic shell) |
| D. EC margins | Observer-robust NEC/WEC/DEC via Hawking--Ellis + BFGS |
| E. Global | Positive ADM mass, asymptotic falloff, tidal forces, invariant transport |

Fuchs constant-velocity shell: $\epsilon_{\mathcal{H}} = 0.165$; 12 of 13
shell-interior points violate ECs under observer-robust certification. The
source-first T-shell drops $\epsilon_{\mathcal{H}}$ to ~$5\times10^{-3}$ (33x
improvement) with positive EC margins in the deep interior. This five-criterion
admissibility study is developed in the companion note (arXiv:2605.25417); the
figures here illustrate the warpax toolkit that feeds it, not a result of the
main observer-robust-certification paper.

## Examples

See [`examples/README.md`](examples/README.md) for runtime estimates, install extras,
and a suggested order for new users.

| Script | Description |
|--------|-------------|
| `01_minkowski_sanity.py` | Flat-space sanity check (all ECs satisfied) |
| `02_schwarzschild_verification.py` | Schwarzschild ground-truth validation |
| `03_alcubierre_analysis.py` | Alcubierre warp drive EC analysis (**quickstart entry**) |
| `04_warp_drive_comparison.py` | Multi-metric comparison (six warp drives) |
| `05_grid_analysis.py` | Grid-based EC verification + comparison figure |
| `06_geodesic_through_warp_bubble.py` | Geodesic integration with tidal forces |
| `07_custom_warp_metric.py` | Custom warp manifold + robust EC validation |
| `08_metric_design.py` | Shape-function metric design (B-spline reproduction) |
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
| `metrics` | Nine warp/shell metrics: Natario, Lentz, Rodal, Van den Broeck, WarpShell, Fuchs, S-shell, T-shell, Garattini--Zatrimaylov (Alcubierre + Minkowski + Schwarzschild ship in `benchmarks`) |
| `constraints` | Hamiltonian + momentum constraint residuals; S-shell and T-shell constraint solvers (pure JAX) |
| `tov` | Anisotropic TOV equilibrium checker |
| `adm` | ADM mass with surface integral and asymptotic falloff verification |
| `junction` | Israel/Darmois junction conditions and surface stress-energy |
| `transport` | Invariant diagnostics: geodesic deviation, null coordinate-time asymmetry, blueshift hazard |
| `optimization` | Bernstein basis, multi-objective loss, EC soft/hard constraints, parameter sweep |
| `geodesics` | Timelike/null geodesic integration via Diffrax, tidal deviation, blueshift extraction |
| `design` | Differentiable shape-function parametrization with constrained BFGS optimizer |
| `analysis` | Eulerian vs. robust comparison, Richardson convergence, kinematic scalars |
| `io` | External metric loaders: WarpFactory (.mat), EinFields (checkpoint), Cactus (HDF5) |
| `visualization` | Matplotlib publication figures, Manim animations, phase diagram plots |
| `classify` | Bobrick--Martire subluminal/superluminal taxonomy |
| `averaged` | ANEC/AWEC null-ray and geodesic line integrals |
| `quantum` | Ford--Roman quantum inequality evaluator |

All metrics implement a common `MetricFunction` interface: a callable `(4,) -> (4,4)` mapping
coordinates $x^\mu$ to the covariant metric tensor $g_{\mu\nu}$.

## Running tests

```bash
pytest                      # Full suite (950+ tests across 34 modules)
pytest -m "not slow"        # Skip @slow grid tests (~50 s with -n auto)
pytest -m smoke             # Visualization import / render smoke tests
pytest -n auto              # Parallel execution
```

## Reproducing results

To pin the exact Python environment used to produce the published results:

```bash
export PYTHON=$(uv run which python)
bash reproduce_all.sh
```

Stages can be run individually:

```bash
bash reproduce_all.sh --stage core      # Core computation
bash reproduce_all.sh --stage ablation  # Ablation studies
bash reproduce_all.sh --stage figures   # Figure generation
```

Use `--keep-cache` to skip cache deletion and only recompute missing results.

For the per-figure, per-claim mapping that backs the warp-shell admissibility
paper (*On the boundary cost of source-consistent warp shells*), see the dedicated how-to guide:
[**Reproducing the warp-shell admissibility paper**](docs/how-to/reproduce_warpshell_paper.md).

## Documentation

warpax ships full documentation in [`docs/`](docs/), organized following the [Diataxis](https://diataxis.fr/) framework:

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
- [**Metric catalog**](docs/reference/metric_catalog.md) -- all ten shipped metrics
- [**Benchmarks**](docs/reference/benchmarks.md) -- asv regression harness

### Explanation

- [**Architecture**](docs/explanation/ARCHITECTURE.md) -- package structure and design decisions
- [**Theory: ADM 3+1 and Hawking--Ellis types**](docs/explanation/theory.md) -- mathematical background
- [**Release notes**](docs/explanation/release_notes.md) -- version history and release summary

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
@article{le2026observer,
  title={Observer-robust energy condition verification for warp drive spacetimes},
  author={Le, An T},
  journal={arXiv preprint arXiv:2602.18023},
  year={2026}
}
```
