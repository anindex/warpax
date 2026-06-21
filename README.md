# warpax

[![arXiv](https://img.shields.io/badge/arXiv-2602.18023-brown)](https://arxiv.org/abs/2602.18023)
[![DOI](https://zenodo.org/badge/1162355401.svg)](https://doi.org/10.5281/zenodo.18715933)
[![CI](https://github.com/anindex/warpax/actions/workflows/ci.yml/badge.svg)](https://github.com/anindex/warpax/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**Observer-robust energy condition verification for warp drive spacetimes.**](https://arxiv.org/abs/2602.18023)

`warpax` certifies the energy-condition structure of warp-drive spacetimes for
*every* observer at once, from the eigenstructure of the mixed stress-energy tensor
$T^a{}_b$, with exact curvature from JAX forward-mode autodiff. Because the
Hawking--Ellis eigenvalue test never builds the Eulerian normal, it stays well-defined
at all warp speeds, including superluminal $v_s \ge 1$ where single-frame (Eulerian)
tools such as WarpFactory break down. The verdict is observer-independent: a Type-I
point is decided exactly for every observer, while a Type-IV point has no rest frame
and violates every condition unconditionally.

![Alcubierre Bubble Collapse](./figures/bubble_collapse.gif)

<p align="center"><em>Geodesic paths through a collapsing Alcubierre warp bubble, computed via warpax's autodiff curvature pipeline.</em></p>

## Highlights

- Frame-independent, all-observer energy-condition certification at every warp speed (including superluminal $v_s \ge 1$), from the eigenstructure of $T^a{}_b$ -- no Eulerian normal, no single-frame blind spots.
- Hawking--Ellis classification (Type I--IV) with explicit Type-IV detection, cross-checked by three eigensolvers and a 50-digit `mpmath` reference.
- Closed-form Type-I worst observer, with a multistart BFGS optimizer kept as a one-sided diagnostic at the residual non-Type-I points.
- Shift-vorticity control of the wall type: the vorticity of the ADM shift sets the Hawking--Ellis type, with universal $v_s$ scaling laws for the wall NEC deficit and curvature.
- Rigorous geodesic-integrated ANEC via a symplectic null integrator (with an on-cone witness), plus a Ford--Roman quantum-inequality diagnostic.
- Bondi four-momentum radiated-flux and Newman--Penrose peeling at null infinity (`warpax.bondi`).
- Exact curvature via forward-mode JAX autodiff -- no finite-difference stencils.
- Ten warp/shell metrics, constraint residuals, anisotropic TOV, ADM mass with falloff, Israel junctions, transport diagnostics, and source-first S-/T-shell construction with a five-criterion admissibility standard.

## Two papers, one toolkit

warpax backs two separate papers with disjoint claims. If you cite a result,
cite the paper it belongs to:

| | Certification paper ([arXiv:2602.18023](https://arxiv.org/abs/2602.18023)) | Companion note ([arXiv:2605.25417](https://arxiv.org/abs/2605.25417)) |
|---|---|---|
| **Question** | Which observers see energy-condition violations, at which warp speeds? | Can source-first shells satisfy the energy conditions at all? |
| **Results** | Frame-free all-velocity certifier; velocity-resolved type map; shift-vorticity -> type control ($f=\kappa\omega$); closed-form worst observer; exoticity ranking + $v_s$ scaling laws | S-/T-shell constructions from the Einstein constraints; five-criterion admissibility standard; boundary-cost analysis |
| **Modules** | `energy_conditions`, `geometry`, `averaged`, `quantum`, `analysis`, `geodesics`, `transport`; metrics Alcubierre / Natário / Van den Broeck / Rodal / Lentz / WarpShell / Garattini | `constraints` (S-/T-shell solvers), `tov`, `adm`, `junction`, `design`, `optimization`; `metrics/sshell.py`, `metrics/tshell.py` |
| **Examples** | 01–07 | 08–10 |

The S-/T-shells are constructed and certified in the companion note, not in the
certification paper; neither paper's results depend on the other's.

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
single geometric quantity, the vorticity of the ADM shift: the irrotational drive
is the unique globally Type-I geometry, while a rotational shift drives the wall to
Type IV. For Rodal's globally Type-I drive the Eulerian frame does not register
~72% of the wall weak-energy and ~73% of the wall dominant-energy violations seen
by boosted observers, an exact
eigenvalue statement rather than an optimizer artifact. A rigorous geodesic-integrated
ANEC (symplectic integrator with an on-cone witness) and a Ford--Roman comparison
preserve the ordering: every drive violates, and the irrotational Rodal geometry is
the mildest by one to two orders of magnitude.

### Invariant exoticity ranking and scaling laws

A boost-invariant ranking (NEC severity, Type-IV fraction, rigorous ANEC minimum)
places the irrotational Rodal drive about a factor of seventy below the bubble-wall
drives, driven by its vanishing Type-IV fraction and tiny averaged-null energy,
not by a milder pointwise NEC. Two universal $v_s$ laws follow: the wall NEC deficit
$\min(\rho+p_i) = -C\,v_s^2$ makes the Santiago--Schuster--Visser no-go quantitative
(measured, not asserted), and the wall curvature splits by the same vorticity
that sets the type: vortical walls grow as $v_s^2$, the irrotational Rodal wall
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
| D. EC margins | Frame-free NEC/WEC/DEC from Hawking--Ellis eigenvalue slacks (exact, cap-free at Type-I; valid at all $v_s$) |
| E. Global | Positive ADM mass, asymptotic falloff, tidal forces, invariant transport |

Fuchs constant-velocity shell: source-aware $\epsilon_{\mathcal{H}} \approx
3\times10^{-8}$; the bulk shell interior is Type-I and EC-compliant (0 of 13
probes violate), while the smoothing tail turns Type-IV. The source-first S-/
T-shells likewise pass criteria A--C and E with positive interior margins; the
binding cost is a cap-free Type-I dominant-energy deficit at the inner shell edge
($\approx -4.4\times10^{-4}$), localized at the smooth source--vacuum transition,
and the tilted T-shell's shift vorticity drives a Type-IV onset at its
low-density edge. These shell results belong to the companion note; see
[The boundary cost of source consistency](docs/explanation/boundary_cost.md).

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
| `metrics` | Nine warp/shell metrics: Natario, Lentz, Rodal, Van den Broeck, WarpShell, Fuchs, S-shell, T-shell, Garattini--Zatrimaylov (Alcubierre, Minkowski, and Schwarzschild ship in `benchmarks`, making ten warp metrics total) |
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
pytest                      # Full suite (1000+ tests across 33 modules)
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

The outer-edge ($r \ge R_2$) Type-IV gate (log-log slope $1.01 \pm 0.01$) and
the ANEC impact-parameter scan are reproduced by
`scripts/run_tshell_typeIV_gate.py` and `scripts/run_anec_impact_scan.py`.

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

@article{le2026boundary,
  title={On the boundary cost of source-consistent warp shells},
  author={Le, An T},
  journal={arXiv preprint arXiv:2605.25417},
  year={2026}
}
```
