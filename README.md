# warpax

[![arXiv](https://img.shields.io/badge/arXiv-2602.18023-brown)](https://arxiv.org/abs/2602.18023)
[![DOI](https://zenodo.org/badge/1162355401.svg)](https://doi.org/10.5281/zenodo.18715933)
[![CI](https://github.com/anindex/warpax/actions/workflows/ci.yml/badge.svg)](https://github.com/anindex/warpax/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

JAX tools for curvature, energy conditions, and geodesics in prescribed
spacetimes. `warpax` uses automatic differentiation to evaluate the Einstein
tensor, then tests energy conditions across observer directions.
See the [changelog](CHANGELOG.md) for version **1.5.0**.

![Alcubierre bubble: Eulerian energy density and normalized null-energy margin](https://raw.githubusercontent.com/anindex/warpax/main/figures/wall_velocity_sweep.gif)

*Alcubierre wall and speed sweep. Height and color use a signed logarithmic
scale; each frame is a separate metric evaluation.*

## Install and run

Python 3.12 or newer:

```bash
pip install warpax
```

```python
from warpax import certify
from warpax.metrics import RodalMetric

metric = RodalMetric(v_s=0.5, R=1.0, sigma=8.0)
result = certify(metric, shape=(16, 16, 16))
print(result.type_fractions)
print(result.invariant_nec_min)  # Minimum over sampled wall Type-I points.
```

The grid is a quick demonstration. Resolve the wall and vary the grid before
interpreting extrema or type fractions. Geometric units are `G = c = 1`, with
metric signature `(-,+,+,+)` and float64 enabled at import.

For development and the examples:

```bash
git clone https://github.com/anindex/warpax.git
cd warpax
uv sync --locked --extra dev --extra viz --extra design --extra solver
uv run python examples/01_minkowski_sanity.py
```

| Extra | Use |
|---|---|
| `dev` | Tests and Ruff |
| `design`, `solver` | Interpolated metrics, shape design, and SciPy solvers |
| `viz`, `manim` | Plotting helpers and animations |
| `interop`, `einfields` | External metric files and model checkpoints |
| `docs`, `bench` | Documentation and performance benchmarks |

## What it computes

- Curvature and stress-energy from callable metrics, including ADM fields.
- Hawking–Ellis classification, Type-I eigenvalue margins, and linear matrix
  inequality (LMI) tests that do not require an algebraic type assignment.
- Continuous observer searches, exact checks for supplied rational certificates,
  and interval bounds on supported metric domains.
- Timelike and null geodesics, finite-segment energy integrals, tidal effects,
  and quantum-inequality reference diagnostics.
- Constraint residuals, TOV equilibrium, ADM mass, junction conditions,
  source-first shell construction, and metric optimization.
- WarpFactory, EinFields, and Cactus imports; Matplotlib and Manim visualization;
  Bondi flux and Newman–Penrose diagnostics.

Numerical classification and LMI searches use finite precision and tolerances.
An unsuccessful certificate search is inconclusive. The exact energy-condition
verdict is independent of the observer, but numerical margin magnitudes depend
on the tetrad and normalization. Observer optimization also depends on its
rapidity cap. The Eulerian normal remains timelike wherever the ADM lapse and
spatial metric are valid, including regions where the coordinate vector
`∂t` is spacelike.

The null integrals cover finite segments. They establish no
complete-geodesic ANEC sign or ranking. Grid spreads and polished extrema are
numerical diagnostics; only reported interval bounds support continuum claims.
Vorticity alone does not determine algebraic type, and the exact pointwise
quadratic speed law requires zero momentum and fixed profiles and domains.

## Documentation

Start with the [quickstart](docs/tutorials/quickstart.md) or
[numbered examples](examples/README.md). Each metric implements a callable
`(4,) -> (4, 4)` from spacetime coordinates to the covariant metric tensor.

| Task | Guide |
|---|---|
| Understand the calculations | [Theory](docs/explanation/theory.md), [architecture](docs/explanation/ARCHITECTURE.md) |
| Choose or implement a metric | [Catalog](docs/reference/metric_catalog.md), [custom metric](docs/how-to/custom_metric_tutorial.md) |
| Read results or load data | [Energy-condition results](docs/how-to/interpreting_ec_results.md), [external metrics](docs/how-to/loading_external_metrics.md) |
| Inspect the API or benchmark it | [API](docs/reference/index.md), [benchmarks](docs/reference/benchmarks.md) |
| Reproduce the papers | [Observer-robust energy conditions](docs/how-to/reproduce_observer_robust_paper.md), [source-consistent shells](docs/how-to/reproduce_warpshell_paper.md) |

[Example 07](examples/07_custom_warp_metric.py) implements a Gaussian warp metric.
[Examples 08–10](examples/README.md) cover shape design and shell diagnostics.
Animation commands and system dependencies are in the
[examples guide](docs/tutorials/examples_tour.md) and
[render script](scripts/render_all_scenes.py).

## Checks and reproduction

```bash
uv run python -m pytest
uv run ruff check src/ tests/ scripts/ benchmarks/ examples/
uv run ruff format --check src/ tests/ scripts/ benchmarks/ examples/
uv sync --locked --extra docs --extra design
uv run mkdocs build --strict
```

Tests run in parallel by default. Use `-n 2` to limit workers, or select a module
such as `tests/test_slemma.py`.

The [script guide](scripts/README.md) lists reproduction commands and required
extras. `reproduce_all.sh` clears generated caches by default; `--keep-cache`
retains them, and `--stage` selects a stage. Interval enclosures are a separate,
expensive stage. The optional paper-number check requires the corresponding
manuscript sources and matching numerical inputs.

## Citation

Use [CITATION.cff](CITATION.cff) for software metadata and cite the relevant paper:

- An T. Le, [Observer-robust energy condition verification for warp drive
  spacetimes](https://arxiv.org/abs/2602.18023) (2026).
- An T. Le, [On the boundary cost of source-consistent warp
  shells](https://arxiv.org/abs/2605.25417) (2026).

The shell construction and admissibility work belongs to the second paper.
See [boundary cost](docs/explanation/boundary_cost.md) for its assumptions.
