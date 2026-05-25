# warpax

Observer-robust energy condition verification for warp-drive spacetimes. A JAX
toolkit built around three pieces: an autodiff curvature chain, Hawking-Ellis
algebraic classification of $T_{\mu\nu}$, and continuous BFGS optimization over
the timelike observer manifold.

## Examples

Numbered scripts under `examples/` in the repository (01-10) with runtime
estimates and a suggested learning path for new users. See the
[examples tour](tutorials/examples_tour.md) for a curated walkthrough.

## Tutorials

Learning-oriented walkthroughs that start from a clean install.

- [Quickstart](tutorials/quickstart.md) - 5-10 minutes from install to seeing
  an energy condition violation on the Alcubierre metric.
- [First curvature computation](tutorials/first_curvature_computation.md) -
  the full curvature chain on Minkowski, as a warm-up.
- [Examples tour](tutorials/examples_tour.md) -
  numbered scripts 01-10 (see also the `examples/` directory in the repository).

## How-To

Task-oriented recipes for readers who know what they want to do.

- [Define a custom warp metric](how-to/custom_metric_tutorial.md)
- [Interpret EC results](how-to/interpreting_ec_results.md)
- [Load an external metric](how-to/loading_external_metrics.md)
- [Reproduce the warp-shell admissibility paper](how-to/reproduce_warpshell_paper.md)

## Reference

Lookup-oriented autodoc of the public API, plus the pinned-default catalog.

- [API reference](reference/index.md)
- [Metric catalog](reference/metric_catalog.md)
- [Benchmarks](reference/benchmarks.md)

## Explanation

Understanding-oriented background on the mathematics and architecture.

- [Architecture overview](explanation/ARCHITECTURE.md)
- [Theory: ADM 3+1 and Hawking-Ellis types](explanation/theory.md)
- [Release notes](explanation/release_notes.md)

## Scripts

- See `scripts/README.md` in the repository for the paper-reproduction and
  research entry points.
