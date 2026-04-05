# warpax

Observer-robust energy condition verification for warp drive spacetimes - a JAX
toolkit that couples automatic-differentiation curvature, Hawking-Ellis algebraic
classification, and continuous BFGS observer optimization.

## Tutorials

Learning-oriented walkthroughs that start from a clean install.

- [Quickstart](tutorials/quickstart.md) - 5–10 minutes from install to seeing
  an energy condition violation on the Alcubierre metric.
- [First curvature computation](tutorials/first_curvature_computation.md) -
  the full curvature chain on Minkowski, as a warm-up.

## How-To

Task-oriented recipes for readers who know what they want to do.

- [Define a custom warp metric](how-to/custom_metric_tutorial.md)
- [Interpret EC results](how-to/interpreting_ec_results.md)
- [Load an external metric](how-to/loading_external_metrics.md)

## Reference

Lookup-oriented autodoc of the public API, plus the pinned-default catalog.

- [API reference](reference/index.md)
- [Metric catalog](reference/metric_catalog.md)
- [v0.1.x API defaults](reference/api_defaults.md)

## Explanation

Understanding-oriented background on the mathematics and architecture.

- [Architecture overview](explanation/ARCHITECTURE.md)
- [Theory: ADM 3+1 and Hawking-Ellis types](explanation/theory.md)
