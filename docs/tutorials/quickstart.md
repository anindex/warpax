# warpax Quickstart

A 5-10 minute path from install to seeing an energy condition violation.

## 1. Install (Python 3.12+)

```bash
conda create -n warpax python=3.12 -y
conda activate warpax
pip install -e ".[dev]"
```

Optional extras:

- `pip install -e ".[viz]"` adds matplotlib visualization helpers
- `pip install -e ".[manim]"` adds animated scene rendering (requires ffmpeg)

## 2. Run the Alcubierre example (under 5 seconds)

```bash
python examples/03_alcubierre_analysis.py
```

This constructs the Alcubierre warp drive at `v_s=0.5, R=1.0, sigma=8.0`,
evaluates curvature at a single bubble-wall point, computes the observer-robust
EC margins via BFGS optimization over the timelike observer manifold, and
prints the result alongside the Eulerian-frame comparison.

Approximate output:

```
Alcubierre Warp Drive Analysis
========================================
Parameters: v_s=0.5, R=1.0, sigma=8.0
Point: (t, x, y, z) = (0.0, 1.0, 0.5, 0.0)

Observer-robust EC margins (negative = violated):
  NEC: -1.23e-01
  WEC: -1.23e-01
  SEC: -4.56e-02
  DEC: -2.34e-01

Eulerian-frame EC margins:
  NEC: -1.05e-02
  WEC: -1.05e-02

NEC/WEC violation confirmed!
The Alcubierre warp drive requires exotic matter (negative energy density).
```

## 3. Read the output

- **Negative margin** = the energy condition is **violated** at the worst-case
  observer.
- The **robust** margin is always less than or equal to the **Eulerian**
  margin: the extra violation is what axis-aligned ADM analysis misses.
- In this run the Eulerian NEC margin is about an order of magnitude less
  negative than the robust NEC margin -- a boosted observer sees a
  substantially worse violation than an Eulerian one.

For a full grid-level view of which regions harbour missed violations, try:

```bash
python examples/07_custom_warp_metric.py
```

This rewrites the pipeline end-to-end on a 16^3 grid of a custom Gaussian
warp bubble (about 22 seconds on a laptop CPU) and emits a 3-panel
PDF comparison at `examples/output/gaussian_warp_comparison.pdf`.

## 4. What next

- [`custom_metric_tutorial.md`](../how-to/custom_metric_tutorial.md) -- define your own
  warp spacetime by subclassing `ADMMetric` and run the full verification
  pipeline (single-point, grid, and wall-restricted diagnostics).
- [`interpreting_ec_results.md`](../how-to/interpreting_ec_results.md) -- deeper
  reference: margin sign convention, Hawking-Ellis Type I-IV semantics,
  `f_miss` vs `f_miss|viol`, wall-restricted vs full-grid statistics, and
  when to trust a number.
- `examples/07_custom_warp_metric.py` in the repo -- the runnable
  custom-metric walkthrough referenced above.
- [`ARCHITECTURE.md`](../explanation/ARCHITECTURE.md) -- how the autodiff
  curvature pipeline turns a metric function into stress-energy and
  observer-robust margins.

## Units convention

warpax uses geometric units throughout: `G = c = 1`. Lengths and times are in
the same unit. The `v_s` parameter is dimensionless (fraction of `c`).

## Requirements

- Python 3.12+ (CPython)
- JAX CPU backend is sufficient; no GPU needed for the examples
- float64 is enforced at import time
  (`jax.config.update("jax_enable_x64", True)` in `warpax/__init__.py`)
- Optional: LaTeX and ffmpeg for publication-quality figures and animations
