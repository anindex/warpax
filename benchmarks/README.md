# warpax asv benchmarks

This top-level `benchmarks/` directory hosts the
[airspeed velocity (`asv`)](https://asv.readthedocs.io) regression harness.
It tracks commit-to-commit performance deltas across the 9 benchmarks
defined below.

## Naming collision

There are **two unrelated things named "benchmarks"** in this repo:

| Location | Type | Purpose |
|----------|------|---------|
| `benchmarks/` (this directory) | asv perf harness | Regression benchmarks |
| `src/warpax/benchmarks/` | Python module | Reference spacetimes (Alcubierre, Schwarzschild, Minkowski) |

The `warpax.benchmarks` module is pinned by the v0.1.0 public API
surface so renaming it would break downstream consumers. The top-level
`benchmarks/` directory follows the SciPy / NumPy / Astropy convention
for asv harnesses. The two co-exist; the naming collision is intentional
and documented here.

## The 9 benchmarks

| # | File | Class | Coverage |
|---|------|-------|----------|
| 1 | `bench_curvature.py` | `CurvatureChain32` | Curvature grid eval, Alcubierre 32³ |
| 2 | `bench_curvature.py` | `CurvatureChain64` | Curvature grid eval, Alcubierre 64³ |
| 3 | `bench_energy_conditions.py` | `NECOptimizer` | NEC multistart (Optimistix BFGS) |
| 4 | `bench_energy_conditions.py` | `WECOptimizer` | WEC multistart |
| 5 | `bench_energy_conditions.py` | `SECOptimizer` | SEC multistart |
| 6 | `bench_energy_conditions.py` | `DECOptimizer` | DEC multistart (three-term min) |
| 7 | `bench_geodesic.py` | `GeodesicIntegration` | Central worldline τ ∈ [0, 10] |
| 8 | `bench_geodesic.py` | `JacobiDeviation` | Co-integrated tidal deviation |
| 9 | `bench_classifier_grid.py` | `ClassifierGrid32` | Float64 Hawking-Ellis on 32³ |

## How to run

### Quick (local development)

```bash
make bench          # asv run --quick --show-stderr (single timing per bench)
make bench-compare  # asv compare HEAD~1 HEAD (per-commit deltas)
```

### Full history

```bash
cd warpax
asv run v0.1.0..HEAD  # time every commit since v0.1.0
asv publish           # HTML report at .asv/html/
asv preview           # local web server for exploring results
```

## Noise budget

`regressions_thresholds.default = 0.20` (20%) in `asv.conf.json` -
regression deltas under 20% are tolerated. This is a single-CPU harness;
CI runners have ±10% variance on tight kernels, so the budget covers
real noise without masking real perf drift.

## JAX platform

`JAX_PLATFORMS=cpu` is set at the top of every `bench_*.py` module for
CPU canonical reproduction. The CUDA matrix axis is documented but not
yet enabled.

## Matrix

| Axis | Values | Notes |
|------|--------|-------|
| `pythons` | 3.12, 3.13, 3.14 | Primary coverage |
| `jax` | 0.10.0 | Pinned to the deps range `<0.11.0` |
| `jax backend` | cpu only | CUDA deferred |
