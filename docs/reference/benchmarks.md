# Performance benchmarks

The [airspeed velocity](https://asv.readthedocs.io) harness measures performance
across commits. It is separate from `src/warpax/benchmarks/`, the public module
containing Minkowski, Schwarzschild, and Alcubierre reference metrics.

## Benchmark catalog

The harness contains 13 benchmark classes; a class may expose several timings.

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
| 10 | `bench_classifier_grid.py` | `ClassifierGrid32Generalized` | Generalized-pencil classifier on 32³ |
| 11 | `bench_ec_four_way_alcubierre.py` | `ECFourWayAlcubierre` | 4-way WEC comparison, single point |
| 12 | `bench_auto_chunk.py` | `AutoChunkMemoryEnvelope` | Full-vmap vs chunked memory envelope |
| 13 | `bench_jit_cache.py` | `JITCacheColdVsWarm` | Persistent JIT cache cold vs warm |

## Run

```bash
JAX_PLATFORMS=cpu make bench  # asv run --quick --show-stderr (single timing per bench)
make bench-compare  # asv compare HEAD~1 HEAD (per-commit deltas)
```

To compare a longer history:

```bash
asv run v1.0.0..HEAD  # historical range; choose the tags of interest
asv publish           # HTML report at .asv/html/
asv preview           # local web server for exploring results
```

## Noise budget

`asv.conf.json` uses a 20% regression threshold. Shared-runner timing noise can
be substantial; confirm regressions on the same hardware and environment.

## JAX platform

The benchmark modules default `JAX_PLATFORMS` to `cpu` only when it is unset.
Set `JAX_PLATFORMS=cpu` explicitly to reproduce the configured CPU runs.
An externally selected backend is not overridden.

## Matrix

| Axis | Values | Notes |
|------|--------|-------|
| `pythons` | 3.12, 3.13, 3.14 | Primary coverage |
| `jax` | 0.10.1 | Pinned to the deps range `<0.11.0` |
| `jax backend` | cpu only | CUDA deferred |
