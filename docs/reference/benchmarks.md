# Performance benchmarks

The [airspeed velocity](https://asv.readthedocs.io) harness measures performance
across commits. It is separate from `src/warpax/benchmarks/`, the public module
containing Minkowski, Schwarzschild, and Alcubierre reference metrics.

## Benchmark catalog

The harness contains 18 benchmark classes; a class may expose several timings.

| # | File | Class | Coverage |
|---|------|-------|----------|
| 1 | `bench_curvature.py` | `CurvatureChain32` | Curvature grid eval, Alcubierre 32³ |
| 2 | `bench_curvature.py` | `CurvatureChain64` | Curvature grid eval, Alcubierre 64³ |
| 3 | `bench_energy_conditions.py` | `NECOptimizer` | NEC multistart (Optimiztix BFGS) |
| 4 | `bench_energy_conditions.py` | `WECOptimizer` | WEC multistart |
| 5 | `bench_energy_conditions.py` | `SECOptimizer` | SEC multistart |
| 6 | `bench_energy_conditions.py` | `DECOptimizer` | DEC multistart (three-term min) |
| 7 | `bench_geodesic.py` | `GeodesicIntegration` | Central worldline τ ∈ [0, 10] |
| 8 | `bench_geodesic.py` | `JacobiDeviation` | Co-integrated tidal deviation |
| 9 | `bench_classifier_grid.py` | `ClassifierGrid32` | Float64 Hawking-Ellis on 32³ |
| 10 | `bench_classifier_grid.py` | `ClassifierGrid32Generalized` | Generalized-pencil classifier on 32³ |
| 11 | `bench_ec_four_way_alcubierre.py` | `ECFourWayAlcubierre` | 4-way WEC comparison, single point |
| 12 | `bench_auto_chunk.py` | `AutoChunkMemoryEnvelope` | Full-vmap vs chunked memory envelope |
| 13 | `bench_jit_cache.py` | `JITCacheColdVsWarm` | Persistent JIT cache cold vs warm (child process per call) |
| 14 | `bench_pipeline.py` | `Certify16` | `warpax.certify` end to end, Alcubierre 16³ at v_s=1.5 |
| 15 | `bench_pipeline.py` | `FrameFreeGrid16` | `certify_grid_frame_free` on 16³ fields, LMI subset non-empty |
| 16 | `bench_pipeline.py` | `VerifyGrid8` | `verify_grid` observer search on 8³ fields |
| 17 | `bench_pipeline.py` | `WallClustered64` | `wall_clustered` 64³ grid plus volume weights |
| 18 | `bench_pipeline.py` | `CurvatureRetrace8` | `evaluate_curvature_grid` with a new `t` per call (retrace guard) |

## Run

```bash
JAX_PLATFORMS=cpu make bench  # asv run --quick --show-stderr (single timing per bench)
make bench-compare  # asv continuous --factor 1.2 --quick --split HEAD~1 HEAD
```

## Regression gate

`make bench-compare` times every benchmark on `HEAD~1` and `HEAD` in the same
environment and exits non-zero when any timing slows by more than a factor of
1.2, matching the 20% threshold in `asv.conf.json`. `--split` prints improved,
unchanged, and regressed benchmarks in separate tables.

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
| `jax` | 0.12.0 | Pinned to the deps range `<0.13` |
| `jax backend` | cpu only | CUDA deferred |
