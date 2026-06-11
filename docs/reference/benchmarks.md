# Benchmarks (asv harness)

warpax ships an [airspeed velocity (`asv`)](https://asv.readthedocs.io)
regression-benchmark harness at the top-level `benchmarks/` directory.
The harness tracks commit-to-commit performance deltas across 13
benchmarks covering curvature grid evaluation, energy-condition
verification, geodesic integration, Hawking-Ellis classification,
JIT-cache warmup, and chunked-memory envelopes.

## Naming distinction

The phrase "benchmarks" is intentionally reused for two unrelated things
in this repo:

- **top-level `benchmarks/`** - the asv perf harness you are reading about.
- **Python module `warpax.benchmarks`** (at `src/warpax/benchmarks/`) -
  a library of *reference spacetimes* (Minkowski, Schwarzschild,
  Alcubierre). Pinned by the public API; renaming is not an
  option.

The top-level directory follows the SciPy / NumPy / Astropy convention
for asv harnesses. The module keeps its original name. See
[`benchmarks/README.md`](https://github.com/anindex/warpax/blob/main/benchmarks/README.md)
for a full table of the benchmarks and their coverage.

## Run instructions

```bash
make bench # asv run --quick (warmup + single timing per bench)
make bench-compare # asv compare HEAD~1 HEAD (per-commit delta)
```

## Noise budget

`regressions_thresholds.default = 0.20` (20%) in `asv.conf.json` -
regression deltas below this threshold are tolerated. Single-CPU CI
runners see ±10% variance on tight kernels; the budget accommodates
real noise without masking real drift.

## JAX platform

`JAX_PLATFORMS=cpu` is set at the top of every `bench_*.py` module
for CPU-canonical reproduction; CUDA benchmarking is deferred.

## Scope

- 13 benchmark classes (curvature × 2, EC × 5, geodesic × 2,
  classifier × 2, JIT cache × 1, memory × 1).
