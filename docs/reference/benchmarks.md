# Benchmarks (asv harness)

The top-level `benchmarks/` directory is an
[airspeed velocity](https://asv.readthedocs.io) harness tracking
commit-to-commit deltas across 13 benchmarks: curvature grid evaluation,
energy-condition verification, geodesic integration, Hawking-Ellis
classification, JIT-cache warmup and chunked-memory envelopes.
[`benchmarks/README.md`](https://github.com/anindex/warpax/blob/main/benchmarks/README.md)
lists them and the noise budget.

```bash
make bench          # asv run --quick
make bench-compare  # asv compare HEAD~1 HEAD
```

`JAX_PLATFORMS=cpu` is set in every `bench_*.py`; CUDA benchmarking is deferred.

Two unrelated things are named "benchmarks" here. The directory above is the perf
harness; `warpax.benchmarks` is the library of reference spacetimes (Minkowski,
Schwarzschild, Alcubierre), whose name is pinned by the public API.
