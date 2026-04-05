"""asv benchmarks - curvature-chain grid evaluation (#1-2).

Two benchmarks covering the full autodiff curvature chain
(metric -> Christoffel -> Riemann -> Ricci -> Einstein -> stress-energy)
on Alcubierre warp spacetimes at two grid sizes.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU canonical

import jax
import jax.numpy as jnp

import warpax  # noqa: F401 -- triggers jax_enable_x64 flag
from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid


class CurvatureChain32:
    """asv benchmark - curvature grid eval on Alcubierre 32^3 (#1)."""

    warmup_time = 1.0
    number = 3
    repeat = (3, 5, 30.0)

    def setup(self) -> None:
        self.metric = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        self.grid = GridSpec(bounds=[(-6.0, 6.0)] * 3, shape=(32, 32, 32))

        def _run(metric: AlcubierreMetric) -> jnp.ndarray:
            result = evaluate_curvature_grid(
                metric, self.grid, t=0.0, compute_invariants=False
            )
            return result.stress_energy

        self.fn = jax.jit(_run)
        _ = self.fn(self.metric)  # JIT warmup

    def time_compute_chain(self) -> None:
        self.fn(self.metric).block_until_ready()


class CurvatureChain64:
    """asv benchmark - curvature grid eval on Alcubierre 64^3 (#2)."""

    warmup_time = 3.0
    number = 2
    repeat = (2, 3, 60.0)
    timeout = 120.0

    def setup(self) -> None:
        self.metric = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        self.grid = GridSpec(bounds=[(-6.0, 6.0)] * 3, shape=(64, 64, 64))

        def _run(metric: AlcubierreMetric) -> jnp.ndarray:
            result = evaluate_curvature_grid(
                metric, self.grid, t=0.0, compute_invariants=False
            )
            return result.stress_energy

        self.fn = jax.jit(_run)
        _ = self.fn(self.metric)  # JIT warmup

    def time_compute_chain(self) -> None:
        self.fn(self.metric).block_until_ready()
