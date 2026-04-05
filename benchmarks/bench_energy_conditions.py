"""asv benchmarks - energy-condition optimiser multistart (#3-6).

Four benchmarks running the Optimistix BFGS multistart over the bounded
timelike observer manifold for NEC / WEC / SEC / DEC on a single
Alcubierre wall point. Uses the default ``n_starts=16`` and
``zeta_max=5.0`` from v0.1.x API defaults.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU canonical

import jax
import jax.numpy as jnp

import warpax  # noqa: F401
from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions import (
    optimize_dec,
    optimize_nec,
    optimize_sec,
    optimize_wec,
)
from warpax.geometry import compute_curvature_chain


def _build_wall_point() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute (T_ab, g_ab) at a representative Alcubierre wall point."""
    metric = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
    coords = jnp.array([0.0, 2.0, 0.5, 0.0])  # near wall
    chain = compute_curvature_chain(metric, coords)
    return chain.stress_energy, chain.metric


class _OptimizerBenchmark:
    """Shared harness for the four EC-optimiser benchmarks."""

    warmup_time = 2.0
    number = 2
    repeat = (2, 3, 45.0)
    timeout = 120.0

    def setup(self) -> None:
        T, g = _build_wall_point()
        self.T = T
        self.g = g
        self.key = jax.random.PRNGKey(0)
        self.fn = jax.jit(self._run)
        _ = self.fn(self.T, self.g, self.key)  # JIT warmup

    def _run(self, T, g, key):  # pragma: no cover - overridden
        raise NotImplementedError


class NECOptimizer(_OptimizerBenchmark):
    """asv benchmark - optimize_nec multistart (#3)."""

    def _run(self, T, g, key):
        return optimize_nec(T, g, key=key).margin

    def time_multistart_nec(self) -> None:
        self.fn(self.T, self.g, self.key).block_until_ready()


class WECOptimizer(_OptimizerBenchmark):
    """asv benchmark - optimize_wec multistart (#4)."""

    def _run(self, T, g, key):
        return optimize_wec(T, g, key=key).margin

    def time_multistart_wec(self) -> None:
        self.fn(self.T, self.g, self.key).block_until_ready()


class SECOptimizer(_OptimizerBenchmark):
    """asv benchmark - optimize_sec multistart (#5)."""

    def _run(self, T, g, key):
        return optimize_sec(T, g, key=key).margin

    def time_multistart_sec(self) -> None:
        self.fn(self.T, self.g, self.key).block_until_ready()


class DECOptimizer(_OptimizerBenchmark):
    """asv benchmark - optimize_dec multistart, three-term-min (#6)."""

    def _run(self, T, g, key):
        # Default v0.1.x mode='three_term_min' (default).
        return optimize_dec(T, g, key=key).margin

    def time_multistart_dec(self) -> None:
        self.fn(self.T, self.g, self.key).block_until_ready()
