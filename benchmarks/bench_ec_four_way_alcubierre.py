"""asv benchmark: 4-way WEC comparison on a matched Alcubierre point.

Compares:
1. ``tanh`` baseline (``strategy='tanh', warm_start='cold'``).
2. ``hard_bound`` alone (``strategy='hard_bound'``).
3. ``hard_bound`` + ``warm_start='spatial_neighbor'``.
4. ``hard_bound`` + ``starts='fibonacci+bfgs_top_k'``.

Single wall-point evaluation (``_build_wall_point`` from
``bench_energy_conditions``); CPU-only with the asv 20 % noise budget.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

import warpax  # noqa: F401
from benchmarks.bench_energy_conditions import _build_wall_point
from warpax.energy_conditions import optimize_wec


class ECFourWayAlcubierre:
    """asv benchmark - 4-way WEC comparison on a single Alcubierre point."""

    warmup_time = 3.0
    number = 1
    repeat = (2, 3, 60.0)
    timeout = 240.0

    def setup(self) -> None:
        self.T, self.g = _build_wall_point()
        self.key = jax.random.PRNGKey(42)
        self.neighbor = optimize_wec(
            self.T, self.g, key=jax.random.PRNGKey(7), strategy="tanh"
        ).worst_observer

    def time_tanh_baseline(self) -> None:
        """Curve 1: tanh + cold + axis+gaussian (baseline)."""
        r = optimize_wec(
            self.T,
            self.g,
            key=self.key,
            strategy="tanh",
            warm_start="cold",
            starts="axis+gaussian",
        )
        r.margin.block_until_ready()

    def time_hard_bound_alone(self) -> None:
        """Curve 2: hard_bound + cold + axis+gaussian."""
        r = optimize_wec(
            self.T,
            self.g,
            key=self.key,
            strategy="hard_bound",
            warm_start="cold",
            starts="axis+gaussian",
        )
        r.margin.block_until_ready()

    def time_hard_bound_plus_spatial_neighbor(self) -> None:
        """Curve 3: hard_bound + spatial_neighbor."""
        r = optimize_wec(
            self.T,
            self.g,
            key=self.key,
            strategy="hard_bound",
            warm_start="spatial_neighbor",
            neighbor_observer=self.neighbor,
            starts="axis+gaussian",
        )
        r.margin.block_until_ready()

    def time_hard_bound_plus_fibonacci_pool(self) -> None:
        """Curve 4: hard_bound + fibonacci+bfgs_top_k."""
        r = optimize_wec(
            self.T,
            self.g,
            key=self.key,
            strategy="hard_bound",
            warm_start="cold",
            starts="fibonacci+bfgs_top_k",
        )
        r.margin.block_until_ready()
