"""asv benchmark - Fig 5 four-way EC comparison on Alcubierre.

Parametrized 4-way comparison on Alcubierre matched parameters:
1. v0.1.x ``tanh`` baseline (``strategy='tanh', warm_start='cold'``)
2. ``hard_bound`` alone (``strategy='hard_bound'``)
3. ``hard_bound + warm_start='spatial_neighbor'``
4. ``hard_bound + starts='fibonacci+bfgs_top_k'``

Target grid: Alcubierre v_s=0.5 R=1.0 σ=8.0 (the
paper §2.1 canonical test spacetime). CPU-only; 20% asv noise budget.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

import warpax  # noqa: F401
from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions import optimize_wec
from warpax.geometry import evaluate_curvature_grid
from warpax.geometry.types import GridSpec


class ECFourWayAlcubierre:
    """asv benchmark - 4-way WEC comparison on a single Alcubierre point."""

    warmup_time = 3.0
    number = 1
    repeat = (2, 3, 60.0)
    timeout = 240.0

    def setup(self) -> None:
        self.metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        # Single-point grid at (t=0, x=0, y=0, z=0) - compact benchmark target
        grid = GridSpec(bounds=[(-0.1, 0.1)] * 3, shape=(1, 1, 1))
        r = evaluate_curvature_grid(self.metric, grid, t=0.0)

        # Extract the one point's stress-energy + metric for optimize_* input
        self.T = r.stress_energy[0, 0, 0]
        self.g = r.metric[0, 0, 0]
        self.key = jax.random.PRNGKey(42)

    def time_tanh_baseline(self) -> None:
        """Curve 1: v0.1.x tanh + cold + axis+gaussian (baseline)."""
        r = optimize_wec(
            self.T, self.g, key=self.key,
            strategy="tanh", warm_start="cold",
            starts="axis+gaussian", backend="cpu",
        )
        r.margin.block_until_ready()

    def time_hard_bound_alone(self) -> None:
        """Curve 2: hard_bound + cold + axis+gaussian."""
        r = optimize_wec(
            self.T, self.g, key=self.key,
            strategy="hard_bound", warm_start="cold",
            starts="axis+gaussian", backend="cpu",
        )
        r.margin.block_until_ready()

    def time_hard_bound_plus_spatial_neighbor(self) -> None:
        """Curve 3: hard_bound + spatial_neighbor."""
        r = optimize_wec(
            self.T, self.g, key=self.key,
            strategy="hard_bound", warm_start="spatial_neighbor",
            starts="axis+gaussian", backend="cpu",
        )
        r.margin.block_until_ready()

    def time_hard_bound_plus_fibonacci_pool(self) -> None:
        """Curve 4: hard_bound + fibonacci+bfgs_top_k."""
        r = optimize_wec(
            self.T, self.g, key=self.key,
            strategy="hard_bound", warm_start="cold",
            starts="fibonacci+bfgs_top_k", backend="cpu",
        )
        r.margin.block_until_ready()
