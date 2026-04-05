"""asv benchmarks - geodesic integration + tidal deviation (#7-8).

Two benchmarks covering the Diffrax-driven ODE path through an Alcubierre
warp spacetime: (7) plain timelike geodesic integration; (8) geodesic
+ Jacobi deviation co-integration.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU canonical

import jax.numpy as jnp

import warpax  # noqa: F401
from warpax.benchmarks import AlcubierreMetric
from warpax.geodesics import (
    integrate_geodesic,
    integrate_geodesic_with_deviation,
    timelike_ic,
)


def _alcubierre_ic(metric: AlcubierreMetric) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a central-axis timelike initial condition."""
    coords0 = jnp.array([0.0, -5.0, 0.0, 0.0])  # behind the bubble
    u_spatial = jnp.array([0.5, 0.0, 0.0])  # aligned with warp direction
    x0, u0 = timelike_ic(metric, coords0, u_spatial)
    return x0, u0


class GeodesicIntegration:
    """asv benchmark - timelike geodesic τ ∈ [0, 10] (#7)."""

    warmup_time = 2.0
    number = 2
    repeat = (2, 3, 30.0)
    timeout = 60.0

    def setup(self) -> None:
        self.metric = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        self.x0, self.u0 = _alcubierre_ic(self.metric)
        # Warm up by running once.
        _ = integrate_geodesic(
            self.metric, self.x0, self.u0, (0.0, 10.0), num_points=200
        )

    def time_geodesic_integration(self) -> None:
        result = integrate_geodesic(
            self.metric, self.x0, self.u0, (0.0, 10.0), num_points=200
        )
        # Access an array field to force materialization.
        result.positions.block_until_ready()


class JacobiDeviation:
    """asv benchmark - co-integrated geodesic + deviation (#8)."""

    warmup_time = 3.0
    number = 2
    repeat = (2, 3, 45.0)
    timeout = 90.0

    def setup(self) -> None:
        self.metric = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        self.x0, self.u0 = _alcubierre_ic(self.metric)
        # Initial deviation vector (spacelike, small).
        self.xi0 = jnp.array([0.0, 0.0, 0.01, 0.0])
        self.xi0_dot = jnp.zeros(4)
        _ = integrate_geodesic_with_deviation(
            self.metric,
            self.x0,
            self.u0,
            self.xi0,
            self.xi0_dot,
            (0.0, 10.0),
            num_points=200,
        )

    def time_jacobi_deviation(self) -> None:
        result = integrate_geodesic_with_deviation(
            self.metric,
            self.x0,
            self.u0,
            self.xi0,
            self.xi0_dot,
            (0.0, 10.0),
            num_points=200,
        )
        result.positions.block_until_ready()
