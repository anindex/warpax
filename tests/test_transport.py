"""Tests for transport diagnostics."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from warpax.transport import (
    blueshift_hazard,
    geodesic_deviation_diagnostic,
    null_round_trip_asymmetry,
)

jax.config.update("jax_enable_x64", True)


def test_null_round_trip_minkowski():
    """Minkowski spacetime has zero null round-trip asymmetry."""
    from warpax.benchmarks.minkowski import MinkowskiMetric

    metric = MinkowskiMetric()
    emitter = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    receiver = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    result = null_round_trip_asymmetry(metric, emitter, receiver)
    assert jnp.abs(result) < 0.1, f"Expected ~0, got {result}"


def test_geodesic_deviation_minkowski():
    """Minkowski has zero geodesic deviation."""
    from warpax.benchmarks.minkowski import MinkowskiMetric

    metric = MinkowskiMetric()
    result = geodesic_deviation_diagnostic(
        metric, jnp.array([0.0, 5.0, 0.0, 0.0], dtype=jnp.float64)
    )
    assert jnp.isclose(result, 0.0, atol=1e-10)


def test_blueshift_hazard_minkowski():
    """Minkowski has zero blueshift hazard."""
    from warpax.benchmarks.minkowski import MinkowskiMetric

    metric = MinkowskiMetric()
    result = blueshift_hazard(
        metric,
        jnp.array([0.0, 5.0, 0.0, 0.0], dtype=jnp.float64),
        n_directions=2,
        tau_max=5.0,
        num_points=50,
    )
    assert result < 0.1, f"Expected ~0, got {result}"
