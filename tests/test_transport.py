"""Tests for transport diagnostics."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from warpax.transport import (
    blueshift_hazard,
    geodesic_deviation_diagnostic,
    null_coord_time_asymmetry,
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


def test_null_round_trip_alias_matches_new_name():
    """null_round_trip_asymmetry is preserved as an alias for the renamed function."""
    assert null_round_trip_asymmetry is null_coord_time_asymmetry


def test_null_coord_time_constant_shift_invariant():
    """delta_t_coord is invariant under a constant time shift t -> t + c.

    Both legs sample the same Minkowski metric with the same null
    geodesics; the observable is the difference of two coordinate-time
    elapses, so a global constant shift cancels exactly. This is the
    one slicing change we actually claim invariance under.
    """
    from warpax.benchmarks.minkowski import MinkowskiMetric

    metric = MinkowskiMetric()
    emitter = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    receiver = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)

    delta_a = null_coord_time_asymmetry(metric, emitter, receiver)

    shifted_emitter = emitter.at[0].add(3.7)
    shifted_receiver = receiver.at[0].add(3.7)
    delta_b = null_coord_time_asymmetry(metric, shifted_emitter, shifted_receiver)

    assert jnp.allclose(delta_a, delta_b, atol=1e-10), (
        f"Constant t-shift should leave delta_t_coord invariant; got "
        f"{float(delta_a)} vs {float(delta_b)}"
    )


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
