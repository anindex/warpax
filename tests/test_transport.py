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


def test_alcubierre_round_trip_future_directed_elapsed_times():
    """Regression: round-trip legs must be future-directed.

    Bug: null_ic returned the past-directed root (k^0 < 0), so both legs of
    null_coord_time_asymmetry ran backward in coordinate time (negative
    elapsed times) and the Alcubierre asymmetry came out ~ -1.892 instead of
    the future-directed ~ -2.75 (the bubble at x_s = v_s t is not t-reversal
    symmetric). Pin positive elapsed coordinate times on both legs and the
    future-directed asymmetry value.
    """
    from warpax.benchmarks import AlcubierreMetric
    from warpax.geodesics import integrate_geodesic, null_ic
    from warpax.transport.diagnostics import _first_local_min_idx

    metric = AlcubierreMetric(v_s=0.5, R=2.0, sigma=2.0)
    emitter = jnp.array([0.0, -8.0, 0.0, 0.0], dtype=jnp.float64)
    receiver = jnp.array([0.0, 8.0, 0.0, 0.0], dtype=jnp.float64)

    # Future-directed asymmetry (past-directed bug gave ~ -1.892).
    delta = float(null_coord_time_asymmetry(metric, emitter, receiver))
    assert -3.0 < delta < -2.4, f"expected ~ -2.75, got {delta}"

    # Replicate both legs and pin positive elapsed coordinate times.
    _, k0_fwd = null_ic(metric, emitter, jnp.array([1.0, 0.0, 0.0]))
    assert float(k0_fwd[0]) > 0.0, "forward null leg must be future-directed"
    sol_fwd = integrate_geodesic(
        metric, emitter, k0_fwd, tau_span=(0.0, 50.0), num_points=500,
        dt0=0.01, rtol=1e-8, atol=1e-8,
    )
    d_fwd = jnp.linalg.norm(sol_fwd.positions[:, 1:] - receiver[1:], axis=1)
    idx_fwd = _first_local_min_idx(d_fwd)
    arrival = sol_fwd.positions[idx_fwd]
    t_elapsed_fwd = float(arrival[0] - emitter[0])
    assert t_elapsed_fwd > 0.0, f"negative forward elapsed time {t_elapsed_fwd}"

    dx_back = emitter[1:] - arrival[1:]
    _, k0_bwd = null_ic(metric, arrival, dx_back / jnp.linalg.norm(dx_back))
    sol_bwd = integrate_geodesic(
        metric, arrival, k0_bwd, tau_span=(0.0, 50.0), num_points=500,
        dt0=0.01, rtol=1e-8, atol=1e-8,
    )
    d_bwd = jnp.linalg.norm(sol_bwd.positions[:, 1:] - emitter[1:], axis=1)
    idx_bwd = _first_local_min_idx(d_bwd)
    t_elapsed_bwd = float(sol_bwd.positions[idx_bwd, 0] - arrival[0])
    assert t_elapsed_bwd > 0.0, f"negative backward elapsed time {t_elapsed_bwd}"
