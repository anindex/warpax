"""Tests for transport diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from warpax.geometry.metric import MetricSpecification
from warpax.transport import (
    blueshift_hazard,
    geodesic_deviation_diagnostic,
    null_coord_time_asymmetry,
)


def test_null_round_trip_minkowski():
    """Minkowski spacetime has zero null round-trip asymmetry (measured exactly
    0.0; the Alcubierre signal scale is ~2.75)."""
    from warpax.benchmarks.minkowski import MinkowskiMetric

    metric = MinkowskiMetric()
    emitter = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    receiver = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    result = null_coord_time_asymmetry(metric, emitter, receiver)
    assert jnp.abs(result) < 1e-6, f"Expected ~0, got {result}"


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
    assert result < 1e-8, f"Expected ~0, got {result}"


def test_alcubierre_round_trip_future_directed_elapsed_times():
    """Both round-trip legs run forward in coordinate time.

    A past-directed root (k^0 < 0) would run both legs of
    null_coord_time_asymmetry backward and give an Alcubierre asymmetry near
    -1.892; the bubble at x_s = v_s t is not t-reversal symmetric, and the
    future-directed value is near -2.75. Both elapsed coordinate times must be
    positive and the asymmetry must match the future-directed value.
    """
    from warpax.benchmarks import AlcubierreMetric
    from warpax.geodesics import integrate_geodesic, null_ic
    from warpax.transport.diagnostics import _first_local_min_idx

    metric = AlcubierreMetric(v_s=0.5, R=2.0, sigma=2.0)
    emitter = jnp.array([0.0, -8.0, 0.0, 0.0], dtype=jnp.float64)
    receiver = jnp.array([0.0, 8.0, 0.0, 0.0], dtype=jnp.float64)

    delta = float(null_coord_time_asymmetry(metric, emitter, receiver))
    assert -3.0 < delta < -2.4, f"expected ~ -2.75, got {delta}"

    _, k0_fwd = null_ic(metric, emitter, jnp.array([1.0, 0.0, 0.0]))
    assert float(k0_fwd[0]) > 0.0, "forward null leg must be future-directed"
    sol_fwd = integrate_geodesic(
        metric,
        emitter,
        k0_fwd,
        tau_span=(0.0, 50.0),
        num_points=500,
        dt0=0.01,
        rtol=1e-8,
        atol=1e-8,
    )
    d_fwd = jnp.linalg.norm(sol_fwd.positions[:, 1:] - receiver[1:], axis=1)
    idx_fwd = _first_local_min_idx(d_fwd)
    arrival = sol_fwd.positions[idx_fwd]
    t_elapsed_fwd = float(arrival[0] - emitter[0])
    assert t_elapsed_fwd > 0.0, f"negative forward elapsed time {t_elapsed_fwd}"

    dx_back = emitter[1:] - arrival[1:]
    _, k0_bwd = null_ic(metric, arrival, dx_back / jnp.linalg.norm(dx_back))
    sol_bwd = integrate_geodesic(
        metric,
        arrival,
        k0_bwd,
        tau_span=(0.0, 50.0),
        num_points=500,
        dt0=0.01,
        rtol=1e-8,
        atol=1e-8,
    )
    d_bwd = jnp.linalg.norm(sol_bwd.positions[:, 1:] - emitter[1:], axis=1)
    idx_bwd = _first_local_min_idx(d_bwd)
    t_elapsed_bwd = float(sol_bwd.positions[idx_bwd, 0] - arrival[0])
    assert t_elapsed_bwd > 0.0, f"negative backward elapsed time {t_elapsed_bwd}"


class _ConstantMetric(MetricSpecification):
    """Constant diagonal metric with g_00 = +0.5 (no static observer)."""

    def __call__(self, coords):
        return jnp.diag(jnp.array([0.5, 1.0, 1.0, 1.0]))

    def shape_function_value(self, coords):
        return jnp.array(0.0)

    def symbolic(self):
        raise NotImplementedError

    def name(self):
        return "Constant"


def test_null_coord_time_nan_when_receiver_not_reached():
    """Receiver 100 units away with tau_max=1 is never reached: result is NaN."""
    from warpax.benchmarks.minkowski import MinkowskiMetric

    emitter = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    receiver = jnp.array([0.0, 100.0, 0.0, 0.0], dtype=jnp.float64)
    result = null_coord_time_asymmetry(MinkowskiMetric(), emitter, receiver, tau_max=1.0)
    assert jnp.isnan(result)


def test_static_observer_diagnostics_nan_when_g00_positive():
    """g_00 > 0 has no static observer: diagnostics are NaN, not ~1e15."""
    x = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    assert jnp.isnan(geodesic_deviation_diagnostic(_ConstantMetric(), x))
    assert jnp.isnan(
        blueshift_hazard(_ConstantMetric(), x, n_directions=1, tau_max=1.0, num_points=10)
    )


def test_blueshift_hazard_rejects_bad_n_directions():
    from warpax.benchmarks.minkowski import MinkowskiMetric

    x = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    with pytest.raises(ValueError):
        blueshift_hazard(MinkowskiMetric(), x, n_directions=7)
    with pytest.raises(ValueError):
        blueshift_hazard(MinkowskiMetric(), x, n_directions=0)


def test_tidal_max_jit_matches_eager_alcubierre():
    """Jitted tidal helper equals the eager max|eigenvalue| to rtol 1e-12."""
    from warpax.benchmarks import AlcubierreMetric
    from warpax.geodesics.deviation import tidal_eigenvalues
    from warpax.transport.diagnostics import _tidal_max

    metric = AlcubierreMetric(v_s=0.5, R=2.0, sigma=2.0)
    x = jnp.array([0.0, 1.5, 0.3, 0.2], dtype=jnp.float64)
    g = metric(x)
    u = jnp.array([1.0 / jnp.sqrt(-g[0, 0]), 0.0, 0.0, 0.0], dtype=jnp.float64)
    expected = jnp.max(jnp.abs(tidal_eigenvalues(metric, x, u)))
    assert expected > 0.0
    assert jnp.allclose(_tidal_max(metric, x, u), expected, rtol=1e-12, atol=0.0)


def test_null_coord_time_sampling_phase_does_not_reject_arrival():
    """A ray through the receiver is accepted even when no saved sample lands on it."""
    from warpax.benchmarks.minkowski import MinkowskiMetric

    emitter = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    receiver = jnp.array([0.0, 24.37, 0.0, 0.0], dtype=jnp.float64)
    result = null_coord_time_asymmetry(
        MinkowskiMetric(), emitter, receiver, tau_max=100.0, num_points=200
    )
    assert jnp.isfinite(result)
    assert jnp.abs(result) < 1e-6


_TRACES = {"n": 0}


class _FloatParamMetric(_ConstantMetric):
    """Flat metric with a Python-float field; counts how often __call__ is traced."""

    a: float

    def __call__(self, coords):
        _TRACES["n"] += 1
        return jnp.diag(jnp.array([-1.0, self.a, self.a, self.a]))


def test_tidal_max_no_retrace_across_float_params():
    """Metrics differing only in Python-float fields share one compiled tidal kernel."""
    from warpax.transport.diagnostics import _tidal_max

    x = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    u = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    assert float(_tidal_max(_FloatParamMetric(a=1.0), x, u)) == 0.0
    n_first = _TRACES["n"]
    assert n_first > 0
    assert float(_tidal_max(_FloatParamMetric(a=2.0), x, u)) == 0.0
    assert _TRACES["n"] == n_first
