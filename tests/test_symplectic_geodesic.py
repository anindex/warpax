"""Tests for the symplectic (canonical Hamiltonian) geodesic integrator.

The headline guarantee: ``g(k,k)`` is conserved to ~machine precision, so a null
geodesic stays on the null cone even for a long crossing of a large warp bubble
where the adaptive Tsit5 integrator drifts off the cone by O(0.1). This is what
makes the rigorous geodesic-integrated ANEC defensible.
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from warpax.benchmarks import (
    AlcubierreMetric,
    MinkowskiMetric,
    SchwarzschildMetric,
)
from warpax.geodesics.initial_conditions import null_ic
from warpax.geodesics.integrator import integrate_geodesic
from warpax.geodesics.observables import monitor_conservation
from warpax.geodesics.symplectic import (
    SymplecticGeodesicResult,
    integrate_geodesic_symplectic,
    null_ic_canonical,
)


def _max_g_kk(result: SymplecticGeodesicResult) -> float:
    # H = 1/2 g(k,k); the witness is max|2H|.
    return float(jnp.max(jnp.abs(2.0 * result.H_values)))


class TestMinkowskiExact:
    def test_H_machine_zero(self):
        m = MinkowskiMetric()
        x0, p0 = null_ic_canonical(m, jnp.array([0.0, 0.0, 0.0, 0.0]),
                                   jnp.array([1.0, 0.0, 0.0]))
        r = integrate_geodesic_symplectic(m, x0, p0, (-5.0, 5.0),
                                          num_steps=512, order=4)
        assert r.complete
        assert r.termination_reason == "complete"
        assert _max_g_kk(r) < 1e-12

    def test_straight_line_trajectory(self):
        m = MinkowskiMetric()
        x0, p0 = null_ic_canonical(m, jnp.array([0.0, 0.0, 0.0, 0.0]),
                                   jnp.array([1.0, 0.0, 0.0]))
        r = integrate_geodesic_symplectic(m, x0, p0, (0.0, 5.0),
                                          num_steps=256, order=4)
        # x(lambda) = x0 + k * lambda (constant velocity): the x-position is
        # linear in the affine parameter.
        x_pos = r.positions[:, 1]
        # Fit a line and check residual is tiny.
        coeffs = jnp.polyfit(r.ts, x_pos, 1)
        residual = jnp.max(jnp.abs(x_pos - (coeffs[0] * r.ts + coeffs[1])))
        assert float(residual) < 1e-9


class TestSymplecticBeatsTsit5:
    """Headline: on a long large-bubble crossing the symplectic scheme holds
    the null cone where adaptive RK drifts off it."""

    def test_alcubierre_long_crossing(self):
        m = AlcubierreMetric(v_s=0.1, R=20.0, sigma=2.0)
        start = jnp.array([0.0, -30.0, 1e-3, 0.0])
        ndir = jnp.array([1.0, 0.0, 0.0])
        span = (0.0, 60.0)

        x0n, k0 = null_ic(m, start, ndir)
        gt = integrate_geodesic(m, x0n, k0, tau_span=span, num_points=512,
                                rtol=1e-10, atol=1e-10)
        tsit5_drift = float(jnp.max(jnp.abs(monitor_conservation(m, gt))))

        x0c, p0 = null_ic_canonical(m, start, ndir)
        rs = integrate_geodesic_symplectic(m, x0c, p0, span,
                                           num_steps=8192, order=4)
        sym_drift = _max_g_kk(rs)

        # The regime must actually stress the RK integrator...
        assert tsit5_drift > 1e-3, f"regime too easy: tsit5_drift={tsit5_drift}"
        # ...and the symplectic scheme must be far better (>=100x) and on-cone.
        assert sym_drift < 1e-2 * tsit5_drift
        assert sym_drift < 1e-7


class TestSchwarzschild:
    def test_radial_null_preserved(self):
        m = SchwarzschildMetric()
        # Radial outward null ray well outside the horizon.
        start = jnp.array([0.0, 8.0, 0.0, 0.0])
        x0c, p0 = null_ic_canonical(m, start, jnp.array([1.0, 0.0, 0.0]))
        r = integrate_geodesic_symplectic(m, x0c, p0, (0.0, 20.0),
                                          num_steps=8192, order=4)
        assert r.complete
        assert _max_g_kk(r) < 1e-8
        # Outward ray: radius increases monotonically.
        radius = r.positions[:, 1]
        assert float(radius[-1]) > float(radius[0])


class TestConvergence:
    def test_order4_improves_with_steps(self):
        m = AlcubierreMetric(v_s=0.1, R=20.0, sigma=2.0)
        start = jnp.array([0.0, -30.0, 1e-3, 0.0])
        x0c, p0 = null_ic_canonical(m, start, jnp.array([1.0, 0.0, 0.0]))
        drifts = []
        for ns in (2048, 4096, 8192):
            r = integrate_geodesic_symplectic(m, x0c, p0, (0.0, 60.0),
                                              num_steps=ns, order=4)
            drifts.append(_max_g_kk(r))
        # Monotone decrease, and a healthy (order-4) improvement factor.
        assert drifts[1] < drifts[0]
        assert drifts[2] < drifts[1]
        assert drifts[0] / drifts[2] > 16.0  # 4x steps twice, 4th order


class TestSentinels:
    def test_nan_momentum_returns_superluminal_sentinel(self):
        m = AlcubierreMetric(v_s=0.5)
        x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
        p0_bad = jnp.array([jnp.nan, 1.0, 0.0, 0.0])
        r = integrate_geodesic_symplectic(m, x0, p0_bad, (-2.0, 2.0),
                                          num_steps=64, order=2)
        assert r.complete is False
        assert r.termination_reason == "superluminal"
        assert bool(jnp.all(jnp.isnan(r.positions)))

    def test_invalid_order_raises(self):
        m = MinkowskiMetric()
        x0, p0 = null_ic_canonical(m, jnp.array([0.0, 0.0, 0.0, 0.0]),
                                   jnp.array([1.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="order"):
            integrate_geodesic_symplectic(m, x0, p0, (-1.0, 1.0),
                                          num_steps=32, order=3)


class TestDtype:
    def test_float64(self):
        m = MinkowskiMetric()
        x0, p0 = null_ic_canonical(m, jnp.array([0.0, 0.0, 0.0, 0.0]),
                                   jnp.array([1.0, 0.0, 0.0]))
        r = integrate_geodesic_symplectic(m, x0, p0, (-1.0, 1.0),
                                          num_steps=64, order=4)
        assert r.positions.dtype == jnp.float64
        assert r.H_values.dtype == jnp.float64
