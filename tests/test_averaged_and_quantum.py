"""Averaged and quantum inequality tests: ANEC, AWEC, Ford-Roman."""

from __future__ import annotations

import functools
import importlib
import math

import jax.numpy as jnp
import pytest

from warpax.averaged import ANECResult, AWECResult, RigorousANEC, anec, anec_rigorous, awec
from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric
from warpax.geodesics.integrator import GeodesicResult
from warpax.geometry.metric import MetricSpecification
from warpax.quantum import QIResult, ford_roman
from warpax.quantum.ford_roman import FORD_ROMAN_CONSTANT_C

fr = importlib.import_module("warpax.quantum.ford_roman")


class TestANEC:
    """ANEC line-integral regression tests ."""

    def test_alcubierre_crossing_sentinel(self):
        """Alcubierre crossing null geodesic (off-axis at y=0.5):
        regression pin on the line integral value. The negative sign is
        corroborated by the rigorous-ANEC sign sentinel. rel=1e-6 because
        adaptive-step ODE results drift across platforms/BLAS; bit-level
        parity is pinned by the golden net on the capture platform."""
        metric = AlcubierreMetric()
        gl = lambda lam: jnp.array([lam, lam, 0.5, 0.0])
        result = anec(metric, gl)
        val = float(result.line_integral)
        assert val == pytest.approx(-0.25806813505162224, rel=1e-6)
        assert val < 0
        assert result.geodesic_complete is True

    def test_incomplete_geodesic_flag(self):
        """Synthetic GeodesicResult with non-success result_code:
        `geodesic_complete` must be False + termination_reason != 'complete'."""
        lam = jnp.linspace(-1.0, 1.0, 16)
        positions = jnp.stack([lam, lam, jnp.zeros_like(lam), jnp.zeros_like(lam)], axis=-1)
        velocities = jnp.broadcast_to(jnp.array([1.0, 1.0, 0.0, 0.0]), positions.shape)
        truncated_geo = GeodesicResult(
            ts=lam,
            positions=positions,
            velocities=velocities,
            result=1,
            event_mask=None,
        )
        result = anec(MinkowskiMetric(), truncated_geo)
        assert result.geodesic_complete is False
        assert result.termination_reason == "max_steps"

    def test_invalid_tangent_norm_raises(self):
        """``tangent_norm='bogus'`` raises ValueError."""
        gl = lambda lam: jnp.array([lam, lam, 0.0, 0.0])
        with pytest.raises(ValueError, match="tangent_norm must be one of"):
            anec(MinkowskiMetric(), gl, tangent_norm="bogus")

    def test_result_namedtuple(self):
        """``ANECResult`` exposes named attributes."""
        gl = lambda lam: jnp.array([lam, lam, 0.0, 0.0])
        result = anec(MinkowskiMetric(), gl)
        li, gc, tr, *_witness = result
        assert float(li) == float(result.line_integral)
        assert gc is result.geodesic_complete
        assert tr == result.termination_reason
        assert hasattr(result, "max_abs_g_kk")
        assert isinstance(result.null_preserved, bool)

    def test_witness_zero_on_minkowski(self):
        """Minkowski null ray: ANEC=0, on-cone witness ~0, and the result
        reports a complete geodesic."""
        gl = lambda lam: jnp.array([lam, lam, 0.0, 0.0])
        r = anec(MinkowskiMetric(), gl)
        assert isinstance(r, ANECResult)
        assert float(jnp.abs(r.line_integral)) < 1e-8
        assert float(r.max_abs_g_kk) < 1e-10
        assert r.null_preserved is True
        assert r.geodesic_complete is True
        assert r.termination_reason == "complete"


class TestRigorousANEC:
    """Symplectic geodesic-integrated ANEC with on-cone witness."""

    def test_minkowski_rigorous_zero(self):
        r = anec_rigorous(
            MinkowskiMetric(),
            jnp.array([0.0, -5.0, 1e-3, 0.0]),
            jnp.array([1.0, 0.0, 0.0]),
            affine_bounds=(0.0, 10.0),
            num_steps=512,
        )
        assert isinstance(r, RigorousANEC)
        assert r.method_used == "symplectic"
        assert r.projection is None
        assert float(jnp.abs(r.symplectic.line_integral)) < 1e-8
        assert r.symplectic.null_preserved is True

    def test_alcubierre_long_bubble_is_rigorous(self):
        """The long-crossing case where Tsit5 drifts off-cone is certified
        on-cone by the symplectic integrator (witness << tol)."""
        m = AlcubierreMetric(v_s=0.1, R=20.0, sigma=2.0)
        r = anec_rigorous(
            m,
            jnp.array([0.0, -30.0, 1e-3, 0.0]),
            jnp.array([1.0, 0.0, 0.0]),
            affine_bounds=(0.0, 60.0),
            num_steps=8192,
        )
        assert r.symplectic.geodesic_complete is True
        assert r.symplectic.null_preserved is True
        assert float(r.symplectic.max_abs_g_kk) < 1e-7
        assert jnp.isfinite(r.symplectic.line_integral)
        r2 = anec_rigorous(
            m,
            jnp.array([0.0, -30.0, 1e-3, 0.0]),
            jnp.array([1.0, 0.0, 0.0]),
            affine_bounds=(0.0, 60.0),
            num_steps=8192,
        )
        assert float(r2.symplectic.line_integral) == float(r.symplectic.line_integral)

    def test_alcubierre_anec_negative_through_bubble(self):
        """Sign sentinel: a null ray through the Alcubierre bubble has
        negative ANEC (NEC violation). Guards against a sign regression in
        the curvature/integrand chain that the Minkowski-zero sentinel
        cannot see.

        Geometry note: with the FUTURE-directed null tangent (null_ic root
        fix) the photon overtakes the bubble at relative speed 1 - v_s, so
        the crossing happens near t ~ 16 and the ANEC minimum over impact
        parameter sits inside the bubble (b = 0.5 here; the old
        past-directed pin at the wall b = 1.121 is positive for the
        future-directed ray)."""
        m = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0, x_s=0.0)
        r = anec_rigorous(
            m,
            jnp.array([0.0, -8.0, 0.5, 0.0]),
            jnp.array([1.0, 0.0, 0.0]),
            affine_bounds=(0.0, 30.0),
            num_steps=8192,
        )
        assert r.symplectic.null_preserved is True
        assert float(r.symplectic.line_integral) < -1e-3


class TestAWEC:
    """AWEC line-integral regression tests ."""

    def test_minkowski_timelike_vacuum_zero(self):
        """Vacuum AWEC on a static inertial worldline is ~ 0."""
        wl = lambda tau: jnp.array([tau, 0.0, 0.0, 0.0])
        result = awec(MinkowskiMetric(), wl)
        assert isinstance(result, AWECResult)
        assert float(jnp.abs(result.line_integral)) < 1e-6
        assert result.geodesic_complete is True

    def test_alcubierre_timelike_sentinel(self):
        """Alcubierre AWEC on an off-axis static worldline (y=0.5):
        regression pin on the line integral value.

        The parameter is not proper time here (the tangent needs
        renormalizing), so the integral carries dtau/dlambda; without that
        Jacobian the value is 0.10535108223138044. rel=1e-6 for cross-platform
        ODE drift, as for the ANEC pin.
        """
        metric = AlcubierreMetric()
        wl = lambda tau: jnp.array([tau, 0.0, 0.5, 0.0])
        result = awec(metric, wl)
        val = float(result.line_integral)
        assert val == pytest.approx(0.09474048865368674, rel=1e-6)
        assert result.timelike_preserved is True
        assert result.geodesic_complete is True
        assert result.termination_reason == "complete"

    def test_invalid_tangent_norm_raises(self):
        """``tangent_norm='bogus'`` raises ValueError."""
        wl = lambda tau: jnp.array([tau, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="tangent_norm must be one of"):
            awec(MinkowskiMetric(), wl, tangent_norm="bogus")

    def test_result_namedtuple(self):
        """``AWECResult`` exposes named attributes."""
        wl = lambda tau: jnp.array([tau, 0.0, 0.0, 0.0])
        result = awec(MinkowskiMetric(), wl)
        li, gc, tr, *_witness = result
        assert float(li) == float(result.line_integral)
        assert gc is result.geodesic_complete
        assert tr == result.termination_reason
        assert result.timelike_preserved is True
        assert float(result.max_u_sq) < 0.0

    def test_spacelike_curve_is_refused(self):
        """A spacelike curve is not an AWEC trajectory and must say so."""
        wl = lambda tau: jnp.array([0.0, tau, 0.0, 0.0])
        result = awec(MinkowskiMetric(), wl)
        assert result.timelike_preserved is False
        assert float(result.max_u_sq) > 0.0


class TestFordRoman:
    """Ford-Roman QI tests.

    - test_C_constant_units: C constant pinned
    - test_minkowski_vacuum_satisfies_qi: on vacuum, margin is positive
    - test_alcubierre_offaxis_pin: pinned-value check
    - test_invalid_sampling_raises: validate input
    - test_odd_rho_integrates_to_zero / test_constant_rho_below_bound_fails /
      test_spacelike_worldline_raises: counterexamples
    """

    def test_C_constant_units(self):
        """C = 3 / (32 pi^2) per Fewster 2012 eq. 2.1."""
        C_ref = 3.0 / (32.0 * math.pi**2)
        assert abs(float(FORD_ROMAN_CONSTANT_C) - C_ref) < 1e-15

    def test_minkowski_vacuum_satisfies_qi(self):
        """Vacuum rho = 0; integral = 0; margin = 0 - (-C/tau0^4) > 0."""
        metric = MinkowskiMetric()
        worldline = lambda t: jnp.array([t, 0.0, 0.0, 0.0])
        result = ford_roman(metric, worldline, tau0=1.0)
        assert isinstance(result, QIResult)
        assert float(result.margin) > 0
        assert float(result.bound) < 0

    def test_alcubierre_offaxis_pin(self):
        """Pinned-value check: static worldline at y=0.5 inside bubble.

        On-axis (y=z=0) is degenerate for the Alcubierre metric; we use a
        small y-offset. The value is the default 256-node proper-time grid
        with the +-1000 tau0 window; it converges to 0.02240506 at 4096
        nodes. Regenerate with ``ford_roman(AlcubierreMetric(), worldline,
        tau0=1.0)``.
        """
        metric = AlcubierreMetric()
        worldline = lambda t: jnp.array([t, 0.0, 0.5, 0.0])
        result = ford_roman(metric, worldline, tau0=1.0)
        margin_val = float(result.margin)
        assert margin_val == pytest.approx(0.022405197133122588, rel=1e-6)
        assert float(result.kernel_weight) == pytest.approx(
            2.0 / math.pi * math.atan(1000.0), abs=1e-5
        )
        result2 = ford_roman(metric, worldline, tau0=1.0)
        assert float(result2.margin) == margin_val

    def test_invalid_sampling_raises(self):
        """sampling='gaussian' must raise ValueError (only lorentzian supported)."""
        metric = MinkowskiMetric()
        worldline = lambda t: jnp.array([t, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="sampling must be 'lorentzian'"):
            ford_roman(metric, worldline, tau0=1.0, sampling="gaussian")

    def test_qi_result_namedtuple(self):
        """``QIResult`` exposes named attributes; ``kernel_weight`` defaults to 1."""
        metric = MinkowskiMetric()
        worldline = lambda t: jnp.array([t, 0.0, 0.0, 0.0])
        result = ford_roman(metric, worldline, tau0=1.0)
        m, b, c, w = result
        assert float(m) == float(result.margin)
        assert float(b) == float(result.bound)
        assert float(c) == float(result.C)
        assert float(w) == float(result.kernel_weight)
        assert float(QIResult(m, b, c).kernel_weight) == 1.0

    def test_odd_rho_integrates_to_zero(self, monkeypatch):
        """rho = lambda with rate 1: the Lorentzian kernel is even and the grid
        symmetric about lambda = 0, so the integral is 0 and margin == -bound.
        Centring on tau[n//2] instead offsets the kernel by 0.039 tau0 for 256
        samples and gives about +0.034 (analytic +0.037)."""
        monkeypatch.setattr(fr, "_rho_and_rate", lambda m, w, lam: (lam, jnp.ones_like(lam)))
        worldline = lambda t: jnp.array([t, 0.0, 0.0, 0.0])
        result = fr.ford_roman(MinkowskiMetric(), worldline, tau0=1.0)
        assert float(result.margin + result.bound) == pytest.approx(0.0, abs=1e-10)

    def test_constant_rho_below_bound_fails(self, monkeypatch):
        """rho = -1.03 C / tau0^4 constant violates the full-line bound, so the
        margin must be negative: (1 - 1.03 kernel_weight) C / tau0^4, about
        -2.8e-4 for tau0 = 1. A +-10 tau0 window captures only
        (2/pi) arctan(10) = 0.9365 of the kernel and reports +0.0355 C."""
        C = float(FORD_ROMAN_CONSTANT_C)
        monkeypatch.setattr(
            fr,
            "_rho_and_rate",
            lambda m, w, lam: (-1.03 * C * jnp.ones_like(lam), jnp.ones_like(lam)),
        )
        worldline = lambda t: jnp.array([t, 0.0, 0.0, 0.0])
        result = fr.ford_roman(MinkowskiMetric(), worldline, tau0=1.0)
        expected = (1.0 - 1.03 * float(result.kernel_weight)) * C
        assert float(result.margin) < 0
        assert float(result.margin) == pytest.approx(expected, abs=1e-5)
        assert expected == pytest.approx(-2.8e-4, abs=2e-6)

    def test_spacelike_worldline_raises(self):
        """x(lambda) = (0, lambda, 0, 0) in Minkowski has g(u,u) = +1, so the
        proper-time rate is NaN and ford_roman must refuse it; the old
        abs(u_sq) accepted it with rate 1."""
        worldline = lambda t: jnp.array([0.0, t, 0.0, 0.0])
        with pytest.raises(ValueError, match="worldline not timelike"):
            ford_roman(MinkowskiMetric(), worldline, tau0=1.0)


class TestProjectToNull:
    """_project_to_null keeps the root closest to 1.

    The naive root ``lam = (-cross + sqrt_disc) / (2 A_s)`` is the reflected
    null branch wherever ``A_t > A_s`` (g_00 > 0, for instance inside a
    superluminal Alcubierre bubble) and would map the exactly-null tangent
    (1, 1, 0, 0) to (1, 3, 0, 0); the root closest to 1 is the identity for
    already-null input.
    """

    def test_exactly_null_tangent_unchanged_in_g00_positive_region(self):
        from warpax.averaged.anec import _project_to_null

        m = AlcubierreMetric(v_s=2.0, R=2.0, sigma=8.0)
        x = jnp.array([0.0, 0.1, 0.0, 0.0])
        g = m(x)
        k = jnp.array([1.0, 1.0, 0.0, 0.0])
        assert float(g[0, 0]) > 0.0
        assert abs(float(jnp.einsum("a,ab,b->", k, g, k))) < 1e-12
        k_proj = _project_to_null(g, k)
        assert float(jnp.max(jnp.abs(k_proj - k))) < 1e-12, (
            f"exactly-null tangent changed: {k_proj}"
        )

    def test_non_null_vector_projects_onto_cone(self):
        from warpax.averaged.anec import _project_to_null

        m = AlcubierreMetric(v_s=2.0, R=2.0, sigma=8.0)
        x = jnp.array([0.0, 0.1, 0.0, 0.0])
        g = m(x)
        u = jnp.array([1.0, 0.7, 0.2, 0.1])
        k_proj = _project_to_null(g, u)
        g_kk = float(jnp.einsum("a,ab,b->", k_proj, g, k_proj))
        assert abs(g_kk) < 1e-12, f"projection off-cone: g(k,k) = {g_kk}"


class TestDiffraxResultCodes:
    """Diffrax result codes map to the reported termination reason.

    The equinox ``EnumerationItem`` carries ``._value``, not ``.value``, and
    a failed conversion must never read as success. The reason table must
    match the installed ``diffrax.RESULTS`` indices; max_steps_reached is
    code 1.
    """

    def test_max_steps_reached_reported_incomplete(self):
        """A max_steps-truncated geodesic must not report 'complete'."""
        from warpax.geodesics import integrate_geodesic, null_ic

        m = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        x0, k0 = null_ic(m, jnp.array([0.0, 5.0, 0.1, 0.0]), jnp.array([-1.0, 0.0, 0.0]))
        geo = integrate_geodesic(m, x0, k0, tau_span=(0.0, 15.0), num_points=32, max_steps=8)
        result = anec(m, geo)
        assert result.geodesic_complete is False
        assert result.termination_reason != "complete"
        assert result.termination_reason == "max_steps"

    def test_unconvertible_result_maps_to_unknown_not_success(self):
        """An unrecognized result object must NOT default to success."""
        from warpax.geodesics._result_codes import (
            RESULT_UNKNOWN,
            result_code_to_int,
            termination_reason,
        )

        class Opaque:
            pass

        code = result_code_to_int(Opaque())
        assert code == RESULT_UNKNOWN
        assert termination_reason(code) == "unknown"

    def test_awec_accepts_real_geodesic_result(self):
        """Regression: awec() crashed with TypeError on any real
        GeodesicResult from integrate_geodesic (``int(geodesic.result)``
        on a diffrax EnumerationItem)."""
        from warpax.geodesics import integrate_geodesic, timelike_ic

        mk = MinkowskiMetric()
        x0, v0 = timelike_ic(mk, jnp.array([0.0, 0.0, 0.0, 0.0]), jnp.array([0.3, 0.0, 0.0]))
        geo = integrate_geodesic(mk, x0, v0, tau_span=(0.0, 5.0), num_points=64)
        result = awec(mk, geo)
        assert result.geodesic_complete is True
        assert result.termination_reason == "complete"
        assert abs(float(result.line_integral)) < 1e-10


class TestDefaultTangentNorm:
    """anec() default tangent_norm is 'null_projected'."""

    def test_witness_measures_the_supplied_ray_not_the_projection(self):
        """The witness reports how far the SUPPLIED curve is off the cone.

        Measuring the projected vector instead made it null by construction,
        so any curve at all, timelike ones included, reported
        ``null_preserved=True``. An Alcubierre coordinate ray at y=0.5 is
        clearly off the cone and must say so.
        """
        m = AlcubierreMetric(v_s=0.5)
        ray = lambda lam: jnp.array([lam, lam, 0.5, 0.0])
        result = anec(m, ray, n_samples=64, affine_bounds=(-3.0, 3.0))
        assert float(result.max_abs_g_kk) > 1e-3
        assert result.null_preserved is False

    def test_witness_vanishes_on_a_genuine_null_ray(self):
        """On Minkowski a coordinate null ray is exactly null."""
        ray = lambda lam: jnp.array([lam, lam, 0.0, 0.0])
        result = anec(MinkowskiMetric(), ray, n_samples=64, affine_bounds=(-3.0, 3.0))
        assert float(result.max_abs_g_kk) <= 1e-10
        assert result.null_preserved is True

    def test_renormalized_still_available(self):
        """The legacy 'renormalized' option remains selectable."""
        gl = lambda lam: jnp.array([lam, lam, 0.0, 0.0])
        result = anec(MinkowskiMetric(), gl, tangent_norm="renormalized")
        assert float(jnp.abs(result.line_integral)) < 1e-6


def test_project_to_null_double_root():
    """With ``g_00 = 0`` the ``t``-axis is already null, so for ``u = (1, 0, 1, 0)``
    the quadratic ``A_s lam^2 + cross lam + A_t`` has ``A_t = cross = 0`` and the
    double root ``lam = 0``: ``k = (1, 0, 0, 0)`` with ``g(k, k) = 0`` (the old
    fallback ``lam = 1`` returned ``u`` itself, with ``g(u, u) = 1``)."""
    from warpax.averaged.anec import _project_to_null

    g = jnp.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    k = _project_to_null(g, jnp.array([1.0, 0.0, 1.0, 0.0]))
    assert float(jnp.max(jnp.abs(k - jnp.array([1.0, 0.0, 0.0, 0.0])))) < 1e-12
    assert abs(float(jnp.einsum("a,ab,b->", k, g, k))) < 1e-12


class _NullAxisMetric(MetricSpecification):
    """Constant metric with g_00 = 0, g_01 = -1: the coordinate t-axis is null."""

    def __call__(self, coords):
        return jnp.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def shape_function_value(self, coords):
        return jnp.array(0.0)

    def symbolic(self):
        raise NotImplementedError

    def name(self):
        return "NullAxis"


def test_project_to_null_fully_degenerate_quadratic_returns_input():
    """For ``u = (1, 0, 0, 0)`` on the null-axis metric the spatial part vanishes,
    so ``A_t = A_s = cross = 0`` and every scaling solves the quadratic; the
    projection returns ``u`` itself with ``g(u, u) = 0``."""
    from warpax.averaged.anec import _project_to_null

    g = _NullAxisMetric()(jnp.zeros(4))
    u = jnp.array([1.0, 0.0, 0.0, 0.0])
    k = _project_to_null(g, u)
    assert float(jnp.max(jnp.abs(k - u))) == 0.0
    assert float(jnp.einsum("a,ab,b->", k, g, k)) == 0.0


def test_anec_along_null_coordinate_axis_is_vacuum():
    """The worldline ``(lambda, 0, 0, 0)`` is null on the constant null-axis
    metric, and a constant metric is flat, so the ANEC integral over
    ``(-1, 1)`` is 0 and the tangent is reported null-preserved."""
    result = anec(
        _NullAxisMetric(), lambda lam: jnp.array([lam, 0.0, 0.0, 0.0]), affine_bounds=(-1.0, 1.0)
    )
    assert float(result.line_integral) == 0.0
    assert result.null_preserved is True


class _ExponentialWallMetric(MetricSpecification):
    """``g = diag(-1, 1, exp(2 a x), exp(2 a x))``: static observers measure
    ``rho = -3 a^2 / (8 pi)`` uniformly, so ``a^2 = 1.03 C 8 pi / 3`` gives
    ``rho = -1.03 C``."""

    def __call__(self, coords):
        a = math.sqrt(1.03 * float(FORD_ROMAN_CONSTANT_C) * 8.0 * math.pi / 3.0)
        e = jnp.exp(2.0 * a * coords[1])
        return jnp.diag(jnp.array([-1.0, 1.0, e, e]))

    def shape_function_value(self, coords):
        return jnp.array(0.0)

    def symbolic(self):
        raise NotImplementedError

    def name(self):
        return "ExponentialWall"


@functools.lru_cache(maxsize=1)
def _ford_roman_by_rate() -> dict[float, QIResult]:
    """``ford_roman`` on the static worldline ``(rate lambda, 0, 0, 0)`` for
    three parameterizations of the same geodesic."""
    return {
        rate: ford_roman(
            _ExponentialWallMetric(),
            lambda lam, rate=rate: jnp.array([rate * lam, 0.0, 0.0, 0.0]),
            tau0=1.0,
        )
        for rate in (1.0, 100.0, 0.001)
    }


def test_ford_roman_margin_invariant_under_reparameterisation():
    """``rho = -1.03 C`` along a static worldline of the exponential-wall metric
    with ``tau0 = 1`` gives margin ``(1 - 1.03 w) C`` with
    ``w = (2/pi) arctan(1000) = 0.99936``, about ``-2.8e-4``, for every rate
    ``dt/dlambda`` in ``{1, 100, 0.001}``: the grid is built in proper time,
    so the three margins agree within 1e-6."""
    margins = [float(r.margin) for r in _ford_roman_by_rate().values()]
    expected = (1.0 - 1.03 * 2.0 / math.pi * math.atan(1000.0)) * float(FORD_ROMAN_CONSTANT_C)
    assert max(margins) - min(margins) < 1e-6
    for m in margins:
        assert m == pytest.approx(expected, abs=5e-6)
        assert m < 0


def test_ford_roman_kernel_weight_invariant_under_reparameterisation():
    """The sampled window reaches ``+-1000 tau0`` in proper time whatever the
    parameter rate, so ``kernel_weight = (2/pi) arctan(1000) = 0.99936`` for
    rates 1, 100 and 0.001 alike, equal within 1e-6."""
    weights = [float(r.kernel_weight) for r in _ford_roman_by_rate().values()]
    assert max(weights) - min(weights) < 1e-6
    for w in weights:
        assert w == pytest.approx(2.0 / math.pi * math.atan(1000.0), abs=1e-6)


@pytest.mark.parametrize(
    "t_of_lam",
    [lambda lam: lam + lam**3 / 3e4, lambda lam: lam**3 / 3 + lam],
    ids=["lam+lam^3/3e4", "lam^3/3+lam"],
)
def test_ford_roman_invariant_under_growing_rate(t_of_lam):
    """Static exponential-wall worldline ``(t(lambda), 0, 0, 0)`` with
    ``dt/dlambda`` growing from 1 at the center to 100 or more by the far
    tail is the same geodesic as ``(lambda, 0, 0, 0)``, so the margin equals
    the rate-1 margin ``(1 - 1.03 w) C`` within 1e-6 and the sampled window
    still reaches ``+-1000 tau0``: ``kernel_weight = (2/pi) arctan(1000)``
    within 1e-6."""
    result = ford_roman(
        _ExponentialWallMetric(),
        lambda lam: jnp.array([t_of_lam(lam), 0.0, 0.0, 0.0]),
        tau0=1.0,
    )
    reference = _ford_roman_by_rate()[1.0]
    assert float(result.margin) == pytest.approx(float(reference.margin), abs=1e-6)
    assert float(result.kernel_weight) == pytest.approx(2.0 / math.pi * math.atan(1000.0), abs=1e-6)
