"""Averaged and quantum inequality tests: ANEC, AWEC, Ford-Roman."""

from __future__ import annotations
from warpax.averaged import ANECResult, RigorousANEC, anec, anec_rigorous
from warpax.averaged import AWECResult, awec
from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric
from warpax.geodesics.integrator import GeodesicResult
from warpax.quantum import QIResult, ford_roman
from warpax.quantum.ford_roman import FORD_ROMAN_CONSTANT_C
import jax.numpy as jnp
import math
import pytest



class TestANEC:
    """ANEC line-integral regression tests ."""

    def test_alcubierre_crossing_sentinel(self):
        """Alcubierre crossing null geodesic (off-axis at y=0.5):
        regression pin on the line integral value. The negative sign is
        corroborated by the rigorous-ANEC sign sentinel."""
        metric = AlcubierreMetric()
        gl = lambda lam: jnp.array([lam, lam, 0.5, 0.0])
        result = anec(metric, gl)
        val = float(result.line_integral)
        # rel=1e-6: adaptive-step ODE results drift across platforms/BLAS;
        # bit-level parity is pinned by the golden net on the capture platform.
        assert val == pytest.approx(-0.25806813505162224, rel=1e-6)
        assert val < 0  # NEC violation, sign backup for the pin
        assert result.geodesic_complete is True

    def test_incomplete_geodesic_flag(self):
        """Synthetic GeodesicResult with non-success result_code:
        `geodesic_complete` must be False + termination_reason != 'complete'."""
        lam = jnp.linspace(-1.0, 1.0, 16)
        positions = jnp.stack(
            [lam, lam, jnp.zeros_like(lam), jnp.zeros_like(lam)], axis=-1
        )
        velocities = jnp.broadcast_to(
            jnp.array([1.0, 1.0, 0.0, 0.0]), positions.shape
        )
        # result=1 => max_steps_reached
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
        # Witness fields appended (append-only NamedTuple extension).
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
            affine_bounds=(0.0, 10.0), num_steps=512,
        )
        assert isinstance(r, RigorousANEC)
        assert r.method_used == "symplectic"
        assert r.projection is None
        assert float(jnp.abs(r.symplectic.line_integral)) < 1e-8
        assert r.symplectic.null_preserved is True

    def test_alcubierre_long_bubble_is_rigorous(self):
        """The long-crossing case where Tsit5 drifts off-cone is now certified
        on-cone by the symplectic integrator (witness << tol)."""
        m = AlcubierreMetric(v_s=0.1, R=20.0, sigma=2.0)
        r = anec_rigorous(
            m, jnp.array([0.0, -30.0, 1e-3, 0.0]), jnp.array([1.0, 0.0, 0.0]),
            affine_bounds=(0.0, 60.0), num_steps=8192,
        )
        assert r.symplectic.geodesic_complete is True
        assert r.symplectic.null_preserved is True
        assert float(r.symplectic.max_abs_g_kk) < 1e-7
        assert jnp.isfinite(r.symplectic.line_integral)
        # Deterministic across invocations.
        r2 = anec_rigorous(
            m, jnp.array([0.0, -30.0, 1e-3, 0.0]), jnp.array([1.0, 0.0, 0.0]),
            affine_bounds=(0.0, 60.0), num_steps=8192,
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
            m, jnp.array([0.0, -8.0, 0.5, 0.0]), jnp.array([1.0, 0.0, 0.0]),
            affine_bounds=(0.0, 30.0), num_steps=8192,
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
        regression pin on the line integral value."""
        metric = AlcubierreMetric()
        wl = lambda tau: jnp.array([tau, 0.0, 0.5, 0.0])
        result = awec(metric, wl)
        val = float(result.line_integral)
        # rel=1e-6: cross-platform ODE drift (see ANEC pin above).
        assert val == pytest.approx(0.10535108223138044, rel=1e-6)
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
        li, gc, tr = result
        assert float(li) == float(result.line_integral)
        assert gc is result.geodesic_complete
        assert tr == result.termination_reason


class TestFordRoman:
    """Ford-Roman QI tests.

    - test_C_constant_units: C constant pinned
    - test_minkowski_vacuum_satisfies_qi: on vacuum, margin is positive
    - test_alcubierre_offaxis_pin: pinned-value check
    - test_invalid_sampling_raises: validate input
    """

    def test_C_constant_units(self):
        """C = 3 / (32 pi^2) per Fewster 2012 eq. 2.1."""
        C_ref = 3.0 / (32.0 * math.pi ** 2)
        assert abs(float(FORD_ROMAN_CONSTANT_C) - C_ref) < 1e-15

    def test_minkowski_vacuum_satisfies_qi(self):
        """Vacuum rho = 0; integral = 0; margin = 0 - (-C/tau0^4) > 0."""
        metric = MinkowskiMetric()
        worldline = lambda t: jnp.array([t, 0.0, 0.0, 0.0])
        result = ford_roman(metric, worldline, tau0=1.0)
        assert isinstance(result, QIResult)
        assert float(result.margin) > 0
        # Bound is negative (Ford-Roman bound is -C/tau0^4).
        assert float(result.bound) < 0

    def test_alcubierre_offaxis_pin(self):
        """Pinned-value check: static worldline at y=0.5 inside bubble.

        On-axis (y=z=0) is degenerate for the Alcubierre metric; we
        use a small y-offset. The test asserts finiteness and repeatability
        of the resulting margin.
        """
        metric = AlcubierreMetric()
        # Off-axis worldline: constant position at y=0.5
        worldline = lambda t: jnp.array([t, 0.0, 0.5, 0.0])
        result = ford_roman(metric, worldline, tau0=1.0)
        margin_val = float(result.margin)
        assert jnp.isfinite(margin_val)
        # Re-run must match for determinism
        result2 = ford_roman(metric, worldline, tau0=1.0)
        assert float(result2.margin) == margin_val

    def test_invalid_sampling_raises(self):
        """sampling='gaussian' must raise ValueError (only lorentzian supported)."""
        metric = MinkowskiMetric()
        worldline = lambda t: jnp.array([t, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="sampling must be 'lorentzian'"):
            ford_roman(metric, worldline, tau0=1.0, sampling="gaussian")

    def test_qi_result_namedtuple(self):
        """``QIResult`` exposes named attributes."""
        metric = MinkowskiMetric()
        worldline = lambda t: jnp.array([t, 0.0, 0.0, 0.0])
        result = ford_roman(metric, worldline, tau0=1.0)
        m, b, c = result
        assert float(m) == float(result.margin)
        assert float(b) == float(result.bound)
        assert float(c) == float(result.C)


class TestProjectToNull:
    """Regression: _project_to_null picked the wrong quadratic root.

    Bug: ``lam = (-cross + sqrt_disc) / (2 A_s)`` selects the *reflected*
    null branch wherever ``A_t > A_s`` (g_00 > 0 regions, e.g. the interior
    of a superluminal Alcubierre bubble), mapping the exactly-null tangent
    (1, 1, 0, 0) to (1, 3, 0, 0). The fix picks the root closest to 1, the
    identity for already-null input.
    """

    def test_exactly_null_tangent_unchanged_in_g00_positive_region(self):
        from warpax.averaged.anec import _project_to_null

        m = AlcubierreMetric(v_s=2.0, R=2.0, sigma=8.0)
        x = jnp.array([0.0, 0.1, 0.0, 0.0])
        g = m(x)
        k = jnp.array([1.0, 1.0, 0.0, 0.0])
        # Interior point: g_00 > 0 and (1,1,0,0) is exactly null there.
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
        u = jnp.array([1.0, 0.7, 0.2, 0.1])  # not null
        k_proj = _project_to_null(g, u)
        g_kk = float(jnp.einsum("a,ab,b->", k_proj, g, k_proj))
        assert abs(g_kk) < 1e-12, f"projection off-cone: g(k,k) = {g_kk}"


class TestDiffraxResultCodes:
    """Regression: diffrax 0.7.2 result codes were silently swallowed.

    Bug: ``int(getattr(raw, 'value', raw))`` raises TypeError on the
    equinox ``EnumerationItem`` (which carries ``._value``, not ``.value``),
    and the ``except`` mapped *every* outcome -- including failures -- to
    success. Also the old reason table did not match the installed
    ``diffrax.RESULTS`` indices.
    """

    def test_max_steps_reached_reported_incomplete(self):
        """A max_steps-truncated geodesic must not report 'complete'."""
        from warpax.geodesics import integrate_geodesic, null_ic

        m = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        x0, k0 = null_ic(
            m, jnp.array([0.0, 5.0, 0.1, 0.0]), jnp.array([-1.0, 0.0, 0.0])
        )
        # max_steps far too small: diffrax reports max_steps_reached (code 1).
        geo = integrate_geodesic(
            m, x0, k0, tau_span=(0.0, 15.0), num_points=32, max_steps=8
        )
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
        x0, v0 = timelike_ic(
            mk, jnp.array([0.0, 0.0, 0.0, 0.0]), jnp.array([0.3, 0.0, 0.0])
        )
        geo = integrate_geodesic(mk, x0, v0, tau_span=(0.0, 5.0), num_points=64)
        result = awec(mk, geo)
        assert result.geodesic_complete is True
        assert result.termination_reason == "complete"
        assert abs(float(result.line_integral)) < 1e-10


class TestDefaultTangentNorm:
    """API change: anec() default tangent_norm is now 'null_projected'."""

    def test_default_path_stays_on_null_cone(self):
        """The default must yield a vanishing on-cone witness on a short
        Alcubierre coordinate ray (projection is exact by construction)."""
        m = AlcubierreMetric(v_s=0.5)
        ray = lambda lam: jnp.array([lam, lam, 0.5, 0.0])
        result = anec(m, ray, n_samples=64, affine_bounds=(-3.0, 3.0))
        assert float(result.max_abs_g_kk) <= 1e-10
        assert result.null_preserved is True

    def test_renormalized_still_available(self):
        """The legacy 'renormalized' option remains selectable."""
        gl = lambda lam: jnp.array([lam, lam, 0.0, 0.0])
        result = anec(MinkowskiMetric(), gl, tangent_norm="renormalized")
        assert float(jnp.abs(result.line_integral)) < 1e-6
