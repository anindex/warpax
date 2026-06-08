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

    def test_minkowski_null_vacuum_zero(self):
        """On Minkowski + null geodesic x^a(lambda) = (lambda, lambda, 0, 0),
        ANEC integral ~ 0 and geodesic_complete=True."""
        gl = lambda lam: jnp.array([lam, lam, 0.0, 0.0])
        result = anec(MinkowskiMetric(), gl)
        assert isinstance(result, ANECResult)
        assert float(jnp.abs(result.line_integral)) < 1e-6
        assert result.geodesic_complete is True
        assert result.termination_reason == "complete"

    def test_alcubierre_crossing_sentinel(self):
        """Alcubierre crossing null geodesic (off-axis at y=0.5):
        regression pin on finite ``line_integral``."""
        metric = AlcubierreMetric()
        gl = lambda lam: jnp.array([lam, lam, 0.5, 0.0])
        result = anec(metric, gl)
        # Regression pin: finite + deterministic across invocations
        val = float(result.line_integral)
        assert jnp.isfinite(val)
        # Re-run: same value
        result2 = anec(metric, gl)
        assert float(result2.line_integral) == val
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

    def test_tangent_norm_renormalized_matches_fixed_to_tol(self):
        """On Minkowski null geodesic (trivial, no tangent drift), both
        modes agree to within 1e-6."""
        gl = lambda lam: jnp.array([lam, lam, 0.0, 0.0])
        r_renorm = anec(MinkowskiMetric(), gl, tangent_norm="renormalized")
        r_fixed = anec(MinkowskiMetric(), gl, tangent_norm="fixed")
        assert (
            abs(float(r_renorm.line_integral) - float(r_fixed.line_integral))
            < 1e-6
        )

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
        """Minkowski null ray: ANEC=0 AND on-cone witness ~0."""
        gl = lambda lam: jnp.array([lam, lam, 0.0, 0.0])
        r = anec(MinkowskiMetric(), gl)
        assert float(jnp.abs(r.line_integral)) < 1e-8
        assert float(r.max_abs_g_kk) < 1e-10
        assert r.null_preserved is True


class TestRigorousANEC:
    """Symplectic geodesic-integrated ANEC with on-cone witness (P0)."""

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

    def test_alcubierre_anec_negative_at_wall(self):
        """Sign sentinel: a null ray through the Alcubierre wall has negative
        ANEC (NEC violation). On-axis the integral is positive; the minimum
        over impact parameter sits at the wall (b ~ R_b, here b=1.121, the
        certified K6b minimum). Guards against a sign regression in the
        curvature/integrand chain that the Minkowski-zero sentinel cannot see."""
        m = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0, x_s=0.0)
        r = anec_rigorous(
            m, jnp.array([0.0, -8.0, 1.121, 0.0]), jnp.array([1.0, 0.0, 0.0]),
            affine_bounds=(0.0, 16.0), num_steps=8192,
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
        regression pin on finite + deterministic value."""
        metric = AlcubierreMetric()
        wl = lambda tau: jnp.array([tau, 0.0, 0.5, 0.0])
        result = awec(metric, wl)
        val = float(result.line_integral)
        assert jnp.isfinite(val)
        result2 = awec(metric, wl)
        assert float(result2.line_integral) == val

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
