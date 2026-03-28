"""Regression tests for warpax.averaged.anec .

Five tests:

- ``test_minkowski_null_vacuum_zero`` - vacuum ANEC is ~0
- ``test_alcubierre_crossing_sentinel`` - regression pin for bubble crossing
- ``test_incomplete_geodesic_flag`` - truncated trajectory gives
  ``geodesic_complete=False``
- ``test_tangent_norm_renormalized_matches_fixed_to_tol`` - Minkowski
  null geodesic: both modes agree within 1e-6
- ``test_invalid_tangent_norm_raises`` - validation
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from warpax.averaged import ANECResult, anec
from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric
from warpax.geodesics.integrator import GeodesicResult


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
        li, gc, tr = result
        assert float(li) == float(result.line_integral)
        assert gc is result.geodesic_complete
        assert tr == result.termination_reason
