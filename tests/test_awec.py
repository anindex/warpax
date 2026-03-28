"""Regression tests for warpax.averaged.awec .

Three tests:

- Vacuum AWEC is ~ 0 on Minkowski timelike worldline
- Alcubierre timelike sentinel (regression pin)
- tangent_norm validation
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from warpax.averaged import AWECResult, awec
from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric


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
