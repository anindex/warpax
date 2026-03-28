"""Regression tests for warpax.quantum.ford_roman.

Four tests:

- ``test_C_constant_unit_audit``: C constant pinned to
  Fewster 2012 eq. 2.1.
- ``test_minkowski_vacuum_satisfies_qi``: on vacuum, margin is positive.
- ``test_alcubierre_offaxis_sentinel``: regression sentinel for an
  off-axis static worldline inside the bubble. (On-axis is degenerate
  at y=z=0; we use a finite y-offset instead.)
- ``test_invalid_sampling_raises``: ``sampling='gaussian'`` raises
  ValueError.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric
from warpax.quantum import QIResult, ford_roman
from warpax.quantum.ford_roman import FORD_ROMAN_CONSTANT_C


class TestFordRoman:
    """Ford-Roman QI regression tests.

    - test_C_constant_unit_audit: C constant pinned
    - test_minkowski_vacuum_satisfies_qi: on vacuum, margin is positive
    - test_alcubierre_offaxis_sentinel: regression sentinel
    - test_invalid_sampling_raises: validate input
    """

    def test_C_constant_unit_audit(self):
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

    def test_alcubierre_offaxis_sentinel(self):
        """Regression sentinel: static worldline at y=0.5 inside bubble.

        On-axis (y=z=0) is degenerate for the Alcubierre metric; we
        use a small y-offset. The exact margin value is pinned by
        reproducibility - the test asserts finiteness + repeatability.
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
