"""Pointwise curvature invariant correctness tests.

Tests curvature invariants at individual spacetime points against
analytical ground truth for Schwarzschild and Minkowski spacetimes.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from warpax.benchmarks.minkowski import MinkowskiMetric
from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.geometry.geometry import compute_curvature_chain
from warpax.geometry.invariants import (
    compute_invariants,
    kretschner_scalar,
    ricci_squared,
    weyl_squared,
)


# ---------------------------------------------------------------------------
# Minkowski: all invariants exactly zero
# ---------------------------------------------------------------------------


class TestMinkowskiInvariants:
    """Minkowski (flat) spacetime: all curvature invariants are zero."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        metric = MinkowskiMetric()
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        self.result = compute_curvature_chain(metric, coords)

    def test_kretschner_minkowski_zero(self):
        """Minkowski: Kretschner scalar K = 0 to machine precision."""
        K = kretschner_scalar(
            self.result.riemann, self.result.metric, self.result.metric_inv
        )
        assert_allclose(float(K), 0.0, atol=1e-14)

    def test_ricci_squared_minkowski_zero(self):
        """Minkowski: R_{ab} R^{ab} = 0 to machine precision."""
        R2 = ricci_squared(self.result.ricci, self.result.metric_inv)
        assert_allclose(float(R2), 0.0, atol=1e-14)

    def test_weyl_squared_minkowski_zero(self):
        """Minkowski: C_{abcd} C^{abcd} = 0 to machine precision."""
        K = kretschner_scalar(
            self.result.riemann, self.result.metric, self.result.metric_inv
        )
        R2 = ricci_squared(self.result.ricci, self.result.metric_inv)
        W2 = weyl_squared(K, R2, self.result.ricci_scalar)
        assert_allclose(float(W2), 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Schwarzschild: analytical ground truth
# ---------------------------------------------------------------------------


class TestSchwarzschildInvariants:
    """Schwarzschild black hole: invariants vs analytical formulas."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.M = 1.0
        metric = SchwarzschildMetric(M=self.M)
        # Test point far from singularity: (t=0, x=0, y=5, z=0)
        self.coords = jnp.array([0.0, 0.0, 5.0, 0.0])
        self.result = compute_curvature_chain(metric, self.coords)

        # Analytical Kretschner: K = 48 * M^2 / r_s^6
        # isotropic r_iso = 5.0, Schwarzschild r_s = r_iso * (1 + M/(2*r_iso))^2
        r_iso = 5.0
        r_s = r_iso * (1.0 + self.M / (2.0 * r_iso)) ** 2
        self.expected_K = 48.0 * self.M**2 / r_s**6

    def test_kretschner_schwarzschild_analytical(self):
        """Kretschner scalar matches K = 48*M^2/r_s^6 for Schwarzschild."""
        K = kretschner_scalar(
            self.result.riemann, self.result.metric, self.result.metric_inv
        )
        assert_allclose(float(K), self.expected_K, rtol=1e-10)

    def test_schwarzschild_ricci_flat(self):
        """Schwarzschild is vacuum: Ricci-flat => ricci_squared=0, weyl_squared=kretschner."""
        K = kretschner_scalar(
            self.result.riemann, self.result.metric, self.result.metric_inv
        )
        R2 = ricci_squared(self.result.ricci, self.result.metric_inv)
        W2 = weyl_squared(K, R2, self.result.ricci_scalar)

        # Ricci tensor vanishes for vacuum => ricci_squared ~ 0
        assert_allclose(float(R2), 0.0, atol=1e-10)
        # Weyl-squared = Kretschner for Ricci-flat spacetimes
        assert_allclose(float(W2), float(K), atol=1e-10)


# ---------------------------------------------------------------------------
# Convenience function and dtype tests
# ---------------------------------------------------------------------------


class TestInvariantConvenience:
    """Test compute_invariants convenience function and dtype enforcement."""

    def test_compute_invariants_convenience(self):
        """compute_invariants matches individual function calls on Schwarzschild."""
        metric = SchwarzschildMetric(M=1.0)
        coords = jnp.array([0.0, 0.0, 5.0, 0.0])
        result = compute_curvature_chain(metric, coords)

        # Individual calls
        K_individual = kretschner_scalar(result.riemann, result.metric, result.metric_inv)
        R2_individual = ricci_squared(result.ricci, result.metric_inv)
        W2_individual = weyl_squared(K_individual, R2_individual, result.ricci_scalar)

        # Convenience function
        K_conv, R2_conv, W2_conv = compute_invariants(result)

        assert_allclose(float(K_conv), float(K_individual), atol=0.0)
        assert_allclose(float(R2_conv), float(R2_individual), atol=0.0)
        assert_allclose(float(W2_conv), float(W2_individual), atol=0.0)

    def test_invariants_float64_dtype(self):
        """All invariant outputs are float64."""
        metric = SchwarzschildMetric(M=1.0)
        coords = jnp.array([0.0, 0.0, 5.0, 0.0])
        result = compute_curvature_chain(metric, coords)

        K = kretschner_scalar(result.riemann, result.metric, result.metric_inv)
        R2 = ricci_squared(result.ricci, result.metric_inv)
        W2 = weyl_squared(K, R2, result.ricci_scalar)

        assert K.dtype == jnp.float64
        assert R2.dtype == jnp.float64
        assert W2.dtype == jnp.float64
