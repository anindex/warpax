"""Tests for Minkowski (flat) spacetime benchmark."""

import jax
import jax.numpy as jnp

from warpax.benchmarks.minkowski import MinkowskiMetric

from .._helpers import ETA


class TestMinkowski:
    """Tests for MinkowskiMetric."""

    def test_minkowski_flat_metric(self, sample_coords):
        """Evaluate at arbitrary coords, verify diag(-1,1,1,1)."""
        m = MinkowskiMetric()
        g = m(sample_coords)
        assert jnp.allclose(g, ETA, atol=1e-15)

    def test_minkowski_jit(self, sample_coords):
        """jax.jit(MinkowskiMetric)(coords) returns correct result."""
        m = MinkowskiMetric()
        g = jax.jit(m)(sample_coords)
        assert jnp.allclose(g, ETA, atol=1e-15)

    def test_minkowski_pytree(self):
        """MinkowskiMetric is a valid pytree with no dynamic leaves."""
        m = MinkowskiMetric()
        leaves = jax.tree.leaves(m)
        assert leaves == []
