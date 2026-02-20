"""Tests for Minkowski (flat) spacetime benchmark."""

import jax
import jax.numpy as jnp

from warpax.benchmarks.minkowski import MinkowskiMetric
from warpax.geometry.metric import SymbolicMetric


class TestMinkowski:
    """Tests for MinkowskiMetric."""

    def test_minkowski_flat_metric(self, sample_coords):
        """Evaluate at arbitrary coords, verify diag(-1,1,1,1)."""
        m = MinkowskiMetric()
        g = m(sample_coords)
        expected = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, expected, atol=1e-15)

    def test_minkowski_jit(self, sample_coords):
        """jax.jit(MinkowskiMetric())(coords) returns correct result."""
        m = MinkowskiMetric()
        g = jax.jit(m)(sample_coords)
        expected = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, expected, atol=1e-15)

    def test_minkowski_float64(self, sample_coords):
        """Output dtype is float64."""
        m = MinkowskiMetric()
        g = m(sample_coords)
        assert g.dtype == jnp.float64

    def test_minkowski_pytree(self):
        """MinkowskiMetric is a valid pytree (jax.tree.leaves returns expected)."""
        m = MinkowskiMetric()
        leaves = jax.tree.leaves(m)
        # MinkowskiMetric has no dynamic fields, so leaves should be empty
        assert isinstance(leaves, list)

    def test_minkowski_symbolic(self):
        """symbolic() returns valid SymbolicMetric."""
        m = MinkowskiMetric()
        sm = m.symbolic()
        assert isinstance(sm, SymbolicMetric)
        assert sm.g.shape == (4, 4)
        assert len(sm.coords) == 4

    def test_minkowski_name(self):
        """name() returns 'Minkowski'."""
        m = MinkowskiMetric()
        assert m.name() == "Minkowski"

    def test_minkowski_at_origin(self, origin_coords):
        """Metric at origin is still flat Minkowski."""
        m = MinkowskiMetric()
        g = m(origin_coords)
        expected = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, expected, atol=1e-15)
