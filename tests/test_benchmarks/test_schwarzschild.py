"""Tests for Schwarzschild black hole benchmark."""

import jax
import jax.numpy as jnp

from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.geometry.metric import SymbolicMetric


class TestSchwarzschild:
    """Tests for SchwarzschildMetric in isotropic Cartesian coordinates."""

    def test_schwarzschild_at_large_r(self):
        """At large r, metric approaches Minkowski (g_00 -> -1, g_ii -> 1).

        In isotropic coords, deviation ~ M/r_iso, so r_iso=1e6 gives ~1e-6.
        """
        m = SchwarzschildMetric(M=1.0)
        far_coords = jnp.array([0.0, 1e6, 0.0, 0.0])
        g = m(far_coords)
        minkowski = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, minkowski, atol=1e-5)

    def test_schwarzschild_symmetry(self, sample_coords):
        """g(t, x, y, z) is diagonal (off-diagonal = 0)."""
        m = SchwarzschildMetric()
        g = m(sample_coords)
        # Off-diagonal elements should be zero
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert jnp.isclose(g[i, j], 0.0, atol=1e-15), (
                        f"g[{i},{j}] = {g[i, j]} != 0"
                    )

    def test_schwarzschild_jit(self, sample_coords):
        """jax.jit compilation works."""
        m = SchwarzschildMetric()
        g_eager = m(sample_coords)
        g_jit = jax.jit(m)(sample_coords)
        assert jnp.allclose(g_eager, g_jit, atol=1e-15)

    def test_schwarzschild_float64(self, sample_coords):
        """Output dtype is float64."""
        m = SchwarzschildMetric()
        g = m(sample_coords)
        assert g.dtype == jnp.float64

    def test_schwarzschild_parameter_change(self, sample_coords):
        """Change M, verify output changes (dynamic field)."""
        m1 = SchwarzschildMetric(M=1.0)
        m2 = SchwarzschildMetric(M=2.0)
        g1 = m1(sample_coords)
        g2 = m2(sample_coords)
        assert not jnp.allclose(g1, g2, atol=1e-10)

    def test_schwarzschild_symbolic(self):
        """symbolic() returns valid SymbolicMetric."""
        m = SchwarzschildMetric()
        sm = m.symbolic()
        assert isinstance(sm, SymbolicMetric)
        assert sm.g.shape == (4, 4)
        assert len(sm.coords) == 4

    def test_schwarzschild_name(self):
        """name() returns 'Schwarzschild'."""
        m = SchwarzschildMetric()
        assert m.name() == "Schwarzschild"

    def test_schwarzschild_g00_negative(self, sample_coords):
        """g_00 should be negative (timelike signature) outside horizon."""
        m = SchwarzschildMetric(M=1.0)
        g = m(sample_coords)
        assert g[0, 0] < 0.0

    def test_schwarzschild_gii_positive(self, sample_coords):
        """g_ii should be positive (spacelike) for i=1,2,3."""
        m = SchwarzschildMetric(M=1.0)
        g = m(sample_coords)
        for i in range(1, 4):
            assert g[i, i] > 0.0
