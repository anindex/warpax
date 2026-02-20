"""Tests for the metric framework: ADM reconstruction, SymbolicMetric."""

import jax
import jax.numpy as jnp
import sympy as sp

from warpax.geometry.metric import SymbolicMetric, adm_to_full_metric


# =========================================================================
# adm_to_full_metric tests
# =========================================================================


class TestADMToFullMetric:
    """Tests for the ADM -> full metric reconstruction."""

    def test_adm_to_full_metric_flat(self):
        """adm_to_full_metric(1, [0,0,0], eye(3)) == diag(-1,1,1,1)."""
        alpha = jnp.array(1.0)
        beta_up = jnp.array([0.0, 0.0, 0.0])
        gamma = jnp.eye(3)

        g = adm_to_full_metric(alpha, beta_up, gamma)
        expected = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, expected, atol=1e-15)

    def test_adm_to_full_metric_shift(self):
        """adm_to_full_metric with nonzero shift, verify g_0i = beta_down_i."""
        alpha = jnp.array(1.0)
        beta_up = jnp.array([0.5, 0.0, 0.0])
        gamma = jnp.eye(3)

        g = adm_to_full_metric(alpha, beta_up, gamma)
        # beta_down = gamma @ beta_up = [0.5, 0, 0]
        # g_00 = -(alpha^2 - beta_down . beta_up) = -(1 - 0.25) = -0.75
        assert jnp.isclose(g[0, 0], -0.75, atol=1e-15)
        # g_01 = beta_down_x = 0.5
        assert jnp.isclose(g[0, 1], 0.5, atol=1e-15)
        assert jnp.isclose(g[1, 0], 0.5, atol=1e-15)
        # g_02, g_03 = 0
        assert jnp.isclose(g[0, 2], 0.0, atol=1e-15)
        assert jnp.isclose(g[0, 3], 0.0, atol=1e-15)
        # spatial block = gamma
        assert jnp.allclose(g[1:, 1:], gamma, atol=1e-15)

    def test_adm_to_full_metric_jit(self):
        """Run through jax.jit, verify same result."""
        alpha = jnp.array(1.0)
        beta_up = jnp.array([0.3, -0.1, 0.2])
        gamma = jnp.eye(3)

        g_eager = adm_to_full_metric(alpha, beta_up, gamma)
        g_jit = jax.jit(adm_to_full_metric)(alpha, beta_up, gamma)
        assert jnp.allclose(g_eager, g_jit, atol=1e-15)

    def test_adm_to_full_metric_float64(self):
        """Verify output dtype is float64."""
        alpha = jnp.array(1.0)
        beta_up = jnp.array([0.0, 0.0, 0.0])
        gamma = jnp.eye(3)

        g = adm_to_full_metric(alpha, beta_up, gamma)
        assert g.dtype == jnp.float64


# =========================================================================
# SymbolicMetric tests
# =========================================================================


class TestSymbolicMetric:
    """Tests for the SymbolicMetric class."""

    def test_symbolic_metric_creation(self):
        """Create SymbolicMetric, verify coords and g."""
        t, x, y, z = sp.symbols("t x y z")
        g = sp.diag(-1, 1, 1, 1)
        sm = SymbolicMetric([t, x, y, z], g)
        assert sm.coords == [t, x, y, z]
        assert sm.g == g
        assert sm.g.shape == (4, 4)

    def test_symbolic_metric_inverse(self):
        """Verify g * g_inv = identity (symbolically)."""
        t, x, y, z = sp.symbols("t x y z")
        g = sp.diag(-1, 1, 1, 1)
        sm = SymbolicMetric([t, x, y, z], g)
        product = sp.simplify(sm.g * sm.g_inv)
        assert product == sp.eye(4)

    def test_symbolic_metric_invalid_coords(self):
        """Verify ValueError for wrong number of coordinates."""
        x, y, z = sp.symbols("x y z")
        g = sp.diag(-1, 1, 1, 1)
        import pytest

        with pytest.raises(ValueError, match="4 coordinate symbols"):
            SymbolicMetric([x, y, z], g)

    def test_symbolic_metric_invalid_shape(self):
        """Verify ValueError for wrong matrix shape."""
        t, x, y, z = sp.symbols("t x y z")
        g = sp.diag(-1, 1, 1)  # 3x3 instead of 4x4
        import pytest

        with pytest.raises(ValueError, match="\\(4, 4\\)"):
            SymbolicMetric([t, x, y, z], g)
