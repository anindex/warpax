"""SymPy-to-JAX bridge cross-validation tests.

For each benchmark metric, verifies that the JAX numeric evaluation matches
the SymPy symbolic form converted via sympy_metric_to_jax, to machine
precision (atol=1e-14).
"""

import jax
import jax.numpy as jnp
import sympy as sp
import pytest

from warpax.geometry.metric import sympy_metric_to_jax
from warpax.benchmarks.minkowski import MinkowskiMetric
from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.benchmarks.alcubierre import AlcubierreMetric


# ---------------------------------------------------------------------------
# Test coordinate points
# ---------------------------------------------------------------------------

TEST_POINTS = [
    jnp.array([0.0, 1.0, 2.0, 3.0]),       # standard
    jnp.array([0.0, 0.01, 0.01, 0.01]),     # near-origin
    jnp.array([0.0, 100.0, 0.0, 0.0]),      # far field
    jnp.array([0.0, 1.0, 1.0, 1.0]),        # off-axis
]


# =========================================================================
# Minkowski bridge tests
# =========================================================================


class TestMinkowskiSympyBridge:
    """SymPy bridge cross-validation for Minkowski."""

    def test_minkowski_sympy_matches_jax(self):
        """Evaluate JAX metric and sympy bridge at several points."""
        m = MinkowskiMetric()
        sm = m.symbolic()
        bridge_fn = sympy_metric_to_jax(sm)

        for coords in TEST_POINTS:
            g_jax = m(coords)
            g_bridge = bridge_fn(coords)
            assert jnp.allclose(g_jax, g_bridge, atol=1e-14), (
                f"Minkowski mismatch at {coords}: max diff = "
                f"{jnp.max(jnp.abs(g_jax - g_bridge))}"
            )

    def test_minkowski_sympy_bridge_float64(self):
        """Bridge function returns float64 arrays."""
        m = MinkowskiMetric()
        bridge_fn = sympy_metric_to_jax(m.symbolic())
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g = bridge_fn(coords)
        assert g.dtype == jnp.float64

    def test_minkowski_sympy_bridge_jit(self):
        """Bridge function output is JIT-compatible."""
        m = MinkowskiMetric()
        bridge_fn = sympy_metric_to_jax(m.symbolic())
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g = jax.jit(bridge_fn)(coords)
        assert g.shape == (4, 4)
        assert g.dtype == jnp.float64


# =========================================================================
# Schwarzschild bridge tests
# =========================================================================


class TestSchwarzschildSympyBridge:
    """SymPy bridge cross-validation for Schwarzschild.

    The symbolic form uses symbol M; we substitute M=1.0 before comparison.
    """

    def _get_bridge_fn(self, M_val: float = 1.0):
        """Build a bridge function with concrete parameter value."""
        m = SchwarzschildMetric(M=M_val)
        sm = m.symbolic()
        # Substitute concrete M value into symbolic expression
        M_sym = sp.Symbol("M", positive=True)
        g_concrete = sm.g.subs(M_sym, M_val)
        from warpax.geometry.metric import SymbolicMetric
        sm_concrete = SymbolicMetric(sm.coords, g_concrete)
        return sympy_metric_to_jax(sm_concrete)

    def test_schwarzschild_sympy_matches_jax(self):
        """Evaluate JAX metric and sympy bridge at several points."""
        m = SchwarzschildMetric(M=1.0)
        bridge_fn = self._get_bridge_fn(1.0)

        for coords in TEST_POINTS:
            g_jax = m(coords)
            g_bridge = bridge_fn(coords)
            assert jnp.allclose(g_jax, g_bridge, atol=1e-14), (
                f"Schwarzschild mismatch at {coords}: max diff = "
                f"{jnp.max(jnp.abs(g_jax - g_bridge))}"
            )

    def test_schwarzschild_sympy_bridge_float64(self):
        """Bridge function returns float64 arrays."""
        bridge_fn = self._get_bridge_fn(1.0)
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g = bridge_fn(coords)
        assert g.dtype == jnp.float64

    def test_schwarzschild_sympy_bridge_jit(self):
        """Bridge function output is JIT-compatible."""
        bridge_fn = self._get_bridge_fn(1.0)
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g = jax.jit(bridge_fn)(coords)
        assert g.shape == (4, 4)
        assert g.dtype == jnp.float64


# =========================================================================
# Alcubierre bridge tests
# =========================================================================


class TestAlcubierreSympyBridge:
    """SymPy bridge cross-validation for Alcubierre.

    The symbolic form uses symbols v_s, R, sigma; we substitute concrete
    values before comparison.
    """

    def _get_bridge_fn(
        self, v_s_val: float = 0.5, R_val: float = 1.0, sigma_val: float = 8.0
    ):
        """Build a bridge function with concrete parameter values."""
        m = AlcubierreMetric(v_s=v_s_val, R=R_val, sigma=sigma_val, x_s=0.0)
        sm = m.symbolic()
        # Substitute concrete parameter values
        v_s_sym = sp.Symbol("v_s", positive=True)
        R_sym = sp.Symbol("R", positive=True)
        sigma_sym = sp.Symbol("sigma", positive=True)
        g_concrete = sm.g.subs({
            v_s_sym: v_s_val,
            R_sym: R_val,
            sigma_sym: sigma_val,
        })
        from warpax.geometry.metric import SymbolicMetric
        sm_concrete = SymbolicMetric(sm.coords, g_concrete)
        return sympy_metric_to_jax(sm_concrete)

    def test_alcubierre_sympy_matches_jax(self):
        """Evaluate JAX metric and sympy bridge at several points.

        Note: Alcubierre symbolic() uses x_s=0 (bubble at origin), matching
        the default AlcubierreMetric(x_s=0.0).
        """
        m = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0, x_s=0.0)
        bridge_fn = self._get_bridge_fn(0.5, 1.0, 8.0)

        for coords in TEST_POINTS:
            g_jax = m(coords)
            g_bridge = bridge_fn(coords)
            assert jnp.allclose(g_jax, g_bridge, atol=1e-14), (
                f"Alcubierre mismatch at {coords}: max diff = "
                f"{jnp.max(jnp.abs(g_jax - g_bridge))}"
            )

    def test_alcubierre_sympy_bridge_float64(self):
        """Bridge function returns float64 arrays."""
        bridge_fn = self._get_bridge_fn(0.5, 1.0, 8.0)
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g = bridge_fn(coords)
        assert g.dtype == jnp.float64

    def test_alcubierre_sympy_bridge_jit(self):
        """Bridge function output is JIT-compatible."""
        bridge_fn = self._get_bridge_fn(0.5, 1.0, 8.0)
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g = jax.jit(bridge_fn)(coords)
        assert g.shape == (4, 4)
        assert g.dtype == jnp.float64
