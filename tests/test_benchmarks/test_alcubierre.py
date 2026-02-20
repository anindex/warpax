"""Tests for Alcubierre warp drive benchmark."""

import jax
import jax.numpy as jnp

from warpax.benchmarks.alcubierre import (
    AlcubierreMetric,
    eulerian_energy_density,
)
from warpax.geometry.metric import SymbolicMetric, adm_to_full_metric


class TestAlcubierre:
    """Tests for AlcubierreMetric."""

    def test_alcubierre_at_origin(self, origin_coords):
        """Evaluate at origin, verify metric structure.

        At the origin with x_s=0, r_s=0, f(0) approx 1 for large sigma.
        g_00 = -(1 - v_s^2 f^2), g_01 = -v_s f, spatial = delta_ij.
        """
        m = AlcubierreMetric()  # v_s=0.5, R=1.0, sigma=8.0, x_s=0.0
        g = m(origin_coords)
        assert g.shape == (4, 4)
        # Spatial block should be flat identity
        assert jnp.allclose(g[1:, 1:], jnp.eye(3), atol=1e-14)

    def test_alcubierre_far_field(self):
        """Evaluate far from bubble (r >> R), verify approaches Minkowski."""
        m = AlcubierreMetric()  # R=1.0
        far_coords = jnp.array([0.0, 100.0, 0.0, 0.0])
        g = m(far_coords)
        minkowski = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, minkowski, atol=1e-10)

    def test_alcubierre_at_center(self):
        """At bubble center (r=0), f(0) approx 1 for large sigma.

        With sigma=8, R=1: f(0) = [tanh(8) - tanh(-8)] / [2*tanh(8)] = 1.0
        So g_00 = -(1 - 0.25) = -0.75, g_01 = -0.5.
        """
        m = AlcubierreMetric()  # v_s=0.5, sigma=8.0, x_s=0.0
        center = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = m(center)
        # f(0) should be very close to 1.0 for sigma=8
        # g_00 = -(1 - v_s^2 * f^2) = -(1 - 0.25) = -0.75
        assert jnp.isclose(g[0, 0], -0.75, atol=1e-10)
        # g_01 = beta_down_x = -v_s * f = -0.5
        assert jnp.isclose(g[0, 1], -0.5, atol=1e-10)

    def test_alcubierre_jit(self, sample_coords):
        """jax.jit compilation works."""
        m = AlcubierreMetric()
        g_eager = m(sample_coords)
        g_jit = jax.jit(m)(sample_coords)
        assert jnp.allclose(g_eager, g_jit, atol=1e-15)

    def test_alcubierre_float64(self, sample_coords):
        """Output dtype is float64."""
        m = AlcubierreMetric()
        g = m(sample_coords)
        assert g.dtype == jnp.float64

    def test_alcubierre_parameter_change(self):
        """Change v_s, verify output changes (dynamic field).

        Use a point inside the bubble where f(r_s) is significant.
        """
        near_center = jnp.array([0.0, 0.3, 0.0, 0.0])  # inside bubble (R=1)
        m1 = AlcubierreMetric(v_s=0.5)
        m2 = AlcubierreMetric(v_s=1.0)
        g1 = m1(near_center)
        g2 = m2(near_center)
        assert not jnp.allclose(g1, g2, atol=1e-10)

    def test_alcubierre_adm_reconstruction(self, sample_coords):
        """Verify __call__ matches manual ADM reconstruction."""
        m = AlcubierreMetric()
        g_call = m(sample_coords)
        g_adm = adm_to_full_metric(
            m.lapse(sample_coords),
            m.shift(sample_coords),
            m.spatial_metric(sample_coords),
        )
        assert jnp.allclose(g_call, g_adm, atol=1e-15)

    def test_alcubierre_symbolic(self):
        """symbolic() returns valid SymbolicMetric."""
        m = AlcubierreMetric()
        sm = m.symbolic()
        assert isinstance(sm, SymbolicMetric)
        assert sm.g.shape == (4, 4)
        assert len(sm.coords) == 4

    def test_alcubierre_eulerian_energy_negative(self):
        """Analytical energy density <= 0 everywhere (ported from legacy)."""
        x = jnp.linspace(-3, 3, 30)
        y = jnp.linspace(-3, 3, 30)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        Z = jnp.zeros_like(X)

        rho = eulerian_energy_density(X, Y, Z, v_s=0.5, R=1.0, sigma=8.0)
        assert jnp.all(rho <= 1e-15), f"max rho = {jnp.max(rho)}"
        assert rho.dtype == jnp.float64
