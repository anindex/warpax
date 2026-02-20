"""Tests for Natario zero-expansion warp drive metric."""

import jax
import jax.numpy as jnp

from warpax.metrics import NatarioMetric
from warpax.geometry import compute_curvature_chain, SymbolicMetric, adm_to_full_metric
from warpax.metrics.natario import natario_eulerian_energy_density


class TestNatario:
    """Tests for NatarioMetric."""

    # ------------------------------------------------------------------
    # Standard 8-test battery
    # ------------------------------------------------------------------

    def test_natario_at_origin(self):
        """Evaluate at origin, verify metric structure.

        At origin (co-moving bubble center): n(0)=0, n'(0)=0, so shift = 0.
        Metric should be Minkowski at the bubble center.
        """
        m = NatarioMetric()  # v_s=0.1, R=100.0, sigma=0.03
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = m(coords)
        assert g.shape == (4, 4)
        minkowski = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, minkowski, atol=1e-10), (
            f"At bubble center, metric should be Minkowski. Got:\n{g}"
        )

    def test_natario_far_field(self):
        """Evaluate far from bubble, verify the actual far-field behavior.

        The Natario metric uses co-moving bubble frame where far field
        has a uniform flow: n(inf) = 1/2, so shift = -v_s * x_hat.
        The metric at far field is NOT Minkowski it has nonzero shift.

        g_00 = -(1 - v_s^2), g_0x = -v_s, g_ij = delta_ij.
        """
        m = NatarioMetric(v_s=0.1)  # R=100.0
        far_coords = jnp.array([0.0, 1000.0, 0.0, 0.0])
        g = m(far_coords)

        # At far field along x-axis: shift = -v_s*(2*n_val) with n(inf)=1/2
        # so beta_x ~ -v_s*(2*0.5) = -v_s (since dn~0 far away)
        # g_00 = -(1 - v_s^2), g_01 = beta_x = -v_s
        v_s = m.v_s
        expected_g00 = -(1.0 - v_s**2)
        expected_g01 = -v_s
        assert jnp.isclose(g[0, 0], expected_g00, atol=1e-4), (
            f"g_00 = {g[0, 0]}, expected {expected_g00}"
        )
        assert jnp.isclose(g[0, 1], expected_g01, atol=1e-4), (
            f"g_01 = {g[0, 1]}, expected {expected_g01}"
        )
        # Spatial block still flat
        assert jnp.allclose(g[1:, 1:], jnp.eye(3), atol=1e-14)

    def test_natario_jit(self):
        """jax.jit compilation works."""
        m = NatarioMetric()
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g_eager = m(coords)
        g_jit = jax.jit(m)(coords)
        assert jnp.allclose(g_eager, g_jit, atol=1e-15)

    def test_natario_float64(self):
        """Output dtype is float64."""
        m = NatarioMetric()
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g = m(coords)
        assert g.dtype == jnp.float64

    def test_natario_parameter_change(self):
        """Change v_s, verify output changes (dynamic field)."""
        coords = jnp.array([0.0, 50.0, 0.0, 0.0])  # inside bubble (R=100)
        m1 = NatarioMetric(v_s=0.1)
        m2 = NatarioMetric(v_s=0.5)
        g1 = m1(coords)
        g2 = m2(coords)
        assert not jnp.allclose(g1, g2, atol=1e-10)

    def test_natario_adm_reconstruction(self):
        """Verify __call__ matches manual ADM reconstruction."""
        m = NatarioMetric()
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g_call = m(coords)
        g_adm = adm_to_full_metric(
            m.lapse(coords),
            m.shift(coords),
            m.spatial_metric(coords),
        )
        assert jnp.allclose(g_call, g_adm, atol=1e-15)

    def test_natario_symbolic(self):
        """symbolic() returns valid SymbolicMetric."""
        m = NatarioMetric()
        sm = m.symbolic()
        assert isinstance(sm, SymbolicMetric)
        assert sm.g.shape == (4, 4)
        assert len(sm.coords) == 4

    def test_natario_curvature_chain(self):
        """Run compute_curvature_chain and verify no NaN, correct shapes.

        Uses a point near the bubble wall where curvature is nontrivial.
        """
        m = NatarioMetric(v_s=0.1, R=100.0, sigma=0.03)
        # Point near bubble wall
        coords = jnp.array([0.0, 100.0, 1.0, 0.0])
        result = compute_curvature_chain(m, coords)

        assert result.metric.shape == (4, 4)
        assert result.christoffel.shape == (4, 4, 4)
        assert result.riemann.shape == (4, 4, 4, 4)
        assert result.ricci.shape == (4, 4)
        assert result.einstein.shape == (4, 4)
        assert result.stress_energy.shape == (4, 4)
        # No NaN
        assert not jnp.any(jnp.isnan(result.metric))
        assert not jnp.any(jnp.isnan(result.christoffel))
        assert not jnp.any(jnp.isnan(result.riemann))
        assert not jnp.any(jnp.isnan(result.einstein))
        assert not jnp.any(jnp.isnan(result.stress_energy))

    # ------------------------------------------------------------------
    # Physics-specific tests
    # ------------------------------------------------------------------

    def test_natario_zero_expansion(self):
        """Verify div(beta) = 0 (zero expansion) at multiple points.

        The Natario metric is constructed so that the trace of the extrinsic
        curvature K = div(beta) = 0 everywhere. We verify this by computing
        the divergence of the shift vector via JAX autodiff.
        """
        m = NatarioMetric(v_s=0.1, R=100.0, sigma=0.03)

        def shift_fn(coords):
            return m.shift(coords)

        # Test at several points including bubble wall region
        test_points = [
            jnp.array([0.0, 50.0, 10.0, 0.0]),
            jnp.array([0.0, 100.0, 5.0, 3.0]),
            jnp.array([0.0, 80.0, 20.0, 10.0]),
        ]

        for coords in test_points:
            # Compute Jacobian of shift w.r.t. spatial coordinates
            # shift returns (3,), coords are (4,), so Jacobian is (3, 4)
            J = jax.jacfwd(shift_fn)(coords)
            # div(beta) = d(beta^x)/dx + d(beta^y)/dy + d(beta^z)/dz
            # Spatial indices are 1,2,3 in coordinates
            div_beta = J[0, 1] + J[1, 2] + J[2, 3]
            assert jnp.isclose(div_beta, 0.0, atol=1e-8), (
                f"div(beta) = {div_beta} at {coords}, expected ~0"
            )

    def test_natario_shape_function(self):
        """Verify n(0)=0, n(inf)=1/2.

        The Natario shape function n(r) = (1/2)*(1 - f_Alc(r)).
        """
        from warpax.metrics.natario import _natario_n

        R, sigma = 100.0, 0.03
        n_0 = _natario_n(jnp.array(0.0), R, sigma)
        n_inf = _natario_n(jnp.array(1000.0), R, sigma)

        assert jnp.isclose(n_0, 0.0, atol=1e-6), f"n(0) = {n_0}, expected ~0"
        assert jnp.isclose(n_inf, 0.5, atol=1e-6), f"n(inf) = {n_inf}, expected ~0.5"

    def test_natario_energy_strictly_negative(self):
        """Verify analytical Eulerian energy density is strictly non-positive.

        rho = -(v_s^2 / kappa) * [...] is negative wherever dn/dr != 0,
        confirming WEC/NEC violation everywhere on the bubble wall.
        """
        x = jnp.linspace(-200, 200, 40)
        y = jnp.linspace(-200, 200, 40)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        Z = jnp.zeros_like(X)

        rho = natario_eulerian_energy_density(X, Y, Z, v_s=0.1, R=100.0, sigma=0.03)
        assert jnp.all(rho <= 1e-15), f"max rho = {jnp.max(rho)}"
        assert rho.dtype == jnp.float64
        # Check that some values are significantly negative (not all zero)
        assert jnp.min(rho) < -1e-10, (
            f"min rho = {jnp.min(rho)}, should be significantly negative"
        )
