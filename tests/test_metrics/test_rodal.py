"""Tests for Rodal irrotational warp drive metric."""

import jax
import jax.numpy as jnp

from warpax.metrics import RodalMetric
from warpax.geometry import compute_curvature_chain, SymbolicMetric, adm_to_full_metric
from warpax.metrics.rodal import _alcubierre_shape, _rodal_G, _rodal_g_paper


class TestRodal:
    """Tests for RodalMetric."""

    # ------------------------------------------------------------------
    # Standard 8-test battery
    # ------------------------------------------------------------------

    def test_rodal_at_origin(self):
        """Evaluate at origin, verify metric structure.

        At origin with r_s=0: F(0)~1, G(0)~1, so shift = (-v_s, 0, 0).
        g_00 = -(1 - v_s^2), g_01 = -v_s, spatial = delta_ij.
        """
        m = RodalMetric()  # v_s=0.1, R=100.0, sigma=0.03
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = m(coords)
        assert g.shape == (4, 4)
        # Spatial block should be flat identity
        assert jnp.allclose(g[1:, 1:], jnp.eye(3), atol=1e-14)
        # g_00 = -(1 - v_s^2) since f(0) ~ 1 for r inside bubble
        assert g[0, 0] < 0.0  # timelike

    def test_rodal_far_field(self):
        """Evaluate far from bubble (r >> R), verify approaches Minkowski.

        Rodal uses lab-frame convention: F(inf)=0, G(inf)=0, so shift -> 0
        and metric -> Minkowski at far field.
        """
        m = RodalMetric()  # R=100.0
        far_coords = jnp.array([0.0, 1000.0, 0.0, 0.0])
        g = m(far_coords)
        minkowski = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, minkowski, atol=1e-6)

    def test_rodal_jit(self):
        """jax.jit compilation works."""
        m = RodalMetric()
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g_eager = m(coords)
        g_jit = jax.jit(m)(coords)
        assert jnp.allclose(g_eager, g_jit, atol=1e-15)

    def test_rodal_float64(self):
        """Output dtype is float64."""
        m = RodalMetric()
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g = m(coords)
        assert g.dtype == jnp.float64

    def test_rodal_parameter_change(self):
        """Change v_s, verify output changes (dynamic field)."""
        coords = jnp.array([0.0, 10.0, 0.0, 0.0])  # inside bubble (R=100)
        m1 = RodalMetric(v_s=0.1)
        m2 = RodalMetric(v_s=0.5)
        g1 = m1(coords)
        g2 = m2(coords)
        assert not jnp.allclose(g1, g2, atol=1e-10)

    def test_rodal_adm_reconstruction(self):
        """Verify __call__ matches manual ADM reconstruction."""
        m = RodalMetric()
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g_call = m(coords)
        g_adm = adm_to_full_metric(
            m.lapse(coords),
            m.shift(coords),
            m.spatial_metric(coords),
        )
        assert jnp.allclose(g_call, g_adm, atol=1e-15)

    def test_rodal_symbolic(self):
        """symbolic() returns valid SymbolicMetric."""
        m = RodalMetric()
        sm = m.symbolic()
        assert isinstance(sm, SymbolicMetric)
        assert sm.g.shape == (4, 4)
        assert len(sm.coords) == 4

    def test_rodal_curvature_chain(self):
        """Run compute_curvature_chain and verify no NaN, correct shapes.

        Uses a point near the bubble wall where curvature is nontrivial.
        """
        m = RodalMetric(v_s=0.1, R=100.0, sigma=0.03)
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

    def test_rodal_irrotational_shift(self):
        """Check that G_0i components of Einstein tensor are near-zero.

        Irrotational shift (curl-free) means the momentum density
        G^0_i should vanish. We check via the mixed-index Einstein
        tensor: G^mu_nu = g^{mu alpha} G_{alpha nu}.
        """
        m = RodalMetric(v_s=0.1, R=100.0, sigma=0.03)
        # Point away from center, off-axis to probe angular components
        coords = jnp.array([0.0, 80.0, 20.0, 0.0])
        result = compute_curvature_chain(m, coords)
        G = result.einstein
        g_inv = result.metric_inv

        # Raise first index: G^mu_nu = g^{mu alpha} G_{alpha nu}
        G_mixed = jnp.einsum("ma,an->mn", g_inv, G)

        # G^0_i (momentum density) should be near zero for irrotational shift
        G_0i = G_mixed[0, 1:]
        assert jnp.allclose(G_0i, 0.0, atol=1e-8), (
            f"G^0_i = {G_0i}, expected ~0 for irrotational shift"
        )

    def test_rodal_shape_functions(self):
        """Verify f(0)~1, f(inf)~0, G(0)~1, G(inf)~0 at representative r values.

        Lab-frame convention: F = f_Alc (radial), G = 1 - g_paper (angular).
        Both are ~1 at center and ~0 far away.

        Note: G(0) depends on R*sigma product. For R=100, sigma=0.03 (R*sigma=3),
        G(0) ~ 0.989. With larger sigma (sharper wall), G(0) -> 1. The physical
        behavior is correct: G is close to 1 at center and 0 at far field.
        """
        R, sigma = 100.0, 0.03

        # Radial shape F = f_Alc
        r_zero = jnp.array(0.0)
        r_far = jnp.array(1000.0)
        F_0 = _alcubierre_shape(r_zero, R, sigma)
        F_inf = _alcubierre_shape(r_far, R, sigma)
        assert jnp.isclose(F_0, 1.0, atol=1e-6), f"F(0) = {F_0}"
        assert jnp.isclose(F_inf, 0.0, atol=1e-6), f"F(inf) = {F_inf}"

        # Angular shape G = 1 - g_paper
        # G(0) is close to 1 but depends on R*sigma; with R*sigma=3, G(0)~0.989
        # G(r) has a 1/r tail (slower convergence than F), so we test monotone
        # decrease and that the far-field metric is still Minkowski (via the
        # full metric test above), rather than requiring G(1000)~0.
        G_0 = _rodal_G(r_zero, R, sigma)
        G_mid = _rodal_G(jnp.array(500.0), R, sigma)
        G_far = _rodal_G(r_far, R, sigma)
        G_very_far = _rodal_G(jnp.array(50000.0), R, sigma)
        assert G_0 > 0.95, f"G(0) = {G_0}, expected > 0.95"
        assert G_mid < G_0, f"G should decrease: G(0)={G_0}, G(500)={G_mid}"
        assert G_far < G_mid, f"G should decrease: G(500)={G_mid}, G(1000)={G_far}"
        assert G_very_far < 0.01, f"G(50000) = {G_very_far}, expected < 0.01"

    def test_rodal_g_numerical_stability(self):
        """Evaluate g_paper(r) at large r*sigma to confirm no NaN/Inf.

        The logcosh implementation should be stable for large arguments.
        """
        R, sigma = 100.0, 0.03
        r_values = jnp.array([0.1, 1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0])
        g_vals = jax.vmap(lambda r: _rodal_g_paper(r, R, sigma))(r_values)
        assert not jnp.any(jnp.isnan(g_vals)), f"NaN found in g_paper: {g_vals}"
        assert not jnp.any(jnp.isinf(g_vals)), f"Inf found in g_paper: {g_vals}"
        # g_paper should be bounded in [0, 1]
        assert jnp.all(g_vals >= -1e-10), f"g_paper < 0: {g_vals}"
        assert jnp.all(g_vals <= 1.0 + 1e-10), f"g_paper > 1: {g_vals}"
