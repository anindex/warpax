"""Tests for Van Den Broeck volume-modified Alcubierre warp drive metric."""

import jax
import jax.numpy as jnp

from warpax.metrics import VanDenBroeckMetric
from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.geometry import compute_curvature_chain, SymbolicMetric, adm_to_full_metric


class TestVanDenBroeck:
    """Tests for VanDenBroeckMetric."""

    # ------------------------------------------------------------------
    # Standard 8-test battery
    # ------------------------------------------------------------------

    def test_vdb_at_origin(self):
        """Evaluate at origin, verify metric structure.

        At origin with r_s=0: f(0)~1, B(0)=1+alpha_vdb.
        Spatial block = B^2 * I. Shift = (-v_s * f, 0, 0).
        """
        m = VanDenBroeckMetric()  # v_s=0.1, R=350, sigma=8, R_tilde=200, alpha_vdb=0.5, sigma_B=8
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = m(coords)
        assert g.shape == (4, 4)
        # Spatial block should be B^2 * I where B(0) = 1 + alpha_vdb = 1.5
        B_sq = (1.0 + m.alpha_vdb) ** 2
        assert jnp.allclose(g[1:, 1:], B_sq * jnp.eye(3), atol=1e-6)
        # Timelike
        assert g[0, 0] < 0.0

    def test_vdb_far_field(self):
        """Evaluate far from bubble (r >> R), verify approaches Minkowski.

        At far field: f(inf)~0, B(inf)~1, so metric -> Minkowski.
        """
        m = VanDenBroeckMetric()  # R=350
        far_coords = jnp.array([0.0, 1000.0, 0.0, 0.0])
        g = m(far_coords)
        minkowski = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, minkowski, atol=1e-6)

    def test_vdb_jit(self):
        """jax.jit compilation works."""
        m = VanDenBroeckMetric()
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g_eager = m(coords)
        g_jit = jax.jit(m)(coords)
        assert jnp.allclose(g_eager, g_jit, atol=1e-15)

    def test_vdb_float64(self):
        """Output dtype is float64."""
        m = VanDenBroeckMetric()
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g = m(coords)
        assert g.dtype == jnp.float64

    def test_vdb_parameter_change(self):
        """Change v_s, verify output changes (dynamic field)."""
        coords = jnp.array([0.0, 10.0, 0.0, 0.0])  # inside bubble (R=350)
        m1 = VanDenBroeckMetric(v_s=0.1)
        m2 = VanDenBroeckMetric(v_s=0.5)
        g1 = m1(coords)
        g2 = m2(coords)
        assert not jnp.allclose(g1, g2, atol=1e-10)

    def test_vdb_adm_reconstruction(self):
        """Verify __call__ matches manual ADM reconstruction."""
        m = VanDenBroeckMetric()
        coords = jnp.array([0.0, 1.0, 2.0, 3.0])
        g_call = m(coords)
        g_adm = adm_to_full_metric(
            m.lapse(coords),
            m.shift(coords),
            m.spatial_metric(coords),
        )
        assert jnp.allclose(g_call, g_adm, atol=1e-15)

    def test_vdb_symbolic(self):
        """symbolic() returns valid SymbolicMetric."""
        m = VanDenBroeckMetric()
        sm = m.symbolic()
        assert isinstance(sm, SymbolicMetric)
        assert sm.g.shape == (4, 4)
        assert len(sm.coords) == 4

    def test_vdb_curvature_chain(self):
        """Run compute_curvature_chain and verify no NaN, correct shapes.

        Uses a point near the conformal bubble wall where curvature is nontrivial.
        """
        m = VanDenBroeckMetric(v_s=0.1, R=350.0, sigma=8.0,
                                R_tilde=200.0, alpha_vdb=0.5, sigma_B=8.0)
        # Point near conformal bubble wall
        coords = jnp.array([0.0, 200.0, 1.0, 0.0])
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

    def test_vdb_conformal_factor(self):
        """At bubble center, B(0) = 1 + alpha_vdb, so gamma_ij = (1+alpha_vdb)^2 * I.

        Verify spatial metric diagonal has the expected conformal factor.
        """
        m = VanDenBroeckMetric()  # alpha_vdb=0.5
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        gamma = m.spatial_metric(coords)
        B_center = 1.0 + m.alpha_vdb  # = 1.5
        expected_gamma = B_center**2 * jnp.eye(3)
        assert jnp.allclose(gamma, expected_gamma, atol=1e-6), (
            f"gamma = {jnp.diag(gamma)}, expected diag = {B_center**2}"
        )

    def test_vdb_shift_matches_alcubierre(self):
        """Verify shift is identical to Alcubierre (same f(r_s) shape).

        Both use the same Alcubierre top-hat shape function for the shift,
        so at matching parameter values the shift vectors should be identical.
        """
        # Use matching parameters (small R so we can test in reasonable coords)
        v_s, R, sigma = 0.1, 350.0, 8.0
        vdb = VanDenBroeckMetric(v_s=v_s, R=R, sigma=sigma)
        alc = AlcubierreMetric(v_s=v_s, R=R, sigma=sigma, x_s=0.0)

        coords = jnp.array([0.0, 100.0, 50.0, 0.0])
        beta_vdb = vdb.shift(coords)
        beta_alc = alc.shift(coords)
        assert jnp.allclose(beta_vdb, beta_alc, atol=1e-14), (
            f"VdB shift = {beta_vdb}, Alcubierre shift = {beta_alc}"
        )

    def test_vdb_energy_density_reduced(self):
        """Verify stress-energy magnitude differs from Alcubierre due to conformal factor.

        The conformal factor B^2 on the spatial metric modifies the curvature
        and hence the stress-energy tensor. At the same point near the bubble
        wall, VdB and Alcubierre should produce different stress-energy.
        """
        v_s, R, sigma = 0.1, 350.0, 8.0
        vdb = VanDenBroeckMetric(v_s=v_s, R=R, sigma=sigma,
                                  R_tilde=200.0, alpha_vdb=0.5, sigma_B=8.0)
        alc = AlcubierreMetric(v_s=v_s, R=R, sigma=sigma, x_s=0.0)

        # Point near the conformal bubble wall where B != 1
        coords = jnp.array([0.0, 200.0, 1.0, 0.0])
        result_vdb = compute_curvature_chain(vdb, coords)
        result_alc = compute_curvature_chain(alc, coords)

        T_vdb = result_vdb.stress_energy
        T_alc = result_alc.stress_energy

        # They should differ due to the conformal factor
        assert not jnp.allclose(T_vdb, T_alc, atol=1e-10), (
            "Stress-energy should differ between VdB and Alcubierre "
            "near conformal bubble wall"
        )
