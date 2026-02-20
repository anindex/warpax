"""Tests for Lentz soliton warp drive metric."""

import jax
import jax.numpy as jnp

from warpax.metrics import LentzMetric
from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.geometry import compute_curvature_chain, SymbolicMetric, adm_to_full_metric


class TestLentz:
    """Tests for LentzMetric."""

    # ------------------------------------------------------------------
    # Standard 8-test battery
    # ------------------------------------------------------------------

    def test_lentz_at_origin(self):
        """Evaluate near origin, verify metric structure.

        Near origin (inside diamond): f~1, shift = (-v_s, 0, 0).
        g_00 = -(1 - v_s^2), g_01 = -v_s, spatial = delta_ij.

        Note: We use a point slightly off origin to avoid the L1 norm
        non-differentiable point at x_rel=0.
        """
        m = LentzMetric()  # v_s=0.1, R=100.0, sigma=8.0
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])  # inside diamond
        g = m(coords)
        assert g.shape == (4, 4)
        # Spatial block should be flat identity
        assert jnp.allclose(g[1:, 1:], jnp.eye(3), atol=1e-14)
        # g_00 = -(1 - v_s^2 * f^2) ~ -(1 - 0.01) = -0.99 for f~1
        assert g[0, 0] < 0.0  # timelike

    def test_lentz_far_field(self):
        """Evaluate far from bubble (d >> R), verify approaches Minkowski."""
        m = LentzMetric()  # R=100.0
        far_coords = jnp.array([0.0, 1000.0, 0.0, 0.0])
        g = m(far_coords)
        minkowski = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, minkowski, atol=1e-6)

    def test_lentz_jit(self):
        """jax.jit compilation works."""
        m = LentzMetric()
        coords = jnp.array([0.0, 5.0, 2.0, 3.0])
        g_eager = m(coords)
        g_jit = jax.jit(m)(coords)
        assert jnp.allclose(g_eager, g_jit, atol=1e-15)

    def test_lentz_float64(self):
        """Output dtype is float64."""
        m = LentzMetric()
        coords = jnp.array([0.0, 5.0, 2.0, 3.0])
        g = m(coords)
        assert g.dtype == jnp.float64

    def test_lentz_parameter_change(self):
        """Change v_s, verify output changes (dynamic field)."""
        coords = jnp.array([0.0, 10.0, 5.0, 0.0])  # inside diamond
        m1 = LentzMetric(v_s=0.1)
        m2 = LentzMetric(v_s=0.5)
        g1 = m1(coords)
        g2 = m2(coords)
        assert not jnp.allclose(g1, g2, atol=1e-10)

    def test_lentz_adm_reconstruction(self):
        """Verify __call__ matches manual ADM reconstruction."""
        m = LentzMetric()
        coords = jnp.array([0.0, 5.0, 2.0, 3.0])
        g_call = m(coords)
        g_adm = adm_to_full_metric(
            m.lapse(coords),
            m.shift(coords),
            m.spatial_metric(coords),
        )
        assert jnp.allclose(g_call, g_adm, atol=1e-15)

    def test_lentz_symbolic(self):
        """symbolic() returns valid SymbolicMetric."""
        m = LentzMetric()
        sm = m.symbolic()
        assert isinstance(sm, SymbolicMetric)
        assert sm.g.shape == (4, 4)
        assert len(sm.coords) == 4

    def test_lentz_curvature_chain(self):
        """Run compute_curvature_chain and verify no NaN, correct shapes.

        Uses a point on the bubble wall where curvature is nontrivial.
        Avoids x_rel=0 (L1 non-differentiable kink).
        """
        m = LentzMetric(v_s=0.1, R=100.0, sigma=8.0)
        # Point on bubble wall, off-axis to avoid L1 kink
        coords = jnp.array([0.0, 50.0, 50.0, 0.0])
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

    def test_lentz_subluminal_g00_negative(self):
        """For v_s=0.5, verify g_00 < 0 at bubble center (timelike preserved)."""
        m = LentzMetric(v_s=0.5, R=100.0, sigma=8.0)
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])  # near center
        g = m(coords)
        assert g[0, 0] < 0.0, (
            f"g_00 = {g[0, 0]}, expected < 0 for subluminal v_s=0.5"
        )

    def test_lentz_superluminal_g00_positive(self):
        """For v_s=2.0, verify g_00 > 0 at bubble center (signature change)."""
        m = LentzMetric(v_s=2.0, R=100.0, sigma=8.0)
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])  # near center
        g = m(coords)
        assert g[0, 0] > 0.0, (
            f"g_00 = {g[0, 0]}, expected > 0 for superluminal v_s=2.0"
        )

    def test_lentz_diamond_geometry(self):
        """Verify diamond (L1) geometry differs from spherical (L2) Alcubierre.

        At a 45-degree diagonal point, L1 distance = |x| + |y| = 100
        while L2 distance = sqrt(x^2 + y^2) ~ 70.7. The Lentz bubble is
        at the wall while Alcubierre is still well inside. This produces
        different shift magnitudes.
        """
        R, sigma, v_s = 100.0, 8.0, 0.1
        lentz = LentzMetric(v_s=v_s, R=R, sigma=sigma)
        alc = AlcubierreMetric(v_s=v_s, R=R, sigma=sigma, x_s=0.0)

        # 45-degree diagonal: L1 = 100 (on wall), L2 = 70.7 (inside)
        diag = jnp.array([0.0, 50.0, 50.0, 0.0])
        beta_lentz = lentz.shift(diag)
        beta_alc = alc.shift(diag)

        # Shifts should differ significantly
        assert not jnp.allclose(beta_lentz, beta_alc, atol=1e-3), (
            f"Lentz shift = {beta_lentz}, Alcubierre shift = {beta_alc}. "
            "Diamond and spherical should produce different shifts at diagonal."
        )

    def test_lentz_warpfactory_comparison(self):
        """Regression baseline for Lentz Eulerian energy density.

        Computes T_ab n^a n^b at representative points and compares against
        stored regression baselines. This serves as a regression test to
        detect any changes in the Lentz metric or curvature pipeline.

        Regression baseline NOT a published WarpFactory comparison.
        Parameter set: v_s=0.1, R=100, sigma=8.
        """
        m = LentzMetric(v_s=0.1, R=100.0, sigma=8.0)

        # Reference values computed from our pipeline (regression baseline)
        # Points chosen to avoid L1 non-differentiable kink at x_rel=0
        test_cases = {
            "near_center": {
                "coords": jnp.array([0.0, 5.0, 5.0, 0.0]),
                "rho_euler": 0.0,  # Deep inside: flat, zero curvature
            },
            "wall_diagonal": {
                "coords": jnp.array([0.0, 50.0, 50.0, 0.0]),
                "rho_euler": -1.5915494309189531e-03,
            },
            "outside": {
                "coords": jnp.array([0.0, 200.0, 200.0, 0.0]),
                "rho_euler": 0.0,  # Far outside: flat, zero curvature
            },
        }

        for name, case in test_cases.items():
            coords = case["coords"]
            result = compute_curvature_chain(m, coords)

            # Eulerian energy density: T_ab n^a n^b
            # For unit lapse and flat spatial: n^a = (1, -beta^i)
            beta = m.shift(coords)
            n_up = jnp.array([1.0, -beta[0], -beta[1], -beta[2]])
            T = result.stress_energy
            rho_euler = jnp.einsum("a,ab,b->", n_up, T, n_up)

            assert jnp.allclose(rho_euler, case["rho_euler"], atol=1e-8), (
                f"Lentz {name}: rho_euler = {rho_euler:.16e}, "
                f"expected {case['rho_euler']:.16e}"
            )
