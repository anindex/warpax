"""Warp-metric construction and contracts: Lentz, Natario, Rodal, VdB, WarpShell."""

from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.geometry import compute_curvature_chain, SymbolicMetric, adm_to_full_metric
from warpax.metrics import LentzMetric
from warpax.metrics import NatarioMetric
from warpax.metrics import RodalMetric
from warpax.metrics import VanDenBroeckMetric
from warpax.metrics import WarpShellMetric, WarpShellPhysical, WarpShellStressTest
from warpax.metrics import tshell_default
from warpax.metrics._common import alcubierre_shape
from warpax.metrics.natario import natario_eulerian_energy_density
from warpax.metrics.rodal import _rodal_G, _rodal_g_paper
import jax
import jax.numpy as jnp
import pytest



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
        """symbolic returns valid SymbolicMetric."""
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

    def test_lentz_regression_baseline(self):
        """Regression baseline for Lentz Eulerian energy density.

        Computes T_ab n^a n^b at representative points and compares against
        stored baselines to detect changes in the Lentz metric or
        curvature pipeline.

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
        """symbolic returns valid SymbolicMetric."""
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
        """symbolic returns valid SymbolicMetric."""
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

        Note: G(0) = 1 exactly for all R*sigma (via the analytic limit
        Delta'(0) = -2*sigma*tanh(sigma*R)). The physical behavior is
        correct: G is 1 at center and 0 at far field.
        """
        R, sigma = 100.0, 0.03

        # Radial shape F = f_Alc
        r_zero = jnp.array(0.0)
        r_far = jnp.array(1000.0)
        F_0 = alcubierre_shape(r_zero, R, sigma)
        F_inf = alcubierre_shape(r_far, R, sigma)
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

    # ------------------------------------------------------------------
    # Cartesian shift & stable G(r) tests
    # ------------------------------------------------------------------

    def test_rodal_cartesian_origin_regularity(self):
        """shift(origin) = (-v_s, 0, 0) without NaN.

        With the Cartesian formula beta = -v_s * [G*x_hat + (F-G)*n_x*n],
        at origin n_x=0 and F(0)=G(0)=1, so beta = (-v_s, 0, 0).
        No jnp.where origin patches needed.
        """
        m = RodalMetric()
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        shift = m.shift(coords)
        assert not jnp.any(jnp.isnan(shift)), f"NaN in shift at origin: {shift}"
        # G(0) ~ 1.0 (exact in analytic limit), so shift_x ~ -v_s
        assert jnp.isclose(shift[0], -m.v_s, atol=1e-8), (
            f"shift[0] = {shift[0]}, expected ~ {-m.v_s}"
        )
        assert jnp.isclose(shift[1], 0.0, atol=1e-10), f"shift[1] = {shift[1]}"
        assert jnp.isclose(shift[2], 0.0, atol=1e-10), f"shift[2] = {shift[2]}"

    def test_rodal_shift_gradient_at_origin(self):
        """jax.jacfwd(m.shift) produces no NaN at origin.

        The Cartesian form is manifestly regular, so forward-mode AD
        should produce finite Jacobian entries everywhere including r=0.
        """
        m = RodalMetric()
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        J = jax.jacfwd(m.shift)(coords)
        assert not jnp.any(jnp.isnan(J)), f"NaN in Jacobian at origin: {J}"

    def test_rodal_G_stable_origin(self):
        """G(0) correct for various R*sigma products via analytic limit.

        The analytic limit Delta'(0) = -2*sigma*tanh(sigma*R) gives
        g_paper(0) = 0 exactly (G(0) = 1) for all R*sigma products.
        """
        # R*sigma=3 (soft wall): G(0) should be very close to 1.0
        G_0_soft = _rodal_G(jnp.array(0.0), 100.0, 0.03)
        assert G_0_soft > 0.99, f"G(0) for R*sigma=3: {G_0_soft}, expected > 0.99"
        assert G_0_soft < 1.01, f"G(0) for R*sigma=3: {G_0_soft}, expected < 1.01"
        # R*sigma=100 (sharp wall): G(0) -> 1 exactly
        G_0_sharp = _rodal_G(jnp.array(0.0), 100.0, 1.0)
        assert jnp.isclose(G_0_sharp, 1.0, atol=1e-6), (
            f"G(0) for R*sigma=100: {G_0_sharp}, expected ~ 1.0"
        )

    def test_rodal_convention_mapping(self):
        """F = 1 - f_paper via f_Alc; G = 1 - g_paper verified."""
        R, sigma = 100.0, 0.03
        for r_val in [0.0, 50.0, 100.0, 200.0, 500.0]:
            r = jnp.array(r_val)
            g_p = _rodal_g_paper(r, R, sigma)
            G = _rodal_G(r, R, sigma)
            assert jnp.isclose(G, 1.0 - g_p, atol=1e-12), (
                f"Convention mismatch at r={r_val}: G={G}, 1-g_paper={1.0-g_p}"
            )

    def test_rodal_cartesian_matches_reference(self):
        """Cartesian form matches reference values at 5 off-axis points.

        Reference values computed from the algebraically equivalent
        spherical-tetrad form at points away from origin.
        """
        m = RodalMetric()  # v_s=0.1, R=100.0, sigma=0.03
        # Pre-computed reference values (from spherical-tetrad implementation)
        test_cases = [
            # (coords, expected_shift) - values from original implementation
            (jnp.array([0.0, 50.0, 10.0, 0.0]),
             jnp.array([-0.09557155, 0.0006476, 0.0])),
            (jnp.array([0.0, 100.0, 1.0, 0.0]),
             jnp.array([-0.0502442, 0.00038641, 0.0])),
            (jnp.array([0.0, 0.0, 50.0, 50.0]),
             jnp.array([-0.09672827, 0.0, 0.0])),
            (jnp.array([0.0, 200.0, 0.0, 0.0]),
             jnp.array([-0.00024849, 0.0, 0.0])),
            (jnp.array([0.0, 30.0, 20.0, 10.0]),
             None),  # just check no NaN
        ]
        for coords, expected in test_cases:
            shift = m.shift(coords)
            assert not jnp.any(jnp.isnan(shift)), (
                f"NaN at coords={coords}: shift={shift}"
            )
            if expected is not None:
                assert jnp.allclose(shift, expected, atol=1e-4), (
                    f"Mismatch at coords={coords}: got {shift}, expected {expected}"
                )


class TestRodalAutodiffAtOrigin:
    """Regression tests for JAX autodiff stability at r_s = 0.

    The Rodal G(r_s) angular profile has a removable 0/0 form at the origin
    (``log_ratio / r_safe`` where both sides go to zero); the analytic
    limit ``lim_{r->0} log_ratio / r = -2*sigma*tanh(sigma*R)`` is applied
    via a ``jnp.where(r < 1e-8, ...)`` branch in ``_rodal_g_paper``. These
    tests ensure the branch keeps the metric, the first-order Jacobian,
    and the Hessian finite across a sweep of coordinate scales.
    """

    def test_metric_finite_at_origin(self):
        m = RodalMetric(v_s=0.5, R=1.0, sigma=0.1)
        g = m(jnp.array([0.0, 0.0, 0.0, 0.0]))
        assert bool(jnp.all(jnp.isfinite(g))), f"non-finite metric at origin: {g}"

    def test_jacfwd_finite_at_origin(self):
        m = RodalMetric(v_s=0.5, R=1.0, sigma=0.1)
        dg = jax.jacfwd(m)(jnp.array([0.0, 0.0, 0.0, 0.0]))
        assert bool(jnp.all(jnp.isfinite(dg))), f"non-finite grad at origin: {dg}"

    def test_hessian_finite_at_origin(self):
        """Second-order autodiff stability: the Hessian of g_00 must be
        finite for the curvature chain (which composes two jacfwd calls)
        to produce a finite Riemann tensor."""
        m = RodalMetric(v_s=0.5, R=1.0, sigma=0.1)
        def g00(c):
            return m(c)[0, 0]
        h = jax.hessian(g00)(jnp.array([0.0, 0.0, 0.0, 0.0]))
        assert bool(jnp.all(jnp.isfinite(h))), f"non-finite Hessian at origin: {h}"

    def test_autodiff_finite_across_small_r_sweep(self):
        """Gradient remains finite across coordinate scales spanning the
        analytic-limit threshold (1e-8)."""
        m = RodalMetric(v_s=0.5, R=1.0, sigma=0.1)
        for r in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 0.0):
            coords = jnp.array([0.0, r, 0.0, 0.0])
            dg = jax.jacfwd(m)(coords)
            assert bool(jnp.all(jnp.isfinite(dg))), (
                f"non-finite grad at r={r!r}: {dg}"
            )

    def test_curvature_chain_finite_at_origin(self):
        """Full curvature chain (metric -> Christoffel -> Riemann -> Ricci
        -> Einstein -> T) stays finite at r_s = 0, so
        ``compute_curvature_chain`` never emits NaN that would corrupt the
        grid aggregate."""
        m = RodalMetric(v_s=0.5, R=1.0, sigma=0.1)
        result = compute_curvature_chain(m, jnp.array([0.0, 0.0, 0.0, 0.0]))
        for name in ("metric", "christoffel", "riemann", "ricci",
                     "einstein", "stress_energy"):
            arr = getattr(result, name)
            assert bool(jnp.all(jnp.isfinite(arr))), (
                f"non-finite {name} at origin"
            )

    def test_g_paper_analytic_limit_matches_numerical_approach(self):
        """At r -> 0 the analytic-limit branch value matches what a
        truncated-but-safe numerical evaluation approaches. This guards
        against a regression where the constant used in the branch
        (``-2*sigma*tanh(sigma*R)``) drifts from the true limit."""
        R, sigma = 1.0, 0.1
        g_at_zero = _rodal_g_paper(jnp.array(0.0), R, sigma)
        # Limit value: derived from Delta'(0) = -2*sigma*tanh(sigma*R)
        # plugged back into g_paper = 1 + cosh(R*sigma) * limit / (2*sigma*sinh(R*sigma))
        # which simplifies to 1 - cosh(R*sigma)*tanh(sigma*R)/sinh(R*sigma) = 0.
        assert jnp.isclose(g_at_zero, 0.0, atol=1e-14)
        # Compare against a tiny-but-nonzero r evaluation to confirm continuity.
        g_near = _rodal_g_paper(jnp.array(1e-6), R, sigma)
        assert jnp.isclose(g_near, 0.0, atol=1e-8)


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
        """symbolic returns valid SymbolicMetric."""
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


class TestWarpShell:
    """Tests for WarpShellMetric."""

    # ------------------------------------------------------------------
    # Standard 8-test battery
    # ------------------------------------------------------------------

    def test_warpshell_at_origin(self):
        """Evaluate at origin, verify metric structure.

        At origin (deep interior, r=0 < R_1): lapse=1, shift=-v_s,
        spatial=identity. The metric should be a Minkowski-like metric
        with uniform shift (co-moving with the bubble).
        """
        m = WarpShellMetric()  # v_s=0.02, R_1=10, R_2=20, R_b=1, r_s_param=5
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = m(coords)
        assert g.shape == (4, 4)
        # At r=0, deep inside, forced to identity by r < 1e-10 guard
        assert jnp.allclose(g[1:, 1:], jnp.eye(3), atol=1e-14)
        # S_warp(0)=1 and lapse=1 exactly, so g_01 = -v_s, g_00 = -(1 - v_s^2)
        assert jnp.isclose(g[0, 1], -m.v_s, atol=1e-12)
        assert jnp.isclose(g[0, 0], -(1.0 - m.v_s**2), atol=1e-12)

    def test_warpshell_far_field(self):
        """Outside the shell (r > R_2), metric is Minkowski.

        Checks both just outside the shell (r=50) and deep far field (r=1000).
        """
        m = WarpShellMetric()  # R_2=20
        minkowski = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        for r in (50.0, 1000.0):
            g = m(jnp.array([0.0, r, 0.0, 0.0]))
            assert jnp.allclose(g, minkowski, atol=1e-6), (
                f"Exterior metric at r={r} should be Minkowski. "
                f"Got g_00={g[0, 0]}, diag_spatial={jnp.diag(g[1:, 1:])}"
            )

    def test_warpshell_jit(self):
        """jax.jit compilation works."""
        m = WarpShellMetric()
        coords = jnp.array([0.0, 15.0, 2.0, 3.0])
        g_eager = m(coords)
        g_jit = jax.jit(m)(coords)
        assert jnp.allclose(g_eager, g_jit, atol=1e-15)

    def test_warpshell_float64(self):
        """Output dtype is float64."""
        m = WarpShellMetric()
        coords = jnp.array([0.0, 15.0, 2.0, 3.0])
        g = m(coords)
        assert g.dtype == jnp.float64

    def test_warpshell_parameter_change(self):
        """Change v_s, verify output changes (dynamic field)."""
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])  # interior (r < R_1)
        m1 = WarpShellMetric(v_s=0.02)
        m2 = WarpShellMetric(v_s=0.1)
        g1 = m1(coords)
        g2 = m2(coords)
        assert not jnp.allclose(g1, g2, atol=1e-10)

    def test_warpshell_adm_reconstruction(self):
        """Verify __call__ matches manual ADM reconstruction."""
        m = WarpShellMetric()
        coords = jnp.array([0.0, 15.0, 2.0, 3.0])
        g_call = m(coords)
        g_adm = adm_to_full_metric(
            m.lapse(coords),
            m.shift(coords),
            m.spatial_metric(coords),
        )
        assert jnp.allclose(g_call, g_adm, atol=1e-15)

    def test_warpshell_symbolic(self):
        """symbolic returns valid SymbolicMetric."""
        m = WarpShellMetric()
        sm = m.symbolic()
        assert isinstance(sm, SymbolicMetric)
        assert sm.g.shape == (4, 4)
        assert len(sm.coords) == 4

    def test_warpshell_curvature_chain(self):
        """Run compute_curvature_chain and verify no NaN, correct shapes.

        Uses a point in the shell region where curvature is nontrivial.
        """
        m = WarpShellMetric(v_s=0.02, R_1=10.0, R_2=20.0, R_b=1.0, r_s_param=5.0)
        # Point in shell region
        coords = jnp.array([0.0, 15.0, 0.0, 0.0])
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

    def test_warpshell_interior_flat(self):
        """At r < R_1 (deep interior), spatial metric = identity, lapse = 1."""
        m = WarpShellMetric()  # R_1=10
        coords = jnp.array([0.0, 5.0, 0.0, 0.0])  # r=5 < R_1=10
        gamma = m.spatial_metric(coords)
        alpha = m.lapse(coords)
        assert jnp.allclose(gamma, jnp.eye(3), atol=1e-10), (
            f"Interior spatial metric should be identity. Got diag = {jnp.diag(gamma)}"
        )
        assert jnp.isclose(alpha, 1.0, atol=1e-10), (
            f"Interior lapse should be 1.0. Got {alpha}"
        )

    def test_warpshell_shell_non_flat(self):
        """At r between R_1 and R_2, spatial metric is NOT identity.

        The Schwarzschild-like radial stretching gives gamma_rr > 1 in the shell.
        """
        m = WarpShellMetric()  # R_1=10, R_2=20
        coords = jnp.array([0.0, 15.0, 0.0, 0.0])  # r=15, in shell
        gamma = m.spatial_metric(coords)
        assert not jnp.allclose(gamma, jnp.eye(3), atol=1e-6), (
            "Shell spatial metric should NOT be identity"
        )
        # Along x-axis, radial direction = x, so gamma_xx should be > 1
        assert gamma[0, 0] > 1.0 + 1e-6, (
            f"gamma_xx = {gamma[0, 0]}, expected > 1 (Schwarzschild radial stretching)"
        )

    def test_warpshell_shell_lapse_reduced(self):
        """At r in shell, lapse < 1 (Schwarzschild time dilation)."""
        m = WarpShellMetric()  # R_1=10, R_2=20
        coords = jnp.array([0.0, 15.0, 0.0, 0.0])  # r=15, in shell
        alpha = m.lapse(coords)
        assert alpha < 1.0 - 1e-6, (
            f"Shell lapse = {alpha}, expected < 1 (Schwarzschild time dilation)"
        )
        assert alpha > 0.0, "Lapse must be positive"

    def test_warpshell_spatial_metric_symmetric(self):
        """At 5+ points in the shell region, verify gamma_ij == gamma_ji.

        The spherical-to-Cartesian tensor transformation must preserve symmetry.
        """
        m = WarpShellMetric()
        # 5 points in the shell region (r between R_1=10 and R_2=20), off-axis
        shell_points = [
            jnp.array([0.0, 12.0, 3.0, 0.0]),
            jnp.array([0.0, 10.0, 10.0, 0.0]),
            jnp.array([0.0, 0.0, 15.0, 0.0]),
            jnp.array([0.0, 8.0, 8.0, 8.0]),
            jnp.array([0.0, 5.0, 5.0, 12.0]),
        ]
        for coords in shell_points:
            gamma = m.spatial_metric(coords)
            assert jnp.allclose(gamma, gamma.T, atol=1e-15), (
                f"Spatial metric not symmetric at {coords[1:]}: "
                f"max diff = {jnp.max(jnp.abs(gamma - gamma.T))}"
            )

    def test_warpshell_spatial_metric_positive_definite(self):
        """At 5+ points in the shell region, all eigenvalues of gamma_ij > 0.

        This catches sign errors in the Schwarzschild-to-Cartesian conversion.
        """
        m = WarpShellMetric()
        shell_points = [
            jnp.array([0.0, 12.0, 3.0, 0.0]),
            jnp.array([0.0, 10.0, 10.0, 0.0]),
            jnp.array([0.0, 0.0, 15.0, 0.0]),
            jnp.array([0.0, 8.0, 8.0, 8.0]),
            jnp.array([0.0, 5.0, 5.0, 12.0]),
        ]
        for coords in shell_points:
            gamma = m.spatial_metric(coords)
            eigvals = jnp.linalg.eigvalsh(gamma)
            assert jnp.all(eigvals > 0.0), (
                f"Spatial metric not positive definite at {coords[1:]}: "
                f"eigenvalues = {eigvals}"
            )

    def test_warpshell_spatial_metric_boundary_identity(self):
        """At r << R_1 (interior) and r >> R_2 (exterior), eigenvalues all 1.0.

        Confirms the transition function correctly reduces to flat space at boundaries.
        """
        m = WarpShellMetric()
        # Interior points
        interior_points = [
            jnp.array([0.0, 3.0, 0.0, 0.0]),
            jnp.array([0.0, 2.0, 2.0, 0.0]),
        ]
        # Exterior points
        exterior_points = [
            jnp.array([0.0, 50.0, 0.0, 0.0]),
            jnp.array([0.0, 100.0, 0.0, 0.0]),
        ]
        for coords in interior_points + exterior_points:
            gamma = m.spatial_metric(coords)
            eigvals = jnp.linalg.eigvalsh(gamma)
            assert jnp.allclose(eigvals, 1.0, atol=1e-6), (
                f"Boundary spatial metric eigenvalues should be 1.0 at r={jnp.sqrt(coords[1]**2+coords[2]**2+coords[3]**2):.1f}. "
                f"Got {eigvals}"
            )

    def test_warpshell_vacuum_shell_energy_density(self):
        """WarpShell Eulerian energy density vanishes everywhere (C2 default).

        With smooth_width=1.2 the shell indicator is exactly 1 throughout
        [R_1, R_2], so the metric is locally Schwarzschild at r=12, 15, 18.
        Schwarzschild is a vacuum solution, so T_ab n^a n^b must be zero up
        to machine epsilon there (measured ~1e-20). Interior and exterior
        are exactly flat.

        Parameter set: v_s=0.02, R_1=10, R_2=20, R_b=1, r_s_param=5.
        Uses default transition_order=2 (C2 quintic smoothstep).
        """
        m = WarpShellMetric(v_s=0.02, R_1=10.0, R_2=20.0, R_b=1.0, r_s_param=5.0)

        points = {
            "interior": jnp.array([0.0, 5.0, 0.0, 0.0]),
            "shell_inner": jnp.array([0.0, 12.0, 0.0, 0.0]),
            "shell_mid": jnp.array([0.0, 15.0, 0.0, 0.0]),
            "shell_outer": jnp.array([0.0, 18.0, 0.0, 0.0]),
            "exterior": jnp.array([0.0, 50.0, 0.0, 0.0]),
        }

        for name, coords in points.items():
            result = compute_curvature_chain(m, coords)

            # Eulerian energy density: T_ab n^a n^b
            # For WarpShell: n^a = (1/alpha, -beta^i/alpha)
            alpha = m.lapse(coords)
            beta = m.shift(coords)
            n_up = jnp.array([
                1.0 / alpha,
                -beta[0] / alpha,
                -beta[1] / alpha,
                -beta[2] / alpha,
            ])
            T = result.stress_energy
            rho_euler = jnp.einsum("a,ab,b->", n_up, T, n_up)

            assert abs(rho_euler) < 1e-15, (
                f"WarpShell {name}: rho_euler = {rho_euler:.16e}, "
                "expected ~0 (locally Schwarzschild shell is vacuum)"
            )

    # ------------------------------------------------------------------
    # C1 continuity tests
    # ------------------------------------------------------------------

    def test_warpshell_c1_continuity(self):
        """Lapse gradient stays bounded across shell transitions (C1)."""
        m = WarpShellMetric()
        # Radial sweep along x-axis from interior through shell to exterior
        r_vals = jnp.linspace(5.0, 25.0, 200)
        coords_fn = lambda r: jnp.array([0.0, r, 0.0, 0.0])

        # Gradient of lapse w.r.t. radial coordinate (x-component)
        dlapse_dr = jax.vmap(
            lambda r: jax.grad(lambda c: m.lapse(c))(coords_fn(r))[1]
        )(r_vals)

        # No NaN
        assert not jnp.any(jnp.isnan(dlapse_dr)), (
            "Lapse gradient contains NaN values"
        )

        # C1 check: second differences should be bounded
        # (no delta-function spikes in the derivative)
        d2 = jnp.diff(dlapse_dr)
        dr = float(r_vals[1] - r_vals[0])
        max_jump = float(jnp.max(jnp.abs(d2))) / dr
        # For a C1 function, the second derivative is bounded.
        # The old C0 functions would have jumps >> 1e3 here.
        assert max_jump < 1e3, (
            f"Lapse gradient has large jump (max_jump/dr={max_jump:.1f}), "
            f"suggesting C0 discontinuity"
        )

    def test_warpshell_c1_shift_continuity(self):
        """Shift gradient stays bounded across shell transitions (C1)."""
        m = WarpShellMetric()
        r_vals = jnp.linspace(5.0, 25.0, 200)
        coords_fn = lambda r: jnp.array([0.0, r, 0.0, 0.0])

        # Gradient of shift_x w.r.t. x-coordinate
        dshift_dr = jax.vmap(
            lambda r: jax.grad(lambda c: m.shift(c)[0])(coords_fn(r))[1]
        )(r_vals)

        # No NaN
        assert not jnp.any(jnp.isnan(dshift_dr)), (
            "Shift gradient contains NaN values"
        )

        # C1 check: bounded second differences
        d2 = jnp.diff(dshift_dr)
        dr = float(r_vals[1] - r_vals[0])
        max_jump = float(jnp.max(jnp.abs(d2))) / dr
        assert max_jump < 1e3, (
            f"Shift gradient has large jump (max_jump/dr={max_jump:.1f}), "
            f"suggesting C0 discontinuity"
        )

    def test_warpshell_smooth_width_parameter(self):
        """smooth_width constructor parameter works and affects transition zone.

        Verifies:
        1. smooth_width can be passed as constructor parameter
        2. Different smooth_width values produce different metrics in transition zone
        3. Interior/exterior flatness is preserved regardless of smooth_width
        """
        m_default = WarpShellMetric()  # smooth_width=None -> 1.2
        m_narrow = WarpShellMetric(smooth_width=0.5)
        m_wide = WarpShellMetric(smooth_width=3.0)

        # Verify smooth_width field values
        assert m_default.smooth_width is None
        assert m_narrow.smooth_width == 0.5
        assert m_wide.smooth_width == 3.0

        # In transition zone (just outside R_1), different smooth_width
        # should produce different lapse values
        coords_transition = jnp.array([0.0, 9.0, 0.0, 0.0])  # r=9 < R_1=10
        alpha_default = m_default.lapse(coords_transition)
        alpha_narrow = m_narrow.lapse(coords_transition)
        alpha_wide = m_wide.lapse(coords_transition)

        # Wide smooth_width should have more blending at r=9
        # (further from R_1 - smooth_width boundary)
        assert not jnp.isclose(alpha_narrow, alpha_wide, atol=1e-6), (
            f"Different smooth_width should produce different lapse in transition zone. "
            f"narrow={alpha_narrow}, wide={alpha_wide}"
        )

        # Interior flatness preserved (r=5 is well inside for all smooth_width values)
        coords_interior = jnp.array([0.0, 5.0, 0.0, 0.0])
        for m_test in [m_default, m_narrow, m_wide]:
            alpha = m_test.lapse(coords_interior)
            gamma = m_test.spatial_metric(coords_interior)
            assert jnp.isclose(alpha, 1.0, atol=1e-10), (
                f"Interior lapse should be 1.0 with smooth_width={m_test.smooth_width}. Got {alpha}"
            )
            assert jnp.allclose(gamma, jnp.eye(3), atol=1e-10), (
                f"Interior spatial metric should be identity with smooth_width={m_test.smooth_width}"
            )

        # Exterior flatness preserved
        coords_exterior = jnp.array([0.0, 50.0, 0.0, 0.0])
        for m_test in [m_default, m_narrow, m_wide]:
            alpha = m_test.lapse(coords_exterior)
            gamma = m_test.spatial_metric(coords_exterior)
            assert jnp.isclose(alpha, 1.0, atol=1e-10), (
                f"Exterior lapse should be 1.0 with smooth_width={m_test.smooth_width}. Got {alpha}"
            )
            assert jnp.allclose(gamma, jnp.eye(3), atol=1e-10), (
                f"Exterior spatial metric should be identity with smooth_width={m_test.smooth_width}"
            )

    # ------------------------------------------------------------------
    # C1 regression tests (lock legacy behavior)
    # ------------------------------------------------------------------

    def test_warpshell_c1_regression(self):
        """C1 regression: pin the transition_order=1 code path.

        Two parts:
        1. Eulerian energy density stays near zero (|rho| < 1e-14) at five
           radial points -- a near-zero smoke test of the C1 curvature
           pipeline, not a baseline lock (shell is locally Schwarzschild
           vacuum, so rho sits at machine epsilon ~1e-20).
        2. Pins that actually distinguish C1 from C2 in the transition
           zone at r=9.2: the cubic and quintic smoothsteps differ at
           O(1e-2) there (C2 lapse is 0.9319294656761363).
        """
        m = WarpShellMetric(
            v_s=0.02, R_1=10.0, R_2=20.0, R_b=1.0, r_s_param=5.0,
            transition_order=1,
        )

        for r in (5.0, 12.0, 15.0, 18.0, 50.0):
            coords = jnp.array([0.0, r, 0.0, 0.0])
            result = compute_curvature_chain(m, coords)

            alpha = m.lapse(coords)
            beta = m.shift(coords)
            n_up = jnp.array([
                1.0 / alpha,
                -beta[0] / alpha,
                -beta[1] / alpha,
                -beta[2] / alpha,
            ])
            T = result.stress_energy
            rho_euler = jnp.einsum("a,ab,b->", n_up, T, n_up)

            assert abs(rho_euler) < 1e-14, (
                f"C1 regression r={r}: rho_euler = {rho_euler:.16e}, expected ~0"
            )

        # C1-distinguishing pins, measured from the current pipeline.
        # Transition zone is [R_1 - smooth_width, R_1] = [8.8, 10].
        seam = jnp.array([0.0, 9.2, 0.0, 0.0])
        lapse_c1 = m.lapse(seam)
        assert jnp.isclose(lapse_c1, 0.9159128693646388, atol=1e-12), (
            f"C1 lapse at r=9.2 = {lapse_c1:.16e}, "
            "expected 0.9159128693646388 (C2 gives 0.9319294656761363)"
        )
        riemann_c1 = compute_curvature_chain(m, seam).riemann
        assert jnp.isclose(riemann_c1[0, 1, 0, 1], 0.20735301228665254, atol=1e-9), (
            f"C1 Riemann[0,1,0,1] at r=9.2 = {riemann_c1[0, 1, 0, 1]:.16e}, "
            "expected 0.20735301228665254 (C2 gives 0.7242653636732740)"
        )

    # ------------------------------------------------------------------
    # C2 transition_order field tests
    # ------------------------------------------------------------------

    def test_warpshell_transition_order_field(self):
        """Default transition_order=2, configurable to 1."""
        m_default = WarpShellMetric()
        m_c1 = WarpShellMetric(transition_order=1)
        assert m_default.transition_order == 2
        assert m_c1.transition_order == 1

    # ------------------------------------------------------------------
    # C2 continuity tests
    # ------------------------------------------------------------------

    def test_warpshell_c2_second_derivative_continuity(self):
        """C2 lapse has bounded third differences (continuous second derivative).

        Sweeps lapse along r from 5 to 25 (200 points), computes second
        derivative via nested jax.grad, and checks that third differences
        are bounded. The C2 quintic should produce significantly smaller
        jumps than C1 cubic at transition seams.
        """
        m = WarpShellMetric(transition_order=2)
        r_vals = jnp.linspace(5.0, 25.0, 200)

        def lapse_of_x(x):
            return m.lapse(jnp.array([0.0, x, 0.0, 0.0]))

        # Second derivative of lapse w.r.t. x
        d2lapse = jax.vmap(jax.grad(jax.grad(lapse_of_x)))(r_vals)

        # No NaN
        assert not jnp.any(jnp.isnan(d2lapse)), (
            "C2 lapse second derivative contains NaN values"
        )

        # Third differences (proxy for third derivative discontinuity)
        d3 = jnp.diff(d2lapse)
        dr = float(r_vals[1] - r_vals[0])
        max_jump = float(jnp.max(jnp.abs(d3))) / dr

        # C2 bound: significantly tighter than C1's < 1e3
        assert max_jump < 100, (
            f"C2 lapse second derivative has large jump (max_jump/dr={max_jump:.1f}), "
            f"expected < 100 for C2 smoothness"
        )

    def test_warpshell_c2_vs_c1_different_at_seam(self):
        """C2 and C1 metrics produce different curvature at transition seam.

        At a point just outside the inner shell boundary (in the
        transition zone), the Riemann tensor components should differ
        between C1 and C2. This confirms the upgrade actually changes
        something.
        """
        m_c1 = WarpShellMetric(transition_order=1)
        m_c2 = WarpShellMetric(transition_order=2)

        # Point in the inner transition zone:
        # R_1 = 10, smooth_width = 1.2, so transition spans [8.8, 10]
        coords = jnp.array([0.0, 9.2, 0.0, 0.0])

        result_c1 = compute_curvature_chain(m_c1, coords)
        result_c2 = compute_curvature_chain(m_c2, coords)

        # Riemann tensors should differ
        riemann_diff = jnp.max(jnp.abs(result_c1.riemann - result_c2.riemann))
        assert riemann_diff > 1e-10, (
            f"C1 and C2 Riemann tensors are identical at seam "
            f"(max diff = {riemann_diff:.2e}). Expected them to differ."
        )


class TestWarpShellPhysical:
    """Tests for the physical-regime WarpShellPhysical class.

    The physical regime requires ``r_s_param < R_1`` so the Schwarzschild
    radius sits inside the flat interior, well clear of the shell where
    the Schwarzschild-like geometry takes over. Without the clamp the
    lapse is a clean ``sqrt(1 - r_s / r)`` in the shell, with no
    ``minimum(ratio, 1.0 - 1e-12)`` or ``maximum(alpha, 1e-12)`` safety
    patches.
    """

    def test_physical_accepts_r_s_less_than_R_1(self):
        """Default r_s_param=5.0 < R_1=10.0 -- construction must succeed."""
        m = WarpShellPhysical()
        assert m.r_s_param < m.R_1

    def test_physical_rejects_r_s_not_less_than_R_1(self):
        """r_s_param >= R_1 must raise at construction."""
        with pytest.raises(ValueError, match="r_s_param"):
            WarpShellPhysical(r_s_param=10.0, R_1=10.0, R_2=20.0)
        with pytest.raises(ValueError, match="r_s_param"):
            WarpShellPhysical(r_s_param=15.0, R_1=10.0, R_2=20.0)

    def test_physical_matches_stress_test_in_physical_regime(self):
        """When r_s_param < R_1, the clamp never triggers, so the two
        variants produce identical metrics at every shell coordinate."""
        phys = WarpShellPhysical(v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0)
        stress = WarpShellStressTest(v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0)
        for x in (5.0, 11.0, 15.0, 19.0, 25.0):
            coords = jnp.array([0.0, x, 0.0, 0.0])
            assert jnp.allclose(phys(coords), stress(coords), atol=1e-14)

    def test_physical_far_field(self):
        """Far from bubble the metric approaches Minkowski."""
        m = WarpShellPhysical()
        g = m(jnp.array([0.0, 1000.0, 0.0, 0.0]))
        minkowski = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, minkowski, atol=1e-6)

    def test_physical_jit(self):
        m = WarpShellPhysical()
        coords = jnp.array([0.0, 15.0, 2.0, 3.0])
        assert jnp.allclose(m(coords), jax.jit(m)(coords), atol=1e-15)

    def test_physical_float64(self):
        g = WarpShellPhysical()(jnp.array([0.0, 15.0, 2.0, 3.0]))
        assert g.dtype == jnp.float64

    def test_physical_lapse_positive_without_floor(self):
        """Within the shell, r_s_param / r < 1 by construction, so the
        lapse is strictly positive without relying on the 1e-12 floor."""
        m = WarpShellPhysical(r_s_param=5.0, R_1=10.0, R_2=20.0)
        for x in (10.0, 12.5, 15.0, 17.5, 20.0):
            coords = jnp.array([0.0, x, 0.0, 0.0])
            alpha = m.lapse(coords)
            # In the shell the lapse is sqrt(1 - r_s/r); well away from 0.
            assert float(alpha) > 0.1

    def test_stress_test_alias_matches_legacy_metric(self):
        """WarpShellStressTest must expose the same behavior as the
        legacy WarpShellMetric class."""
        legacy = WarpShellMetric()
        renamed = WarpShellStressTest()
        coords = jnp.array([0.0, 15.0, 2.0, 3.0])
        assert jnp.allclose(legacy(coords), renamed(coords), atol=1e-14)


class TestTShell:
    """Smoke tests for source-first T-shell metric."""

    def test_tshell_on_axis_finite(self):
        m = tshell_default()
        coords = jnp.array([0.0, 12.0, 0.0, 0.0])
        g = m(coords)
        assert jnp.all(jnp.isfinite(g))
        assert g[0, 0] < 0.0

    def test_tshell_curvature_finite(self):
        m = tshell_default()
        coords = jnp.array([0.0, 12.0, 0.0, 0.0])
        chain = compute_curvature_chain(m, coords)
        assert jnp.all(jnp.isfinite(chain.stress_energy))
        assert jnp.all(jnp.isfinite(chain.metric))


class TestShellTotalMassArrayLeaf:
    """Regression (total_mass static-leaf retrace): ``total_mass`` was a
    Python float pytree leaf, so ``eqx.partition(metric, eqx.is_array)``
    placed it in the static partition and ``eqx.filter_jit`` retraced for
    every new mass value. It must live in the array partition.
    """

    def test_sshell_total_mass_in_array_partition(self):
        import equinox as eqx
        from warpax.metrics.sshell import sshell_from_profiles
        from warpax.metrics.sshell_profiles import constant_density_profiles

        profiles = constant_density_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4)
        m = sshell_from_profiles(profiles, n_grid=128)
        arrays, static = eqx.partition(m, eqx.is_array)
        assert eqx.is_array(arrays.total_mass), (
            f"total_mass not in array partition: {type(arrays.total_mass)}"
        )
        assert static.total_mass is None
        assert float(arrays.total_mass) > 0.0

    def test_tshell_total_mass_in_array_partition(self):
        import equinox as eqx
        from warpax.metrics.tshell import tshell_from_profiles
        from warpax.metrics.tshell_profiles import constant_velocity_profiles

        profiles = constant_velocity_profiles(
            R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=0.1,
        )
        m = tshell_from_profiles(profiles, n_grid=128)
        arrays, static = eqx.partition(m, eqx.is_array)
        assert eqx.is_array(arrays.total_mass), (
            f"total_mass not in array partition: {type(arrays.total_mass)}"
        )
        assert static.total_mass is None
        assert float(arrays.total_mass) > 0.0
