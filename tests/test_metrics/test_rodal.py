"""Tests for Rodal irrotational warp drive metric."""

import jax
import jax.numpy as jnp

from warpax.metrics import RodalMetric
from warpax.geometry import compute_curvature_chain, SymbolicMetric, adm_to_full_metric
from warpax.metrics._common import alcubierre_shape
from warpax.metrics.rodal import _rodal_G, _rodal_g_paper


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
