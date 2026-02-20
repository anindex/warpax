"""Tests for WarpShell warp drive metric."""

import jax
import jax.numpy as jnp

from warpax.metrics import WarpShellMetric
from warpax.geometry import compute_curvature_chain, SymbolicMetric, adm_to_full_metric


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

    def test_warpshell_far_field(self):
        """Evaluate far from bubble (r >> R_2), verify approaches Minkowski."""
        m = WarpShellMetric()  # R_2=20
        far_coords = jnp.array([0.0, 1000.0, 0.0, 0.0])
        g = m(far_coords)
        minkowski = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, minkowski, atol=1e-6)

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
        """symbolic() returns valid SymbolicMetric."""
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

    def test_warpshell_exterior_minkowski(self):
        """At r >> R_2, metric approaches Minkowski."""
        m = WarpShellMetric()  # R_2=20
        coords = jnp.array([0.0, 50.0, 0.0, 0.0])  # r=50 >> R_2=20
        g = m(coords)
        minkowski = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g, minkowski, atol=1e-6), (
            f"Exterior metric should be Minkowski. Got g_00={g[0,0]}, diag_spatial={jnp.diag(g[1:,1:])}"
        )

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

    def test_warpshell_warpfactory_comparison(self):
        """Regression baseline for WarpShell Eulerian energy density (C2 default).

        Computes T_ab n^a n^b at representative points and compares against
        stored regression baselines. This serves as a regression test to
        detect any changes in the WarpShell metric or curvature pipeline.

        Regression baseline NOT a published WarpFactory comparison.
        Parameter set: v_s=0.02, R_1=10, R_2=20, R_b=1, r_s_param=5.
        Uses default transition_order=2 (C2 quintic smoothstep).

        Note: With smooth_width=1.2, the shell indicator is exactly 1
        throughout [R_1, R_2], so the metric is locally Schwarzschild at
        r=12,15,18. Schwarzschild is a vacuum solution (zero stress-energy),
        hence shell-interior energy densities are near machine epsilon.
        """
        m = WarpShellMetric(v_s=0.02, R_1=10.0, R_2=20.0, R_b=1.0, r_s_param=5.0)

        # Reference values computed with C2 quintic (transition_order=2)
        # Interior/exterior: exactly 0 (flat space)
        # Shell interior (r=12,15,18): near machine epsilon (~1e-19 to 1e-21)
        test_cases = {
            "interior": {
                "coords": jnp.array([0.0, 5.0, 0.0, 0.0]),
                "rho_euler": 0.0,  # Flat interior
            },
            "shell_inner": {
                "coords": jnp.array([0.0, 12.0, 0.0, 0.0]),
                "rho_euler": -1.1159659080075407e-19,
            },
            "shell_mid": {
                "coords": jnp.array([0.0, 15.0, 0.0, 0.0]),
                "rho_euler": -3.0572595439959903e-20,
            },
            "shell_outer": {
                "coords": jnp.array([0.0, 18.0, 0.0, 0.0]),
                "rho_euler": 3.4456208415896368e-21,
            },
            "exterior": {
                "coords": jnp.array([0.0, 50.0, 0.0, 0.0]),
                "rho_euler": 0.0,  # Flat exterior
            },
        }

        for name, case in test_cases.items():
            coords = case["coords"]
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

            assert jnp.allclose(rho_euler, case["rho_euler"], atol=1e-8), (
                f"WarpShell {name}: rho_euler = {rho_euler:.16e}, "
                f"expected {case['rho_euler']:.16e}"
            )

    # ------------------------------------------------------------------
    # C1 continuity tests
    # ------------------------------------------------------------------

    def test_warpshell_c1_continuity(self):
        """Metric gradient is continuous across shell boundaries (C1).

        Uses a fine radial sweep along the x-axis and checks that the
        lapse gradient has no delta-function spikes (bounded second
        differences). The old C0 transitions produced jumps >> 1e3;
        C1 Hermite smoothstep should keep them well below that.
        """
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
        """Shift gradient is continuous across shell boundaries (C1).

        Same approach as lapse continuity: radial sweep, compute gradient
        of shift[0] w.r.t. x-coordinate, verify bounded second differences.
        """
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
        """C1 regression: WarpShellMetric(transition_order=1) reproduces legacy baselines.

        Uses the same parameter set and test points as the original C1
        regression test to confirm that the refactored code path produces
        identical results (within atol=1e-14).
        """
        m = WarpShellMetric(
            v_s=0.02, R_1=10.0, R_2=20.0, R_b=1.0, r_s_param=5.0,
            transition_order=1,
        )

        # C1 regression baselines (from original test before C2 upgrade)
        test_cases = {
            "interior": {
                "coords": jnp.array([0.0, 5.0, 0.0, 0.0]),
                "rho_euler": 0.0,
            },
            "shell_inner": {
                "coords": jnp.array([0.0, 12.0, 0.0, 0.0]),
                "rho_euler": -6.5486234734598095e-20,
            },
            "shell_mid": {
                "coords": jnp.array([0.0, 15.0, 0.0, 0.0]),
                "rho_euler": -1.6848014716480068e-20,
            },
            "shell_outer": {
                "coords": jnp.array([0.0, 18.0, 0.0, 0.0]),
                "rho_euler": 1.0580467215165777e-20,
            },
            "exterior": {
                "coords": jnp.array([0.0, 50.0, 0.0, 0.0]),
                "rho_euler": 0.0,
            },
        }

        for name, case in test_cases.items():
            coords = case["coords"]
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

            assert jnp.allclose(rho_euler, case["rho_euler"], atol=1e-14), (
                f"C1 regression {name}: rho_euler = {rho_euler:.16e}, "
                f"expected {case['rho_euler']:.16e}"
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
