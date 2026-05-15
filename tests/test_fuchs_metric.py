"""Tests for Fuchs et al. constant-velocity warp shell metric.

Verifies the FuchsMetric implementation against arXiv:2405.02709.
Covers paper-exact parameters, ADM decomposition (interior/shell/
exterior), shell source profiles, curvature chain at the shell, the
warpax EC pipeline, constraint residuals, source consistency, TOV
equilibrium, transport diagnostics, ADM mass and falloff. Also runs
Rodal and Lentz through the same diagnostics for comparison.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class TestFuchsMetric:
    """Verify the Fuchs metric has paper-correct structure."""

    def test_paper_parameters(self):
        """Default parameters match Section 4 of arXiv:2405.02709."""
        from warpax.metrics import fuchs_default

        metric = fuchs_default()
        assert metric.v_s == 0.02
        assert metric.R_1 == 10.0
        assert metric.R_2 == 20.0

    def test_interior_adm_decomposition(self):
        """Deep interior (r << R_1): non-unit lapse (gravitational well),
        shift = -v_s, and Schwarzschild-modified spatial metric.

        The iteratively-smoothed construction has nonzero mass at all r,
        so the interior is not exactly Minkowski.
        """
        from warpax.metrics import fuchs_default

        metric = fuchs_default()
        coords = jnp.array([0.0, 1.0, 0.0, 0.0])

        alpha = metric.lapse(coords)
        beta = metric.shift(coords)

        # Lapse < 1 inside the gravitational well (Schwarzschild-like)
        assert 0.0 < float(alpha) < 1.0, \
            f"Interior lapse should be < 1 (gravitational well), got {alpha}"
        # Shift is -v_s inside the bubble
        assert jnp.isclose(beta[0], -0.02, atol=1e-3), \
            f"Interior shift should be ~-v_s=-0.02, got {beta[0]}"
        assert jnp.allclose(beta[1:], 0.0, atol=1e-10)

    def test_shell_schwarzschild_structure(self):
        """Shell region (R_1 < r < R_2): non-unit lapse and curved spatial.

        Schwarzschild-like structure: lapse < 1 (gravitational time
        dilation) and gamma_xx > 1 (radial stretching).
        """
        from warpax.metrics import fuchs_default

        metric = fuchs_default()
        coords = jnp.array([0.0, 15.0, 0.0, 0.0])

        alpha = metric.lapse(coords)
        gamma = metric.spatial_metric(coords)

        assert 0.0 < alpha < 1.0 - 1e-6, \
            f"Shell lapse should be in (0, 1), got {alpha}"
        assert gamma[0, 0] > 1.0 + 1e-6, \
            f"gamma_xx = {gamma[0, 0]}, expected > 1 (radial stretching)"

    def test_shell_curvature_chain(self):
        """Curvature chain at a shell point is NaN-free with nontrivial values.

        This is the most important single-point test: the shell region
        is where curvature is nontrivial and numerical issues (division
        by zero near r_s, NaN from autodiff) would manifest.
        """
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import fuchs_default

        result = compute_curvature_chain(
            fuchs_default(),
            jnp.array([0.0, 15.0, 0.0, 0.0]),
        )

        assert result.stress_energy.shape == (4, 4)
        assert not jnp.any(jnp.isnan(result.riemann))
        assert not jnp.any(jnp.isnan(result.stress_energy))

        # Curvature should be nontrivial (not all zeros)
        assert jnp.max(jnp.abs(result.riemann)) > 1e-10, \
            "Shell Riemann tensor should be nontrivial"

    def test_regularity_report(self):
        """Full regularity report passes C^2 for the Fuchs metric.

        C^2 smoothness is required for pointwise energy condition
        evaluation (Barzegar et al. arXiv:2602.16495).
        """
        from warpax.geometry import regularity_report
        from warpax.metrics import fuchs_default

        report = regularity_report(fuchs_default(), r_min=5.0, r_max=25.0)
        assert report.is_c2, \
            "Fuchs metric should be C^2. Jumps: " + \
            ", ".join(f"{k}: {v.c2_max_jump:.1f}"
                      for k, v in report.components.items())




class TestFuchsShellProfiles:
    """Verify analytical shell source profiles."""

    def test_profiles_construction(self):
        """Analytical shell profiles return correct type with paper params."""
        from warpax.metrics import FuchsShellProfiles, fuchs_shell_profiles

        profiles = fuchs_shell_profiles()
        assert isinstance(profiles, FuchsShellProfiles)
        assert profiles.R_1 == 10.0
        assert profiles.R_2 == 20.0
        assert profiles.total_mass > 0.0

    def test_density_compact_support(self):
        """Density is zero outside [R_1, R_2] and positive inside."""
        from warpax.metrics import fuchs_shell_profiles

        profiles = fuchs_shell_profiles()
        assert float(profiles.density(jnp.array(5.0))) == 0.0
        assert float(profiles.density(jnp.array(25.0))) == 0.0
        assert float(profiles.density(jnp.array(15.0))) > 0.0

    def test_mass_integral_consistency(self):
        """4pi int rho r^2 dr from R_1 to R_2 equals total_mass.

        This validates that density and cumulative_mass are mutually
        consistent, not just individually plausible.
        """
        from warpax.metrics import fuchs_shell_profiles

        profiles = fuchs_shell_profiles()

        r_vals = jnp.linspace(10.0, 20.0, 1000)
        dr = float(r_vals[1] - r_vals[0])
        rho_vals = jax.vmap(profiles.density)(r_vals)
        M_integrated = float(jnp.sum(4.0 * jnp.pi * rho_vals * r_vals**2) * dr)

        assert jnp.isclose(M_integrated, profiles.total_mass, rtol=1e-2), \
            f"Integrated mass {M_integrated:.4f} != total {profiles.total_mass:.4f}"

    def test_cumulative_mass_monotonic(self):
        """m(r) is zero before R_1, monotonic in shell, equals M at R_2."""
        from warpax.metrics import fuchs_shell_profiles

        profiles = fuchs_shell_profiles()

        assert float(profiles.cumulative_mass(jnp.array(5.0))) == 0.0
        m_mid = float(profiles.cumulative_mass(jnp.array(15.0)))
        m_outer = float(profiles.cumulative_mass(jnp.array(20.0)))
        assert 0.0 < m_mid < m_outer
        assert jnp.isclose(m_outer, profiles.total_mass, rtol=1e-6)

    def test_pressure_boundary(self):
        """Radial pressure vanishes at R_2 (free-surface boundary condition)."""
        from warpax.metrics import fuchs_shell_profiles

        profiles = fuchs_shell_profiles()
        assert abs(float(profiles.radial_pressure(jnp.array(20.0)))) < 1e-10


# -- EC pipeline --------------------------------------------------------


class TestFuchsEC:
    """Energy condition pipeline on the Fuchs shell."""

    def test_interior_type_i(self):
        """Interior (r=1) classifies as Hawking-Ellis Type I."""
        from warpax.energy_conditions import classify_mixed_tensor
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import fuchs_default

        curv = compute_curvature_chain(
            fuchs_default(), jnp.array([0.0, 1.0, 0.0, 0.0]),
        )
        cls = classify_mixed_tensor(
            curv.stress_energy, curv.metric, curv.metric_inv,
        )
        assert int(cls.he_type) == 1

    def test_shell_classification_and_margins(self):
        """Shell (r=15): classify, verify finite margins, compare Eulerian.

        At v_s=0.02 the shell eigenvalues are ~1e-7 (near-degenerate).
        """
        from warpax.energy_conditions import (
            classify_mixed_tensor,
            compute_eulerian_ec,
            verify_point,
        )
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import fuchs_default

        curv = compute_curvature_chain(
            fuchs_default(), jnp.array([0.0, 15.0, 0.0, 0.0]),
        )
        T, g, gi = curv.stress_energy, curv.metric, curv.metric_inv

        cls = classify_mixed_tensor(T, g, gi)
        assert int(cls.he_type) in (1, 4)
        assert jnp.all(jnp.isfinite(cls.eigenvalues))

        ec = verify_point(T, g, gi, n_starts=16)
        for name, m in [("NEC", ec.nec_margin), ("WEC", ec.wec_margin),
                        ("SEC", ec.sec_margin), ("DEC", ec.dec_margin)]:
            assert jnp.isfinite(m), f"{name} not finite"

        eul = compute_eulerian_ec(T, g, gi)
        for k in ["nec", "wec", "sec", "dec"]:
            assert jnp.isfinite(eul[k]), f"Eulerian {k} not finite"

    def test_interior_ec_non_negative(self):
        """Interior (r=1, near-vacuum) has non-negative EC margins."""
        from warpax.energy_conditions import verify_point
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import fuchs_default

        curv = compute_curvature_chain(
            fuchs_default(), jnp.array([0.0, 1.0, 0.0, 0.0]),
        )
        ec = verify_point(
            curv.stress_energy, curv.metric, curv.metric_inv, n_starts=16,
        )
        assert float(ec.nec_margin) >= -1e-8
        assert float(ec.wec_margin) >= -1e-8
        assert float(ec.sec_margin) >= -1e-8
        assert float(ec.dec_margin) >= -1e-8


# -- Constraint residuals -----------------------------------------------


class TestFuchsConstraints:
    """ADM constraint residuals on Fuchs initial data."""

    def test_interior_constraints_vanish(self):
        """Interior (r=1): flat Minkowski, constraints vanish."""
        from warpax.constraints import normalized_residuals
        from warpax.metrics import fuchs_default

        res = normalized_residuals(
            fuchs_default(), jnp.array([0.0, 1.0, 0.0, 0.0]),
        )
        assert float(res["epsilon_H"]) < 1e-8
        assert float(res["epsilon_M"]) < 1e-8

    def test_exterior_constraints_small(self):
        """Exterior (r=30): near-vacuum, small constraint residuals.

        Gaussian-smoothed profiles have exponentially decaying tails,
        so residuals are small but not machine-zero at r=30.
        """
        from warpax.constraints import normalized_residuals
        from warpax.metrics import fuchs_default

        res = normalized_residuals(
            fuchs_default(), jnp.array([0.0, 30.0, 0.0, 0.0]),
        )
        assert float(res["epsilon_H"]) < 1e-2
        assert float(res["epsilon_M"]) < 1e-2


# -- Source consistency --------------------------------------------------


class TestFuchsSourceConsistency:
    """Input-vs-derived stress-energy comparison."""

    def test_input_stress_energy_construction(self):
        """T_input is symmetric (4,4) float64; nonzero at shell, zero outside."""
        from warpax.metrics._fuchs_legacy import _fuchs_analytical_default
        from warpax.metrics import fuchs_input_stress_energy

        metric = _fuchs_analytical_default()

        T_shell = fuchs_input_stress_energy(
            metric, jnp.array([0.0, 15.0, 0.0, 0.0]),
        )
        assert T_shell.shape == (4, 4)
        assert T_shell.dtype == jnp.float64
        assert jnp.allclose(T_shell, T_shell.T, atol=1e-15)
        assert jnp.max(jnp.abs(T_shell)) > 1e-15

        T_ext = fuchs_input_stress_energy(
            metric, jnp.array([0.0, 30.0, 0.0, 0.0]),
        )
        assert jnp.max(jnp.abs(T_ext)) < 1e-10

    def test_exterior_source_consistency(self):
        """Exterior (r=30): both T_input and G/8pi are ~0."""
        from warpax.constraints import stress_energy_residual
        from warpax.metrics._fuchs_legacy import _fuchs_analytical_default
        from warpax.metrics import fuchs_input_stress_energy

        metric = _fuchs_analytical_default()
        coords = jnp.array([0.0, 30.0, 0.0, 0.0])
        T_input = fuchs_input_stress_energy(metric, coords)

        sc = stress_energy_residual(metric, coords, T_input=T_input)
        assert float(sc["max_residual"]) < 1e-8


# -- TOV residuals -------------------------------------------------------


class TestFuchsTOV:
    """Anisotropic TOV equilibrium residuals on the analytical Fuchs shell."""

    def test_tov_interior_vanishes(self):
        """Interior (r=1): zero density/pressure -> TOV residual vanishes."""
        from warpax.metrics._fuchs_legacy import _fuchs_analytical_default
        from warpax.tov import tov_residual_from_metric

        metric = _fuchs_analytical_default()
        profiles = metric.shell_profiles()
        res = tov_residual_from_metric(
            metric, jnp.array(1.0),
            profiles.density, profiles.radial_pressure,
            profiles.tangential_pressure,
        )
        assert jnp.isfinite(res)
        assert float(jnp.abs(res)) < 1e-8

    def test_tov_exterior_vanishes(self):
        """Exterior (r=30): vacuum -> TOV residual vanishes."""
        from warpax.metrics._fuchs_legacy import _fuchs_analytical_default
        from warpax.tov import tov_residual_from_metric

        metric = _fuchs_analytical_default()
        profiles = metric.shell_profiles()
        res = tov_residual_from_metric(
            metric, jnp.array(30.0),
            profiles.density, profiles.radial_pressure,
            profiles.tangential_pressure,
        )
        assert jnp.isfinite(res)
        assert float(jnp.abs(res)) < 1e-8

    def test_tov_radial_sweep(self):
        """TOV residual is finite and bounded at points through the shell."""
        from warpax.metrics._fuchs_legacy import _fuchs_analytical_default
        from warpax.tov import tov_residual_from_metric

        metric = _fuchs_analytical_default()
        profiles = metric.shell_profiles()
        max_residual = 0.0

        for r_val in jnp.linspace(5.0, 25.0, 10):
            res = tov_residual_from_metric(
                metric, jnp.array(float(r_val)),
                profiles.density, profiles.radial_pressure,
                profiles.tangential_pressure,
            )
            assert jnp.isfinite(res), f"TOV not finite at r={float(r_val)}"
            max_residual = max(max_residual, float(jnp.abs(res)))

        assert max_residual < 1e2


# -- Transport diagnostics ----------------------------------------------


class TestFuchsTransport:
    """Geodesic deviation and blueshift diagnostics on the Fuchs shell."""

    def test_geodesic_deviation_interior_vanishes(self):
        """Interior (r=1): flat -> zero tidal acceleration."""
        from warpax.metrics import fuchs_default
        from warpax.transport import geodesic_deviation_diagnostic

        metric = fuchs_default()
        A_geo = geodesic_deviation_diagnostic(
            metric, jnp.array([0.0, 1.0, 0.0, 0.0]),
        )
        assert jnp.isfinite(A_geo)
        assert float(A_geo) < 1e-8

    def test_geodesic_deviation_shell_nonzero(self):
        """Shell (r=15): curved region has nonzero tidal acceleration."""
        from warpax.metrics import fuchs_default
        from warpax.transport import geodesic_deviation_diagnostic

        metric = fuchs_default()
        A_geo = geodesic_deviation_diagnostic(
            metric, jnp.array([0.0, 15.0, 0.0, 0.0]),
        )
        assert jnp.isfinite(A_geo)
        assert float(A_geo) > 1e-15

    def test_blueshift_interior_vanishes(self):
        """Interior (r=1): flat -> trivial blueshift."""
        from warpax.metrics import fuchs_default
        from warpax.transport import blueshift_hazard

        metric = fuchs_default()
        B = blueshift_hazard(
            metric, jnp.array([0.0, 1.0, 0.0, 0.0]),
            n_directions=2, tau_max=5.0, num_points=50,
        )
        assert jnp.isfinite(B)
        assert float(B) < 0.5


# -- ADM mass and falloff ------------------------------------------------


class TestFuchsADMMass:
    """ADM mass and asymptotic falloff for the Fuchs shell."""

    def test_adm_mass_positive(self):
        """ADM mass at r=R_2 is positive."""
        from warpax.adm import adm_mass
        from warpax.metrics import fuchs_default

        M = adm_mass(fuchs_default(), r_surface=20.0, n_theta=8, n_phi=16)
        assert jnp.isfinite(M)
        assert float(M) > 0

    def test_metric_approaches_flat_far(self):
        """Metric deviation from Minkowski decreases with radius."""
        from warpax.metrics import fuchs_default

        metric = fuchs_default()
        eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

        g_25 = metric(jnp.array([0.0, 25.0, 0.0, 0.0]))
        g_29 = metric(jnp.array([0.0, 29.0, 0.0, 0.0]))

        dev_25 = float(jnp.max(jnp.abs(g_25 - eta)))
        dev_29 = float(jnp.max(jnp.abs(g_29 - eta)))

        # Deviation should decrease as we move outward
        assert dev_29 < dev_25, \
            f"Metric not approaching Minkowski: dev(29)={dev_29:.4e} >= dev(25)={dev_25:.4e}"


# -- Comparison cases: Rodal, Lentz --------------------------------------


class TestComparison:
    """Rodal and Lentz through the same diagnostics as Fuchs."""

    def test_rodal_ec(self):
        """Rodal at bubble center: Type I, finite EC margins."""
        from warpax.energy_conditions import classify_mixed_tensor, verify_point
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import RodalMetric

        metric = RodalMetric(v_s=0.1, R=100.0, sigma=0.03)
        curv = compute_curvature_chain(metric, jnp.array([0.0, 0.0, 0.0, 0.0]))
        T, g, gi = curv.stress_energy, curv.metric, curv.metric_inv

        cls = classify_mixed_tensor(T, g, gi)
        assert int(cls.he_type) == 1

        ec = verify_point(T, g, gi, n_starts=4)
        for name, m in [("NEC", ec.nec_margin), ("WEC", ec.wec_margin),
                        ("SEC", ec.sec_margin), ("DEC", ec.dec_margin)]:
            assert jnp.isfinite(m), f"Rodal {name} not finite"

    def test_rodal_constraints(self):
        """Rodal at bubble center: constraint residuals are finite."""
        from warpax.constraints import normalized_residuals
        from warpax.metrics import RodalMetric

        metric = RodalMetric(v_s=0.1, R=100.0, sigma=0.03)
        res = normalized_residuals(metric, jnp.array([0.0, 0.0, 0.0, 0.0]))
        assert jnp.isfinite(res["epsilon_H"])
        assert jnp.isfinite(res["epsilon_M"])

    def test_rodal_wall_ec(self):
        """Rodal at bubble wall (r=R): EC margins are finite."""
        from warpax.energy_conditions import verify_point
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import RodalMetric

        metric = RodalMetric(v_s=0.1, R=100.0, sigma=0.03)
        curv = compute_curvature_chain(metric, jnp.array([0.0, 100.0, 0.0, 0.0]))
        ec = verify_point(curv.stress_energy, curv.metric, curv.metric_inv, n_starts=4)
        for name, m in [("NEC", ec.nec_margin), ("WEC", ec.wec_margin)]:
            assert jnp.isfinite(m), f"Rodal wall {name} not finite"

    def test_lentz_ec(self):
        """Lentz at bubble center: classify and verify EC margins."""
        from warpax.energy_conditions import classify_mixed_tensor, verify_point
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import LentzMetric

        metric = LentzMetric(v_s=0.1, R=100.0, sigma=8.0)
        curv = compute_curvature_chain(metric, jnp.array([0.0, 0.0, 0.0, 0.0]))
        T, g, gi = curv.stress_energy, curv.metric, curv.metric_inv

        cls = classify_mixed_tensor(T, g, gi)
        assert int(cls.he_type) in (1, 4)

        ec = verify_point(T, g, gi, n_starts=4)
        for name, m in [("NEC", ec.nec_margin), ("WEC", ec.wec_margin),
                        ("SEC", ec.sec_margin), ("DEC", ec.dec_margin)]:
            assert jnp.isfinite(m), f"Lentz {name} not finite"

    def test_lentz_constraints(self):
        """Lentz near bubble center: constraint residuals are finite.

        Off-axis probe avoids jnp.abs(x_rel) cusp at x_rel=0.
        """
        from warpax.constraints import normalized_residuals
        from warpax.metrics import LentzMetric

        metric = LentzMetric(v_s=0.1, R=100.0, sigma=8.0)
        res = normalized_residuals(metric, jnp.array([0.0, 0.1, 0.1, 0.0]))
        assert jnp.isfinite(res["epsilon_H"])
        assert jnp.isfinite(res["epsilon_M"])

    def test_lentz_wall_ec(self):
        """Lentz at diamond wall (d=R): EC margins are finite."""
        from warpax.energy_conditions import verify_point
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import LentzMetric

        metric = LentzMetric(v_s=0.1, R=100.0, sigma=8.0)
        curv = compute_curvature_chain(metric, jnp.array([0.0, 100.0, 0.0, 0.0]))
        ec = verify_point(curv.stress_energy, curv.metric, curv.metric_inv, n_starts=4)
        for name, m in [("NEC", ec.nec_margin), ("WEC", ec.wec_margin)]:
            assert jnp.isfinite(m), f"Lentz wall {name} not finite"

    def test_far_field_source_consistency(self):
        """Far-field source consistency: G/8pi ~ 0 for Rodal and Lentz."""
        from warpax.constraints import stress_energy_residual
        from warpax.metrics import RodalMetric, LentzMetric

        rodal = RodalMetric()
        sc = stress_energy_residual(rodal, jnp.array([0.0, 500.0, 0.0, 0.0]))
        assert float(sc["max_residual"]) < 1e-6

        # Lentz: probe along y-axis to avoid jnp.abs cusp
        lentz = LentzMetric()
        sc = stress_energy_residual(lentz, jnp.array([0.0, 0.0, 500.0, 0.0]))
        assert float(sc["max_residual"]) < 1e-6
