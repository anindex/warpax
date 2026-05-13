"""Tests for the S-shell source-first warp shell metric.

Covers source profiles, constraint solver, metric construction,
constraint residuals, energy conditions, TOV, and transport.
"""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class TestSShellProfiles:
    """Source profile construction and properties."""

    def test_constant_density(self):
        """Constant density: nonzero in shell, zero outside."""
        from warpax.metrics import constant_density_profiles

        profiles = constant_density_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4)
        assert profiles.R_1 == 10.0
        assert profiles.R_2 == 20.0
        assert profiles.total_mass > 0
        assert float(profiles.density(jnp.asarray(15.0))) > 1e-6
        assert float(jnp.abs(profiles.density(jnp.asarray(5.0)))) < 1e-10
        assert float(jnp.abs(profiles.density(jnp.asarray(25.0)))) < 1e-10

    def test_parabolic_density(self):
        """Parabolic density: peaks at center, vanishes at boundaries."""
        from warpax.metrics import parabolic_density_profiles

        profiles = parabolic_density_profiles(R_1=10.0, R_2=20.0, rho_max=1e-4)
        assert profiles.total_mass > 0
        assert float(profiles.density(jnp.asarray(15.0))) > float(profiles.density(jnp.asarray(10.0)))
        assert float(jnp.abs(profiles.density(jnp.asarray(10.0)))) < 1e-10
        assert float(jnp.abs(profiles.density(jnp.asarray(20.0)))) < 1e-10

    def test_bernstein_density(self):
        """Bernstein density: compact support, differentiable."""
        from warpax.metrics import bernstein_density_profiles

        profiles = bernstein_density_profiles(R_1=10.0, R_2=20.0)
        assert profiles.total_mass > 0
        assert float(jnp.abs(profiles.density(jnp.asarray(10.0)))) < 1e-10
        assert float(jnp.abs(profiles.density(jnp.asarray(20.0)))) < 1e-10
        assert float(profiles.density(jnp.asarray(15.0))) > 0

    def test_cumulative_mass_monotone(self):
        """m(r) is non-decreasing through the shell."""
        from warpax.metrics import constant_density_profiles

        profiles = constant_density_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4)
        r_vals = jnp.linspace(5.0, 25.0, 20)
        m_vals = jnp.array([float(profiles.cumulative_mass(jnp.asarray(r))) for r in r_vals])
        assert jnp.all(jnp.diff(m_vals) >= -1e-10)

    def test_pressure_non_negative(self):
        """p_r(r) >= 0 everywhere (enforced by TOV integration)."""
        from warpax.metrics import constant_density_profiles

        profiles = constant_density_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4)
        r_vals = jnp.linspace(5.0, 25.0, 50)
        p_vals = jnp.array([float(profiles.radial_pressure(jnp.asarray(r))) for r in r_vals])
        assert jnp.all(p_vals >= -1e-15)


class TestConstraintSolver:
    """Constraint solver: source profiles -> metric potentials."""

    def test_schwarzschild_exterior(self):
        """Exterior potentials match exact Schwarzschild."""
        from warpax.constraints import solve_sshell_potentials
        from warpax.metrics import constant_density_profiles

        profiles = constant_density_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4)
        pots = solve_sshell_potentials(
            rho=profiles.density, p_r=profiles.radial_pressure,
            R_1=10.0, R_2=20.0, n_grid=2048,
        )
        M = pots.total_mass
        assert M > 0

        r_test = jnp.asarray(25.0)
        Phi_exact = 0.5 * jnp.log(1.0 - 2.0 * M / r_test)
        assert float(jnp.abs(pots.Phi_fn(r_test) - Phi_exact)) < 1e-4

        Lambda_exact = -0.5 * jnp.log(1.0 - 2.0 * M / r_test)
        assert float(jnp.abs(pots.Lambda_fn(r_test) - Lambda_exact)) < 1e-4

    def test_mass_conservation(self):
        """Total mass agrees with analytical (4/3)pi rho_0 (R2^3-R1^3)."""
        from warpax.constraints import solve_sshell_potentials
        from warpax.metrics import constant_density_profiles

        profiles = constant_density_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4)
        pots = solve_sshell_potentials(
            rho=profiles.density, p_r=profiles.radial_pressure,
            R_1=10.0, R_2=20.0,
        )
        M_analytic = (4.0 / 3.0) * jnp.pi * 1e-4 * (20.0**3 - 10.0**3)
        assert abs(pots.total_mass - float(M_analytic)) / float(M_analytic) < 0.1

    def test_interior_flat(self):
        """Interior (r < R_1): Lambda ~ 0 (no enclosed mass)."""
        from warpax.constraints import solve_sshell_potentials
        from warpax.metrics import constant_density_profiles

        profiles = constant_density_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4)
        pots = solve_sshell_potentials(
            rho=profiles.density, p_r=profiles.radial_pressure,
            R_1=10.0, R_2=20.0,
        )
        assert abs(float(pots.Lambda_fn(jnp.asarray(5.0)))) < 1e-6

    def test_potentials_finite(self):
        """All potential grids are finite."""
        from warpax.constraints import solve_sshell_potentials
        from warpax.metrics import constant_density_profiles

        profiles = constant_density_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4)
        pots = solve_sshell_potentials(
            rho=profiles.density, p_r=profiles.radial_pressure,
            R_1=10.0, R_2=20.0,
        )
        assert jnp.all(jnp.isfinite(pots.Phi_grid))
        assert jnp.all(jnp.isfinite(pots.Lambda_grid))
        assert jnp.all(jnp.isfinite(pots.m_grid))

    def test_trapped_surface_rejection(self):
        """Compactness 2m/r >= 1 raises ValueError."""
        from warpax.constraints import solve_sshell_potentials
        from warpax.metrics import constant_density_profiles

        profiles = constant_density_profiles(R_1=10.0, R_2=20.0, rho_0=1.0)
        with pytest.raises(ValueError, match="trapped surface"):
            solve_sshell_potentials(
                rho=profiles.density, p_r=profiles.radial_pressure,
                R_1=10.0, R_2=20.0,
            )


class TestSShellMetric:
    """SShellMetric construction and ADM components."""

    def test_metric_shape_and_symmetry(self):
        """Returns (4,4) float64 symmetric tensor at all probe radii."""
        from warpax.metrics import sshell_default

        metric = sshell_default()
        for r_val in [1.0, 10.0, 15.0, 20.0, 30.0]:
            g = metric(jnp.array([0.0, r_val, 0.0, 0.0]))
            assert g.shape == (4, 4)
            assert g.dtype == jnp.float64
            assert jnp.all(jnp.isfinite(g))
            assert jnp.allclose(g, g.T, atol=1e-14), f"Not symmetric at r={r_val}"

    def test_interior_flat_spatial(self):
        """Interior (r=1): flat spatial metric, g_tt < 0, no shift."""
        from warpax.metrics import sshell_default

        metric = sshell_default()
        g = metric(jnp.array([0.0, 1.0, 0.0, 0.0]))
        assert jnp.allclose(g[1:, 1:], jnp.eye(3), atol=1e-6)
        assert float(g[0, 0]) < 0
        assert jnp.allclose(g[0, 1:], 0.0, atol=1e-10)

    def test_exterior_schwarzschild_structure(self):
        """Exterior (r=25): g_tt < 0, g_xx > 1 (radial stretching)."""
        from warpax.metrics import sshell_default

        metric = sshell_default()
        g = metric(jnp.array([0.0, 25.0, 0.0, 0.0]))
        assert float(g[0, 0]) < 0
        assert float(g[1, 1]) > 1.0 - 1e-6
        assert float(jnp.abs(g[0, 1])) < 1e-10

    def test_lapse_positive(self):
        """Lapse is positive at all radii."""
        from warpax.metrics import sshell_default

        metric = sshell_default()
        for r_val in [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
            assert float(metric.lapse(jnp.array([0.0, r_val, 0.0, 0.0]))) > 0

    def test_spatial_metric_positive_definite(self):
        """gamma_{ij} has positive eigenvalues."""
        from warpax.metrics import sshell_default

        metric = sshell_default()
        for r_val in [5.0, 15.0, 25.0]:
            gamma = metric.spatial_metric(jnp.array([0.0, r_val, 0.0, 0.0]))
            assert jnp.all(jnp.linalg.eigvalsh(gamma) > 0)

    def test_shift_zero_vs_nonzero(self):
        """v_s=0: zero shift; v_s=0.02: nonzero inside, decays outside."""
        from warpax.metrics import sshell_default

        m0 = sshell_default(v_s=0.0)
        for r_val in [5.0, 15.0, 25.0]:
            assert jnp.allclose(m0.shift(jnp.array([0.0, r_val, 0.0, 0.0])), 0.0, atol=1e-15)

        ms = sshell_default(v_s=0.02)
        assert float(jnp.abs(ms.shift(jnp.array([0.0, 5.0, 0.0, 0.0]))[0])) > 1e-4
        assert float(jnp.abs(ms.shift(jnp.array([0.0, 30.0, 0.0, 0.0]))[0])) < 1e-10


class TestSShellConstraints:
    """ADM constraint residuals (numerical accuracy of solver chain)."""

    def test_interior_constraints(self):
        """Interior (r=5): eps_H, eps_M < 1e-4."""
        from warpax.constraints import normalized_residuals
        from warpax.metrics import sshell_default

        res = normalized_residuals(sshell_default(), jnp.array([0.0, 5.0, 0.0, 0.0]))
        assert float(res["epsilon_H"]) < 1e-4
        assert float(res["epsilon_M"]) < 1e-4

    def test_shell_constraints(self):
        """Shell (r=15): finite and smaller than Fuchs (eps_H=0.165)."""
        from warpax.constraints import normalized_residuals
        from warpax.metrics import sshell_default

        res = normalized_residuals(sshell_default(), jnp.array([0.0, 15.0, 0.0, 0.0]))
        assert jnp.isfinite(res["epsilon_H"])
        assert jnp.isfinite(res["epsilon_M"])
        assert float(res["epsilon_H"]) < 0.165

    def test_exterior_constraints(self):
        """Exterior (r=25): eps_H, eps_M < 1e-4."""
        from warpax.constraints import normalized_residuals
        from warpax.metrics import sshell_default

        res = normalized_residuals(sshell_default(), jnp.array([0.0, 25.0, 0.0, 0.0]))
        assert float(res["epsilon_H"]) < 1e-4
        assert float(res["epsilon_M"]) < 1e-4


class TestSShellEC:
    """Energy condition evaluation on the S-shell."""

    def test_interior_ec(self):
        """Interior (r=5): all EC margins >= -1e-6."""
        from warpax.energy_conditions import verify_point
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import sshell_default

        curv = compute_curvature_chain(sshell_default(), jnp.array([0.0, 5.0, 0.0, 0.0]))
        ec = verify_point(curv.stress_energy, curv.metric, curv.metric_inv, n_starts=4)
        for name, m in [("NEC", ec.nec_margin), ("WEC", ec.wec_margin),
                        ("SEC", ec.sec_margin), ("DEC", ec.dec_margin)]:
            assert float(m) >= -1e-6, f"Interior {name} = {float(m)}"

    def test_exterior_ec(self):
        """Exterior (r=25): all EC margins >= -1e-6 (vacuum)."""
        from warpax.energy_conditions import verify_point
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import sshell_default

        curv = compute_curvature_chain(sshell_default(), jnp.array([0.0, 25.0, 0.0, 0.0]))
        ec = verify_point(curv.stress_energy, curv.metric, curv.metric_inv, n_starts=4)
        for name, m in [("NEC", ec.nec_margin), ("WEC", ec.wec_margin),
                        ("SEC", ec.sec_margin), ("DEC", ec.dec_margin)]:
            assert float(m) >= -1e-6, f"Exterior {name} = {float(m)}"

    def test_shell_ec_finite(self):
        """Shell (r=15): all EC margins finite."""
        from warpax.energy_conditions import verify_point
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import sshell_default

        curv = compute_curvature_chain(sshell_default(), jnp.array([0.0, 15.0, 0.0, 0.0]))
        ec = verify_point(curv.stress_energy, curv.metric, curv.metric_inv, n_starts=4)
        for name, m in [("NEC", ec.nec_margin), ("WEC", ec.wec_margin),
                        ("SEC", ec.sec_margin), ("DEC", ec.dec_margin)]:
            assert jnp.isfinite(m), f"Shell {name} not finite"

    def test_shell_he_classification(self):
        """Shell (r=15): Hawking-Ellis type is well-defined."""
        from warpax.energy_conditions import classify_mixed_tensor
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import sshell_default

        curv = compute_curvature_chain(sshell_default(), jnp.array([0.0, 15.0, 0.0, 0.0]))
        cls = classify_mixed_tensor(curv.stress_energy, curv.metric, curv.metric_inv)
        assert int(cls.he_type) in (1, 2, 3, 4)
        assert jnp.all(jnp.isfinite(cls.eigenvalues))


class TestSShellTOV:
    """TOV equilibrium residuals."""

    def test_tov_interior(self):
        """Interior (r=5): vacuum, TOV residual vanishes."""
        from warpax.metrics import sshell_default, constant_density_profiles
        from warpax.tov import tov_residual_from_metric

        profiles = constant_density_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4)
        res = tov_residual_from_metric(
            sshell_default(), jnp.array(5.0),
            profiles.density, profiles.radial_pressure,
            profiles.tangential_pressure,
        )
        assert jnp.isfinite(res)
        assert float(jnp.abs(res)) < 1e-6

    def test_tov_shell(self):
        """Shell (r=15): TOV residual is finite."""
        from warpax.metrics import sshell_default, constant_density_profiles
        from warpax.tov import tov_residual_from_metric

        profiles = constant_density_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4)
        res = tov_residual_from_metric(
            sshell_default(), jnp.array(15.0),
            profiles.density, profiles.radial_pressure,
            profiles.tangential_pressure,
        )
        assert jnp.isfinite(res)


class TestSShellTransport:
    """Geodesic deviation diagnostics."""

    def test_geodesic_deviation_interior(self):
        """Interior (r=5): zero tidal acceleration."""
        from warpax.metrics import sshell_default
        from warpax.transport import geodesic_deviation_diagnostic

        A = geodesic_deviation_diagnostic(sshell_default(), jnp.array([0.0, 5.0, 0.0, 0.0]))
        assert jnp.isfinite(A)
        assert float(A) < 1e-6

    def test_geodesic_deviation_shell(self):
        """Shell (r=15): nonzero tidal acceleration."""
        from warpax.metrics import sshell_default
        from warpax.transport import geodesic_deviation_diagnostic

        A = geodesic_deviation_diagnostic(sshell_default(), jnp.array([0.0, 15.0, 0.0, 0.0]))
        assert jnp.isfinite(A)
        assert float(A) > 1e-15

    def test_shift_changes_deviation(self):
        """Adding shift (v_s=0.02) produces a finite geodesic deviation."""
        from warpax.metrics import sshell_default
        from warpax.transport import geodesic_deviation_diagnostic

        A_static = geodesic_deviation_diagnostic(
            sshell_default(v_s=0.0), jnp.array([0.0, 15.0, 0.0, 0.0]),
        )
        A_shifted = geodesic_deviation_diagnostic(
            sshell_default(v_s=0.02), jnp.array([0.0, 15.0, 0.0, 0.0]),
        )
        assert jnp.isfinite(A_static)
        assert jnp.isfinite(A_shifted)
        # Shift should change the deviation (not necessarily increase it)
        assert not jnp.allclose(A_static, A_shifted, atol=1e-12)


class TestSShellProfileComparison:
    """Compare different density profiles."""

    def test_parabolic_produces_valid_metric(self):
        """Parabolic density produces a finite, well-formed metric."""
        from warpax.metrics import sshell_from_profiles, parabolic_density_profiles

        profiles = parabolic_density_profiles(R_1=10.0, R_2=20.0, rho_max=1e-4)
        metric = sshell_from_profiles(profiles)
        g = metric(jnp.array([0.0, 15.0, 0.0, 0.0]))
        assert g.shape == (4, 4)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.allclose(g, g.T, atol=1e-14)
