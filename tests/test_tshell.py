"""Tests for the T-shell source-first warp shell metric."""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class TestTShellProfiles:
    """Source profile construction and Eulerian projections."""

    def test_constant_velocity_compact_support(self):
        from warpax.metrics import constant_velocity_profiles

        profiles = constant_velocity_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=0.1)
        assert profiles.R_1 == 10.0
        assert profiles.R_2 == 20.0
        assert float(jnp.abs(profiles.velocity_x(jnp.asarray(15.0)))) > 0.05
        assert float(jnp.abs(profiles.velocity_x(jnp.asarray(5.0)))) < 1e-10
        assert float(jnp.abs(profiles.velocity_x(jnp.asarray(25.0)))) < 1e-10

    def test_lorentz_factor_bounds(self):
        from warpax.metrics import constant_velocity_profiles

        profiles = constant_velocity_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=0.1)
        r_vals = jnp.linspace(5.0, 25.0, 50)
        for r in r_vals:
            assert float(profiles.lorentz_factor(jnp.asarray(r))) >= 1.0 - 1e-15
        assert float(profiles.lorentz_factor(jnp.asarray(15.0))) > 1.0 + 1e-6

    def test_eulerian_energy_exceeds_comoving(self):
        from warpax.metrics import constant_velocity_profiles

        profiles = constant_velocity_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=0.1)
        r_mid = jnp.asarray(15.0)
        assert float(profiles.eulerian_energy(r_mid)) >= float(profiles.density(r_mid)) - 1e-15

    def test_momentum_density_compact_support(self):
        from warpax.metrics import constant_velocity_profiles

        profiles = constant_velocity_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=0.1)
        assert float(jnp.abs(profiles.momentum_density_x(jnp.asarray(15.0)))) > 1e-8
        assert float(jnp.abs(profiles.momentum_density_x(jnp.asarray(5.0)))) < 1e-15
        assert float(jnp.abs(profiles.momentum_density_x(jnp.asarray(25.0)))) < 1e-15

    def test_subluminal_enforcement(self):
        from warpax.metrics import constant_velocity_profiles

        profiles = constant_velocity_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=0.9)
        r_vals = jnp.linspace(5.0, 25.0, 50)
        for r in r_vals:
            assert float(jnp.abs(profiles.velocity_x(jnp.asarray(r)))) < 1.0
        with pytest.raises(ValueError, match="superluminal"):
            constant_velocity_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=1.5)

    def test_parabolic_velocity(self):
        from warpax.metrics import parabolic_velocity_profiles

        profiles = parabolic_velocity_profiles(R_1=10.0, R_2=20.0, rho_max=1e-4, v_0=0.1)
        assert profiles.total_mass > 0
        assert float(profiles.velocity_x(jnp.asarray(15.0))) > 0.05
        assert float(jnp.abs(profiles.velocity_x(jnp.asarray(10.0)))) < 1e-10
        assert float(jnp.abs(profiles.velocity_x(jnp.asarray(20.0)))) < 1e-10

    def test_bernstein_velocity(self):
        from warpax.metrics import bernstein_velocity_profiles

        profiles = bernstein_velocity_profiles(R_1=10.0, R_2=20.0, v_0=0.1)
        assert profiles.total_mass > 0
        assert float(jnp.abs(profiles.velocity_x(jnp.asarray(10.0)))) < 1e-10
        assert float(jnp.abs(profiles.velocity_x(jnp.asarray(20.0)))) < 1e-10
        assert float(profiles.velocity_x(jnp.asarray(15.0))) > 0

    def test_parabolic_produces_valid_metric(self):
        from warpax.metrics import tshell_from_profiles, parabolic_velocity_profiles

        profiles = parabolic_velocity_profiles(R_1=10.0, R_2=20.0, rho_max=1e-4, v_0=0.1)
        metric = tshell_from_profiles(profiles)
        g = metric(jnp.array([0.0, 15.0, 0.0, 0.0]))
        assert g.shape == (4, 4)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.allclose(g, g.T, atol=1e-14)
        assert float(jnp.abs(metric.shift(jnp.array([0.0, 15.0, 0.0, 0.0]))[0])) > 1e-8

    def test_bernstein_produces_valid_metric(self):
        from warpax.metrics import tshell_from_profiles, bernstein_velocity_profiles

        profiles = bernstein_velocity_profiles(R_1=10.0, R_2=20.0, v_0=0.1)
        metric = tshell_from_profiles(profiles)
        g = metric(jnp.array([0.0, 15.0, 0.0, 0.0]))
        assert g.shape == (4, 4)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.allclose(g, g.T, atol=1e-14)


class TestTShellConstraintSolver:
    """Constraint solver: source profiles -> potentials (including beta^x)."""

    def test_schwarzschild_exterior(self):
        from warpax.constraints import solve_tshell_potentials
        from warpax.metrics import constant_velocity_profiles

        profiles = constant_velocity_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=0.1)
        pots = solve_tshell_potentials(
            rho=profiles.density, p_r=profiles.radial_pressure,
            v_x=profiles.velocity_x,
            R_1=10.0, R_2=20.0, n_grid=2048,
        )
        M = pots.total_mass
        assert M > 0

        r_test = jnp.asarray(25.0)
        assert float(jnp.abs(pots.Phi_fn(r_test) - 0.5 * jnp.log(1.0 - 2.0 * M / r_test))) < 1e-4
        assert float(jnp.abs(pots.Lambda_fn(r_test) + 0.5 * jnp.log(1.0 - 2.0 * M / r_test))) < 1e-4

    def test_shift_nonzero_in_shell(self):
        from warpax.constraints import solve_tshell_potentials
        from warpax.metrics import constant_velocity_profiles

        profiles = constant_velocity_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=0.1)
        pots = solve_tshell_potentials(
            rho=profiles.density, p_r=profiles.radial_pressure,
            v_x=profiles.velocity_x, R_1=10.0, R_2=20.0,
        )
        assert float(jnp.abs(pots.beta_x_fn(jnp.asarray(15.0)))) > 1e-6
        assert float(jnp.abs(pots.beta_x_fn(jnp.asarray(30.0)))) < float(
            jnp.abs(pots.beta_x_fn(jnp.asarray(15.0)))
        )

    def test_interior_flat(self):
        from warpax.constraints import solve_tshell_potentials
        from warpax.metrics import constant_velocity_profiles

        profiles = constant_velocity_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=0.1)
        pots = solve_tshell_potentials(
            rho=profiles.density, p_r=profiles.radial_pressure,
            v_x=profiles.velocity_x, R_1=10.0, R_2=20.0,
        )
        assert abs(float(pots.Lambda_fn(jnp.asarray(5.0)))) < 1e-6

    def test_potentials_finite(self):
        from warpax.constraints import solve_tshell_potentials
        from warpax.metrics import constant_velocity_profiles

        profiles = constant_velocity_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=0.1)
        pots = solve_tshell_potentials(
            rho=profiles.density, p_r=profiles.radial_pressure,
            v_x=profiles.velocity_x, R_1=10.0, R_2=20.0,
        )
        for grid in [pots.Phi_grid, pots.Lambda_grid, pots.m_grid, pots.beta_x_grid]:
            assert jnp.all(jnp.isfinite(grid))

    def test_trapped_surface_rejection(self):
        from warpax.constraints import solve_tshell_potentials
        from warpax.metrics import constant_velocity_profiles

        profiles = constant_velocity_profiles(R_1=10.0, R_2=20.0, rho_0=1.0, v_0=0.1)
        with pytest.raises(ValueError, match="trapped surface"):
            solve_tshell_potentials(
                rho=profiles.density, p_r=profiles.radial_pressure,
                v_x=profiles.velocity_x, R_1=10.0, R_2=20.0,
            )


class TestTShellMetric:
    """TShellMetric ADM components."""

    def test_metric_shape_and_symmetry(self):
        from warpax.metrics import tshell_default

        metric = tshell_default()
        for r_val in [1.0, 10.0, 15.0, 20.0, 30.0]:
            g = metric(jnp.array([0.0, r_val, 0.0, 0.0]))
            assert g.shape == (4, 4)
            assert g.dtype == jnp.float64
            assert jnp.all(jnp.isfinite(g))
            assert jnp.allclose(g, g.T, atol=1e-14)

    def test_interior_flat_spatial(self):
        from warpax.metrics import tshell_default

        g = tshell_default()(jnp.array([0.0, 1.0, 0.0, 0.0]))
        assert jnp.allclose(g[1:, 1:], jnp.eye(3), atol=1e-4)
        assert float(g[0, 0]) < 0

    def test_exterior_schwarzschild(self):
        from warpax.metrics import tshell_default

        g = tshell_default()(jnp.array([0.0, 25.0, 0.0, 0.0]))
        assert float(g[0, 0]) < 0
        assert float(g[1, 1]) > 1.0 - 1e-6

    def test_lapse_positive(self):
        from warpax.metrics import tshell_default

        metric = tshell_default()
        for r_val in [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
            assert float(metric.lapse(jnp.array([0.0, r_val, 0.0, 0.0]))) > 0

    def test_spatial_metric_positive_definite(self):
        from warpax.metrics import tshell_default

        metric = tshell_default()
        for r_val in [5.0, 15.0, 25.0]:
            gamma = metric.spatial_metric(jnp.array([0.0, r_val, 0.0, 0.0]))
            assert jnp.all(jnp.linalg.eigvalsh(gamma) > 0)

    def test_shift_from_constraint(self):
        from warpax.metrics import tshell_default

        beta = tshell_default().shift(jnp.array([0.0, 15.0, 0.0, 0.0]))
        assert float(jnp.abs(beta[0])) > 1e-6
        assert float(jnp.abs(beta[1])) < 1e-15
        assert float(jnp.abs(beta[2])) < 1e-15


class TestTShellConstraints:
    """ADM constraint residuals."""

    def test_interior(self):
        from warpax.constraints import normalized_residuals
        from warpax.metrics import tshell_default

        res = normalized_residuals(tshell_default(), jnp.array([0.0, 5.0, 0.0, 0.0]))
        assert float(res["epsilon_H"]) < 1e-3
        assert float(res["epsilon_M"]) < 1e-2

    def test_shell(self):
        from warpax.constraints import normalized_residuals
        from warpax.metrics import tshell_default

        res = normalized_residuals(tshell_default(), jnp.array([0.0, 15.0, 0.0, 0.0]))
        assert jnp.isfinite(res["epsilon_H"])
        assert jnp.isfinite(res["epsilon_M"])
        assert float(res["epsilon_H"]) < 0.1

    def test_exterior(self):
        from warpax.constraints import normalized_residuals
        from warpax.metrics import tshell_default

        res = normalized_residuals(tshell_default(), jnp.array([0.0, 25.0, 0.0, 0.0]))
        assert float(res["epsilon_H"]) < 1e-3
        assert float(res["epsilon_M"]) < 1e-2


class TestTShellEC:
    """Energy condition evaluation."""

    def test_interior_ec_finite(self):
        """Interior: EC margins finite. NEC near zero; WEC/DEC may be
        slightly negative from shift-field curvature in vacuum region."""
        from warpax.energy_conditions import verify_point
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import tshell_default

        curv = compute_curvature_chain(tshell_default(), jnp.array([0.0, 5.0, 0.0, 0.0]))
        ec = verify_point(curv.stress_energy, curv.metric, curv.metric_inv, n_starts=4)
        for m in [ec.nec_margin, ec.wec_margin, ec.sec_margin, ec.dec_margin]:
            assert jnp.isfinite(m)
        assert float(jnp.abs(ec.nec_margin)) < 1e-3

    def test_exterior_ec_finite(self):
        """Exterior: same shift-curvature effect as interior."""
        from warpax.energy_conditions import verify_point
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import tshell_default

        curv = compute_curvature_chain(tshell_default(), jnp.array([0.0, 25.0, 0.0, 0.0]))
        ec = verify_point(curv.stress_energy, curv.metric, curv.metric_inv, n_starts=4)
        for m in [ec.nec_margin, ec.wec_margin, ec.sec_margin, ec.dec_margin]:
            assert jnp.isfinite(m)
        assert float(jnp.abs(ec.nec_margin)) < 1e-3

    def test_shell_ec_finite(self):
        from warpax.energy_conditions import verify_point
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import tshell_default

        curv = compute_curvature_chain(tshell_default(), jnp.array([0.0, 15.0, 0.0, 0.0]))
        ec = verify_point(curv.stress_energy, curv.metric, curv.metric_inv, n_starts=4)
        for m in [ec.nec_margin, ec.wec_margin, ec.sec_margin, ec.dec_margin]:
            assert jnp.isfinite(m)

    def test_shell_he_type(self):
        from warpax.energy_conditions import classify_mixed_tensor
        from warpax.geometry import compute_curvature_chain
        from warpax.metrics import tshell_default

        curv = compute_curvature_chain(tshell_default(), jnp.array([0.0, 15.0, 0.0, 0.0]))
        cls = classify_mixed_tensor(curv.stress_energy, curv.metric, curv.metric_inv)
        assert int(cls.he_type) in (1, 2, 3, 4)
        assert jnp.all(jnp.isfinite(cls.eigenvalues))


class TestTShellTransport:
    """Geodesic deviation diagnostics."""

    def test_deviation_interior(self):
        from warpax.metrics import tshell_default
        from warpax.transport import geodesic_deviation_diagnostic

        A = geodesic_deviation_diagnostic(tshell_default(), jnp.array([0.0, 5.0, 0.0, 0.0]))
        assert jnp.isfinite(A)

    def test_deviation_shell(self):
        from warpax.metrics import tshell_default
        from warpax.transport import geodesic_deviation_diagnostic

        A = geodesic_deviation_diagnostic(tshell_default(), jnp.array([0.0, 15.0, 0.0, 0.0]))
        assert jnp.isfinite(A)
        assert float(A) > 1e-15

    def test_differs_from_sshell(self):
        from warpax.metrics import sshell_default, tshell_default
        from warpax.transport import geodesic_deviation_diagnostic

        coords = jnp.array([0.0, 15.0, 0.0, 0.0])
        A_s = geodesic_deviation_diagnostic(sshell_default(v_s=0.0), coords)
        A_t = geodesic_deviation_diagnostic(tshell_default(v_0=0.1), coords)
        assert jnp.isfinite(A_s) and jnp.isfinite(A_t)
        assert not jnp.allclose(A_s, A_t, atol=1e-10)


class TestTShellVsSShell:
    """Limiting cases and scaling."""

    def test_zero_velocity_mass_match(self):
        from warpax.metrics import constant_velocity_profiles, constant_density_profiles

        sshell = constant_density_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4)
        tshell = constant_velocity_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=1e-6)
        assert abs(sshell.total_mass - tshell.total_mass) / sshell.total_mass < 1e-6

    def test_small_velocity_constraints(self):
        from warpax.constraints import normalized_residuals
        from warpax.metrics import sshell_default, tshell_default

        coords = jnp.array([0.0, 15.0, 0.0, 0.0])
        res_s = normalized_residuals(sshell_default(v_s=0.0), coords)
        res_t = normalized_residuals(tshell_default(v_0=0.01), coords)
        assert float(res_t["epsilon_H"]) < 10.0 * max(float(res_s["epsilon_H"]), 1e-6)

    def test_momentum_density_linear_scaling(self):
        from warpax.metrics import constant_velocity_profiles

        p1 = constant_velocity_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=0.01)
        p2 = constant_velocity_profiles(R_1=10.0, R_2=20.0, rho_0=1e-4, v_0=0.02)
        r_mid = jnp.asarray(15.0)
        ratio = float(p2.momentum_density_x(r_mid)) / float(p1.momentum_density_x(r_mid))
        assert abs(ratio - 2.0) < 0.1
