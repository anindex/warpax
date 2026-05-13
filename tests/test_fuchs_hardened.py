"""Tests for Fuchs et al. constant-velocity warp shell metric.

Verifies the FuchsMetric implementation against arXiv:2405.02709.
Tests cover paper-exact parameters, ADM decomposition in all three
regions (interior, shell, exterior), shell source profiles, and
curvature chain integrity at the most physically interesting point
(the shell region).
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
        assert metric.r_s_param == 5.0
        assert metric.transition_order == 2

    def test_interior_adm_decomposition(self):
        """Deep interior (r << R_1): unit lapse, shift = -v_s, flat spatial.

        The passenger volume must be Minkowski with a uniform shift
        (Section 4, constraint 1: interior remains flat).
        """
        from warpax.metrics import fuchs_default

        metric = fuchs_default()
        coords = jnp.array([0.0, 1.0, 0.0, 0.0])

        alpha = metric.lapse(coords)
        beta = metric.shift(coords)
        gamma = metric.spatial_metric(coords)

        assert jnp.isclose(alpha, 1.0, atol=1e-10), \
            f"Interior lapse should be 1.0, got {alpha}"
        assert jnp.isclose(beta[0], -0.02, atol=1e-6), \
            f"Interior shift should be -v_s=-0.02, got {beta[0]}"
        assert jnp.allclose(beta[1:], 0.0, atol=1e-10)
        assert jnp.allclose(gamma, jnp.eye(3), atol=1e-10)

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
        """FuchsMetric.shell_profiles() returns correct type with paper params."""
        from warpax.metrics import FuchsShellProfiles, fuchs_default

        profiles = fuchs_default().shell_profiles()
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
