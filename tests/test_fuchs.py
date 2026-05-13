"""Tests for the Fuchs constant-velocity warp shell (arXiv:2405.02709).

Covers the warpax EC pipeline, constraint residuals, source consistency,
TOV equilibrium, transport diagnostics, ADM mass, and falloff on the
Fuchs shell. Also runs Rodal and Lentz through the same diagnostics
for comparison.
"""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


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

    def test_shell_constraints_finite(self):
        """Shell (r=15): constraints are finite."""
        from warpax.constraints import normalized_residuals
        from warpax.metrics import fuchs_default

        res = normalized_residuals(
            fuchs_default(), jnp.array([0.0, 15.0, 0.0, 0.0]),
        )
        assert jnp.isfinite(res["epsilon_H"])
        assert jnp.isfinite(res["epsilon_M"])

    def test_exterior_constraints_vanish(self):
        """Exterior (r=30): flat Minkowski, constraints vanish."""
        from warpax.constraints import normalized_residuals
        from warpax.metrics import fuchs_default

        res = normalized_residuals(
            fuchs_default(), jnp.array([0.0, 30.0, 0.0, 0.0]),
        )
        assert float(res["epsilon_H"]) < 1e-8
        assert float(res["epsilon_M"]) < 1e-8


# -- Source consistency --------------------------------------------------


class TestFuchsSourceConsistency:
    """Input-vs-derived stress-energy comparison."""

    def test_input_stress_energy_construction(self):
        """T_input is symmetric (4,4) float64; nonzero at shell, zero outside."""
        from warpax.metrics import fuchs_default, fuchs_input_stress_energy

        metric = fuchs_default()

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
        from warpax.metrics import fuchs_default, fuchs_input_stress_energy

        metric = fuchs_default()
        coords = jnp.array([0.0, 30.0, 0.0, 0.0])
        T_input = fuchs_input_stress_energy(metric, coords)

        sc = stress_energy_residual(metric, coords, T_input=T_input)
        assert float(sc["max_residual"]) < 1e-8

    def test_shell_source_consistency_finite(self):
        """Shell (r=15): source consistency residual is finite.

        Nonzero residual expected from the isotropic pressure approximation.
        """
        from warpax.constraints import stress_energy_residual
        from warpax.metrics import fuchs_default, fuchs_input_stress_energy

        metric = fuchs_default()
        coords = jnp.array([0.0, 15.0, 0.0, 0.0])
        T_input = fuchs_input_stress_energy(metric, coords)

        sc = stress_energy_residual(metric, coords, T_input=T_input)
        assert jnp.isfinite(sc["max_residual"])
        assert jnp.isfinite(sc["relative_residual"])


# -- TOV residuals -------------------------------------------------------


class TestFuchsTOV:
    """Anisotropic TOV equilibrium residuals on the Fuchs shell."""

    def test_tov_interior_vanishes(self):
        """Interior (r=1): zero density/pressure -> TOV residual vanishes."""
        from warpax.metrics import fuchs_default
        from warpax.tov import tov_residual_from_metric

        metric = fuchs_default()
        profiles = metric.shell_profiles()
        res = tov_residual_from_metric(
            metric, jnp.array(1.0),
            profiles.density, profiles.radial_pressure,
            profiles.tangential_pressure,
        )
        assert jnp.isfinite(res)
        assert float(jnp.abs(res)) < 1e-8

    def test_tov_shell_finite(self):
        """Shell (r=15): TOV residual is finite.

        Nonzero residual expected from the isotropic pressure approximation.
        """
        from warpax.metrics import fuchs_default
        from warpax.tov import tov_residual_from_metric

        metric = fuchs_default()
        profiles = metric.shell_profiles()
        res = tov_residual_from_metric(
            metric, jnp.array(15.0),
            profiles.density, profiles.radial_pressure,
            profiles.tangential_pressure,
        )
        assert jnp.isfinite(res)

    def test_tov_exterior_vanishes(self):
        """Exterior (r=30): vacuum -> TOV residual vanishes."""
        from warpax.metrics import fuchs_default
        from warpax.tov import tov_residual_from_metric

        metric = fuchs_default()
        profiles = metric.shell_profiles()
        res = tov_residual_from_metric(
            metric, jnp.array(30.0),
            profiles.density, profiles.radial_pressure,
            profiles.tangential_pressure,
        )
        assert jnp.isfinite(res)
        assert float(jnp.abs(res)) < 1e-8

    def test_tov_radial_sweep(self):
        """TOV residual is finite at all points through the shell."""
        from warpax.metrics import fuchs_default
        from warpax.tov import tov_residual_from_metric

        metric = fuchs_default()
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

    @pytest.mark.slow
    def test_blueshift_shell_finite(self):
        """Shell (r=15): blueshift hazard is finite."""
        from warpax.metrics import fuchs_default
        from warpax.transport import blueshift_hazard

        metric = fuchs_default()
        B = blueshift_hazard(
            metric, jnp.array([0.0, 15.0, 0.0, 0.0]),
            n_directions=2, tau_max=5.0, num_points=50,
        )
        assert jnp.isfinite(B)


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

    def test_falloff(self):
        """Metric falloff at r=50 is consistent with asymptotic flatness."""
        from warpax.adm import falloff_check
        from warpax.metrics import fuchs_default

        results = falloff_check(fuchs_default(), r_test=50.0)
        for name, passed in results.items():
            assert passed, f"Falloff failed for {name}"

    def test_asymptotic_flatness_report(self):
        """Full asymptotic flatness report passes."""
        from warpax.adm import asymptotic_flatness_report
        from warpax.metrics import fuchs_default

        report = asymptotic_flatness_report(
            fuchs_default(), radii=[30.0, 50.0, 100.0],
        )
        assert report["is_asymptotically_flat"]


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
