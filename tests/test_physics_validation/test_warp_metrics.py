"""Physics validation tests for warp drive metrics against published literature.

Validates:
  Natario zero-expansion property (Natario 2001)
  Rodal globally Hawking-Ellis Type I (Rodal 2025)
  Rodal energy deficit vs Alcubierre (Rodal 2025)
  Lentz WEC violation at bubble wall (Celmaster-Rubin 2025)
  Santiago observer-dependence theorem (Santiago-Schuster-Visser 2022)
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from warpax.benchmarks import AlcubierreMetric
from warpax.metrics import NatarioMetric, RodalMetric, LentzMetric
from warpax.geometry import (
    GridSpec,
    compute_curvature_chain,
    evaluate_curvature_grid,
    kretschner_scalar,
)
from warpax.analysis import compute_kinematic_scalars, compute_kinematic_scalars_grid
from warpax.energy_conditions import (
    classify_hawking_ellis,
    compute_eulerian_ec,
    verify_point,
)


# =========================================================================
# Natario zero-expansion property
# =========================================================================


class TestNatarioZeroExpansion:
    """Natario (2001): zero-expansion warp drive by construction.

    The Natario metric has theta = -K = 0 everywhere, meaning no spatial
    volume changes. This is the defining property of the Natario metric.
    """

    @pytest.mark.slow
    def test_natario_zero_expansion_grid_50_cubed(self):
        """Natario expansion theta = 0 across a 50^3 grid (125,000 points).

        Natario (2001): zero-expansion warp drive by construction.
        theta = -K = 0 everywhere.
        """
        metric = NatarioMetric(v_s=0.1, R=1.0, sigma=8.0)
        grid = GridSpec(
            bounds=[(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
            shape=(50, 50, 50),
        )
        theta_grid, sigma_sq_grid, omega_sq_grid = compute_kinematic_scalars_grid(
            metric, grid, t=0.0, batch_size=5000
        )
        npt.assert_allclose(
            np.asarray(theta_grid),
            0.0,
            atol=1e-8,
            err_msg="Natario expansion theta should be zero everywhere",
        )

    def test_natario_zero_expansion_pointwise(self):
        """Natario expansion theta = 0 at 10 specific points.

        Tests bubble center, wall region, and exterior. Tighter tolerance
        (1e-10) than the grid test since no vmap accumulation.
        """
        metric = NatarioMetric(v_s=0.1, R=1.0, sigma=8.0)
        test_points = [
            jnp.array([0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.5, 0.0, 0.0]),
            jnp.array([0.0, 1.0, 0.0, 0.0]),
            jnp.array([0.0, 1.0, 0.5, 0.0]),
            jnp.array([0.0, 0.0, 1.0, 0.5]),
            jnp.array([0.0, 2.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 2.0]),
            jnp.array([0.0, 1.5, 1.5, 0.0]),
            jnp.array([0.0, -1.0, 0.5, 0.3]),
            jnp.array([0.0, 2.5, 0.0, 0.0]),
        ]
        for coords in test_points:
            theta, sigma_sq, omega_sq = compute_kinematic_scalars(metric, coords)
            npt.assert_allclose(
                float(theta),
                0.0,
                atol=1e-10,
                err_msg=f"Natario theta != 0 at coords {coords}",
            )

    def test_natario_vorticity_zero(self):
        """Eulerian vorticity omega^2 = 0 across a 50^3 grid.

        Eulerian congruence is hypersurface-orthogonal: omega = 0
        by Frobenius theorem.
        """
        metric = NatarioMetric(v_s=0.1, R=1.0, sigma=8.0)
        grid = GridSpec(
            bounds=[(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
            shape=(50, 50, 50),
        )
        theta_grid, sigma_sq_grid, omega_sq_grid = compute_kinematic_scalars_grid(
            metric, grid, t=0.0, batch_size=5000
        )
        npt.assert_allclose(
            np.asarray(omega_sq_grid),
            0.0,
            atol=1e-10,
            err_msg="Eulerian vorticity should be identically zero",
        )


# =========================================================================
# Rodal globally Hawking-Ellis Type I
# =========================================================================


class TestRodalTypeI:
    """Rodal (2025): stress-energy is globally Type I (perfect fluid form)."""

    def test_rodal_globally_type_i(self):
        """Rodal metric produces globally Hawking-Ellis Type I on 8^3 grid.

        Rodal (2025): stress-energy is globally Type I (perfect fluid form).
        Points in the flat exterior have T_ab = 0, which classifies as
        Type I trivially (rho=0, p_i=0). All non-trivial points with
        |T_ab| > threshold should also be Type I.
        """
        metric = RodalMetric(v_s=0.1, R=100.0, sigma=0.03)
        grid = GridSpec(
            bounds=[(-200, 200)] * 3,
            shape=(8, 8, 8),
        )
        result = evaluate_curvature_grid(metric, grid, compute_invariants=False)

        # Flatten grid arrays for classification
        n_points = 8 * 8 * 8
        flat_T = result.stress_energy.reshape(n_points, 4, 4)
        flat_g = result.metric.reshape(n_points, 4, 4)
        flat_g_inv = result.metric_inv.reshape(n_points, 4, 4)

        # Classify all points via vmap
        flat_T_mixed = jax.vmap(jnp.matmul)(flat_g_inv, flat_T)
        cls_results = jax.vmap(classify_hawking_ellis)(flat_T_mixed, flat_g)

        he_types = np.asarray(cls_results.he_type)

        # All points should be Type I (he_type == 1)
        assert np.all(he_types == 1.0), (
            f"Found non-Type-I points: unique types = {np.unique(he_types)}, "
            f"counts = {dict(zip(*np.unique(he_types, return_counts=True)))}"
        )


# =========================================================================
# Rodal energy deficit vs Alcubierre
# =========================================================================


class TestRodalEnergyDeficit:
    """Rodal (2025): peak energy deficit ~38x smaller than Alcubierre at matched params."""

    def test_rodal_energy_deficit_vs_alcubierre(self):
        """Rodal peak energy deficit is 10x-100x smaller than Alcubierre.

        Rodal (2025) claims ~38x peak energy deficit reduction vs Alcubierre
        at matched parameters. We accept the order-of-magnitude range [10, 100].

        Uses MATCHED parameters: v_s=0.1, R=100.0, sigma=0.20.
        Note: sigma=0.20 produces a moderately sharp wall (R*sigma=20) where
        the Rodal improvement is most pronounced. The Rodal paper's ~38x claim
        corresponds to this wall-sharpness regime. Very diffuse walls (sigma=0.03)
        show less improvement because both metrics approach zero curvature.

        Sampling strategy: scan radii along the equatorial plane (x=0) where
        Alcubierre energy density is maximal (sin^2(theta)=1), finding the peak
        violation for each metric independently.
        """
        v_s, R, sigma = 0.1, 100.0, 0.20
        rodal = RodalMetric(v_s=v_s, R=R, sigma=sigma)
        alcubierre = AlcubierreMetric(v_s=v_s, R=R, sigma=sigma, x_s=0.0)

        # Scan radii along the equatorial plane (x=0, y=r, z=0)
        # to find peak WEC violation for each metric independently.
        # The Alcubierre peak is near r=R; Rodal peak may shift outward.
        r_values = np.linspace(R * 0.5, R * 1.5, 40)

        peak_rho_rodal = 0.0
        peak_rho_alcubierre = 0.0

        for r_test in r_values:
            coords = jnp.array([0.0, 0.0, float(r_test), 0.0])

            # Rodal
            result_r = compute_curvature_chain(rodal, coords)
            euler_r = compute_eulerian_ec(result_r.stress_energy, result_r.metric)
            val_r = float(euler_r["wec"])
            if val_r < peak_rho_rodal:
                peak_rho_rodal = val_r

            # Alcubierre
            result_a = compute_curvature_chain(alcubierre, coords)
            euler_a = compute_eulerian_ec(result_a.stress_energy, result_a.metric)
            val_a = float(euler_a["wec"])
            if val_a < peak_rho_alcubierre:
                peak_rho_alcubierre = val_a

        # Both should be negative (energy condition violations)
        assert peak_rho_rodal < 0, f"Rodal peak rho = {peak_rho_rodal}, expected < 0"
        assert peak_rho_alcubierre < 0, (
            f"Alcubierre peak rho = {peak_rho_alcubierre}, expected < 0"
        )

        # Ratio: both negative, so ratio > 1 means Rodal has smaller violation
        reduction_factor = peak_rho_alcubierre / peak_rho_rodal
        assert 10.0 < reduction_factor < 100.0, (
            f"Reduction factor = {reduction_factor:.2f}, expected in [10, 100]. "
            f"Rodal peak = {peak_rho_rodal:.6e}, Alcubierre peak = {peak_rho_alcubierre:.6e}"
        )


# =========================================================================
# Lentz WEC violation at bubble wall
# =========================================================================


class TestLentzWECViolation:
    """Celmaster-Rubin (2025): Lentz metric does NOT satisfy WEC despite original claims."""

    def test_lentz_eulerian_wec_violated_at_wall(self):
        """Lentz Eulerian energy density is negative at bubble wall.

        Celmaster-Rubin (2025): Lentz metric does NOT satisfy WEC despite
        original claims. Eulerian energy density is negative at bubble wall.

        Tests at the wall_diagonal point from existing regression test,
        plus additional wall points.
        """
        metric = LentzMetric(v_s=0.1, R=100.0, sigma=8.0)

        wall_points = [
            jnp.array([0.0, 50.0, 50.0, 0.0]),   # wall diagonal (from regression)
            jnp.array([0.0, 80.0, 20.0, 0.0]),    # wall off-diagonal
            jnp.array([0.0, 30.0, 70.0, 0.0]),    # wall off-diagonal
        ]

        for coords in wall_points:
            result = compute_curvature_chain(metric, coords)
            rho_euler = compute_eulerian_ec(
                result.stress_energy, result.metric
            )["wec"]
            assert float(rho_euler) < 0, (
                f"Lentz WEC should be violated at {coords}: "
                f"rho_euler = {float(rho_euler):.6e}"
            )

    def test_lentz_wec_satisfied_interior_and_exterior(self):
        """Lentz rho_euler = 0 (trivially satisfied) deep inside and far outside.

        The interior is flat Minkowski (T_ab = 0) and the exterior is also
        flat. WEC is satisfied trivially at these locations.
        """
        metric = LentzMetric(v_s=0.1, R=100.0, sigma=8.0)

        trivial_points = [
            jnp.array([0.0, 5.0, 5.0, 0.0]),       # deep inside bubble
            jnp.array([0.0, 200.0, 200.0, 0.0]),    # far outside bubble
        ]

        for coords in trivial_points:
            result = compute_curvature_chain(metric, coords)
            rho_euler = compute_eulerian_ec(
                result.stress_energy, result.metric
            )["wec"]
            assert abs(float(rho_euler)) < 1e-8, (
                f"Lentz rho_euler should be ~0 at {coords}: "
                f"rho_euler = {float(rho_euler):.6e}"
            )


# =========================================================================
# Santiago observer-dependence theorem
# =========================================================================


class TestSantiagoObserverDependence:
    """Santiago-Schuster-Visser (2022): observer-dependent NEC violations.

    For nonzero-expansion warp metrics, NEC is violated for some observer
    at every point where expansion is nonzero. This is the paper's core
    demonstration: Eulerian-frame-only checks are insufficient.
    """

    @pytest.mark.slow
    def test_santiago_alcubierre_nec_violated_at_wall(self):
        """NEC violated for some observer at all 20 sampled Alcubierre wall points.

        Santiago et al. (2022): For nonzero-expansion warp metrics, NEC is
        violated for some observer at every point with theta != 0. Alcubierre
        has nonzero expansion everywhere on the bubble wall.
        """
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0, x_s=0.0)

        n_points = 20
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        # Points on a ring in the x-y plane at distance ~R from center
        wall_points = [
            jnp.array([0.0, 1.0 * np.cos(a), 1.0 * np.sin(a), 0.0])
            for a in angles
        ]

        n_violated = 0
        for coords in wall_points:
            result = compute_curvature_chain(metric, coords)
            ec_result = verify_point(
                result.stress_energy, result.metric, n_starts=16,
            )
            if float(ec_result.nec_margin) < 0:
                n_violated += 1

        assert n_violated == n_points, (
            f"Santiago theorem: NEC should be violated at all {n_points} wall "
            f"points, but only {n_violated}/{n_points} show violation."
        )

    @pytest.mark.slow
    def test_santiago_rodal_nec_violated_at_wall(self):
        """NEC violated for some observer at all 10 sampled Rodal wall points.

        Rodal metric also has nonzero expansion, so Santiago theorem applies.
        Uses R=1.0, sigma=8.0 for compact bubble (fast evaluation).
        """
        metric = RodalMetric(v_s=0.5, R=1.0, sigma=8.0)

        n_points = 10
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        # Points on equatorial ring at r ~ R, avoiding on-axis (sin(a)!=0)
        # Offset z=0.1 to stay off the coordinate axis
        wall_points = [
            jnp.array([0.0, 1.0 * np.cos(a), 1.0 * np.sin(a), 0.1])
            for a in angles
        ]

        n_violated = 0
        for coords in wall_points:
            result = compute_curvature_chain(metric, coords)
            ec_result = verify_point(
                result.stress_energy, result.metric, n_starts=16,
            )
            if float(ec_result.nec_margin) < 0:
                n_violated += 1

        assert n_violated == n_points, (
            f"Santiago theorem: NEC should be violated at all {n_points} Rodal "
            f"wall points, but only {n_violated}/{n_points} show violation."
        )

    @pytest.mark.slow
    def test_santiago_eulerian_may_miss_violations(self):
        """Observer-robust NEC margin is worse than Eulerian at majority of points.

        Core paper result: observer-robust NEC margin is worse (more negative)
        than Eulerian NEC margin, demonstrating that Eulerian-only checks are
        insufficient for detecting all energy condition violations.
        """
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0, x_s=0.0)

        # Use 5 wall points (subset of the 20-point ring)
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        wall_points = [
            jnp.array([0.0, 1.0 * np.cos(a), 1.0 * np.sin(a), 0.0])
            for a in angles
        ]

        n_robust_worse = 0
        for coords in wall_points:
            result = compute_curvature_chain(metric, coords)

            # Observer-robust NEC margin
            ec_result = verify_point(
                result.stress_energy, result.metric, n_starts=16,
            )
            robust_nec = float(ec_result.nec_margin)

            # Eulerian-only NEC margin
            euler_ec = compute_eulerian_ec(result.stress_energy, result.metric)
            euler_nec = float(euler_ec["nec"])

            # Observer-robust should find worse (more negative) violations
            if robust_nec < euler_nec:
                n_robust_worse += 1

        # Majority (>= 3 out of 5) should show observer-robust is worse
        assert n_robust_worse >= 3, (
            f"Observer-robust NEC should be worse than Eulerian at majority "
            f"of points, but only {n_robust_worse}/5 showed this."
        )
