"""Schwarzschild geodesic validation and warp drive smoke tests.

Tests the core geodesic integrator against known analytical solutions
for Schwarzschild spacetime and verifies NaN-free integration for all
warp drive metrics.

Schwarzschild uses isotropic Cartesian coordinates in this project.
All analytical reference values are converted to isotropic coordinates
before comparison.

"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks import (
    AlcubierreMetric,
    MinkowskiMetric,
    SchwarzschildMetric,
)
from warpax.geodesics import (
    GeodesicResult,
    bounding_box_event,
    circular_orbit_ic,
    horizon_event,
    integrate_geodesic,
    integrate_geodesic_family,
    make_event,
    monitor_conservation,
    null_ic,
    radial_infall_ic,
    timelike_ic,
)
from warpax.metrics import (
    LentzMetric,
    NatarioMetric,
    RodalMetric,
    VanDenBroeckMetric,
    WarpShellMetric,
)


# ---------------------------------------------------------------------------
# Helper: isotropic <-> Schwarzschild coordinate conversions
# ---------------------------------------------------------------------------


def _schw_r_to_iso(r_schw: float, M: float = 1.0) -> float:
    """Convert Schwarzschild radial coordinate to isotropic."""
    return (r_schw - M + np.sqrt(r_schw**2 - 2 * M * r_schw)) / 2.0


def _iso_to_schw_r(r_iso: float, M: float = 1.0) -> float:
    """Convert isotropic radial coordinate to Schwarzschild."""
    return r_iso * (1 + M / (2 * r_iso)) ** 2


# ===========================================================================
# Schwarzschild validation benchmarks
# ===========================================================================


class TestMinkowskiStraightLine:
    """Test 1: Sanity check straight-line geodesic in flat spacetime."""

    def test_minkowski_straight_line(self):
        """Timelike geodesic in Minkowski should be a straight line."""
        metric = MinkowskiMetric()
        x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
        v_spatial = jnp.array([0.5, 0.0, 0.0])
        x0, v0 = timelike_ic(metric, x0, v_spatial)

        tau_end = 10.0
        sol = integrate_geodesic(metric, x0, v0, tau_span=(0.0, tau_end), num_points=200)

        # Final position should be x0 + v0 * tau_end
        expected_final = x0 + v0 * tau_end
        actual_final = sol.positions[-1]

        np.testing.assert_allclose(
            actual_final, expected_final, atol=1e-8,
            err_msg="Minkowski straight-line geodesic deviated from analytical trajectory",
        )


class TestSchwarzschildCircularOrbit:
    """Test 2: Circular orbit at r_schw=10M.

    Uses circular_orbit_ic which computes the exact 4-velocity from
    Schwarzschild conserved quantities (E, L) converted to isotropic
    Cartesian coordinates.
    """

    def test_circular_orbit_period_and_conservation(self):
        """Circular orbit should return near starting position after one period."""
        M = 1.0
        r_schw = 10.0
        metric = SchwarzschildMetric(M=M)

        x0, v0 = circular_orbit_ic(metric, r_schw=r_schw, M=M)

        # Orbital period in Schwarzschild coordinate time
        T_schw = 2.0 * np.pi * r_schw ** 1.5 / np.sqrt(M)
        # Convert to proper time
        T_proper = T_schw * np.sqrt(1.0 - 3.0 * M / r_schw)

        # Integrate for one full orbital period
        sol = integrate_geodesic(
            metric, x0, v0,
            tau_span=(0.0, T_proper),
            num_points=2000,
            max_steps=65536,
        )

        # Check radial distance oscillation: compute isotropic r at all points
        r_iso_traj = jnp.sqrt(
            sol.positions[:, 1] ** 2
            + sol.positions[:, 2] ** 2
            + sol.positions[:, 3] ** 2
        )
        r_iso_start = float(jnp.sqrt(x0[1] ** 2 + x0[2] ** 2 + x0[3] ** 2))

        # With correct ICs from conserved quantities, radial variation
        # should be extremely small (sub-ppm level).
        max_radial_variation = float(
            jnp.max(jnp.abs(r_iso_traj - r_iso_start)) / r_iso_start
        )
        assert max_radial_variation < 0.01, (
            f"Radial variation {max_radial_variation:.6%} exceeds 1% threshold. "
            f"r_iso_start={r_iso_start:.4f}"
        )

        # Check 4-velocity norm conservation
        norms = monitor_conservation(metric, sol)
        max_norm_error = float(jnp.max(jnp.abs(norms + 1.0)))
        assert max_norm_error < 1e-6, (
            f"4-velocity norm drift {max_norm_error:.2e} exceeds 1e-6 threshold"
        )


class TestSchwarzschildRadialInfall:
    """Test 3: Radial infall from rest conservation and monotonic r decrease."""

    def test_radial_infall_conservation(self):
        """Radial infall should conserve 4-velocity norm and monotonically decrease r."""
        M = 1.0
        r_start_schw = 20.0
        metric = SchwarzschildMetric(M=M)

        x0, v0 = radial_infall_ic(metric, r_start_schw=r_start_schw, M=M)

        # Integrate use horizon event to avoid coordinate singularity
        event = make_event(horizon_event)

        sol = integrate_geodesic(
            metric, x0, v0,
            tau_span=(0.0, 100.0),
            num_points=2000,
            max_steps=32768,
            event=event,
        )

        # Compute isotropic radius along trajectory
        r_iso_traj = jnp.sqrt(
            sol.positions[:, 1] ** 2
            + sol.positions[:, 2] ** 2
            + sol.positions[:, 3] ** 2
        )

        # After event termination, remaining save-points repeat the last valid
        # state. Use the first 500 points which should all be well above the horizon.
        r_valid = r_iso_traj[:500]

        # Verify radius monotonically decreases (with small tolerance for numerics)
        dr = jnp.diff(r_valid)
        # Allow small positive bumps up to 1e-10 (numerical noise)
        n_increasing = int(jnp.sum(dr > 1e-10))
        assert n_increasing == 0, (
            f"Radius increased at {n_increasing} points during radial infall"
        )

        # Check 4-velocity norm conservation (first 500 points)
        norms = monitor_conservation(metric, sol)
        norms_valid = norms[:500]
        max_norm_error = float(jnp.max(jnp.abs(norms_valid + 1.0)))
        assert max_norm_error < 1e-6, (
            f"4-velocity norm drift {max_norm_error:.2e} exceeds 1e-6 during radial infall"
        )


class TestSchwarzschildPhotonSphere:
    """Test 4: Null geodesic at photon sphere (r_schw=3M)."""

    def test_null_photon_sphere_approximately_stable(self):
        """Null geodesic at r=3M should stay approximately circular for short time."""
        M = 1.0
        metric = SchwarzschildMetric(M=M)

        # Photon sphere: r_schw = 3M
        # Convert to isotropic: r_iso = (3 - 1 + sqrt(9 - 6)) / 2 = (2 + sqrt(3)) / 2
        r_schw_ps = 3.0 * M
        r_iso_ps = _schw_r_to_iso(r_schw_ps, M)

        # Position on x-axis, tangential null direction (y-direction)
        x0 = jnp.array([0.0, r_iso_ps, 0.0, 0.0])
        n_spatial = jnp.array([0.0, 1.0, 0.0])  # tangential direction

        x0, k0 = null_ic(metric, x0, n_spatial)

        # Integrate for a short time (unstable orbit)
        sol = integrate_geodesic(
            metric, x0, k0,
            tau_span=(0.0, 5.0),
            num_points=500,
            max_steps=16384,
        )

        # Compute isotropic radius along trajectory
        r_iso_traj = jnp.sqrt(
            sol.positions[:, 1] ** 2
            + sol.positions[:, 2] ** 2
            + sol.positions[:, 3] ** 2
        )

        # For short integration, radius should stay within 10% of the photon sphere
        # (unstable orbit, so it will deviate)
        max_variation = float(jnp.max(jnp.abs(r_iso_traj - r_iso_ps)) / r_iso_ps)
        assert max_variation < 0.10, (
            f"Photon sphere radius variation {max_variation:.4%} exceeds 10% "
            f"threshold for short integration"
        )

        # Null norm conservation
        norms = monitor_conservation(metric, sol)
        max_null_error = float(jnp.max(jnp.abs(norms)))
        assert max_null_error < 1e-6, (
            f"Null norm drift {max_null_error:.2e} exceeds 1e-6 at photon sphere"
        )


class TestGeodesicFamilyBatch:
    """Test 5: Batched geodesic integration via vmap."""

    def test_geodesic_family_batch(self):
        """integrate_geodesic_family should produce correct shapes and trajectories."""
        metric = MinkowskiMetric()

        # 4 different initial conditions in Minkowski
        x0_batch = jnp.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        # Use timelike_ic for proper normalization
        _, v0_0 = timelike_ic(metric, x0_batch[0], jnp.array([0.5, 0.0, 0.0]))
        _, v0_1 = timelike_ic(metric, x0_batch[1], jnp.array([0.0, 0.5, 0.0]))
        _, v0_2 = timelike_ic(metric, x0_batch[2], jnp.array([0.0, 0.0, 0.5]))
        _, v0_3 = timelike_ic(metric, x0_batch[3], jnp.array([0.3, 0.3, 0.0]))

        v0_batch = jnp.stack([v0_0, v0_1, v0_2, v0_3])

        num_pts = 100
        sol = integrate_geodesic_family(
            metric, x0_batch, v0_batch,
            tau_span=(0.0, 5.0),
            num_points=num_pts,
        )

        # Check output shapes
        assert sol.positions.shape == (4, num_pts, 4), (
            f"Expected positions shape (4, {num_pts}, 4), got {sol.positions.shape}"
        )
        assert sol.velocities.shape == (4, num_pts, 4), (
            f"Expected velocities shape (4, {num_pts}, 4), got {sol.velocities.shape}"
        )

        # Verify each trajectory is correct (straight line in Minkowski)
        for i in range(4):
            expected_final = x0_batch[i] + v0_batch[i] * 5.0
            actual_final = sol.positions[i, -1]
            np.testing.assert_allclose(
                actual_final, expected_final, atol=1e-6,
                err_msg=f"Geodesic family member {i} deviates from straight line",
            )


class TestEventDetection:
    """Test 6: Bounding box event termination.

    Note: With Diffrax SaveAt(ts=...), after event termination the
    remaining save-points are filled with the final valid state (not
    extrapolated). We test that the trajectory does not exceed the
    bounding box significantly.
    """

    def test_event_detection_bounding_box(self):
        """Geodesic should terminate when reaching bounding box radius."""
        metric = MinkowskiMetric()
        x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
        # Large outward velocity
        _, v0 = timelike_ic(metric, x0, jnp.array([0.9, 0.0, 0.0]))

        R_max = 50.0

        def bb_event(t, y, args, **kwargs):
            x = y[:4]
            r = jnp.sqrt(x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
            return R_max - r

        event = make_event(bb_event)

        # Use a shorter tau_span so it doesn't run too long.
        # At v=0.9c (spatial), v_x = 0.9 proper, reaching R=50 takes
        # tau ~ 50/0.9 ~ 55.6 proper time.
        sol = integrate_geodesic(
            metric, x0, v0,
            tau_span=(0.0, 80.0),
            num_points=1000,
            event=event,
        )

        # Compute radius along trajectory
        r_traj = jnp.sqrt(
            sol.positions[:, 1] ** 2
            + sol.positions[:, 2] ** 2
            + sol.positions[:, 3] ** 2
        )

        # After event termination, Diffrax fills remaining save-points
        # past the event time with inf. Filter to valid (non-inf) points.
        valid_mask = jnp.isfinite(r_traj)
        r_valid = r_traj[valid_mask]

        # The event should have triggered (some points are inf)
        assert int(jnp.sum(~valid_mask)) > 0, (
            "Event did not trigger all save points are finite"
        )

        # The last valid radius should be near R_max (within a few percent)
        last_valid_r = float(r_valid[-1])
        assert abs(last_valid_r - R_max) / R_max < 0.05, (
            f"Last valid radius {last_valid_r:.1f} not near bounding box {R_max:.1f}"
        )


# ===========================================================================
# Warp drive smoke tests
# ===========================================================================


class TestAlcubierreGeodesicSmoke:
    """Test 7: Alcubierre metric geodesic no NaN/Inf."""

    def test_alcubierre_geodesic_smoke(self):
        """Geodesic in Alcubierre spacetime should produce valid (non-NaN) output."""
        metric = AlcubierreMetric(v_s=0.1, R=1.0, sigma=8.0)
        # Start well outside the bubble
        x0 = jnp.array([0.0, 5.0, 0.0, 0.0])
        _, v0 = timelike_ic(metric, x0, jnp.array([0.0, 0.0, 0.0]))

        sol = integrate_geodesic(
            metric, x0, v0,
            tau_span=(0.0, 1.0),
            num_points=100,
            max_steps=4096,
        )

        # Verify no NaN or Inf
        assert not jnp.any(jnp.isnan(sol.positions)), (
            "Alcubierre geodesic produced NaN in positions"
        )
        assert not jnp.any(jnp.isinf(sol.positions)), (
            "Alcubierre geodesic produced Inf in positions"
        )
        assert not jnp.any(jnp.isnan(sol.velocities)), (
            "Alcubierre geodesic produced NaN in velocities"
        )


class TestWarpMetricsGeodesicSmoke:
    """Test 8: Parametrized warp drive smoke tests.

    All warp metrics are tested with small bubble radius (R=1.0) and a
    starting point well outside the bubble at (0, 5, 0.1, 0).

    The y=0.1 offset avoids the y=z=0 axis where some metrics (Rodal,
    Lentz) have coordinate singularities from spherical-to-Cartesian
    conversion or L1 norm non-differentiability. This is a known
    limitation of the Cartesian formulation, not a bug.
    """

    @pytest.mark.parametrize(
        "metric_cls,kwargs",
        [
            (RodalMetric, dict(v_s=0.1, R=1.0, sigma=8.0)),
            (NatarioMetric, dict(v_s=0.1, R=1.0, sigma=8.0)),
            (VanDenBroeckMetric, dict(v_s=0.1, R=1.0, sigma=8.0)),
            (LentzMetric, dict(v_s=0.1, R=1.0, sigma=8.0)),
            (WarpShellMetric, dict(v_s=0.1, R_1=0.5, R_2=1.5)),
        ],
        ids=["Rodal", "Natario", "VanDenBroeck", "Lentz", "WarpShell"],
    )
    def test_warp_metric_geodesic_no_nan(self, metric_cls, kwargs):
        """Each warp metric should produce NaN-free geodesics."""
        metric = metric_cls(**kwargs)
        # Start well outside the bubble with slight y-offset to avoid
        # axis singularity in spherical-to-Cartesian conversion
        x0 = jnp.array([0.0, 5.0, 0.1, 0.0])
        _, v0 = timelike_ic(metric, x0, jnp.array([0.0, 0.0, 0.0]))

        sol = integrate_geodesic(
            metric, x0, v0,
            tau_span=(0.0, 1.0),
            num_points=100,
            max_steps=4096,
        )

        assert not jnp.any(jnp.isnan(sol.positions)), (
            f"{metric_cls.__name__} geodesic produced NaN in positions"
        )
        assert not jnp.any(jnp.isinf(sol.positions)), (
            f"{metric_cls.__name__} geodesic produced Inf in positions"
        )
        assert not jnp.any(jnp.isnan(sol.velocities)), (
            f"{metric_cls.__name__} geodesic produced NaN in velocities"
        )
