"""Geodesic integrator, deviation/tidal eigenvalues, blueshift / observables."""

from __future__ import annotations
from warpax.benchmarks import (
    AlcubierreMetric,
    MinkowskiMetric,
    SchwarzschildMetric,
)
from warpax.geodesics import (
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
from warpax.geodesics import (
    compute_blueshift,
    integrate_geodesic_with_deviation,
    tidal_eigenvalues,
)
from warpax.geodesics import tidal_tensor
from warpax.metrics import (
    LentzMetric,
    NatarioMetric,
    RodalMetric,
    VanDenBroeckMetric,
    WarpShellMetric,
)
import jax.numpy as jnp
import numpy as np
import pytest



# Helper: isotropic <-> Schwarzschild coordinate conversions


def _schw_r_to_iso(r_schw: float, M: float = 1.0) -> float:
    """Convert Schwarzschild radial coordinate to isotropic."""
    return (r_schw - M + np.sqrt(r_schw**2 - 2 * M * r_schw)) / 2.0


def _iso_to_schw_r(r_iso: float, M: float = 1.0) -> float:
    """Convert isotropic radial coordinate to Schwarzschild."""
    return r_iso * (1 + M / (2 * r_iso)) ** 2


# Schwarzschild validation benchmarks


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
    remaining save-points are filled with inf. We filter to the finite
    points and check the last valid radius sits near the bounding box.
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


# Warp drive smoke tests


class TestWarpMetricsGeodesicSmoke:
    """Test 7: Parametrized warp drive smoke tests.

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
            (AlcubierreMetric, dict(v_s=0.1, R=1.0, sigma=8.0)),
            (RodalMetric, dict(v_s=0.1, R=1.0, sigma=8.0)),
            (NatarioMetric, dict(v_s=0.1, R=1.0, sigma=8.0)),
            (VanDenBroeckMetric, dict(v_s=0.1, R=1.0, sigma=8.0)),
            (LentzMetric, dict(v_s=0.1, R=1.0, sigma=8.0)),
            (WarpShellMetric, dict(v_s=0.1, R_1=0.5, R_2=1.5)),
        ],
        ids=["Alcubierre", "Rodal", "Natario", "VanDenBroeck", "Lentz", "WarpShell"],
    )
    def test_warp_metric_geodesic_no_nan(self, metric_cls, kwargs):
        """Each warp metric should produce NaN-free, norm-conserving geodesics."""
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

        # IC is timelike, so g(v,v) should hold at -1 along the trajectory
        norms = monitor_conservation(metric, sol)
        assert float(jnp.max(jnp.abs(norms + 1.0))) < 1e-6, (
            f"{metric_cls.__name__} 4-velocity norm drifted off -1"
        )


# Tidal force tests


class TestSchwarzschildTidalEigenvalues:
    """Test 1: Tidal eigenvalues at a Schwarzschild point.

    Analytical tidal eigenvalues for a static observer at Schwarzschild r:
        - Radial: -2M/r_schw^3 (stretching)
        - Transverse: +M/r_schw^3 each (compression, x2)
        - Zero eigenvalue from time direction

    The tidal tensor K^mu_rho = R^mu_{nu rho sigma} v^nu v^sigma is
    computed via nested JAX autodiff of the isotropic Schwarzschild metric.
    """

    def test_schwarzschild_tidal_eigenvalues(self):
        """Tidal eigenvalues should match -2M/r^3 and +M/r^3 analytical values."""
        M = 1.0
        metric = SchwarzschildMetric(M=M)

        # Use r_iso=10 as test point (well outside horizon)
        r_iso = 10.0
        r_schw = _iso_to_schw_r(r_iso, M)

        # Position on x-axis
        x = jnp.array([0.0, r_iso, 0.0, 0.0])

        # Static observer 4-velocity: v = (1/sqrt(-g_00), 0, 0, 0)
        # For Schwarzschild: g_00 = -alpha^2 = -((1-M/(2r_iso))/(1+M/(2r_iso)))^2
        g = metric(x)
        v_t = 1.0 / jnp.sqrt(-g[0, 0])
        v = jnp.array([v_t, 0.0, 0.0, 0.0])

        # Compute tidal eigenvalues (sorted ascending)
        eigs = tidal_eigenvalues(metric, x, v)
        eigs_np = np.array(eigs)

        # Analytical values (in Schwarzschild coords, coordinate-invariant)
        radial_eig = -2.0 * M / r_schw**3
        transverse_eig = M / r_schw**3

        # Expected sorted: [radial, 0, transverse, transverse]
        # but the zero eigenvalue may not be exactly zero
        # Sort by absolute value to identify them
        sorted_by_abs = sorted(eigs_np, key=abs)

        # The smallest eigenvalue by absolute value should be near 0
        assert abs(sorted_by_abs[0]) < abs(transverse_eig) * 0.5, (
            f"Smallest eigenvalue {sorted_by_abs[0]:.6e} not near zero"
        )

        # The most negative eigenvalue should be near -2M/r^3
        most_negative = min(eigs_np)
        np.testing.assert_allclose(
            most_negative, radial_eig, rtol=1e-6,
            err_msg=(
                f"Radial tidal eigenvalue {most_negative:.6e} does not match "
                f"analytical {radial_eig:.6e}"
            ),
        )

        # The two most positive eigenvalues should be near +M/r^3
        positive_eigs = sorted([e for e in eigs_np if e > abs(radial_eig) * 0.1])
        assert len(positive_eigs) >= 2, (
            f"Expected at least 2 positive eigenvalues, got {len(positive_eigs)}"
        )
        for pe in positive_eigs[:2]:
            np.testing.assert_allclose(
                pe, transverse_eig, rtol=1e-6,
                err_msg=(
                    f"Transverse tidal eigenvalue {pe:.6e} does not match "
                    f"analytical {transverse_eig:.6e}"
                ),
            )


class TestTidalTensorFlatSpace:
    """Test 2: Tidal tensor in Minkowski should be zero."""

    def test_tidal_tensor_flat_space(self):
        """All tidal eigenvalues should be zero in flat spacetime."""
        metric = MinkowskiMetric()
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        v = jnp.array([1.0, 0.0, 0.0, 0.0])  # static observer

        eigs = tidal_eigenvalues(metric, x, v)

        # All eigenvalues should be zero (Riemann = 0 in flat space)
        np.testing.assert_allclose(
            eigs, 0.0, atol=1e-14,
            err_msg="Tidal eigenvalues not zero in flat Minkowski spacetime",
        )


class TestDeviationCoIntegration:
    """Test 3: Geodesic deviation co-integration in flat space.

    In Minkowski spacetime, the Riemann tensor is zero, so the geodesic
    deviation equation reduces to:
        D^2 xi / D tau^2 = 0

    This means xi(tau) = xi0 + w0 * tau (linear growth).
    In Cartesian flat space, the Christoffel terms also vanish, so
    coordinate deviation is exactly linear.
    """

    def test_deviation_linear_growth_minkowski(self):
        """Deviation vector should grow linearly in flat space."""
        metric = MinkowskiMetric()

        # Base geodesic: static observer at origin
        x0 = jnp.array([0.0, 1.0, 0.0, 0.0])
        v0 = jnp.array([1.0, 0.0, 0.0, 0.0])  # purely timelike

        # Initial deviation: small spatial offset
        xi0 = jnp.array([0.0, 0.1, 0.0, 0.0])
        # Deviation velocity: expand in y-direction
        w0 = jnp.array([0.0, 0.0, 0.05, 0.0])

        tau_end = 5.0
        sol = integrate_geodesic_with_deviation(
            metric, x0, v0, xi0, w0,
            tau_span=(0.0, tau_end),
            num_points=500,
        )

        # Final deviation should be xi0 + w0 * tau_end
        expected_final_xi = xi0 + w0 * tau_end
        actual_final_xi = sol.deviations[-1]

        np.testing.assert_allclose(
            actual_final_xi, expected_final_xi, atol=1e-8,
            err_msg="Deviation vector did not grow linearly in flat space",
        )


# Blueshift tests


class TestBlueshiftMinkowskiNoShift:
    """Test 4: Blueshift in Minkowski = 1.0 (no frequency shift)."""

    def test_blueshift_minkowski_no_shift(self):
        """Static emitter and receiver in flat space: blueshift = 1.0 exactly."""
        metric = MinkowskiMetric()

        # Static emitter at one location
        x_emit = jnp.array([0.0, 0.0, 0.0, 0.0])
        u_emit = jnp.array([1.0, 0.0, 0.0, 0.0])  # static observer

        # Same null vector (photon propagating in x-direction)
        k_emit = jnp.array([1.0, 1.0, 0.0, 0.0])  # null: -1 + 1 = 0

        # Static receiver at another location
        x_recv = jnp.array([0.0, 10.0, 0.0, 0.0])
        u_recv = jnp.array([1.0, 0.0, 0.0, 0.0])  # static observer
        k_recv = jnp.array([1.0, 1.0, 0.0, 0.0])  # same photon 4-momentum

        z_plus_1 = compute_blueshift(
            metric, k_emit, u_emit, x_emit, k_recv, u_recv, x_recv,
        )

        np.testing.assert_allclose(
            float(z_plus_1), 1.0, atol=1e-14,
            err_msg="Blueshift in flat Minkowski should be exactly 1.0",
        )


class TestBlueshiftMinkowskiDoppler:
    """Test 5: Relativistic Doppler shift in Minkowski.

    Emitter moving with velocity v along x-axis, receiver static.
    Photon emitted in x-direction (head-on).
    Relativistic Doppler: z+1 = sqrt((1+v)/(1-v)).
    """

    def test_blueshift_doppler(self):
        """Blueshift factor should match relativistic Doppler formula."""
        metric = MinkowskiMetric()
        v = 0.3  # emitter velocity (fraction of c)

        # Emitter moving with velocity v in x-direction
        # 4-velocity: u = gamma * (1, v, 0, 0)
        gamma = 1.0 / np.sqrt(1.0 - v**2)
        x_emit = jnp.array([0.0, 0.0, 0.0, 0.0])
        u_emit = jnp.array([gamma, gamma * v, 0.0, 0.0])

        # Photon propagating in +x direction
        k_emit = jnp.array([1.0, 1.0, 0.0, 0.0])

        # Static receiver
        x_recv = jnp.array([0.0, 10.0, 0.0, 0.0])
        u_recv = jnp.array([1.0, 0.0, 0.0, 0.0])
        k_recv = jnp.array([1.0, 1.0, 0.0, 0.0])

        z_plus_1 = compute_blueshift(
            metric, k_emit, u_emit, x_emit, k_recv, u_recv, x_recv,
        )

        # Relativistic Doppler for head-on approach:
        # omega_recv / omega_emit = 1 / (gamma * (1 - v))
        # which equals sqrt((1+v)/(1-v))
        expected = np.sqrt((1.0 + v) / (1.0 - v))

        np.testing.assert_allclose(
            float(z_plus_1), expected, rtol=1e-10,
            err_msg=(
                f"Doppler blueshift {float(z_plus_1):.10f} does not match "
                f"analytical {expected:.10f} for v={v}"
            ),
        )


class TestBlueshiftSchwarzschildGravitational:
    """Test 6: Gravitational redshift in Schwarzschild spacetime.

    For static observers connected by a radial photon:
        omega_recv / omega_emit = sqrt(-g_00(emit)) / sqrt(-g_00(recv))

    Photon emitted from deeper in the gravitational well is redshifted
    when received farther out.
    """

    def test_gravitational_redshift(self):
        """Blueshift factor should match gravitational redshift formula."""
        M = 1.0
        metric = SchwarzschildMetric(M=M)

        # Emitter at r_iso = 3 (closer to the hole)
        r_iso_emit = 3.0
        x_emit = jnp.array([0.0, r_iso_emit, 0.0, 0.0])
        g_emit = metric(x_emit)
        # Static observer 4-velocity: u = (1/alpha, 0, 0, 0)
        alpha_emit = jnp.sqrt(-g_emit[0, 0])
        u_emit = jnp.array([1.0 / alpha_emit, 0.0, 0.0, 0.0])

        # Receiver at r_iso = 20 (farther out)
        r_iso_recv = 20.0
        x_recv = jnp.array([0.0, r_iso_recv, 0.0, 0.0])
        g_recv = metric(x_recv)
        alpha_recv = jnp.sqrt(-g_recv[0, 0])
        u_recv = jnp.array([1.0 / alpha_recv, 0.0, 0.0, 0.0])

        # Radial null vector at emission: k = (k^t, k^x, 0, 0) with
        # g_ab k^a k^b = 0. Set k^t = 1 at emission for convenience.
        k_x_emit = jnp.sqrt(-g_emit[0, 0] / g_emit[1, 1])
        k_emit = jnp.array([1.0, k_x_emit, 0.0, 0.0])

        # At reception: the photon's conserved energy along the static
        # Killing vector is E = -g_00 k^0 = alpha^2 k^0. By energy
        # conservation: alpha_emit^2 * k^0_emit = alpha_recv^2 * k^0_recv.
        # So k^0_recv = (alpha_emit / alpha_recv)^2 * k^0_emit.
        k_t_recv = (alpha_emit / alpha_recv) ** 2
        k_x_recv = jnp.sqrt(-g_recv[0, 0] / g_recv[1, 1]) * k_t_recv
        k_recv = jnp.array([k_t_recv, k_x_recv, 0.0, 0.0])

        z_plus_1 = compute_blueshift(
            metric, k_emit, u_emit, x_emit, k_recv, u_recv, x_recv,
        )

        # Analytical gravitational redshift for static observers:
        # omega_recv / omega_emit = alpha_emit / alpha_recv
        # Emitter is deeper in the potential well, so alpha_emit < alpha_recv,
        # giving z+1 < 1 (gravitational redshift).
        expected = float(alpha_emit / alpha_recv)

        np.testing.assert_allclose(
            float(z_plus_1), expected, rtol=1e-8,
            err_msg=(
                f"Gravitational blueshift {float(z_plus_1):.10f} does not match "
                f"analytical {expected:.10f}"
            ),
        )


# Conservation monitoring tests


class TestConservationMonitoringTimelike:
    """Test 7: monitor_conservation for a Schwarzschild timelike geodesic."""

    def test_conservation_monitoring_timelike(self):
        """Timelike norm should stay near -1 throughout integration."""
        M = 1.0
        metric = SchwarzschildMetric(M=M)

        # Use radial infall from r=15M
        from warpax.geodesics import make_event, horizon_event
        x0, v0 = radial_infall_ic(metric, r_start_schw=15.0, M=M)
        event = make_event(horizon_event)

        sol = integrate_geodesic(
            metric, x0, v0,
            tau_span=(0.0, 50.0),
            num_points=1000,
            max_steps=16384,
            event=event,
        )

        norms = monitor_conservation(metric, sol)

        # Use first half of points (well before any event termination)
        norms_valid = norms[:500]
        max_norm_error = float(jnp.max(jnp.abs(norms_valid + 1.0)))
        assert max_norm_error < 1e-6, (
            f"Timelike norm drift {max_norm_error:.2e} exceeds 1e-6"
        )


class TestConservationMonitoringNull:
    """Test 8: monitor_conservation for a null geodesic."""

    def test_conservation_monitoring_null(self):
        """Null norm should stay near 0 throughout integration."""
        metric = MinkowskiMetric()

        # Null geodesic in Minkowski
        x0 = jnp.array([0.0, 0.0, 0.0, 0.0])
        n_spatial = jnp.array([1.0, 0.0, 0.0])
        x0, k0 = null_ic(metric, x0, n_spatial)

        sol = integrate_geodesic(
            metric, x0, k0,
            tau_span=(0.0, 10.0),
            num_points=500,
        )

        norms = monitor_conservation(metric, sol)
        max_null_error = float(jnp.max(jnp.abs(norms)))
        assert max_null_error < 1e-8, (
            f"Null norm drift {max_null_error:.2e} exceeds 1e-8"
        )

    def test_conservation_monitoring_null_schwarzschild(self):
        """Null norm should stay near 0 in curved spacetime too."""
        M = 1.0
        metric = SchwarzschildMetric(M=M)

        # Null geodesic starting outside Schwarzschild
        r_iso = 10.0
        x0 = jnp.array([0.0, r_iso, 0.0, 0.0])
        n_spatial = jnp.array([0.0, 1.0, 0.0])  # tangential
        x0, k0 = null_ic(metric, x0, n_spatial)

        sol = integrate_geodesic(
            metric, x0, k0,
            tau_span=(0.0, 10.0),
            num_points=500,
        )

        norms = monitor_conservation(metric, sol)
        max_null_error = float(jnp.max(jnp.abs(norms)))
        assert max_null_error < 1e-6, (
            f"Schwarzschild null norm drift {max_null_error:.2e} exceeds 1e-6"
        )


def test_boosted_observer_trace_is_invariant():
    """Tidal trace ``K^mu_mu = R^mu_{nu mu sigma} v^nu v^sigma`` is invariant under boost.

    Pure Ricci contraction: in vacuum, ``R_{nu sigma} = 0`` so the tidal
    tensor is traceless for any timelike unit ``v``. A boosted observer
    must therefore also give zero trace within autodiff precision.
    """
    M = 1.0
    r_iso = 10.0
    metric = SchwarzschildMetric(M=M)
    x = jnp.array([0.0, r_iso, 0.0, 0.0])
    g = metric(x)

    rapidity = 0.3
    v_t = jnp.cosh(rapidity) / jnp.sqrt(-g[0, 0])
    v_y = jnp.sinh(rapidity) / jnp.sqrt(g[2, 2])
    v = jnp.array([v_t, 0.0, v_y, 0.0])

    norm = jnp.einsum("a,ab,b->", v, g, v)
    assert abs(float(norm) + 1.0) < 1e-10

    K = tidal_tensor(metric, x, v)
    trace = float(jnp.einsum("mm->", K))
    assert abs(trace) < 1e-9, f"tidal trace {trace} should vanish in vacuum"

    eigs = np.real(np.linalg.eigvals(np.asarray(K)))
    r_schw = _iso_to_schw_r(r_iso, M)
    bound = 4.0 * M / r_schw**3 * (np.cosh(float(rapidity)) ** 2)
    assert np.all(np.abs(eigs) < bound + 1e-9)


class TestFutureDirectedICs:
    """Regression: timelike_ic / null_ic must select the FUTURE-directed root.

    Bug: both helpers deliberately returned the past-directed root
    (``u^0 < 0`` / ``k^0 < 0`` for ``g_00 < 0``) with a docstring claiming
    sign-insensitivity. That claim is false for time-dependent metrics:
    an Alcubierre bubble at ``x_s = v_s t`` is not symmetric under
    ``t -> -t`` (which maps ``v_s -> -v_s``), so a past-directed leg
    traces a physically different trajectory.
    """

    # (metric, starting point) cases including a time-dependent metric
    CASES = [
        ("Minkowski", MinkowskiMetric(), jnp.array([0.0, 0.0, 0.0, 0.0])),
        ("Schwarzschild", SchwarzschildMetric(M=1.0), jnp.array([0.0, 10.0, 0.0, 0.0])),
        ("Alcubierre", AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0),
         jnp.array([0.0, 5.0, 0.1, 0.0])),
    ]

    @pytest.mark.parametrize("name,metric,x0", CASES, ids=[c[0] for c in CASES])
    def test_timelike_ic_future_directed(self, name, metric, x0):
        """v^0 > 0 and g(v, v) = -1 for the returned timelike IC."""
        _, v0 = timelike_ic(metric, x0, jnp.array([0.3, 0.0, 0.0]))
        assert float(v0[0]) > 0.0, f"{name}: past-directed v^0 = {float(v0[0])}"
        norm = float(jnp.einsum("a,ab,b->", v0, metric(x0), v0))
        assert abs(norm + 1.0) < 1e-12, f"{name}: norm {norm} != -1"

    @pytest.mark.parametrize("name,metric,x0", CASES, ids=[c[0] for c in CASES])
    def test_null_ic_future_directed(self, name, metric, x0):
        """k^0 > 0 and g(k, k) = 0 for the returned null IC."""
        _, k0 = null_ic(metric, x0, jnp.array([1.0, 0.0, 0.0]))
        assert float(k0[0]) > 0.0, f"{name}: past-directed k^0 = {float(k0[0])}"
        norm = float(jnp.einsum("a,ab,b->", k0, metric(x0), k0))
        assert abs(norm) < 1e-12, f"{name}: null norm {norm} != 0"

    def test_superluminal_nan_sentinel_preserved(self):
        """Future-root selection keeps the disc < 0 NaN sentinel intact.

        A static observer (zero 3-velocity) does not exist inside a
        superluminal Alcubierre bubble (g_00 > 0 there), so timelike_ic
        must still return the NaN sentinel rather than a spurious root.
        """
        m = AlcubierreMetric(v_s=2.0, R=2.0, sigma=8.0)
        _, v0 = timelike_ic(m, jnp.array([0.0, 0.1, 0.0, 0.0]),
                            jnp.array([0.0, 0.0, 0.0]))
        assert bool(jnp.isnan(v0[0])), "expected NaN sentinel for no real root"
