"""Tidal force validation, blueshift tests, and conservation diagnostics.

Tests the geodesic observable extraction modules (deviation.py and
observables.py) against known analytical solutions for Schwarzschild
spacetime and flat Minkowski spacetime.

"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks import (
    MinkowskiMetric,
    SchwarzschildMetric,
)
from warpax.geodesics import (
    compute_blueshift,
    integrate_geodesic,
    integrate_geodesic_with_deviation,
    monitor_conservation,
    null_ic,
    radial_infall_ic,
    tidal_eigenvalues,
    tidal_tensor,
    timelike_ic,
    velocity_norm,
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
# Tidal force tests
# ===========================================================================


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
            most_negative, radial_eig, rtol=0.10,
            err_msg=(
                f"Radial tidal eigenvalue {most_negative:.6e} does not match "
                f"analytical {radial_eig:.6e} (10% tolerance)"
            ),
        )

        # The two most positive eigenvalues should be near +M/r^3
        positive_eigs = sorted([e for e in eigs_np if e > abs(radial_eig) * 0.1])
        assert len(positive_eigs) >= 2, (
            f"Expected at least 2 positive eigenvalues, got {len(positive_eigs)}"
        )
        for pe in positive_eigs[:2]:
            np.testing.assert_allclose(
                pe, transverse_eig, rtol=0.10,
                err_msg=(
                    f"Transverse tidal eigenvalue {pe:.6e} does not match "
                    f"analytical {transverse_eig:.6e} (10% tolerance)"
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


# ===========================================================================
# Blueshift tests
# ===========================================================================


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


# ===========================================================================
# Conservation monitoring tests
# ===========================================================================


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
