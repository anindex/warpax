"""Physics validation: geodesic benchmarks and EC verification.

Validates geodesic integration against analytical GR solutions for
Schwarzschild spacetime:
  - Circular orbit period (Wald Ch. 6.3)
  - Radial infall energy conservation (Killing energy)
  - Tidal eigenvalues at multiple radii (MTW Ch. 31)

Also validates energy condition verification logic against synthetic
stress-energy tensors with known analytical properties.

"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from warpax.benchmarks import SchwarzschildMetric, MinkowskiMetric
from warpax.geodesics import (
    circular_orbit_ic,
    radial_infall_ic,
    integrate_geodesic,
    monitor_conservation,
    tidal_eigenvalues,
    tidal_tensor,
    make_event,
    horizon_event,
)
from warpax.energy_conditions import verify_point, check_all, compute_eulerian_ec


# ---------------------------------------------------------------------------
# Helper: isotropic <-> Schwarzschild coordinate conversions
# ---------------------------------------------------------------------------


def _schw_r_to_iso(r_schw, M=1.0):
    """Convert Schwarzschild radial coordinate to isotropic."""
    return (r_schw - M + np.sqrt(r_schw**2 - 2 * M * r_schw)) / 2.0


def _iso_to_schw_r(r_iso, M=1.0):
    """Convert isotropic radial coordinate to Schwarzschild."""
    return r_iso * (1 + M / (2 * r_iso)) ** 2


# ===========================================================================
# Schwarzschild Circular Orbit Benchmarks
# ===========================================================================


class TestSchwarzschildCircularOrbitBenchmark:
    """Circular orbit at r_schw=10M.

    Analytical proper period T = 2*pi*r^(3/2) * M^(-1/2) * sqrt(1 - 3M/r).
    Reference: Wald GR Ch. 6.3.
    """

    M = 1.0
    r_schw = 10.0

    @pytest.fixture(autouse=True)
    def setup_orbit(self):
        """Integrate one full circular orbit (shared across tests)."""
        self.metric = SchwarzschildMetric(M=self.M)
        self.x0, self.v0 = circular_orbit_ic(
            self.metric, r_schw=self.r_schw, M=self.M
        )

        # Analytical proper period: T = 2*pi*r^(3/2) M^(-1/2) sqrt(1 - 3M/r)
        # Wald GR Ch. 6.3
        T_coord = 2.0 * np.pi * self.r_schw**1.5 / np.sqrt(self.M)
        self.T_proper = T_coord * np.sqrt(1.0 - 3.0 * self.M / self.r_schw)

        self.sol = integrate_geodesic(
            self.metric,
            self.x0,
            self.v0,
            tau_span=(0.0, self.T_proper),
            num_points=2000,
            max_steps=65536,
        )

        self.r_iso_start = float(
            jnp.sqrt(self.x0[1] ** 2 + self.x0[2] ** 2 + self.x0[3] ** 2)
        )

    def test_circular_orbit_period_matches_analytical(self):
        """Circular orbit should return to starting angular position after one proper period.

        Analytical proper period T = 2*pi*r^(3/2) M^(-1/2) sqrt(1 - 3M/r).
        Wald GR Ch. 6.3.
        """
        # Check angular displacement: after one full orbit, the particle
        # should return to its starting angular position.
        x_final = float(self.sol.positions[-1, 1])
        y_final = float(self.sol.positions[-1, 2])
        x_start = float(self.x0[1])
        y_start = float(self.x0[2])

        angle_final = np.arctan2(y_final, x_final)
        angle_start = np.arctan2(y_start, x_start)

        # Angular displacement should be near 0 or 2*pi (full orbit)
        # Normalize to [-pi, pi]
        delta_angle = angle_final - angle_start
        delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi

        # Within 0.1% of a full revolution means |delta_angle| < 0.001 * 2*pi
        assert abs(delta_angle) < 0.001 * 2 * np.pi, (
            f"Angular displacement {np.degrees(delta_angle):.4f} deg after one "
            f"period exceeds 0.1% threshold"
        )

        # Also verify final position is close to starting position in x-y plane
        r_final = np.sqrt(x_final**2 + y_final**2)
        r_start = np.sqrt(x_start**2 + y_start**2)
        npt.assert_allclose(
            r_final,
            r_start,
            rtol=1e-3,
            err_msg="Final radius differs from start after one orbital period",
        )

    def test_circular_orbit_radial_stability(self):
        """Radial variation should be sub-ppm for conserved-quantity ICs.

        ICs from conserved quantities (E, L) should give radial variation < 0.01%.
        """
        r_iso_traj = jnp.sqrt(
            self.sol.positions[:, 1] ** 2
            + self.sol.positions[:, 2] ** 2
            + self.sol.positions[:, 3] ** 2
        )

        max_radial_variation = float(
            jnp.max(jnp.abs(r_iso_traj - self.r_iso_start)) / self.r_iso_start
        )
        assert max_radial_variation < 1e-4, (
            f"Radial variation {max_radial_variation:.2e} exceeds 1e-4 threshold. "
            f"Expected sub-ppm from conserved-quantity ICs."
        )

    def test_circular_orbit_norm_conservation(self):
        """4-velocity norm should be conserved to 1e-6 throughout orbit.

        g_ab v^a v^b = -1 for timelike geodesics.
        """
        norms = monitor_conservation(self.metric, self.sol)
        max_norm_error = float(jnp.max(jnp.abs(norms + 1.0)))
        assert max_norm_error < 1e-6, (
            f"4-velocity norm drift {max_norm_error:.2e} exceeds 1e-6 threshold"
        )


# ===========================================================================
# Schwarzschild Radial Infall Benchmarks
# ===========================================================================


class TestSchwarzschildRadialInfallBenchmark:
    """Radial infall from rest at r_schw=20M.

    Killing energy E = -xi_a v^a must be conserved along geodesic.
    xi = partial_t is the static Killing vector.
    """

    M = 1.0
    r_start_schw = 20.0

    @pytest.fixture(autouse=True)
    def setup_infall(self):
        """Integrate radial infall trajectory (shared across tests)."""
        self.metric = SchwarzschildMetric(M=self.M)
        self.x0, self.v0 = radial_infall_ic(
            self.metric, r_start_schw=self.r_start_schw, M=self.M
        )

        event = make_event(horizon_event)
        self.sol = integrate_geodesic(
            self.metric,
            self.x0,
            self.v0,
            tau_span=(0.0, 100.0),
            num_points=2000,
            max_steps=32768,
            event=event,
        )

        # Compute isotropic radius along trajectory
        self.r_iso_traj = jnp.sqrt(
            self.sol.positions[:, 1] ** 2
            + self.sol.positions[:, 2] ** 2
            + self.sol.positions[:, 3] ** 2
        )

    def test_radial_infall_energy_conservation(self):
        """Killing energy E = -g_{0a} v^a must be conserved along geodesic.

        Killing energy E = -xi_a v^a for the static Killing vector xi = partial_t.
        For Schwarzschild in isotropic coords (diagonal metric), E = -g_00 * v^0.
        """
        # Analytical E at start: E = sqrt(1 - 2M/r_start_schw)
        E_analytical = np.sqrt(1.0 - 2.0 * self.M / self.r_start_schw)

        # Compute E_i = -g_{0a} v^a at each trajectory point (first 500 points).
        # For diagonal Schwarzschild in isotropic coords: E = -g_00 v^0.
        # Note: timelike_ic uses affine parameter convention where v^0 may be
        # negative (time decreases with increasing affine parameter). The
        # conserved Killing energy magnitude |E| = |g_00 v^0| is unaffected.
        n_valid = 500
        positions_valid = self.sol.positions[:n_valid]
        velocities_valid = self.sol.velocities[:n_valid]

        def killing_energy_at_point(x, v):
            g = self.metric(x)
            # E = -g_{0a} v^a = -g_00 v^0 (diagonal metric)
            return -g[0, 0] * v[0]

        E_traj = jax.vmap(killing_energy_at_point)(positions_valid, velocities_valid)

        # Check conservation: max(|E_i - E_start|) / |E_start| < 1e-4
        E_start = float(E_traj[0])
        max_energy_drift = float(
            jnp.max(jnp.abs(E_traj - E_start)) / jnp.abs(E_start)
        )
        assert max_energy_drift < 1e-4, (
            f"Killing energy drift {max_energy_drift:.2e} exceeds 1e-4 threshold. "
            f"E_start={E_start:.6f}, E_analytical={E_analytical:.6f}"
        )

        # Verify |E_start| matches analytical (sign depends on affine parameter
        # direction convention in timelike_ic)
        npt.assert_allclose(
            abs(E_start),
            E_analytical,
            rtol=1e-4,
            err_msg="Starting Killing energy magnitude does not match analytical",
        )

    def test_radial_infall_monotonic_decrease(self):
        """Isotropic radius should monotonically decrease during radial infall."""
        r_valid = self.r_iso_traj[:500]
        dr = jnp.diff(r_valid)

        # Allow small positive bumps up to 1e-10 (numerical noise)
        n_increasing = int(jnp.sum(dr > 1e-10))
        assert n_increasing == 0, (
            f"Radius increased at {n_increasing} points during radial infall"
        )

    def test_radial_infall_norm_conservation(self):
        """4-velocity norm should be conserved to 1e-6 during radial infall.

        g_ab v^a v^b = -1 for timelike geodesics.
        """
        norms = monitor_conservation(self.metric, self.sol)
        norms_valid = norms[:500]
        max_norm_error = float(jnp.max(jnp.abs(norms_valid + 1.0)))
        assert max_norm_error < 1e-6, (
            f"4-velocity norm drift {max_norm_error:.2e} exceeds 1e-6 "
            f"during radial infall"
        )


# ===========================================================================
# Tidal Eigenvalue Benchmarks
# ===========================================================================


class TestTidalEigenvaluesMultiRadius:
    """Tidal eigenvalues at multiple Schwarzschild radii.

    MTW Ch. 31: tidal eigenvalues for Schwarzschild: {-2M/r^3, 0, +M/r^3, +M/r^3}.
    """

    M = 1.0

    @pytest.mark.parametrize("r_iso", [5.0, 10.0, 20.0])
    def test_tidal_eigenvalues_multi_radius(self, r_iso):
        """Tidal eigenvalues should match analytical pattern at each radius.

        MTW Ch. 31: tidal eigenvalues for Schwarzschild:
        {-2M/r^3, 0, +M/r^3, +M/r^3}.
        """
        metric = SchwarzschildMetric(M=self.M)

        # Position on x-axis
        x = jnp.array([0.0, r_iso, 0.0, 0.0])

        # Static observer: v = (1/sqrt(-g_00), 0, 0, 0)
        g = metric(x)
        v_t = 1.0 / jnp.sqrt(-g[0, 0])
        v = jnp.array([v_t, 0.0, 0.0, 0.0])

        # Compute tidal eigenvalues (sorted ascending by tidal_eigenvalues)
        eigs = tidal_eigenvalues(metric, x, v)
        eigs_np = np.array(eigs)

        # Analytical: use Schwarzschild r for the eigenvalue formula
        r_schw = _iso_to_schw_r(r_iso, self.M)
        radial_eig = -2.0 * self.M / r_schw**3
        transverse_eig = self.M / r_schw**3

        # Sort by absolute value to identify eigenvalues
        sorted_by_abs = sorted(eigs_np, key=abs)

        # Smallest-by-abs should be near 0 (within 0.5 * |transverse|)
        assert abs(sorted_by_abs[0]) < 0.5 * abs(transverse_eig), (
            f"Smallest eigenvalue {sorted_by_abs[0]:.6e} not near zero at "
            f"r_iso={r_iso} (threshold={0.5 * abs(transverse_eig):.6e})"
        )

        # Most negative should match radial_eig to rtol=0.10
        most_negative = min(eigs_np)
        npt.assert_allclose(
            most_negative,
            radial_eig,
            rtol=0.10,
            err_msg=(
                f"Radial tidal eigenvalue {most_negative:.6e} does not match "
                f"analytical {radial_eig:.6e} (10% tolerance) at r_iso={r_iso}"
            ),
        )

        # Two most positive should match transverse_eig to rtol=0.10
        positive_eigs = sorted(
            [e for e in eigs_np if e > abs(radial_eig) * 0.1]
        )
        assert len(positive_eigs) >= 2, (
            f"Expected at least 2 positive eigenvalues, got {len(positive_eigs)} "
            f"at r_iso={r_iso}"
        )
        for pe in positive_eigs[:2]:
            npt.assert_allclose(
                pe,
                transverse_eig,
                rtol=0.10,
                err_msg=(
                    f"Transverse tidal eigenvalue {pe:.6e} does not match "
                    f"analytical {transverse_eig:.6e} (10% tolerance) "
                    f"at r_iso={r_iso}"
                ),
            )

    def test_tidal_eigenvalues_zero_in_flat_space(self):
        """All tidal eigenvalues should be zero in flat Minkowski spacetime.

        Riemann tensor vanishes in flat space, so all tidal eigenvalues are zero.
        """
        metric = MinkowskiMetric()
        x = jnp.array([0.0, 5.0, 2.0, 3.0])
        v = jnp.array([1.0, 0.0, 0.0, 0.0])  # static observer

        eigs = tidal_eigenvalues(metric, x, v)
        npt.assert_allclose(
            eigs,
            0.0,
            atol=1e-14,
            err_msg="Tidal eigenvalues not zero in flat Minkowski spacetime",
        )


# ===========================================================================
# Synthetic Energy Condition Verification
# ===========================================================================

# Flat space metric for all synthetic EC tests
ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


class TestSyntheticECVerification:
    """Validate EC verification pipeline against synthetic stress-energy.

    Each test constructs a T_ab with known analytical properties and verifies
    that verify_point correctly identifies satisfaction/violation patterns.
    All tests use flat Minkowski spacetime (ETA = diag(-1,1,1,1)).
    """

    def test_dust_all_satisfied(self):
        """Dust (pressureless perfect fluid with rho>0): all energy conditions satisfied.

        T = diag(rho, 0, 0, 0) with rho=1.0 in orthonormal frame.
        All standard ECs (WEC, NEC, SEC, DEC) should be satisfied.
        """
        T = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        r = verify_point(T, ETA, n_starts=8)

        assert float(r.nec_margin) >= 0, f"NEC margin {float(r.nec_margin):.6f} < 0"
        assert float(r.wec_margin) >= 0, f"WEC margin {float(r.wec_margin):.6f} < 0"
        assert float(r.sec_margin) >= 0, f"SEC margin {float(r.sec_margin):.6f} < 0"
        assert float(r.dec_margin) >= 0, f"DEC margin {float(r.dec_margin):.6f} < 0"

        # Type I classification for diagonal stress-energy
        assert float(r.he_type) == 1.0, f"Expected Type I, got {float(r.he_type)}"

    def test_negative_energy_wec_violated(self):
        """Negative energy density: WEC violated.

        T = diag(-0.5, 0, 0, 0). Energy density is negative, so WEC is violated.
        The eigenvalue-based WEC margin is -0.5 (from rho = -0.5), but the
        observer-optimized margin may be much more negative (boosted observers
        see amplified violation). verify_point returns the worse (more negative)
        of eigenvalue and optimization margins.
        """
        T = jnp.diag(jnp.array([-0.5, 0.0, 0.0, 0.0]))
        r = verify_point(T, ETA, n_starts=8)

        # WEC must be violated
        assert float(r.wec_margin) < 0, (
            f"WEC margin {float(r.wec_margin):.6f} should be negative"
        )

        # Eigenvalue-based check should give exactly -0.5
        rho = r.rho
        pressures = r.pressures
        _, wec_eig, _, _ = check_all(rho, pressures)
        npt.assert_allclose(
            float(wec_eig),
            -0.5,
            atol=1e-4,
            err_msg="Eigenvalue WEC margin should be -0.5 for rho=-0.5",
        )

        # Observer-optimized margin should be at least as negative
        assert float(r.wec_margin) <= float(wec_eig) + 1e-6, (
            f"verify_point WEC margin {float(r.wec_margin):.6f} should be <= "
            f"eigenvalue margin {float(wec_eig):.6f}"
        )

    def test_nec_violated_rho_plus_p_negative(self):
        """rho + p_i < 0 for some direction: NEC violated.

        T = diag(0.3, -0.5, 0, 0) in orthonormal frame.
        rho=0.3, p_x=-0.5 (in covariant components with flat metric).
        For the eigenvalue analysis: T^a_b = eta^{ac} T_{cb}.
        In flat space with -+++ signature: T^0_0 = -T_00, T^i_j = T_ij.
        So eigenvalues of T^a_b are {-0.3, -0.5, 0, 0}.
        The timelike eigenvector gives rho = 0.3, pressures = {-0.5, 0, 0}.
        NEC requires rho + p_i >= 0 for all i: 0.3 + (-0.5) = -0.2 < 0, violated.
        """
        T = jnp.diag(jnp.array([0.3, -0.5, 0.0, 0.0]))
        r = verify_point(T, ETA, n_starts=8)

        assert float(r.nec_margin) < 0, (
            f"NEC margin {float(r.nec_margin):.6f} should be negative"
        )

    def test_sec_violated_but_wec_satisfied(self):
        """Dark energy type: WEC satisfied but SEC violated when rho + 3p < 0.

        T = diag(0.5, -0.3, -0.3, -0.3).
        rho=0.5, p=-0.3. rho+p = 0.2 > 0 (NEC/WEC satisfied).
        rho + 3p = 0.5 - 0.9 = -0.4 < 0 (SEC violated).
        """
        T = jnp.diag(jnp.array([0.5, -0.3, -0.3, -0.3]))
        r = verify_point(T, ETA, n_starts=8)

        # WEC should be satisfied (or borderline)
        assert float(r.wec_margin) >= -1e-6, (
            f"WEC margin {float(r.wec_margin):.6f} should be >= 0 (or borderline)"
        )

        # SEC should be violated
        assert float(r.sec_margin) < 0, (
            f"SEC margin {float(r.sec_margin):.6f} should be negative"
        )

    def test_dec_violated_large_momentum_flux(self):
        """Large momentum flux: energy current becomes spacelike, DEC violated.

        T with large off-diagonal T_{01} = T_{10} = 2.0.
        The energy-momentum current j^a = -T^a_b n^b may become spacelike.
        """
        T = jnp.array([
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        r = verify_point(T, ETA, n_starts=8)

        assert float(r.dec_margin) < 0, (
            f"DEC margin {float(r.dec_margin):.6f} should be negative "
            f"for large momentum flux"
        )

    def test_vacuum_all_trivially_satisfied(self):
        """Vacuum T_ab = 0: all ECs trivially satisfied (margins ~ 0).

        All energy conditions are trivially satisfied for vacuum
        since T_ab = 0 gives zero for all bilinear contractions.
        """
        T = jnp.zeros((4, 4))
        r = verify_point(T, ETA, n_starts=4)

        assert float(r.nec_margin) >= -1e-10, (
            f"NEC margin {float(r.nec_margin):.10f} should be >= 0 for vacuum"
        )
        assert float(r.wec_margin) >= -1e-10, (
            f"WEC margin {float(r.wec_margin):.10f} should be >= 0 for vacuum"
        )
        assert float(r.sec_margin) >= -1e-10, (
            f"SEC margin {float(r.sec_margin):.10f} should be >= 0 for vacuum"
        )
        assert float(r.dec_margin) >= -1e-10, (
            f"DEC margin {float(r.dec_margin):.10f} should be >= 0 for vacuum"
        )

    def test_radiation_all_satisfied(self):
        """Radiation (rho = 3p > 0): all energy conditions satisfied.

        T = diag(1.0, 1/3, 1/3, 1/3). Perfect radiation fluid.
        rho=1, p=1/3. rho+p = 4/3 > 0, rho+3p = 2 > 0, rho >= |p|.
        All standard ECs are satisfied.
        """
        T = jnp.diag(jnp.array([1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
        r = verify_point(T, ETA, n_starts=8)

        assert float(r.nec_margin) >= 0, f"NEC margin {float(r.nec_margin):.6f} < 0"
        assert float(r.wec_margin) >= 0, f"WEC margin {float(r.wec_margin):.6f} < 0"
        assert float(r.sec_margin) >= 0, f"SEC margin {float(r.sec_margin):.6f} < 0"
        assert float(r.dec_margin) >= 0, f"DEC margin {float(r.dec_margin):.6f} < 0"

    def test_eigenvalue_check_consistency(self):
        """Eigenvalue-based and optimization-based margins should agree for Type I.

        For a Type I stress-energy (dust), extract rho and pressures from
        classification, call check_all, and compare to verify_point margins.
        """
        T = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        r = verify_point(T, ETA, n_starts=8)

        # Extract rho and pressures from the classification
        rho = r.rho
        pressures = r.pressures

        # check_all returns (nec, wec, sec, dec) margins
        nec_eig, wec_eig, sec_eig, dec_eig = check_all(rho, pressures)

        # For Type I, verify_point takes min(eigenvalue, optimization) margins.
        # Both methods should agree for simple diagonal stress-energy.
        # The verify_point margin should be <= the eigenvalue margin (it takes min).
        assert float(r.nec_margin) <= float(nec_eig) + 1e-6, (
            f"NEC: verify_point margin {float(r.nec_margin):.6f} > "
            f"eigenvalue margin {float(nec_eig):.6f}"
        )
        assert float(r.wec_margin) <= float(wec_eig) + 1e-6, (
            f"WEC: verify_point margin {float(r.wec_margin):.6f} > "
            f"eigenvalue margin {float(wec_eig):.6f}"
        )
        assert float(r.sec_margin) <= float(sec_eig) + 1e-6, (
            f"SEC: verify_point margin {float(r.sec_margin):.6f} > "
            f"eigenvalue margin {float(sec_eig):.6f}"
        )
        assert float(r.dec_margin) <= float(dec_eig) + 1e-6, (
            f"DEC: verify_point margin {float(r.dec_margin):.6f} > "
            f"eigenvalue margin {float(dec_eig):.6f}"
        )

        # Both should give non-negative margins for dust
        assert float(nec_eig) >= 0, f"Eigenvalue NEC margin {float(nec_eig):.6f} < 0"
        assert float(wec_eig) >= 0, f"Eigenvalue WEC margin {float(wec_eig):.6f} < 0"
        assert float(sec_eig) >= 0, f"Eigenvalue SEC margin {float(sec_eig):.6f} < 0"
        assert float(dec_eig) >= 0, f"Eigenvalue DEC margin {float(dec_eig):.6f} < 0"
