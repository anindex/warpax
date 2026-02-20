"""Physics validation: curvature chain cross-checks against analytical GR formulas.

Schwarzschild vacuum at multiple radii
    - Ricci tensor = 0 (vacuum solution)
    - Einstein tensor = 0 (vacuum)
    - Stress-energy tensor = 0 (vacuum)
    - Kretschner scalar K = 48 M^2 / r_schw^6

Minkowski exact zero at asymmetric off-axis points
    - Full curvature chain identically zero to machine precision
    - Metric is diag(-1, 1, 1, 1) everywhere

Alcubierre Eulerian energy density
    - Autodiff T_{ab} n^a n^b matches analytical rho = -(v_s^2/32pi)(df/dr_s)^2(y^2+z^2)/r_s^2
    - Energy density always non-positive (WEC violation)
    - Zero on bubble axis (y=z=0) and far from bubble

References:
    - Misner, Thorne & Wheeler, *Gravitation*, Ch. 31 (Schwarzschild)
    - Alcubierre, Class. Quantum Grav. 11 (1994) L73 (warp drive energy density)
"""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from warpax.benchmarks.alcubierre import AlcubierreMetric, eulerian_energy_density
from warpax.benchmarks.minkowski import MinkowskiMetric
from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.energy_conditions import compute_eulerian_ec
from warpax.geometry import compute_curvature_chain, kretschner_scalar


# ---------------------------------------------------------------------------
# Schwarzschild multi-radius validation
# ---------------------------------------------------------------------------

SCHWARZSCHILD_RADII = [1.0, 2.0, 5.0, 10.0, 50.0, 100.0]


class TestSchwarzschildMultiRadius:
    """Schwarzschild vacuum at 6 isotropic radii.

    Schwarzschild is an exact vacuum solution of the Einstein field equations:
    R_{ab} = 0, G_{ab} = 0, T_{ab} = 0 everywhere outside the singularity.
    The Kretschner scalar K = 48 M^2 / r_schw^6 is the only nonzero
    curvature invariant (MTW Ch. 31).
    """

    M = 1.0
    metric = SchwarzschildMetric(M=1.0)

    @pytest.mark.parametrize("r_iso", SCHWARZSCHILD_RADII)
    def test_ricci_zero(self, r_iso: float) -> None:
        """Schwarzschild is a vacuum solution: R_{ab} = 0, R = 0 at all radii."""
        coords = jnp.array([0.0, r_iso, 0.0, 0.0])
        result = compute_curvature_chain(self.metric, coords)
        npt.assert_allclose(
            np.array(result.ricci), 0.0, atol=1e-10,
            err_msg=f"Ricci tensor nonzero at r_iso={r_iso}",
        )
        npt.assert_allclose(
            float(result.ricci_scalar), 0.0, atol=1e-10,
            err_msg=f"Ricci scalar nonzero at r_iso={r_iso}",
        )

    @pytest.mark.parametrize("r_iso", SCHWARZSCHILD_RADII)
    def test_kretschner(self, r_iso: float) -> None:
        """Kretschner scalar K = 48 M^2 / r_schw^6 (MTW Ch. 31)."""
        coords = jnp.array([0.0, r_iso, 0.0, 0.0])
        result = compute_curvature_chain(self.metric, coords)
        K = kretschner_scalar(result.riemann, result.metric, result.metric_inv)

        # Convert isotropic r_iso to Schwarzschild r
        r_schw = r_iso * (1.0 + self.M / (2.0 * r_iso)) ** 2
        K_expected = 48.0 * self.M**2 / r_schw**6

        npt.assert_allclose(
            float(K), K_expected, rtol=1e-8,
            err_msg=f"Kretschner mismatch at r_iso={r_iso} (r_schw={r_schw:.6f})",
        )

    @pytest.mark.parametrize("r_iso", SCHWARZSCHILD_RADII)
    def test_einstein_zero(self, r_iso: float) -> None:
        """G_{ab} = 0 for vacuum."""
        coords = jnp.array([0.0, r_iso, 0.0, 0.0])
        result = compute_curvature_chain(self.metric, coords)
        npt.assert_allclose(
            np.array(result.einstein), 0.0, atol=1e-10,
            err_msg=f"Einstein tensor nonzero at r_iso={r_iso}",
        )

    @pytest.mark.parametrize("r_iso", SCHWARZSCHILD_RADII)
    def test_stress_energy_zero(self, r_iso: float) -> None:
        """T_{ab} = 0 for vacuum (G = 8pi T)."""
        coords = jnp.array([0.0, r_iso, 0.0, 0.0])
        result = compute_curvature_chain(self.metric, coords)
        npt.assert_allclose(
            np.array(result.stress_energy), 0.0, atol=1e-10,
            err_msg=f"Stress-energy tensor nonzero at r_iso={r_iso}",
        )


# ---------------------------------------------------------------------------
# Minkowski exact zero
# ---------------------------------------------------------------------------

MINKOWSKI_COORDS = [
    jnp.array([1.1, 2.3, -0.7, 3.14]),     # fully asymmetric
    jnp.array([0.0, 0.0, 0.0, 0.0]),        # origin
    jnp.array([-5.0, 7.1, -2.2, 0.9]),      # negative coords
    jnp.array([0.0, 100.0, -50.0, 25.0]),    # large values
    jnp.array([3.0, 0.01, -0.01, 0.001]),    # near-zero spatial
]


class TestMinkowskiExactZero:
    """Flat spacetime must produce identically zero curvature.

    Minkowski spacetime has zero Christoffel symbols, zero Riemann tensor,
    and therefore zero Ricci, Einstein, and stress-energy tensors everywhere.
    These tests use asymmetric off-axis coordinates to expose potential
    transpose or index ordering bugs in the curvature chain.
    """

    metric = MinkowskiMetric()

    @pytest.mark.parametrize("coords", MINKOWSKI_COORDS)
    def test_full_chain_exact_zero(self, coords: jnp.ndarray) -> None:
        """Flat spacetime must produce identically zero curvature at machine precision."""
        result = compute_curvature_chain(self.metric, coords)

        npt.assert_allclose(
            np.array(result.christoffel), 0.0, atol=1e-14,
            err_msg="Christoffel symbols nonzero in Minkowski",
        )
        npt.assert_allclose(
            np.array(result.riemann), 0.0, atol=1e-14,
            err_msg="Riemann tensor nonzero in Minkowski",
        )
        npt.assert_allclose(
            np.array(result.ricci), 0.0, atol=1e-14,
            err_msg="Ricci tensor nonzero in Minkowski",
        )
        npt.assert_allclose(
            float(result.ricci_scalar), 0.0, atol=1e-14,
            err_msg="Ricci scalar nonzero in Minkowski",
        )
        npt.assert_allclose(
            np.array(result.einstein), 0.0, atol=1e-14,
            err_msg="Einstein tensor nonzero in Minkowski",
        )
        npt.assert_allclose(
            np.array(result.stress_energy), 0.0, atol=1e-14,
            err_msg="Stress-energy tensor nonzero in Minkowski",
        )

    @pytest.mark.parametrize("coords", MINKOWSKI_COORDS)
    def test_metric_is_minkowski(self, coords: jnp.ndarray) -> None:
        """Metric must be diag(-1, 1, 1, 1) everywhere."""
        result = compute_curvature_chain(self.metric, coords)
        eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

        npt.assert_allclose(
            np.array(result.metric), np.array(eta), atol=1e-15,
            err_msg="Metric is not Minkowski",
        )
        npt.assert_allclose(
            np.array(result.metric_inv), np.array(eta), atol=1e-15,
            err_msg="Inverse metric is not Minkowski",
        )


# ---------------------------------------------------------------------------
# Alcubierre Eulerian energy density
# ---------------------------------------------------------------------------

BUBBLE_WALL_POINTS = [
    (0.8, 0.5, 0.0),   # bubble wall, y-offset
    (0.8, 0.0, 0.5),   # bubble wall, z-offset
    (1.0, 0.3, 0.3),   # on-wall, both y and z
    (0.5, 0.7, 0.0),   # inner wall region
    (1.2, 0.4, 0.0),   # outer wall region
]


class TestAlcubierreEulerianEnergyDensity:
    """Alcubierre Eulerian energy density cross-validation.

    Cross-validates the full autodiff curvature chain (metric -> Christoffel
    -> Riemann -> Ricci -> Einstein -> T_{ab}) against the closed-form
    analytical formula from Alcubierre (1994) Eq. 11:

        rho = -(v_s^2 / 32pi) * (df/dr_s)^2 * (y^2 + z^2) / r_s^2

    The Eulerian energy density is T_{ab} n^a n^b where n^a is the unit
    normal to constant-time hypersurfaces (the Eulerian observer).
    """

    v_s = 0.5
    R = 1.0
    sigma = 8.0
    metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0, x_s=0.0)

    def _compute_rho_autodiff(self, x: float, y: float, z: float) -> float:
        """Compute Eulerian energy density via full autodiff chain."""
        coords = jnp.array([0.0, x, y, z])
        result = compute_curvature_chain(self.metric, coords)
        ec = compute_eulerian_ec(result.stress_energy, result.metric)
        return float(ec["wec"])

    def _compute_rho_analytical(self, x: float, y: float, z: float) -> float:
        """Compute Eulerian energy density via analytical formula."""
        return float(eulerian_energy_density(
            jnp.array(x), jnp.array(y), jnp.array(z),
            v_s=self.v_s, R=self.R, sigma=self.sigma, x_s=0.0,
        ))

    @pytest.mark.parametrize("x,y,z", BUBBLE_WALL_POINTS)
    def test_energy_density_at_bubble_wall_points(
        self, x: float, y: float, z: float,
    ) -> None:
        """Cross-validates autodiff T_ab against Alcubierre (1994) Eq. 11:
        rho = -(v_s^2/32pi)(df/dr_s)^2(y^2+z^2)/r_s^2."""
        rho_autodiff = self._compute_rho_autodiff(x, y, z)
        rho_analytical = self._compute_rho_analytical(x, y, z)

        npt.assert_allclose(
            rho_autodiff, rho_analytical, rtol=1e-6,
            err_msg=(
                f"Eulerian energy density mismatch at ({x}, {y}, {z}): "
                f"autodiff={rho_autodiff:.10e}, analytical={rho_analytical:.10e}"
            ),
        )

    @pytest.mark.parametrize("x,y,z", BUBBLE_WALL_POINTS)
    def test_energy_density_always_nonpositive(
        self, x: float, y: float, z: float,
    ) -> None:
        """Alcubierre WEC violation: energy density is always <= 0."""
        rho_autodiff = self._compute_rho_autodiff(x, y, z)
        rho_analytical = self._compute_rho_analytical(x, y, z)

        assert rho_analytical <= 1e-15, (
            f"Analytical rho should be <= 0 at ({x}, {y}, {z}), got {rho_analytical}"
        )
        assert rho_autodiff <= 1e-15, (
            f"Autodiff rho should be <= 0 at ({x}, {y}, {z}), got {rho_autodiff}"
        )

    @pytest.mark.parametrize("x,y,z", [
        (0.8, 0.0, 0.0),   # on x-axis near wall
        (1.0, 0.0, 0.0),   # on x-axis at wall
    ])
    def test_energy_density_zero_on_axis(
        self, x: float, y: float, z: float,
    ) -> None:
        """Energy density is zero on the bubble axis (y=z=0) because rho ~ y^2+z^2."""
        rho_autodiff = self._compute_rho_autodiff(x, y, z)
        rho_analytical = self._compute_rho_analytical(x, y, z)

        npt.assert_allclose(
            rho_analytical, 0.0, atol=1e-12,
            err_msg=f"Analytical rho should be zero on axis at ({x}, {y}, {z})",
        )
        npt.assert_allclose(
            rho_autodiff, 0.0, atol=1e-12,
            err_msg=f"Autodiff rho should be zero on axis at ({x}, {y}, {z})",
        )

    def test_energy_density_zero_far_away(self) -> None:
        """Energy density is zero far from the bubble (df/dr_s ~ 0)."""
        x, y, z = 10.0, 1.0, 0.0
        rho_autodiff = self._compute_rho_autodiff(x, y, z)
        rho_analytical = self._compute_rho_analytical(x, y, z)

        npt.assert_allclose(
            rho_analytical, 0.0, atol=1e-12,
            err_msg=f"Analytical rho should be zero far from bubble at ({x}, {y}, {z})",
        )
        npt.assert_allclose(
            rho_autodiff, 0.0, atol=1e-12,
            err_msg=f"Autodiff rho should be zero far from bubble at ({x}, {y}, {z})",
        )
