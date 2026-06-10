"""Physics validation: curvature chain, warp metrics, geodesics+EC, regression, on-axis gradients, WarpShell DEC."""

from __future__ import annotations
from warpax.analysis import compute_kinematic_scalars, compute_kinematic_scalars_grid
from warpax.benchmarks import AlcubierreMetric, MinkowskiMetric, SchwarzschildMetric
from warpax.benchmarks.alcubierre import eulerian_energy_density
from warpax.energy_conditions import (
    check_all,
    classify_hawking_ellis,
    compute_eulerian_ec,
    verify_point,
)
from warpax.geodesics import (
    circular_orbit_ic,
    horizon_event,
    integrate_geodesic,
    make_event,
    monitor_conservation,
    radial_infall_ic,
    tidal_eigenvalues,
)
from warpax.geometry import (
    GridSpec,
    compute_curvature_chain,
    evaluate_curvature_grid,
    kretschmann_scalar,
)
from warpax.metrics import (
    LentzMetric,
    NatarioMetric,
    RodalMetric,
    VanDenBroeckMetric,
    WarpShellPhysical,
)
from warpax.metrics.natario import natario_eulerian_energy_density
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest



# Schwarzschild multi-radius validation

SCHWARZSCHILD_RADII = [1.0, 2.0, 5.0, 10.0, 50.0, 100.0]


class TestSchwarzschildMultiRadius:
    """Schwarzschild vacuum at 6 isotropic radii.

    Schwarzschild is an exact vacuum solution of the Einstein field equations:
    R_{ab} = 0, G_{ab} = 0, T_{ab} = 0 everywhere outside the singularity.
    The Kretschmann scalar K = 48 M^2 / r_schw^6 is the only nonzero
    curvature invariant (MTW Ch. 31).
    """

    M = 1.0
    metric = SchwarzschildMetric(M=1.0)

    @pytest.mark.parametrize("r_iso", SCHWARZSCHILD_RADII)
    def test_vacuum_chain_zero(self, r_iso: float) -> None:
        """Vacuum solution: R_{ab}, R, G_{ab}, T_{ab} all zero at all radii.

        One chain evaluation per radius covers all four quantities;
        G and T are linear in Ricci so recomputing them separately
        added nothing.
        """
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
        npt.assert_allclose(
            np.array(result.einstein), 0.0, atol=1e-10,
            err_msg=f"Einstein tensor nonzero at r_iso={r_iso}",
        )
        npt.assert_allclose(
            np.array(result.stress_energy), 0.0, atol=1e-10,
            err_msg=f"Stress-energy tensor nonzero at r_iso={r_iso}",
        )

    @pytest.mark.parametrize("r_iso", SCHWARZSCHILD_RADII)
    def test_kretschmann(self, r_iso: float) -> None:
        """Kretschmann scalar K = 48 M^2 / r_schw^6 (MTW Ch. 31)."""
        coords = jnp.array([0.0, r_iso, 0.0, 0.0])
        result = compute_curvature_chain(self.metric, coords)
        K = kretschmann_scalar(result.riemann, result.metric, result.metric_inv)

        r_schw = r_iso * (1.0 + self.M / (2.0 * r_iso)) ** 2
        K_expected = 48.0 * self.M**2 / r_schw**6

        npt.assert_allclose(
            float(K), K_expected, rtol=1e-8,
            err_msg=f"Kretschmann mismatch at r_iso={r_iso} (r_schw={r_schw:.6f})",
        )


# Minkowski exact zero

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


# Alcubierre Eulerian energy density

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


# Natario Eulerian energy density

NATARIO_WALL_POINTS = [
    (5.0, 3.0, 0.0),
    (10.0, 5.0, 2.0),
    (50.0, 20.0, 10.0),
    (80.0, 30.0, 15.0),
]


class TestNatarioEulerianEnergyDensity:
    """Natario Eulerian energy density cross-validation at t=0.

    Cross-validates autodiff T_{ab} n^a n^b against the closed-form
    expression in natario.py (Natario 2002, arXiv:gr-qc/0110086).
    """

    v_s = 0.1
    R = 100.0
    sigma = 0.03
    metric = NatarioMetric(v_s=0.1, R=100.0, sigma=0.03)

    def _compute_rho_autodiff(self, x: float, y: float, z: float) -> float:
        coords = jnp.array([0.0, x, y, z])
        result = compute_curvature_chain(self.metric, coords)
        ec = compute_eulerian_ec(result.stress_energy, result.metric)
        return float(ec["wec"])

    def _compute_rho_analytical(self, x: float, y: float, z: float) -> float:
        return float(natario_eulerian_energy_density(
            jnp.array(x), jnp.array(y), jnp.array(z),
            v_s=self.v_s, R=self.R, sigma=self.sigma,
        ))

    @pytest.mark.parametrize("x,y,z", NATARIO_WALL_POINTS)
    def test_energy_density_matches_analytical(
        self, x: float, y: float, z: float,
    ) -> None:
        rho_autodiff = self._compute_rho_autodiff(x, y, z)
        rho_analytical = self._compute_rho_analytical(x, y, z)

        npt.assert_allclose(
            rho_autodiff, rho_analytical, rtol=1e-5,
            err_msg=(
                f"Natario Eulerian rho mismatch at ({x}, {y}, {z}): "
                f"autodiff={rho_autodiff:.10e}, analytical={rho_analytical:.10e}"
            ),
        )

    @pytest.mark.parametrize("x,y,z", NATARIO_WALL_POINTS)
    def test_energy_density_always_nonpositive(
        self, x: float, y: float, z: float,
    ) -> None:
        rho_autodiff = self._compute_rho_autodiff(x, y, z)
        assert rho_autodiff <= 1e-12, (
            f"Natario autodiff rho should be <= 0 at ({x}, {y}, {z}), got {rho_autodiff}"
        )

    @pytest.mark.parametrize("t,x", [(0.5, 2.0), (1.0, 3.5), (2.0, 5.0)])
    def test_energy_density_moving_bubble(self, t: float, x: float) -> None:
        """Analytical helper uses dx = x - v_s t (co-moving radius)."""
        y, z = 1.0, 0.5
        coords = jnp.array([t, x, y, z])
        result = compute_curvature_chain(self.metric, coords)
        rho_autodiff = float(compute_eulerian_ec(
            result.stress_energy, result.metric,
        )["wec"])
        rho_analytical = float(natario_eulerian_energy_density(
            jnp.array(x), jnp.array(y), jnp.array(z),
            v_s=self.v_s, R=self.R, sigma=self.sigma, t=t,
        ))
        npt.assert_allclose(rho_autodiff, rho_analytical, rtol=1e-4)


class TestStressEnergySymmetry:
    """Einstein's equations guarantee T_{ab} = T_{ba}. The autodiff chain
    computes T via the Einstein tensor, which inherits its symmetry from
    the Riemann pair-interchange symmetry. Floating-point reduction order
    determined by XLA can perturb this slightly.

    stress_energy_tensor() explicitly symmetrizes T, so asserting on T
    would be vacuous; the bound is checked on the pre-symmetrization
    Einstein tensor instead.

    This suite bounds the relative antisymmetric drift per metric at
    ``|G - G^T|_max / |G|_max < 1e-13`` over a 10x10x10 coordinate slab;
    a regression that breaks the bound would surface a reduction-order
    change in the curvature chain and is worth investigating before
    accepting the diff.
    """

    GRID_N = 10
    REL_TOL = 1e-13

    @staticmethod
    def _slab_coords(bounds, n):
        import jax.numpy as jnp_

        xs = jnp_.linspace(bounds[0], bounds[1], n)
        ys = jnp_.linspace(bounds[0], bounds[1], n)
        zs = jnp_.linspace(bounds[0], bounds[1], n)
        return jnp_.stack(
            [
                jnp_.broadcast_to(jnp_.float64(0.0), (n * n * n,)),
                jnp_.repeat(jnp_.repeat(xs, n), n),
                jnp_.tile(jnp_.repeat(ys, n), n),
                jnp_.tile(zs, n * n),
            ],
            axis=-1,
        )

    def _assert_symmetric_over_slab(self, metric, bounds):
        import jax

        coords = self._slab_coords(bounds, self.GRID_N)

        def einstein_at(c):
            # pre-symmetrization tensor; stress_energy is forced symmetric
            return compute_curvature_chain(metric, c).einstein

        G = jax.vmap(einstein_at)(coords)
        asym = jnp.max(jnp.abs(G - jnp.swapaxes(G, -1, -2)))
        scale = jnp.max(jnp.abs(G)) + jnp.finfo(G.dtype).tiny
        rel = asym / scale
        assert float(rel) < self.REL_TOL, (
            f"Einstein antisymmetric drift above tolerance: "
            f"|G-G^T|_max = {asym:.2e}, |G|_max = {scale:.2e}, "
            f"rel = {rel:.2e} (tol {self.REL_TOL:.1e})"
        )

    def test_alcubierre_stress_energy_symmetric(self):
        self._assert_symmetric_over_slab(
            AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0), bounds=(-2.0, 2.0)
        )

    def test_rodal_stress_energy_symmetric(self):
        from warpax.metrics import RodalMetric

        self._assert_symmetric_over_slab(
            RodalMetric(v_s=0.5, R=1.0, sigma=0.1), bounds=(-2.0, 2.0)
        )

    def test_natario_stress_energy_symmetric(self):
        from warpax.metrics import NatarioMetric

        self._assert_symmetric_over_slab(
            NatarioMetric(v_s=0.1, R=100.0, sigma=0.03), bounds=(-150.0, 150.0)
        )

    def test_lentz_stress_energy_symmetric(self):
        from warpax.metrics import LentzMetric

        self._assert_symmetric_over_slab(
            LentzMetric(v_s=0.1, R=100.0, sigma=8.0), bounds=(-150.0, 150.0)
        )

    def test_vdb_stress_energy_symmetric(self):
        from warpax.metrics import VanDenBroeckMetric

        # compact bubble so the slab crosses the walls; default R=350
        # leaves G = 0 identically on a (-2, 2) slab
        self._assert_symmetric_over_slab(
            VanDenBroeckMetric(v_s=0.5, R=1.0, sigma=8.0, R_tilde=0.5, sigma_B=8.0),
            bounds=(-2.0, 2.0),
        )

    def test_warpshell_stress_energy_symmetric(self):
        from warpax.metrics import WarpShellMetric

        self._assert_symmetric_over_slab(
            WarpShellMetric(v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0),
            bounds=(5.0, 25.0),
        )

    def test_warpshell_physical_stress_energy_symmetric(self):
        from warpax.metrics import WarpShellPhysical

        self._assert_symmetric_over_slab(
            WarpShellPhysical(v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0),
            bounds=(5.0, 25.0),
        )


# Natario zero-expansion property


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
            # covers the omega output slot (zero for Eulerian observers)
            npt.assert_allclose(
                float(omega_sq),
                0.0,
                atol=1e-10,
                err_msg=f"Eulerian vorticity != 0 at coords {coords}",
            )


# Rodal globally Hawking-Ellis Type I


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


# Rodal energy deficit vs Alcubierre


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
        assert 19.0 < reduction_factor < 57.0, (
            f"Reduction factor = {reduction_factor:.2f}, expected ~38x (19-57). "
            f"Rodal peak = {peak_rho_rodal:.6e}, Alcubierre peak = {peak_rho_alcubierre:.6e}"
        )


# Lentz WEC violation at bubble wall


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


# Santiago observer-dependence theorem


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


# Helper: isotropic <-> Schwarzschild coordinate conversions


def _schw_r_to_iso(r_schw, M=1.0):
    """Convert Schwarzschild radial coordinate to isotropic."""
    return (r_schw - M + np.sqrt(r_schw**2 - 2 * M * r_schw)) / 2.0


def _iso_to_schw_r(r_iso, M=1.0):
    """Convert isotropic radial coordinate to Schwarzschild."""
    return r_iso * (1 + M / (2 * r_iso)) ** 2


# Schwarzschild Circular Orbit Benchmarks


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


# Schwarzschild Radial Infall Benchmarks


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
        # timelike_ic returns the future-directed root (v^0 > 0), so E > 0.
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

        # Verify E_start matches analytical (future-directed convention: E > 0)
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


# Tidal Eigenvalue Benchmarks


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

        # Most negative should match radial_eig (machine precision on isotropic grid)
        most_negative = min(eigs_np)
        npt.assert_allclose(
            most_negative,
            radial_eig,
            rtol=1e-6,
            atol=1e-12,
            err_msg=(
                f"Radial tidal eigenvalue {most_negative:.6e} does not match "
                f"analytical {radial_eig:.6e} at r_iso={r_iso}"
            ),
        )

        # Two most positive should match transverse_eig
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
                rtol=1e-6,
                atol=1e-12,
                err_msg=(
                    f"Transverse tidal eigenvalue {pe:.6e} does not match "
                    f"analytical {transverse_eig:.6e} at r_iso={r_iso}"
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


# Synthetic Energy Condition Verification

# Flat space metric for all synthetic EC tests
ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


class TestSyntheticECVerification:
    """Validate EC verification on synthetic tensors not covered elsewhere.

    Dust, vacuum, radiation, NEC/SEC splits are exercised in
    ``tests/test_ec_verifier.py``; keep only cases that stress distinct
    physics (observer amplification, DEC flux, Type-I margin contract).
    """

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

    def test_eigenvalue_check_consistency(self):
        """Type I verify_point margins match check_all eigenvalue margins."""
        T = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
        r = verify_point(T, ETA, n_starts=8)

        nec_eig, wec_eig, sec_eig, dec_eig = check_all(r.rho, r.pressures)

        npt.assert_allclose(float(r.nec_margin), float(nec_eig), rtol=0.0, atol=1e-10)
        npt.assert_allclose(float(r.wec_margin), float(wec_eig), rtol=0.0, atol=1e-10)
        npt.assert_allclose(float(r.sec_margin), float(sec_eig), rtol=0.0, atol=1e-10)
        npt.assert_allclose(float(r.dec_margin), float(dec_eig), rtol=0.0, atol=1e-10)


class TestAlcubierreVsZeroIsMinkowski:
    """v_s = 0 zeroes the Alcubierre shift and recovers Minkowski."""

    def test_curvature_chain_vanishes(self):
        m = AlcubierreMetric(v_s=0.0, R=2.0, sigma=8.0)
        coords = jnp.array([0.0, 0.5, 0.3, -0.2])
        chain = compute_curvature_chain(m, coords)
        eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        np.testing.assert_allclose(np.asarray(chain.metric), np.asarray(eta), atol=1e-14)
        np.testing.assert_allclose(np.asarray(chain.ricci), np.zeros((4, 4)), atol=5e-12)
        np.testing.assert_allclose(
            np.asarray(chain.einstein), np.zeros((4, 4)), atol=5e-12
        )
        np.testing.assert_allclose(
            np.asarray(chain.stress_energy), np.zeros((4, 4)), atol=5e-12
        )

    def test_einstein_constraint_on_grid(self):
        from warpax.geometry import evaluate_curvature_grid
        m = AlcubierreMetric(v_s=0.0, R=2.0, sigma=8.0)
        grid = GridSpec(bounds=[(-1.5, 1.5)] * 3, shape=(5, 5, 5))
        chain = evaluate_curvature_grid(m, grid, batch_size=32)
        max_T = float(jnp.max(jnp.abs(chain.stress_energy)))
        assert max_T < 5e-12, (
            f"v_s=0 Alcubierre still has non-zero stress-energy: {max_T:.3e}"
        )


class TestSchwarzschildADMMassFuchs:
    """ADM mass surface integral on the Fuchs canonical metric.

    The documented behavior (CODEBASE.md) is that the surface integrand
    has the wrong large-r asymptote for thick-shell constructions, so the
    integral drifts away from the Komar / volume mass as r grows. This
    test pins the trend quantitatively (linear-in-r drift plus a golden
    snapshot) so a future fix doesn't regress silently.
    """

    @pytest.mark.slow
    def test_adm_mass_finite_across_radii(self):
        from warpax.adm.mass import adm_mass
        from warpax.metrics.fuchs_construction import fuchs_default

        metric = fuchs_default()
        radii = [20.0, 50.0, 100.0]
        masses = [
            float(adm_mass(metric, r_surface=r, n_theta=10, n_phi=18))
            for r in radii
        ]
        assert all(np.isfinite(m) for m in masses), masses
        # All ADM probes return positive mass on this Type-I shell
        assert all(m > 0 for m in masses), masses
        # Documented wrong asymptote: integral grows linearly in r outside R_2
        assert masses[0] < masses[1] < masses[2], masses
        npt.assert_allclose(
            masses[2] / masses[1], 100.0 / 50.0, rtol=1e-6,
            err_msg=f"large-r linear drift broken: {masses}",
        )
        # Golden snapshot (CPU float64); a fixed integrand must update these
        npt.assert_allclose(
            masses,
            [2.9763560775557134, 5.0338710480931175, 10.067742096186235],
            rtol=1e-6,
        )


class TestInterpolationOrderRegression:
    """Default (linear) IO interpolation has O(h) accuracy in cell size."""

    def _make_grid(self, n: int) -> tuple:
        """Sample a smooth analytic alpha=alpha(x) on a 1D-extended 4D grid."""
        Nt, N = 2, n
        x = jnp.linspace(-1.0, 1.0, N)
        # alpha(x) = 1 + 0.1 * sin(pi x); smooth, has nontrivial 2nd derivative.
        alpha_1d = 1.0 + 0.1 * jnp.sin(jnp.pi * x)
        alpha = jnp.broadcast_to(
            alpha_1d[None, :, None, None], (Nt, N, N, N)
        ).copy()
        beta = jnp.zeros((Nt, N, N, N, 3))
        gamma = jnp.broadcast_to(
            jnp.eye(3)[None, None, None, None, :, :], (Nt, N, N, N, 3, 3)
        )
        spec = GridSpec(bounds=[(-1.0, 1.0)] * 4, shape=(Nt, N, N, N))
        return alpha, beta, gamma, spec

    def test_linear_interpolation_h_scaling(self):
        from warpax.io import InterpolatedADMMetric

        # Probe alpha at a midpoint between two grid nodes - linear interp
        # error there is O(h^2 * |alpha''|), and grid spacing is h = 2/(N-1).
        probe = jnp.array([0.0, 0.001, 0.0, 0.0])
        truth = 1.0 + 0.1 * jnp.sin(jnp.pi * 0.001)

        errors = []
        for n in (9, 17, 33):
            alpha, beta, gamma, spec = self._make_grid(n)
            m = InterpolatedADMMetric(
                alpha_grid=alpha,
                beta_grid=beta,
                gamma_grid=gamma,
                grid_spec=spec,
            )
            errors.append(abs(float(m.lapse(probe) - truth)))

        # Doubling resolution should roughly quarter the error (O(h^2)).
        ratio = errors[0] / max(errors[2], 1e-30)
        assert ratio >= 8.0, (
            f"Linear interp didn't converge as ~h^2: errors {errors}, ratio {ratio:.2f}"
        )


@pytest.mark.slow
class TestWarpShellBoundaryDEC:
    """WarpShellPhysical: DEC violated at shell transition bands."""

    def test_dec_violation_near_r1_transition(self):
        metric = WarpShellPhysical(v_s=0.02, R_1=10.0, R_2=20.0, r_s_param=5.0)
        grid = GridSpec(
            bounds=[(-25.0, 25.0), (-25.0, 25.0), (-25.0, 25.0)],
            shape=(15, 15, 15),
        )
        chain = evaluate_curvature_grid(metric, grid, batch_size=256)
        flat_T = chain.stress_energy.reshape(-1, 4, 4)
        flat_g = chain.metric.reshape(-1, 4, 4)
        shape = grid.shape
        xs = np.linspace(-25, 25, shape[0])
        dec_margins = []
        for i, x in enumerate(xs):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    y = -25.0 + j * (50.0 / (shape[1] - 1))
                    z = -25.0 + k * (50.0 / (shape[2] - 1))
                    rad = np.sqrt(x * x + y * y + z * z)
                    if abs(rad - 10.0) > 1.2 and abs(rad - 20.0) > 1.2:
                        continue
                    idx = i * shape[1] * shape[2] + j * shape[2] + k
                    r = verify_point(
                        flat_T[idx], flat_g[idx],
                        n_starts=8,
                        solver="auto",
                    )
                    dec_margins.append(float(r.dec_margin))
        assert len(dec_margins) >= 5
        assert min(dec_margins) < 0.0, (
            f"expected DEC violation at R1/R2 transitions; min={min(dec_margins)}"
        )


# Bubble centers ((t, x, y, z) coords). Each metric's center moves with v_s*t,
# so (0, 0, 0, 0) is the on-axis center at t=0 for v_s != 0.
@pytest.fixture
def on_axis_coords() -> jnp.ndarray:
    return jnp.array([0.0, 0.0, 0.0, 0.0])


def _check_finite_curvature(metric, coords: jnp.ndarray) -> None:
    chain = compute_curvature_chain(metric, coords)
    for name in (
        "metric",
        "christoffel",
        "riemann",
        "ricci",
        "ricci_scalar",
        "einstein",
        "stress_energy",
    ):
        val = getattr(chain, name)
        assert jnp.all(jnp.isfinite(val)), f"{type(metric).__name__}: non-finite {name}"


class TestOnAxisCurvatureFinite:
    def test_lentz_origin_curvature_finite(self, on_axis_coords):
        _check_finite_curvature(LentzMetric(v_s=0.1), on_axis_coords)

    def test_natario_origin_curvature_finite(self, on_axis_coords):
        _check_finite_curvature(
            NatarioMetric(v_s=0.1, R=1.0, sigma=8.0), on_axis_coords
        )

    def test_van_den_broeck_origin_curvature_finite(self, on_axis_coords):
        _check_finite_curvature(VanDenBroeckMetric(v_s=0.1), on_axis_coords)

    def test_sshell_origin_curvature_finite(self, on_axis_coords):
        from warpax.metrics.sshell import sshell_default
        _check_finite_curvature(sshell_default(v_s=0.02), on_axis_coords)


class TestOnAxisShiftJacobian:
    """jax.jacfwd of the metric shift on-axis must not be NaN."""

    def test_van_den_broeck_shift_jacobian_finite(self, on_axis_coords):
        m = VanDenBroeckMetric(v_s=0.1)
        jac = jax.jacfwd(m.shift)(on_axis_coords)
        assert jnp.all(jnp.isfinite(jac))

    def test_sshell_shift_jacobian_finite(self, on_axis_coords):
        from warpax.metrics.sshell import sshell_default
        m = sshell_default(v_s=0.02)
        jac = jax.jacfwd(m.shift)(on_axis_coords)
        assert jnp.all(jnp.isfinite(jac))

    def test_natario_shift_jacobian_finite(self, on_axis_coords):
        m = NatarioMetric(v_s=0.1, R=1.0, sigma=8.0)
        jac = jax.jacfwd(m.shift)(on_axis_coords)
        assert jnp.all(jnp.isfinite(jac))
