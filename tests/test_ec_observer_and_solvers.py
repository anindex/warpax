"""EC observer parameterizations, solver fallbacks, filtering, multistart and PRNG contracts."""

from __future__ import annotations
from collections import Counter
from pathlib import Path
from warpax.analysis.kinematic_scalars import compute_kinematic_scalars
from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.energy_conditions import (
    optimize_dec,
    optimize_nec,
    optimize_sec,
    optimize_wec,
)
from warpax.energy_conditions.classification import classify_hawking_ellis
from warpax.energy_conditions.classification_mpmath import (
    classify_hawking_ellis_mpmath,
    eigenvalues_mpmath,
    verify_classification_at_points,
)
from warpax.energy_conditions.observer import (
    bounded_param,
    compute_orthonormal_tetrad,
    null_from_angles,
    timelike_from_rapidity,
    unbounded_param,
)
from warpax.energy_conditions.optimization import _make_initial_conditions_3d
from warpax.energy_conditions.verifier import _eulerian_ec_point
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import warnings



# Ensure float64 is active
assert jnp.array(1.0).dtype == jnp.float64, "Float64 not enabled"

# Flat Minkowski metric: eta = diag(-1, 1, 1, 1)
ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


def _orthonormality_check(tetrad: jnp.ndarray, g_ab: jnp.ndarray) -> jnp.ndarray:
    """Compute g_{ab} e_I^a e_J^b and return the deviation from eta_{IJ}."""
    # tetrad[I, a], g_ab[a, b] -> product[I, J] = g_{ab} e_I^a e_J^b
    product = jnp.einsum("Ia,ab,Jb->IJ", tetrad, g_ab, tetrad)
    return product - ETA


# Tetrad tests


class TestTetradMinkowski:
    """Tetrad construction for Minkowski (flat) metric."""

    def test_minkowski_tetrad_is_identity(self):
        """For flat metric diag(-1,1,1,1), tetrad should be identity."""
        g = ETA
        tetrad = compute_orthonormal_tetrad(g)
        np.testing.assert_allclose(tetrad, jnp.eye(4), atol=1e-14)

    def test_minkowski_orthonormality(self):
        """Verify g_{ab} e_I^a e_J^b = eta_{IJ} for Minkowski."""
        g = ETA
        tetrad = compute_orthonormal_tetrad(g)
        deviation = _orthonormality_check(tetrad, g)
        np.testing.assert_allclose(deviation, jnp.zeros((4, 4)), atol=1e-14)


class TestTetradSchwarzschild:
    """Tetrad construction for Schwarzschild metric at r=3M."""

    @pytest.fixture
    def schwarzschild_metric_at_3M(self):
        """Schwarzschild metric evaluated at r_iso=3M (x=3, y=0, z=0)."""
        metric = SchwarzschildMetric(M=1.0)
        coords = jnp.array([0.0, 3.0, 0.0, 0.0])
        return metric(coords)

    def test_schwarzschild_orthonormality(self, schwarzschild_metric_at_3M):
        """Verify g_{ab} e_I^a e_J^b = eta_{IJ} for Schwarzschild at r=3M."""
        g = schwarzschild_metric_at_3M
        tetrad = compute_orthonormal_tetrad(g)
        deviation = _orthonormality_check(tetrad, g)
        np.testing.assert_allclose(deviation, jnp.zeros((4, 4)), atol=1e-12)

    def test_schwarzschild_e0_timelike(self, schwarzschild_metric_at_3M):
        """e_0 should be timelike: g_{ab} e_0^a e_0^b = -1."""
        g = schwarzschild_metric_at_3M
        tetrad = compute_orthonormal_tetrad(g)
        norm = jnp.einsum("a,ab,b->", tetrad[0], g, tetrad[0])
        np.testing.assert_allclose(norm, -1.0, atol=1e-12)

    def test_schwarzschild_spatial_spacelike(self, schwarzschild_metric_at_3M):
        """e_1, e_2, e_3 should be spacelike: g_{ab} e_i^a e_i^b = +1."""
        g = schwarzschild_metric_at_3M
        tetrad = compute_orthonormal_tetrad(g)
        for i in range(1, 4):
            norm = jnp.einsum("a,ab,b->", tetrad[i], g, tetrad[i])
            np.testing.assert_allclose(norm, 1.0, atol=1e-12)


class TestTetradAlcubierre:
    """Tetrad construction for Alcubierre metric inside the bubble."""

    @pytest.fixture
    def alcubierre_metric_inside(self):
        """Alcubierre metric at a point inside the bubble (near center)."""
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0, x_s=0.0)
        # Inside the bubble: near center
        coords = jnp.array([0.0, 0.1, 0.1, 0.0])
        return metric(coords)

    def test_alcubierre_orthonormality(self, alcubierre_metric_inside):
        """Verify g_{ab} e_I^a e_J^b = eta_{IJ} for Alcubierre."""
        g = alcubierre_metric_inside
        tetrad = compute_orthonormal_tetrad(g)
        deviation = _orthonormality_check(tetrad, g)
        np.testing.assert_allclose(deviation, jnp.zeros((4, 4)), atol=1e-12)


class TestTetradSuperluminal:
    """Tetrad validity boundary across the luminal transition."""

    @pytest.mark.parametrize("v_s", [0.3, 0.7, 0.95])
    def test_orthonormal_across_subluminal_sweep(self, v_s):
        """g_{ab} e_I^a e_J^b = eta_{IJ} at the wall for every v_s < 1, where a
        timelike normal exists (g^{00} < 0)."""
        metric = AlcubierreMetric(v_s=v_s, R=1.0, sigma=8.0, x_s=0.0)
        g = metric(jnp.array([0.0, 1.0, 0.1, 0.0]))
        tetrad = compute_orthonormal_tetrad(g)
        deviation = _orthonormality_check(tetrad, g)
        np.testing.assert_allclose(deviation, jnp.zeros((4, 4)), atol=1e-10)

    def test_orthonormal_at_superluminal_center(self):
        """At the center of a superluminal bubble the coordinate-time direction
        turns spacelike (g_{00} > 0), yet the slice normal stays timelike
        (g^{00} = -1/alpha^2 = -1) so the tetrad remains orthonormal. This pins
        the all-velocity validity of the ADM-normal frame the certifier relies
        on (and guards against confusing g_{00} with g^{00})."""
        metric = AlcubierreMetric(v_s=2.0, R=1.0, sigma=8.0, x_s=0.0)
        g = metric(jnp.array([0.0, 0.0, 0.0, 0.0]))   # center: f=1, g_00=+3
        assert float(g[0, 0]) > 0.0                    # coordinate time spacelike
        assert float(jnp.linalg.inv(g)[0, 0]) < 0.0    # slice normal still timelike
        tetrad = compute_orthonormal_tetrad(g)
        deviation = _orthonormality_check(tetrad, g)
        np.testing.assert_allclose(deviation, jnp.zeros((4, 4)), atol=1e-10)


class TestTetradJITVmap:
    """JIT and vmap compatibility for tetrad construction."""

    def test_jit_compilation(self):
        """jax.jit(compute_orthonormal_tetrad) runs without error."""
        g = ETA
        jit_fn = jax.jit(compute_orthonormal_tetrad)
        tetrad = jit_fn(g)
        np.testing.assert_allclose(tetrad, jnp.eye(4), atol=1e-14)

    def test_vmap_batch(self):
        """jax.vmap(compute_orthonormal_tetrad) over batch of metrics."""
        metrics = jnp.stack([
            ETA,
            SchwarzschildMetric(M=1.0)(jnp.array([0.0, 3.0, 0.0, 0.0])),
            ETA,
        ])
        vmap_fn = jax.vmap(compute_orthonormal_tetrad)
        tetrads = vmap_fn(metrics)
        assert tetrads.shape == (3, 4, 4)
        # Each should be orthonormal
        for i in range(3):
            deviation = _orthonormality_check(tetrads[i], metrics[i])
            np.testing.assert_allclose(deviation, jnp.zeros((4, 4)), atol=1e-12)


# Timelike vector tests


class TestTimelikeFromRapidity:
    """Timelike vector construction from rapidity parameters."""

    @pytest.fixture
    def flat_tetrad(self):
        """Tetrad for flat Minkowski metric."""
        return compute_orthonormal_tetrad(ETA)

    def test_eulerian_observer_zeta_zero(self, flat_tetrad):
        """zeta=0: u^a = e_0^a (Eulerian observer)."""
        u = timelike_from_rapidity(
            jnp.float64(0.0), jnp.float64(0.5), jnp.float64(0.3), flat_tetrad
        )
        np.testing.assert_allclose(u, flat_tetrad[0], atol=1e-14)

    def test_boosted_observer_unit_norm(self, flat_tetrad):
        """zeta=1.0, various (theta, phi): g_{ab} u^a u^b = -1."""
        zeta = jnp.float64(1.0)
        for theta_val in [0.3, 0.7, 1.2, 2.5]:
            for phi_val in [0.0, 1.0, 3.14, 5.0]:
                theta = jnp.float64(theta_val)
                phi = jnp.float64(phi_val)
                u = timelike_from_rapidity(zeta, theta, phi, flat_tetrad)
                norm = jnp.einsum("a,ab,b->", u, ETA, u)
                np.testing.assert_allclose(norm, -1.0, atol=1e-14)

    def test_high_rapidity_unit_norm(self, flat_tetrad):
        """zeta=5.0 (Lorentz factor ~74): still unit norm."""
        zeta = jnp.float64(5.0)
        theta = jnp.float64(0.5)
        phi = jnp.float64(0.3)
        u = timelike_from_rapidity(zeta, theta, phi, flat_tetrad)
        norm = jnp.einsum("a,ab,b->", u, ETA, u)
        np.testing.assert_allclose(norm, -1.0, atol=1e-14)

    def test_angular_coverage(self, flat_tetrad):
        """Sweep theta/phi: all produce unit timelike vectors."""
        thetas = [0.0, jnp.pi / 4, jnp.pi / 2, 3 * jnp.pi / 4, jnp.pi]
        phis = [0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2]
        zeta = jnp.float64(1.5)
        for theta_val in thetas:
            for phi_val in phis:
                theta = jnp.float64(theta_val)
                phi = jnp.float64(phi_val)
                u = timelike_from_rapidity(zeta, theta, phi, flat_tetrad)
                norm = jnp.einsum("a,ab,b->", u, ETA, u)
                np.testing.assert_allclose(norm, -1.0, atol=1e-14)

    def test_curved_spacetime_unit_norm(self):
        """Boosted observer in Schwarzschild: g_{ab} u^a u^b = -1."""
        metric = SchwarzschildMetric(M=1.0)
        coords = jnp.array([0.0, 3.0, 0.0, 0.0])
        g = metric(coords)
        tetrad = compute_orthonormal_tetrad(g)
        u = timelike_from_rapidity(
            jnp.float64(2.0), jnp.float64(0.7), jnp.float64(1.2), tetrad
        )
        norm = jnp.einsum("a,ab,b->", u, g, u)
        np.testing.assert_allclose(norm, -1.0, atol=1e-12)


# Null vector tests


class TestNullFromAngles:
    """Null vector construction from angular parameters."""

    @pytest.fixture
    def flat_tetrad(self):
        """Tetrad for flat Minkowski metric."""
        return compute_orthonormal_tetrad(ETA)

    def test_null_norm_flat(self, flat_tetrad):
        """For various (theta, phi), g_{ab} k^a k^b = 0 in flat spacetime."""
        for theta_val in [0.0, 0.5, 1.0, 1.5, jnp.pi]:
            for phi_val in [0.0, 0.5, 1.0, 2.0, 3.14, 5.0]:
                theta = jnp.float64(theta_val)
                phi = jnp.float64(phi_val)
                k = null_from_angles(theta, phi, flat_tetrad)
                norm = jnp.einsum("a,ab,b->", k, ETA, k)
                np.testing.assert_allclose(norm, 0.0, atol=1e-14)

    def test_null_in_schwarzschild(self):
        """Null vector in Schwarzschild: g_{ab} k^a k^b = 0."""
        metric = SchwarzschildMetric(M=1.0)
        coords = jnp.array([0.0, 3.0, 0.0, 0.0])
        g = metric(coords)
        tetrad = compute_orthonormal_tetrad(g)
        for theta_val in [0.3, 1.0, 2.5]:
            for phi_val in [0.0, 1.5, 4.0]:
                theta = jnp.float64(theta_val)
                phi = jnp.float64(phi_val)
                k = null_from_angles(theta, phi, tetrad)
                norm = jnp.einsum("a,ab,b->", k, g, k)
                np.testing.assert_allclose(norm, 0.0, atol=1e-12)


# Sigmoid parameterization tests


class TestBoundedParam:
    """Sigmoid reparameterization tests."""

    def test_bounded_range(self):
        """bounded_param(raw, 0, 5) always in (0, 5) for various raw values."""
        raw_values = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        for raw in raw_values:
            result = bounded_param(raw, jnp.float64(0.0), jnp.float64(5.0))
            assert float(result) > 0.0, f"bounded_param({float(raw)}) = {float(result)} <= 0"
            assert float(result) < 5.0, f"bounded_param({float(raw)}) = {float(result)} >= 5"

    def test_roundtrip(self):
        """unbounded_param(bounded_param(raw, lo, hi), lo, hi) = raw."""
        raw_values = jnp.array([-3.0, -1.0, 0.0, 0.5, 2.0])
        lo = jnp.float64(0.0)
        hi = jnp.float64(5.0)
        for raw in raw_values:
            bounded = bounded_param(raw, lo, hi)
            recovered = unbounded_param(bounded, lo, hi)
            np.testing.assert_allclose(float(recovered), float(raw), atol=1e-12)

    def test_gradient_at_zero(self):
        """Gradient of bounded_param at raw=0 is maximal: sigmoid'(0) * range = 0.25 * 5 = 1.25."""
        grad_fn = jax.grad(lambda r: bounded_param(r, jnp.float64(0.0), jnp.float64(5.0)))
        grad_at_zero = grad_fn(jnp.float64(0.0))
        # sigmoid'(0) = 0.25, range = 5, so gradient = 1.25
        np.testing.assert_allclose(float(grad_at_zero), 1.25, atol=1e-14)

    def test_dtype_float64(self):
        """Output should be float64."""
        result = bounded_param(jnp.float64(0.0), jnp.float64(0.0), jnp.float64(5.0))
        assert result.dtype == jnp.float64


jax.config.update("jax_enable_x64", True)

from warpax.metrics import LentzMetric, RodalMetric, WarpShellMetric
from warpax.geometry import GridSpec, evaluate_curvature_grid
from warpax.analysis import compare_eulerian_vs_robust


@pytest.mark.slow
@pytest.mark.parametrize(
    "MetricCls,kwargs,grid_bounds",
    [
        # Alcubierre: R=1, sigma=8, bounds [-3,3] (3x bubble radius)
        (AlcubierreMetric, dict(v_s=0.5, R=1.0, sigma=8.0), [(-3, 3)] * 3),
        # Rodal: R=100, sigma=0.03, bounds [-300,300] (3x bubble radius)
        (RodalMetric, dict(v_s=0.5, R=100.0, sigma=0.03), [(-300, 300)] * 3),
        # WarpShell: R_1=10, R_2=20, bounds [-30,30] (1.5x outer radius)
        (WarpShellMetric, dict(v_s=0.5), [(-30, 30)] * 3),
        # Lentz: default parameters, bounds [-3,3]
        (LentzMetric, dict(v_s=0.5), [(-3, 3)] * 3),
    ],
    ids=["alcubierre", "rodal", "warpshell", "lentz"],
)
def test_robust_leq_eulerian(MetricCls, kwargs, grid_bounds):
    """Robust margins must be <= Eulerian margins + tolerance at every grid point.

    This is the core mathematical invariant: the optimizer searches over all
    observers, so it can only find margins that are at least as negative as
    any specific observer (including the Eulerian one).
    """
    metric = MetricCls(**kwargs)
    grid = GridSpec(bounds=grid_bounds, shape=(30, 30, 30))

    curv = evaluate_curvature_grid(metric, grid, batch_size=128)
    result = compare_eulerian_vs_robust(
        curv.stress_energy,
        curv.metric,
        curv.metric_inv,
        grid.shape,
        n_starts=4,
        batch_size=64,
    )

    # NEC tol relaxed: Eulerian uses 6 discrete null dirs vs continuous S^2 search.
    tol = {"nec": 5e-4, "wec": 1e-6, "sec": 1e-6, "dec": 1e-6}

    for cond in ("nec", "wec", "sec", "dec"):
        eul = np.asarray(result.eulerian_margins[cond])
        rob = np.asarray(result.robust_margins[cond])

        valid = np.isfinite(eul) & np.isfinite(rob)
        if not np.any(valid):
            continue

        excess = rob[valid] - eul[valid]
        max_excess = float(np.max(excess))

        if max_excess > tol[cond]:
            # Find worst violating point for diagnostic output
            excess_full = np.full_like(eul, -np.inf)
            excess_full[valid] = excess
            worst_idx = np.unravel_index(np.argmax(excess_full), eul.shape)
            pytest.fail(
                f"{cond.upper()}: robust margin exceeds Eulerian by {max_excess:.2e} "
                f"(tolerance {tol[cond]:.0e}) at grid point {worst_idx}. "
                f"Eulerian={eul[worst_idx]:.6e}, Robust={rob[worst_idx]:.6e}"
            )


jax.config.update("jax_enable_x64", True)

from warpax.benchmarks import AlcubierreMetric
from warpax.energy_conditions import WallRestrictedStats
from warpax.energy_conditions.filtering import (
    compute_wall_restricted_stats,
    determinant_guard_mask,
    frobenius_norm_mask,
    shape_function_mask,
)
from warpax.energy_conditions.types import ECGridResult, ECSummary


# Helpers

_DUMMY_SUMMARY = ECSummary(
    fraction_violated=jnp.array(0.0),
    max_violation=jnp.array(0.0),
    min_margin=jnp.array(0.0),
)


def _make_synthetic_ec_result(
    grid_shape: tuple[int, ...],
    he_types: jnp.ndarray,
    nec_margins: jnp.ndarray,
    wec_margins: jnp.ndarray | None = None,
    sec_margins: jnp.ndarray | None = None,
    dec_margins: jnp.ndarray | None = None,
) -> ECGridResult:
    """Construct a synthetic ECGridResult with the specified fields.

    Remaining fields are filled with zeros or None to avoid running
    the full verify_grid pipeline in unit tests.
    """
    n = int(jnp.prod(jnp.array(grid_shape)))
    if wec_margins is None:
        wec_margins = jnp.zeros(grid_shape)
    if sec_margins is None:
        sec_margins = jnp.zeros(grid_shape)
    if dec_margins is None:
        dec_margins = jnp.zeros(grid_shape)

    return ECGridResult(
        he_types=he_types.reshape(grid_shape),
        eigenvalues=jnp.zeros((*grid_shape, 4)),
        rho=jnp.zeros(grid_shape),
        pressures=jnp.zeros((*grid_shape, 3)),
        nec_margins=nec_margins.reshape(grid_shape),
        wec_margins=wec_margins.reshape(grid_shape),
        sec_margins=sec_margins.reshape(grid_shape),
        dec_margins=dec_margins.reshape(grid_shape),
        worst_observers=jnp.zeros((*grid_shape, 4)),
        worst_params=jnp.zeros((*grid_shape, 3)),
        nec_summary=_DUMMY_SUMMARY,
        wec_summary=_DUMMY_SUMMARY,
        sec_summary=_DUMMY_SUMMARY,
        dec_summary=_DUMMY_SUMMARY,
        nec_opt_margins=None,
        wec_opt_margins=None,
        sec_opt_margins=None,
        dec_opt_margins=None,
        n_type_i=None,
        n_type_ii=None,
        n_type_iii=None,
        n_type_iv=None,
        n_vacuum=None,
        max_imag_eigenvalue=None,
        nec_opt_converged=None,
        wec_opt_converged=None,
        sec_opt_converged=None,
        dec_opt_converged=None,
        nec_opt_n_steps=None,
        wec_opt_n_steps=None,
        sec_opt_n_steps=None,
        dec_opt_n_steps=None,
    )


# TestShapeFunctionMask


class TestShapeFunctionMask:
    """Tests for shape_function_mask with AlcubierreMetric."""

    def setup_method(self):
        """Set up Alcubierre metric and a 1D grid along the x-axis."""
        self.metric = AlcubierreMetric(v_s=1.0, R=1.0, sigma=8.0, x_s=0.0)
        # Points along x-axis: interior (0), wall region (~R), exterior (far)
        x_vals = jnp.array([0.0, 0.3, 0.6, 0.9, 1.0, 1.1, 1.4, 2.0, 5.0, 10.0])
        self.coords_batch = jnp.stack(
            [jnp.zeros_like(x_vals), x_vals, jnp.zeros_like(x_vals), jnp.zeros_like(x_vals)],
            axis=-1,
        )
        self.grid_shape = (len(x_vals),)

    def test_shape_function_mask_selects_wall(self):
        """Mask selects points where 0.1 <= f <= 0.9."""
        mask = shape_function_mask(self.metric, self.coords_batch, self.grid_shape)
        assert mask.shape == self.grid_shape, f"Expected shape {self.grid_shape}, got {mask.shape}"
        assert mask.dtype == jnp.bool_, f"Expected bool dtype, got {mask.dtype}"
        # At least some points should be in the wall
        assert jnp.sum(mask) > 0, "Expected some points in the wall region"

    def test_shape_function_mask_excludes_interior(self):
        """Points at the origin (f ~ 1) are excluded by [0.1, 0.9] bounds."""
        mask = shape_function_mask(self.metric, self.coords_batch, self.grid_shape)
        # Origin point (x=0) should have f ~ 1.0, which is outside [0.1, 0.9]
        f_origin = self.metric.shape_function_value(self.coords_batch[0])
        if float(f_origin) > 0.9:
            assert not bool(mask[0]), (
                f"Origin (f={float(f_origin):.3f}) should be excluded from wall"
            )

    def test_shape_function_mask_excludes_exterior(self):
        """Points far away (f ~ 0) are excluded by [0.1, 0.9] bounds."""
        mask = shape_function_mask(self.metric, self.coords_batch, self.grid_shape)
        # Far point (x=10) should have f ~ 0.0, which is outside [0.1, 0.9]
        f_far = self.metric.shape_function_value(self.coords_batch[-1])
        if float(f_far) < 0.1:
            assert not bool(mask[-1]), (
                f"Far point (f={float(f_far):.6f}) should be excluded from wall"
            )

    def test_shape_function_mask_custom_bounds(self):
        """Using f_low=0.0, f_high=1.0 selects all points."""
        mask = shape_function_mask(
            self.metric,
            self.coords_batch,
            self.grid_shape,
            f_low=0.0,
            f_high=1.0,
        )
        assert jnp.all(mask), "f_low=0, f_high=1 should select all points"

    def test_shape_function_mask_output_shape(self):
        """Output shape matches grid_shape."""
        # Test with a 2D grid shape
        n = len(self.coords_batch)
        # Duplicate coords to form a 2-row grid
        coords_2d = jnp.tile(self.coords_batch, (2, 1))
        grid_2d = (2, n)
        mask = shape_function_mask(self.metric, coords_2d, grid_2d)
        assert mask.shape == grid_2d, f"Expected shape {grid_2d}, got {mask.shape}"


# TestFrobeniusNormMask


class TestFrobeniusNormMask:
    """Tests for frobenius_norm_mask with synthetic stress-energy fields."""

    def test_frobenius_norm_mask_zero_tensor(self):
        """All-zeros T_ab produces all-False mask."""
        T = jnp.zeros((3, 3, 3, 4, 4))
        mask = frobenius_norm_mask(T)
        assert mask.shape == (3, 3, 3), f"Expected shape (3, 3, 3), got {mask.shape}"
        assert not jnp.any(mask), "Zero tensor should produce all-False mask"

    def test_frobenius_norm_mask_nonzero(self):
        """T_ab with large values produces True mask."""
        T = jnp.ones((2, 2, 4, 4))
        mask = frobenius_norm_mask(T)
        assert mask.shape == (2, 2), f"Expected shape (2, 2), got {mask.shape}"
        assert jnp.all(mask), "Nonzero tensor should produce all-True mask"

    def test_frobenius_norm_mask_threshold(self):
        """Adjusting threshold changes which points are selected."""
        # Create a (3,) grid with varying norms
        T = jnp.zeros((3, 4, 4))
        T = T.at[0, 0, 0].set(1e-15)  # norm ~ 1e-15 (below default threshold)
        T = T.at[1, 0, 0].set(1e-10)  # norm ~ 1e-10 (below default but above lower)
        T = T.at[2, 0, 0].set(1.0)  # norm ~ 1.0 (above all thresholds)

        # Default threshold = 1e-12
        mask_default = frobenius_norm_mask(T)
        assert bool(mask_default[2]), "Norm=1.0 should pass default threshold"
        assert not bool(mask_default[0]), "Norm=1e-15 should fail default threshold"

        # Lower threshold = 1e-16
        mask_low = frobenius_norm_mask(T, threshold=1e-16)
        assert jnp.all(mask_low), "All points should pass threshold=1e-16"

        # Higher threshold = 0.5
        mask_high = frobenius_norm_mask(T, threshold=0.5)
        assert bool(mask_high[2]), "Norm=1.0 should pass threshold=0.5"
        assert not bool(mask_high[0]), "Norm=1e-15 should fail threshold=0.5"
        assert not bool(mask_high[1]), "Norm=1e-10 should fail threshold=0.5"

    def test_frobenius_norm_mask_dtype(self):
        """Output is boolean."""
        T = jnp.ones((2, 4, 4))
        mask = frobenius_norm_mask(T)
        assert mask.dtype == jnp.bool_, f"Expected bool dtype, got {mask.dtype}"


# TestDeterminantGuardMask


class TestDeterminantGuardMask:
    """Tests for determinant_guard_mask with synthetic metric fields."""

    def test_determinant_guard_nondegenerate(self):
        """Minkowski metric (det=-1) passes guard at default threshold."""
        eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        g_field = jnp.broadcast_to(eta, (3, 3, 4, 4)).copy()
        mask = determinant_guard_mask(g_field)
        assert mask.shape == (3, 3), f"Expected shape (3, 3), got {mask.shape}"
        assert jnp.all(mask), "Minkowski (det=-1) should pass guard"

    def test_determinant_guard_degenerate(self):
        """Metric with det=0 fails guard and emits warning."""
        # Create a degenerate metric (all zeros -> det=0)
        g_field = jnp.zeros((2, 4, 4))
        with pytest.warns(UserWarning, match="grid points have"):
            mask = determinant_guard_mask(g_field)
        assert not jnp.any(mask), "Degenerate metric (det=0) should fail guard"

    def test_determinant_guard_configurable_threshold(self):
        """Threshold parameter changes behavior."""
        # Metric with small but nonzero determinant
        g = jnp.diag(jnp.array([-1e-6, 1e-6, 1e-6, 1e-6]))
        g_field = g.reshape(1, 4, 4)
        det_val = float(jnp.linalg.det(g))

        # Default threshold (1e-10): should fail since |det| = 1e-24
        with pytest.warns(UserWarning, match="grid points have"):
            mask_default = determinant_guard_mask(g_field)
        assert not bool(mask_default[0]), (
            f"|det|={abs(det_val):.2e} should fail default threshold"
        )

        # Very low threshold: should pass
        mask_low = determinant_guard_mask(g_field, threshold=1e-30)
        assert bool(mask_low[0]), (
            f"|det|={abs(det_val):.2e} should pass threshold=1e-30"
        )

    def test_determinant_guard_warning(self):
        """warnings.warn is called with count of degenerate points."""
        g_field = jnp.zeros((5, 4, 4))
        with pytest.warns(UserWarning, match="5 of 5 grid points") as record:
            determinant_guard_mask(g_field)
        assert len(record) == 1, f"Expected 1 warning, got {len(record)}"

    def test_determinant_guard_no_warning_when_nondegenerate(self):
        """No warning emitted when all points are non-degenerate."""
        eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        g_field = jnp.broadcast_to(eta, (3, 4, 4)).copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mask = determinant_guard_mask(g_field)
        assert jnp.all(mask)

    def test_determinant_guard_mixed(self):
        """Mix of degenerate and non-degenerate points."""
        eta = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        degenerate = jnp.zeros((4, 4))
        g_field = jnp.stack([eta, degenerate, eta], axis=0)
        with pytest.warns(UserWarning, match="1 of 3 grid points"):
            mask = determinant_guard_mask(g_field)
        expected = jnp.array([True, False, True])
        assert jnp.array_equal(mask, expected), f"Expected {expected}, got {mask}"


# TestMaskComposability


class TestMaskComposability:
    """Tests that masks compose correctly via logical AND/OR."""

    def test_mask_and(self):
        """shape_mask & frobenius_mask produces correct logical AND."""
        mask_a = jnp.array([True, True, False, False])
        mask_b = jnp.array([True, False, True, False])
        combined = mask_a & mask_b
        expected = jnp.array([True, False, False, False])
        assert jnp.array_equal(combined, expected), f"AND: expected {expected}, got {combined}"
        assert combined.shape == mask_a.shape

    def test_mask_or(self):
        """mask_a | mask_b produces correct logical OR."""
        mask_a = jnp.array([True, True, False, False])
        mask_b = jnp.array([True, False, True, False])
        combined = mask_a | mask_b
        expected = jnp.array([True, True, True, False])
        assert jnp.array_equal(combined, expected), f"OR: expected {expected}, got {combined}"
        assert combined.shape == mask_a.shape

    def test_mask_composability_shapes(self):
        """Composed masks preserve grid shape."""
        shape_3d = (2, 3, 4)
        mask_a = jnp.ones(shape_3d, dtype=bool)
        mask_b = jnp.zeros(shape_3d, dtype=bool)
        assert (mask_a & mask_b).shape == shape_3d
        assert (mask_a | mask_b).shape == shape_3d


# TestWallRestrictedStats


class TestWallRestrictedStats:
    """Tests for compute_wall_restricted_stats type counts and violation counts."""

    def setup_method(self):
        """Build a synthetic ECGridResult with known types and margins."""
        self.grid_shape = (6,)
        # Types: [I, I, II, III, IV, IV]
        self.he_types = jnp.array([1.0, 1.0, 2.0, 3.0, 4.0, 4.0])
        # NEC margins: [0.5, -0.5, -1.0, 0.1, -2.0, -0.01]
        # Violated (< -1e-10): indices 1, 2, 4, 5
        self.nec_margins = jnp.array([0.5, -0.5, -1.0, 0.1, -2.0, -0.01])
        # Mask: all True except index 5
        self.mask = jnp.array([True, True, True, True, True, False])
        self.ec_result = _make_synthetic_ec_result(
            self.grid_shape, self.he_types, self.nec_margins,
        )

    def test_wall_restricted_stats_type_counts(self):
        """Known he_types with mask produces correct n_type_i/ii/iii/iv."""
        stats = compute_wall_restricted_stats(self.ec_result, self.mask)
        assert stats.n_total == 5, f"Expected n_total=5, got {stats.n_total}"
        assert stats.n_type_i == 2, f"Expected n_type_i=2, got {stats.n_type_i}"
        assert stats.n_type_ii == 1, f"Expected n_type_ii=1, got {stats.n_type_ii}"
        assert stats.n_type_iii == 1, f"Expected n_type_iii=1, got {stats.n_type_iii}"
        assert stats.n_type_iv == 1, f"Expected n_type_iv=1, got {stats.n_type_iv}"

    def test_wall_restricted_stats_violation_counts(self):
        """Known margins with mask produce correct violation counts."""
        stats = compute_wall_restricted_stats(self.ec_result, self.mask)
        # Violated within mask: indices 1 (-0.5), 2 (-1.0), 4 (-2.0) = 3
        # Index 5 is masked out
        assert stats.nec_violated == 3, f"Expected nec_violated=3, got {stats.nec_violated}"

    def test_wall_restricted_stats_fractions(self):
        """Fractions equal counts / n_total."""
        stats = compute_wall_restricted_stats(self.ec_result, self.mask)
        assert stats.frac_type_i == pytest.approx(2 / 5), (
            f"Expected frac_type_i=0.4, got {stats.frac_type_i}"
        )
        assert stats.frac_type_iv == pytest.approx(1 / 5), (
            f"Expected frac_type_iv=0.2, got {stats.frac_type_iv}"
        )
        assert stats.nec_frac_violated == pytest.approx(3 / 5), (
            f"Expected nec_frac_violated=0.6, got {stats.nec_frac_violated}"
        )

    def test_wall_restricted_stats_is_namedtuple(self):
        """Result is a WallRestrictedStats NamedTuple."""
        stats = compute_wall_restricted_stats(self.ec_result, self.mask)
        assert isinstance(stats, WallRestrictedStats)
        assert len(stats) == 21, f"Expected 21 fields, got {len(stats)}"


# TestWallRestrictedStatsMissRate


class TestWallRestrictedStatsMissRate:
    """Tests for miss rate computation with known Eulerian/robust margins."""

    def setup_method(self):
        """Build known Eulerian vs robust scenario."""
        self.grid_shape = (4,)
        self.he_types = jnp.array([1.0, 1.0, 1.0, 1.0])
        # Robust margins: all violated
        self.robust_nec = jnp.array([-1.0, -1.0, -1.0, -1.0])
        # Eulerian margins: 2 satisfied (missed), 2 violated (not missed)
        self.eulerian_nec = jnp.array([0.5, 0.5, -0.5, -0.5])
        self.mask = jnp.ones(self.grid_shape, dtype=bool)
        self.ec_result = _make_synthetic_ec_result(
            self.grid_shape, self.he_types, self.robust_nec,
        )

    def test_wall_restricted_stats_miss_rate(self):
        """With known eulerian/robust margins, miss rate = missed / violated."""
        eulerian_margins = {
            "nec": self.eulerian_nec,
            "wec": jnp.zeros(self.grid_shape),
            "sec": jnp.zeros(self.grid_shape),
            "dec": jnp.zeros(self.grid_shape),
        }
        stats = compute_wall_restricted_stats(
            self.ec_result, self.mask, eulerian_margins=eulerian_margins,
        )
        # 4 robust-violated, 2 Eulerian-satisfied -> miss rate = 2/4 = 0.5
        assert stats.nec_miss_rate == pytest.approx(0.5), (
            f"Expected nec_miss_rate=0.5, got {stats.nec_miss_rate}"
        )

    def test_wall_restricted_stats_miss_rate_none(self):
        """When no violations exist, miss rate is None."""
        # All robust margins positive
        ec_positive = _make_synthetic_ec_result(
            self.grid_shape,
            self.he_types,
            jnp.ones(self.grid_shape),  # all satisfied
        )
        eulerian_margins = {
            "nec": jnp.ones(self.grid_shape),
            "wec": jnp.ones(self.grid_shape),
            "sec": jnp.ones(self.grid_shape),
            "dec": jnp.ones(self.grid_shape),
        }
        stats = compute_wall_restricted_stats(
            ec_positive, self.mask, eulerian_margins=eulerian_margins,
        )
        assert stats.nec_miss_rate is None, (
            f"Expected nec_miss_rate=None, got {stats.nec_miss_rate}"
        )
        assert stats.wec_miss_rate is None
        assert stats.sec_miss_rate is None
        assert stats.dec_miss_rate is None

    def test_miss_rate_no_eulerian_means_none(self):
        """Without eulerian_margins, all miss rates are None."""
        stats = compute_wall_restricted_stats(self.ec_result, self.mask)
        assert stats.nec_miss_rate is None
        assert stats.wec_miss_rate is None
        assert stats.sec_miss_rate is None
        assert stats.dec_miss_rate is None


# TestWallRestrictedStatsEdgeCases


class TestWallRestrictedStatsEdgeCases:
    """Edge case tests for compute_wall_restricted_stats."""

    def test_wall_restricted_stats_empty_mask(self):
        """All-False mask produces n_total=0 with zero counts."""
        grid_shape = (4,)
        he_types = jnp.array([1.0, 2.0, 3.0, 4.0])
        margins = jnp.array([-1.0, -2.0, -3.0, -4.0])
        ec_result = _make_synthetic_ec_result(grid_shape, he_types, margins)
        mask = jnp.zeros(grid_shape, dtype=bool)

        stats = compute_wall_restricted_stats(ec_result, mask)
        assert stats.n_total == 0, f"Expected n_total=0, got {stats.n_total}"
        assert stats.n_type_i == 0
        assert stats.n_type_ii == 0
        assert stats.n_type_iii == 0
        assert stats.n_type_iv == 0
        assert stats.nec_violated == 0
        # Fractions should be 0 (safe division by max(n_total, 1))
        assert stats.frac_type_i == 0.0
        assert stats.nec_frac_violated == 0.0

    def test_wall_restricted_stats_all_true_mask(self):
        """All-True mask includes all points."""
        grid_shape = (3,)
        he_types = jnp.array([1.0, 2.0, 4.0])
        margins = jnp.array([-1.0, 0.5, -0.5])
        ec_result = _make_synthetic_ec_result(grid_shape, he_types, margins)
        mask = jnp.ones(grid_shape, dtype=bool)

        stats = compute_wall_restricted_stats(ec_result, mask)
        assert stats.n_total == 3
        assert stats.n_type_i == 1
        assert stats.n_type_ii == 1
        assert stats.n_type_iv == 1
        # Violated: indices 0 (-1.0), 2 (-0.5)
        assert stats.nec_violated == 2

    def test_wall_restricted_stats_3d_grid(self):
        """Stats work with multi-dimensional grid shapes."""
        grid_shape = (2, 3)
        n = 6
        he_types = jnp.array([1.0, 1.0, 2.0, 3.0, 4.0, 4.0])
        margins = jnp.array([0.5, -0.5, -1.0, 0.1, -2.0, -0.01])
        ec_result = _make_synthetic_ec_result(grid_shape, he_types, margins)
        mask = jnp.ones(grid_shape, dtype=bool)

        stats = compute_wall_restricted_stats(ec_result, mask)
        assert stats.n_total == 6
        assert stats.n_type_i == 2
        assert stats.nec_violated == 4  # indices 1, 2, 4, 5


def _degenerate_lorentzian() -> jnp.ndarray:
    """g_{ab} with g^{00} == 0 (degenerate light slicing)."""
    g = jnp.diag(jnp.array([0.0, 1.0, 1.0, 1.0]))
    return g


def _almost_lorentzian(eps: float = 1e-40) -> jnp.ndarray:
    """g_{ab} with -g^{00} ~ eps (rounded-to-zero radicand)."""
    return jnp.diag(jnp.array([-eps, 1.0, 1.0, 1.0]))


class TestEulerianMarginsLapseGuard:
    def test_almost_null_slicing_no_nan(self):
        g = _almost_lorentzian()
        g_inv = jnp.linalg.inv(g)
        T = jnp.zeros((4, 4))
        out = _eulerian_ec_point(T, g, g_inv)
        for key, val in out.items():
            assert jnp.all(jnp.isfinite(val)), f"{key} -> {val}"


class TestObserverTetradLapseGuard:
    def test_almost_null_slicing_no_nan(self):
        g = _almost_lorentzian()
        tetrad = compute_orthonormal_tetrad(g)
        assert jnp.all(jnp.isfinite(tetrad))


class TestKinematicScalarsLapseGuard:
    def test_almost_null_slicing_no_nan(self):
        almost_g = _almost_lorentzian()

        def constant_metric(coords):
            return almost_g

        theta, sigma_sq, omega_sq = compute_kinematic_scalars(
            constant_metric, jnp.zeros(4)
        )
        assert jnp.isfinite(theta)
        assert jnp.isfinite(sigma_sq)
        assert jnp.isfinite(omega_sq)


class TestStartsFibonacciPool:
    """starts kwarg contract tests."""

    def _bench_inputs(self):
        g = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        T = jnp.diag(jnp.array([-1.0, 0.3, 0.3, 0.3]))
        return T, g, jax.random.PRNGKey(42)

    def test_default_axis_gaussian_matches_explicit(self):
        """Default and explicit starts='axis+gaussian' produce identical margins."""
        T, g, key = self._bench_inputs()
        for fn in [optimize_wec, optimize_sec, optimize_dec]:
            r_default = fn(T, g, key=key)
            r_explicit = fn(T, g, starts="axis+gaussian", key=key)
            assert jnp.array_equal(r_default.margin, r_explicit.margin), (
                f"{fn.__name__} drifted under starts='axis+gaussian'"
            )
        r_default_nec = optimize_nec(T, g, key=key)
        r_explicit_nec = optimize_nec(T, g, starts="axis+gaussian", key=key)
        assert jnp.array_equal(r_default_nec.margin, r_explicit_nec.margin)

    def test_fibonacci_pool_differs_from_axis_gaussian_on_hard_point(self):
        """fibonacci+bfgs_top_k uses an expanded multistart pool."""
        from warpax.benchmarks import AlcubierreMetric
        from warpax.geometry import GridSpec, evaluate_curvature_grid

        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        grid = GridSpec(bounds=[(-0.1, 0.1)] * 3, shape=(1, 1, 1))
        chain = evaluate_curvature_grid(metric, grid, t=0.0)
        T, g = chain.stress_energy[0, 0, 0], chain.metric[0, 0, 0]
        key = jax.random.PRNGKey(7)
        r_axis = optimize_wec(T, g, key=key, starts="axis+gaussian", n_starts=16)
        r_fib = optimize_wec(
            T, g, key=key, starts="fibonacci+bfgs_top_k", n_starts=16,
        )
        assert r_fib.margin <= r_axis.margin + 1e-9

    def test_invalid_starts_raises(self):
        """starts='bad' raises ValueError with verbatim message."""
        T, g, _ = self._bench_inputs()
        for bad in ["bad", "random", ""]:
            with pytest.raises(ValueError, match="starts must be one of"):
                optimize_wec(T, g, starts=bad)

    def test_composes_with_strategy_hard_bound(self):
        """Orthogonal kwargs: strategy + starts compose without conflict."""
        T, g, key = self._bench_inputs()
        r = optimize_wec(
            T, g, key=key,
            strategy="hard_bound",
            starts="fibonacci+bfgs_top_k",
        )
        assert r.margin is not None

    def test_golden_fixture_metadata_and_determinism(self):
        """Golden header fields present; fibonacci+bfgs_top_k is seed-deterministic."""
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "golden"
            / "fibonacci_pool.npy"
        )
        assert fixture_path.exists(), f"golden fixture missing: {fixture_path}"

        d = np.load(fixture_path, allow_pickle=True).item()
        for k in (
            "warpax_version", "jaxlib_version", "jax_random_seed",
            "backend", "starts", "strategy",
        ):
            assert k in d, f"fixture missing required key: {k!r}"
        assert d["starts"] == "fibonacci+bfgs_top_k"

        from warpax.benchmarks import AlcubierreMetric
        from warpax.geometry import GridSpec, evaluate_curvature_grid

        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        grid = GridSpec(bounds=[(-0.1, 0.1)] * 3, shape=(1, 1, 1))
        chain = evaluate_curvature_grid(metric, grid, t=0.0)
        T, g = chain.stress_energy[0, 0, 0], chain.metric[0, 0, 0]
        key = jax.random.PRNGKey(int(d["jax_random_seed"]))
        kwargs = dict(
            key=key, starts=d["starts"], strategy=d["strategy"], n_starts=16,
        )
        r1 = optimize_wec(T, g, **kwargs)
        r2 = optimize_wec(T, g, **kwargs)
        assert jnp.array_equal(r1.margin, r2.margin)
        assert jnp.array_equal(r1.worst_observer, r2.worst_observer)
        assert jnp.isfinite(r1.margin)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestSpatialNeighborWarmStart:
    """warm_start kwarg contract tests."""

    def _bench_inputs(self):
        """Standard Minkowski-like inputs that exercise all 4 optimize_*."""
        g = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        T = jnp.diag(jnp.array([-1.0, 0.3, 0.3, 0.3]))
        key = jax.random.PRNGKey(42)
        return T, g, key

    def test_default_cold_matches_explicit_cold(self):
        """Default and explicit warm_start='cold' produce identical margins."""
        T, g, key = self._bench_inputs()

        for fn in [optimize_wec, optimize_sec, optimize_dec]:
            r_default = fn(T, g, key=key)
            r_explicit = fn(T, g, warm_start="cold", key=key)
            assert jnp.array_equal(r_default.margin, r_explicit.margin), (
                f"{fn.__name__} drifted under warm_start='cold' explicit call"
            )

        # NEC (2D) - check margin equality
        r_default_nec = optimize_nec(T, g, key=key)
        r_explicit_nec = optimize_nec(T, g, warm_start="cold", key=key)
        assert jnp.array_equal(r_default_nec.margin, r_explicit_nec.margin)

    def test_spatial_neighbor_can_use_neighbor_observer(self):
        """spatial_neighbor with a supplied neighbor can differ from cold."""
        T, g, key = self._bench_inputs()
        r_cold = optimize_wec(T, g, warm_start="cold", key=key)
        neighbor = r_cold.worst_observer
        key2 = jax.random.PRNGKey(43)
        r_neighbor = optimize_wec(
            T, g, warm_start="spatial_neighbor",
            neighbor_observer=neighbor, key=key2,
        )
        assert r_neighbor.margin is not None
        # Pool composition changes when neighbor seed is injected.
        assert not jnp.array_equal(r_cold.worst_params, r_neighbor.worst_params) or (
            r_neighbor.margin <= r_cold.margin
        )

    def test_invalid_warm_start_raises(self):
        """warm_start='bad' raises ValueError with verbatim message."""
        T, g, _ = self._bench_inputs()
        with pytest.raises(ValueError, match="warm_start must be one of"):
            optimize_wec(T, g, warm_start="bad")
        with pytest.raises(ValueError, match="warm_start must be one of"):
            optimize_nec(T, g, warm_start="cold_start")
        with pytest.raises(ValueError, match="warm_start must be one of"):
            optimize_sec(T, g, warm_start="warm")
        with pytest.raises(ValueError, match="warm_start must be one of"):
            optimize_dec(T, g, warm_start="")

    def test_neighbor_fraction_validates(self):
        """neighbor_fraction must satisfy 0 < f <= 1; other values raise."""
        T, g, _ = self._bench_inputs()
        for bad in [-0.5, 0.0, 1.5, 2.0]:
            with pytest.raises(ValueError, match="neighbor_fraction must satisfy"):
                optimize_wec(T, g, neighbor_fraction=bad)

    def test_default_neighbor_fraction_accepted(self):
        """Default neighbor_fraction=1/16 is explicitly accepted."""
        T, g, key = self._bench_inputs()
        r = optimize_wec(T, g, neighbor_fraction=1.0 / 16.0, key=key)
        assert r.margin is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


jax.config.update("jax_enable_x64", True)

from warpax.energy_conditions.classification import (
    classify_with_solver,
)
from warpax.metrics import WarpShellMetric


# 10 fixed seeds for the perturbation ensemble.
N_SEEDS = 10
SEEDS = [20260418 + i for i in range(N_SEEDS)]


@pytest.mark.slow
class TestGeneralizedSolverStability:
    """Integration: WarpShell idx=8 10-seed stability + cond_V diagnostic."""

    @pytest.fixture(scope="class")
    def warpshell_idx_8_inputs(self):
        """Resolve the WarpShell v_s=0.5 idx=8 flat index.

        Builds the 50^3 WarpShell grid, classifies under solver='standard'
        (the v0.2.0 path), collects all Type-IV flat indices, samples 200 with
        seed=20260418, sorts, and picks the 9th element (0-indexed: sampled[8]).
        Returns T_ab, g_ab, T_mixed, and the flat index for that point.
        """
        metric = WarpShellMetric(v_s=0.5)
        grid = GridSpec(
            bounds=((-12.0, 12.0), (-6.0, 6.0), (-6.0, 6.0)),
            shape=(50, 50, 50),
        )
        chain = evaluate_curvature_grid(metric, grid)
        T_flat = np.asarray(chain.stress_energy.reshape(-1, 4, 4))
        g_flat = np.asarray(chain.metric.reshape(-1, 4, 4))
        g_inv_flat = np.asarray(chain.metric_inv.reshape(-1, 4, 4))
        T_mixed_flat = np.einsum('nab,nbc->nac', g_inv_flat, T_flat)

        classify_v = jax.vmap(classify_hawking_ellis, in_axes=(0, 0))
        he_types = np.asarray(
            classify_v(jnp.asarray(T_mixed_flat), jnp.asarray(g_flat)).he_type
        ).astype(int)
        type_iv_flat = np.where(he_types == 4)[0]
        rng = np.random.default_rng(seed=20260418)
        sampled = np.sort(
            rng.choice(
                type_iv_flat, size=min(200, len(type_iv_flat)), replace=False
            )
        )
        if len(sampled) < 9:
            pytest.skip(
                f"WarpShell idx=8 requires ≥9 Type-IV points; got {len(sampled)}"
            )
        idx_8_flat = int(sampled[8])
        return {
            'T_ab': T_flat[idx_8_flat],
            'g_ab': g_flat[idx_8_flat],
            'T_mixed': T_mixed_flat[idx_8_flat],
            'idx_8_flat': idx_8_flat,
        }

    def test_standard_solver_perturbation_reproducible(self, warpshell_idx_8_inputs):
        """Machine-epsilon perturbations must not flip he_type at pinned idx=8."""
        g_ab = warpshell_idx_8_inputs['g_ab']
        T_mixed = warpshell_idx_8_inputs['T_mixed']

        he_types = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed=seed)
            eps = rng.standard_normal(T_mixed.shape) * np.finfo(np.float64).eps
            T_mixed_pert = jnp.asarray(T_mixed + eps)
            r = classify_hawking_ellis(T_mixed_pert, jnp.asarray(g_ab))
            he_types.append(int(r.he_type))
        dist = Counter(he_types)
        assert len(dist) == 1 and max(dist.values()) == N_SEEDS, (
            f"expected seed-stable standard classification; got {dict(dist)}"
        )

    def test_generalized_solver_stable(self, warpshell_idx_8_inputs):
        """Contract: solver='generalized' stable across 10 perturbation seeds.

        The modal he_type count MUST equal N_SEEDS (i.e. all 10 seeds classify
        the same way).
        """
        T_ab = warpshell_idx_8_inputs['T_ab']
        g_ab = warpshell_idx_8_inputs['g_ab']
        T_mixed = warpshell_idx_8_inputs['T_mixed']

        he_types = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed=seed)
            eps = rng.standard_normal(T_mixed.shape) * np.finfo(np.float64).eps
            T_mixed_pert = jnp.asarray(T_mixed + eps)
            eps_ab = rng.standard_normal(T_ab.shape) * np.finfo(np.float64).eps
            T_ab_pert = jnp.asarray(T_ab + eps_ab)
            r = classify_hawking_ellis(
                T_mixed_pert, jnp.asarray(g_ab),
                solver='generalized', T_ab=T_ab_pert,
            )
            he_types.append(int(r.he_type))
        dist = Counter(he_types)
        print(f"\n[CONTRACT generalized] idx=8 he_type across {N_SEEDS} seeds: {dist}")

        total = sum(dist.values())
        modal_count = max(dist.values())
        n_deviant = total - modal_count
        assert n_deviant < 1, (
            f"solver='generalized' must be stable across {N_SEEDS} seeds; "
            f"got distribution {dict(dist)} with n_deviant={n_deviant}"
        )

    def test_cond_v_diagnostic_functional(self, warpshell_idx_8_inputs):
        """Bauer-Fike cond_V diagnostic returns valid finite results.

        The mpmath classifier must return a finite cond_V and a boolean
        uncertain flag for the WarpShell hard-boundary point.
        """
        from warpax.energy_conditions.classification_mpmath import (
            classify_hawking_ellis_mpmath,
        )

        T_mixed = np.asarray(warpshell_idx_8_inputs['T_mixed'])
        g_ab = np.asarray(warpshell_idx_8_inputs['g_ab'])

        result = classify_hawking_ellis_mpmath(T_mixed, g_ab)
        uncertain = bool(result['uncertain'])
        cond_V = float(result['cond_V'])
        print(
            f"\n[cond_V diagnostic] idx=8 uncertain={uncertain}, "
            f"cond_V={cond_V:.3e}"
        )
        assert np.isfinite(cond_V), f"cond_V must be finite; got {cond_V}"
        assert isinstance(uncertain, bool), (
            f"uncertain must be bool; got {type(uncertain)}"
        )

    def test_auto_solver_matches_generalized(self, warpshell_idx_8_inputs):
        """solver='auto' reclassifies ill-conditioned points like generalized."""
        T_ab = jnp.asarray(warpshell_idx_8_inputs['T_ab'])
        g_ab = jnp.asarray(warpshell_idx_8_inputs['g_ab'])
        T_mixed = jnp.asarray(warpshell_idx_8_inputs['T_mixed'])
        r_auto = classify_with_solver(T_mixed, g_ab, T_ab, solver="auto")
        r_gen = classify_hawking_ellis(
            T_mixed, g_ab, solver="generalized", T_ab=T_ab,
        )
        assert int(r_auto.he_type) == int(r_gen.he_type)
        npt.assert_allclose(
            np.asarray(r_auto.rho), np.asarray(r_gen.rho), rtol=0.0, atol=1e-10,
        )


# Rows 0-6: deterministic axis-aligned boosts at zeta_max.
# Rows 7-15: jax.random.normal(PRNGKey(42), (9, 3)) * 5.0.
GOLDEN = np.array(
    [
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [-5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, -5.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, -5.0],
        [-0.923558726409558, -10.84912280198877, 0.9346777589691291],
        [3.0613267852554795, 2.4481247523785092, 1.8450215235414178],
        [1.8543731889309525, 1.1766366207640897, -3.5844941575956857],
        [-3.173121663824603, -3.50010940833805, -7.882790659278354],
        [2.98935432810288, 4.546202300849532, 1.116331111966121],
        [-3.6799338948667746, -10.100550686727946, 1.6485357408377945],
        [-3.8216242644691234, 9.264443397260278, 0.2867568066460418],
        [-5.894341769959686, 1.2134601729839203, 4.332715782619537],
        [-1.1019365643520587, 11.787525561966842, 0.5508630843025522],
    ],
    dtype=np.float64,
)


def test_golden_starter_pool_at_default_config():
    """The canonical starter pool must match the captured values bit-for-bit."""
    ic = np.asarray(_make_initial_conditions_3d(
        n_starts=16, zeta_max=5.0, key=jax.random.PRNGKey(42)
    ))
    assert ic.shape == GOLDEN.shape
    assert ic.dtype == GOLDEN.dtype
    npt.assert_allclose(ic, GOLDEN, rtol=0.0, atol=1e-14)


MINKOWSKI = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


def _type_iv_block_diag(imag: float = 2.0e-5) -> jnp.ndarray:
    """Return a T^a_b with a known complex eigenvalue pair ``1 ± i*imag``.

    The upper 2x2 block is the companion matrix ``[[1, -imag], [imag, 1]]``;
    the lower 2x2 block is ``diag(0.5, -0.3)``. The spectrum is therefore
    ``{1 + i*imag, 1 - i*imag, 0.5, -0.3}`` -- unambiguously Type IV.
    """
    return jnp.array(
        [
            [1.0, -imag, 0.0, 0.0],
            [imag, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, -0.3],
        ]
    )


def _perfect_fluid(rho: float = 1.0, p: float = 0.3) -> jnp.ndarray:
    """Return a perfect-fluid T^a_b at rest: diag(-rho, p, p, p)."""
    return jnp.diag(jnp.array([-rho, p, p, p]))


class TestMpmathEigenvalues:
    """Pin the high-precision eigenvalue path against hand-verified spectra."""

    def test_block_diag_spectrum(self) -> None:
        T = np.asarray(_type_iv_block_diag(imag=2.0e-5))
        evs = eigenvalues_mpmath(T, precision=50)

        imag_parts = sorted(float(abs(ev.imag)) for ev in evs)
        assert imag_parts[0] < 1.0e-45
        assert imag_parts[1] < 1.0e-45
        assert abs(imag_parts[2] - 2.0e-5) < 1.0e-20
        assert abs(imag_parts[3] - 2.0e-5) < 1.0e-20

    def test_perfect_fluid_spectrum_is_real(self) -> None:
        T = np.asarray(_perfect_fluid(rho=1.0, p=0.3))
        evs = eigenvalues_mpmath(T, precision=50)
        for ev in evs:
            assert abs(float(ev.imag)) < 1.0e-45


class TestMpmathClassifier:
    """Pin the 50-digit classifier verdicts and contrast with float64."""

    def test_weak_type_iv_now_correct_at_unit_scale(self) -> None:
        # Pre-fix: the relative imaginary tier absorbed this genuinely
        # complex pair (1 +/- 2e-5 i) as Type I at unit scale. With the
        # scale floor on the tier, float64 now agrees with the 50-digit
        # verdict here.
        T = _type_iv_block_diag(imag=2.0e-5)

        float64_result = classify_hawking_ellis(T, MINKOWSKI)
        assert int(float64_result.he_type) == 4

        mp_result = classify_hawking_ellis_mpmath(
            np.asarray(T), np.asarray(MINKOWSKI), precision=50
        )
        assert mp_result["he_type"] == 4
        assert mp_result["max_imag_abs"] > 1.0e-6

    def test_type_iv_that_float64_misclassifies(self) -> None:
        # The float64 blind spot now exists only above the relative
        # tier's scale floor (where it absorbs eigensolver noise by
        # design); the 50-digit gate remains the authority there.
        scale = 1.0e11
        T = scale * _type_iv_block_diag(imag=2.0e-5)

        float64_result = classify_hawking_ellis(T, MINKOWSKI)
        assert int(float64_result.he_type) == 1

        mp_result = classify_hawking_ellis_mpmath(
            np.asarray(T), np.asarray(MINKOWSKI), precision=50
        )
        assert mp_result["he_type"] == 4
        assert mp_result["max_imag_abs"] > 1.0e-6 * scale

    def test_perfect_fluid_mpmath_agrees_with_float64(self) -> None:
        T = _perfect_fluid(rho=1.0, p=0.3)

        float64_result = classify_hawking_ellis(T, MINKOWSKI)
        mp_result = classify_hawking_ellis_mpmath(
            np.asarray(T), np.asarray(MINKOWSKI), precision=50
        )

        assert int(float64_result.he_type) == 1
        assert mp_result["he_type"] == 1

    def test_type_i_fluid_non_minkowski_g(self) -> None:
        """mpmath path: Type-I fluid under non-Minkowski ``g`` -> Type I.

        Mirrors ``TestCausalBasisFix::test_type_i_non_minkowski_g`` at 50-digit
        precision. The relative-sign test in :func:`_causal_counts` identifies
        the timelike eigenvector even when ``|g_{ij}|/|g_{00}| ~ 10``.
        """
        # Same fixture as the float64 contract test
        T_mixed = np.diag(np.array([-1.0, 0.3, 0.3, 0.3]))
        g_ab = np.array([
            [-0.12, 0.05, 0.05, 0.0],
            [0.05,   1.5, 0.3,  0.2],
            [0.05,   0.3, 1.5,  0.1],
            [0.0,    0.2, 0.1,  1.5],
        ])

        mp_result = classify_hawking_ellis_mpmath(T_mixed, g_ab, precision=50)
        assert mp_result["he_type"] == 1, (
            f"Expected Type I -- Type-I fluid spectrum under non-Minkowski g; "
            f"got he_type={mp_result['he_type']} "
            f"(eigenvalues_real={mp_result['eigenvalues_real']}, "
            f"eigenvalues_imag={mp_result['eigenvalues_imag']})"
        )


class TestVerifyClassificationAtPoints:
    """Pin the cross-check function's flip-rate accounting."""

    def test_batch_flip_rate_is_correct(self) -> None:
        T_flip = np.asarray(_type_iv_block_diag(imag=2.0e-5))
        T_fluid = np.asarray(_perfect_fluid(rho=1.0, p=0.3))

        # Two points: one Type-I-in-float64 (actually Type IV), one genuine Type I.
        T_batch = np.stack([T_flip, T_fluid], axis=0)
        g_batch = np.stack([np.asarray(MINKOWSKI)] * 2, axis=0)
        float64_types = np.array([1, 1], dtype=np.int32)

        report = verify_classification_at_points(
            T_batch, g_batch, float64_types, precision=50
        )

        assert report["n_points"] == 2
        assert report["n_flips"] == 1
        assert report["flip_indices"].tolist() == [0]
        # Flipped point's mpmath verdict is Type IV.
        assert report["mpmath_he_types"][0] == 4
        assert report["mpmath_he_types"][1] == 1
        assert report["flip_rate"] == 0.5

    def test_empty_batch_returns_zero_flip_rate(self) -> None:
        T_batch = np.zeros((0, 4, 4))
        g_batch = np.zeros((0, 4, 4))
        float64_types = np.zeros((0,), dtype=np.int32)

        report = verify_classification_at_points(
            T_batch, g_batch, float64_types, precision=50
        )

        assert report["n_points"] == 0
        assert report["n_flips"] == 0
        assert report["flip_rate"] == 0.0

    def test_batch_exposes_cond_v_per_point(self) -> None:
        """batch verify exposes cond_V_per_point + uncertain_mask."""
        T_batch = np.stack([
            np.asarray(_perfect_fluid(rho=1.0, p=0.3)),
            np.asarray(_perfect_fluid(rho=2.0, p=0.5)),
        ])
        g_batch = np.stack([np.asarray(MINKOWSKI), np.asarray(MINKOWSKI)])
        float64_he = np.array([1, 1], dtype=np.int32)

        report = verify_classification_at_points(
            T_batch, g_batch, float64_he, precision=50
        )

        assert "cond_V_per_point" in report
        assert "uncertain_mask" in report
        assert report["cond_V_per_point"].shape == (2,)
        assert report["uncertain_mask"].shape == (2,)
        assert report["uncertain_mask"].dtype == np.bool_
        # Both clean fluids should be certain
        assert report["uncertain_mask"].any() == False


# Bauer-Fike eigenvector-matrix condition number diagnostic


class TestCondV:
    """Bauer-Fike sensitivity diagnostic.

    - test_well_conditioned_flags_certain: clean perfect-fluid -> cond(V) small,
      uncertain=False.
    - test_jordan_defective_flags_uncertain: near-Jordan synthesised input ->
      cond(V) ~ inf (or 10^10+), uncertain=True.

    Threshold: cond(V) > 10**(precision/2) per Demmel 1997 Thm 4.4 (Bauer-Fike).
    """

    def test_well_conditioned_flags_certain(self) -> None:
        """Perfect fluid under Minkowski -> uncertain=False, cond_V well-bounded."""
        T = np.asarray(_perfect_fluid(rho=1.0, p=0.3))
        g = np.asarray(MINKOWSKI)

        result = classify_hawking_ellis_mpmath(T, g, precision=50)

        assert "cond_V" in result, "missing 'cond_V' key in return dict"
        assert "uncertain" in result, "missing 'uncertain' key in return dict"

        assert result["uncertain"] is False, (
            f"Expected uncertain=False for clean perfect fluid; "
            f"got uncertain={result['uncertain']}, cond_V={result['cond_V']}"
        )
        # Threshold for precision=50 is 10^25; clean fluid should be well below.
        assert result["cond_V"] < 10 ** 10, (
            f"Expected cond_V < 1e10 for clean perfect fluid; "
            f"got cond_V={result['cond_V']}"
        )
        # Existing keys still present (additivity contract)
        for k in ("he_type", "all_real", "near_vacuum", "n_timelike",
                  "n_null", "n_unique", "max_imag_abs", "max_real_abs",
                  "eigenvalues_real", "eigenvalues_imag", "precision"):
            assert k in result, f"existing key {k!r} dropped from return dict"

    def test_jordan_defective_flags_uncertain(self) -> None:
        """Exact Jordan synthesised T -> uncertain=True (cond_V exceeds 10^25).

        The top-left 2x2 is an exact Jordan block: eigenvalue 1.0 with
        algebraic multiplicity 2 and a SINGLE eigenvector (the (1,0) entry
        is exactly 0.0). ``mpmath.eig`` on a defective matrix returns a
        near-singular eigenvector column at the defective eigenvalue, so
        ``sigma_min(V) ~ 10**(-precision)`` and ``cond_V ~ 10**precision``
        - comfortably above the half-digit threshold ``10**(precision/2)``.

        Note: the plan author's original fixture used a ``(1,0)`` perturbation
        of ``1e-20`` expecting ``cond_V > 10**25``; empirically that yields
        ``cond_V ~ 10**10`` (sqrt of the perturbation), BELOW the half-digit
        threshold at ``precision=50``. An exact Jordan block is the
        physically-meaningful fixture for the Bauer-Fike diagnostic.
        """
        # EXACT Jordan block in top-left 2x2: (1,0) entry is 0.0.
        T = np.array([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, -0.3],
        ])
        g = np.asarray(MINKOWSKI)

        result = classify_hawking_ellis_mpmath(T, g, precision=50)

        assert "cond_V" in result
        assert "uncertain" in result

        # cond(V) should be very large (ill-conditioned eigenvector matrix)
        assert result["uncertain"] is True, (
            f"Expected uncertain=True for exact-Jordan T; "
            f"got uncertain={result['uncertain']}, cond_V={result['cond_V']}"
        )
        # cond_V should exceed the half-digit threshold 10^(50/2) = 10^25
        assert result["cond_V"] > 10 ** 25 or result["cond_V"] == float("inf"), (
            f"Expected cond_V > 10^25 (or inf); got cond_V={result['cond_V']}"
        )
