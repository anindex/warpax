"""Tests for wall-restricted filtering and statistics.

Covers all three mask builders (shape function, Frobenius norm, determinant
guard), mask composability, wall-restricted stats (type counts, violation
counts, fractions, miss rates), and edge cases (empty mask, no violations).
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TestShapeFunctionMask
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestFrobeniusNormMask
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestDeterminantGuardMask
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestMaskComposability
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestWallRestrictedStats
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestWallRestrictedStatsMissRate
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestWallRestrictedStatsEdgeCases
# ---------------------------------------------------------------------------


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
