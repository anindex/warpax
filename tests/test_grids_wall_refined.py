"""Tests for :func:`warpax.grids.wall_refined` (, )."""
from __future__ import annotations

import jax.numpy as jnp

from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import GridSpec
from warpax.grids import RefinedGrid, wall_refined


def _mask_near_wall(coords_flat, wall_r: float = 2.0, half_width: float = 0.5):
    """Select points within ``half_width`` of the wall radius ``wall_r``."""
    r = jnp.linalg.norm(coords_flat[:, 1:], axis=-1)
    return jnp.abs(r - wall_r) < half_width


class TestWallRefined:
    """Contract tests for the 2-level AMR wall-refined grid."""

    def test_returns_refined_grid(self):
        m = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        rg = wall_refined(
            m,
            bounds=((-10.0, 10.0),) * 3,
            shape=(16, 16, 16),
            refine_where=_mask_near_wall,
        )
        assert isinstance(rg, RefinedGrid)
        assert isinstance(rg.base, GridSpec)
        assert isinstance(rg.fine, GridSpec)

    def test_fine_patch_higher_resolution(self):
        m = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        rg = wall_refined(
            m,
            bounds=((-10.0, 10.0),) * 3,
            shape=(8, 8, 8),
            refine_where=_mask_near_wall,
            refine_factor=3,
        )
        assert rg.base.shape == (8, 8, 8)
        assert rg.fine.shape == (24, 24, 24)

    def test_fine_bounds_tighter_than_base(self):
        """Fine patch bounds should be a sub-region of base bounds."""
        m = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        rg = wall_refined(
            m,
            bounds=((-10.0, 10.0),) * 3,
            shape=(16, 16, 16),
            refine_where=_mask_near_wall,
        )
        for axis_i in range(3):
            base_lo, base_hi = rg.base.bounds[axis_i]
            fine_lo, fine_hi = rg.fine.bounds[axis_i]
            assert base_lo <= fine_lo, (
                f"axis {axis_i}: fine_lo {fine_lo} < base_lo {base_lo}"
            )
            assert fine_hi <= base_hi, (
                f"axis {axis_i}: fine_hi {fine_hi} > base_hi {base_hi}"
            )
            assert (fine_hi - fine_lo) <= (base_hi - base_lo)

    def test_mask_shape_matches_base(self):
        m = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        rg = wall_refined(
            m,
            bounds=((-10.0, 10.0),) * 3,
            shape=(8, 8, 8),
            refine_where=_mask_near_wall,
        )
        assert rg.mask.shape == (8, 8, 8)
        assert rg.mask.dtype == jnp.bool_

    def test_empty_mask_fallback(self):
        """If refine_where selects nothing, fine bounds fall back to base bounds."""
        m = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)

        def _mask_empty(coords_flat):
            return jnp.zeros(coords_flat.shape[0], dtype=jnp.bool_)

        rg = wall_refined(
            m,
            bounds=((-10.0, 10.0),) * 3,
            shape=(8, 8, 8),
            refine_where=_mask_empty,
        )
        # Bounds comparison - base.bounds may be list, convert for comparison.
        assert list(rg.fine.bounds) == list(rg.base.bounds)
