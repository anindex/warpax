"""Tests for :func:`warpax.grids.wall_clustered` (, )."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import GridSpec
from warpax.grids import wall_clustered


class TestWallClustered:
    """Contract tests for the cosh-stretched wall-clustered grid."""

    def test_returns_gridspec(self):
        m = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        g = wall_clustered(m, bounds=((-10.0, 10.0),) * 3, shape=(16, 16, 16))
        assert isinstance(g, GridSpec)

    def test_shape_and_bounds_preserved(self):
        m = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        g = wall_clustered(m, bounds=((-10.0, 10.0),) * 3, shape=(16, 16, 16))
        assert g.shape == (16, 16, 16)
        assert list(g.bounds) == [(-10.0, 10.0)] * 3

    def test_non_uniform_spacing(self):
        """Clustered coord arrays have non-uniform spacing."""
        m = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        g = wall_clustered(
            m,
            bounds=((-10.0, 10.0),) * 3,
            shape=(16, 16, 16),
            wall_radius=2.0,
        )
        assert g.coord_arrays is not None, (
            "Grid must expose coord_arrays for non-uniform spacing"
        )
        x = np.asarray(g.coord_arrays[0])
        dx = np.diff(x)
        # max / min spacing ratio > 1.2 (clustering is real)
        assert dx.max() / dx.min() > 1.2

    def test_volume_weights_present(self):
        m = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        g = wall_clustered(m, bounds=((-10.0, 10.0),) * 3, shape=(16, 16, 16))
        assert g.volume_weights is not None
        vw = g.volume_weights_array
        assert vw is not None
        assert vw.shape == (16, 16, 16)

    def test_volume_weights_sum_approximates_total_volume(self):
        """: sum(volume_weights) approximates total bounded volume."""
        m = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        g = wall_clustered(m, bounds=((-10.0, 10.0),) * 3, shape=(32, 32, 32))
        total = float(jnp.sum(g.volume_weights_array))
        expected = 20.0**3  # bounded cube of side 20
        # 10% tolerance on trapezoidal quadrature for clustered grid
        assert 0.9 * expected < total < 1.1 * expected

    def test_unknown_clustering_raises(self):
        m = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        with pytest.raises(ValueError, match="clustering"):
            wall_clustered(
                m,
                bounds=((-10.0, 10.0),) * 3,
                shape=(8, 8, 8),
                clustering="bad",
            )

    def test_bounds_shape_length_mismatch_raises(self):
        m = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        with pytest.raises(ValueError, match="same length"):
            wall_clustered(
                m, bounds=((-10.0, 10.0),) * 3, shape=(8, 8)  # 2-tuple
            )

    def test_static_field_for_jit_cache(self):
        """: grid_spec is hashable / static."""
        m = AlcubierreMetric(v_s=0.5, R=2.0, sigma=8.0)
        g = wall_clustered(m, bounds=((-10.0, 10.0),) * 3, shape=(16, 16, 16))
        assert isinstance(g.shape, tuple)
        assert isinstance(g.coord_arrays, tuple)
        # Each coord_array element is itself a tuple of floats (hashable).
        for arr in g.coord_arrays:
            assert isinstance(arr, tuple)
