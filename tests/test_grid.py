"""Grid evaluation tests: shapes, dtypes, chunking, and field correctness.

Tests the grid evaluation pipeline (build_coord_batch, evaluate_curvature_grid)
for correct output shapes, float64 enforcement, batch_size chunking consistency,
and curvature field correctness on Minkowski and Schwarzschild spacetimes.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from warpax.benchmarks.alcubierre import AlcubierreMetric
from warpax.benchmarks.minkowski import MinkowskiMetric
from warpax.benchmarks.schwarzschild import SchwarzschildMetric, kretschner_isotropic
from warpax.geometry.geometry import CurvatureResult
from warpax.geometry.grid import (
    GridCurvatureResult,
    build_coord_batch,
    evaluate_curvature_grid,
)
from warpax.geometry.types import GridSpec


# ---------------------------------------------------------------------------
# Coordinate batch construction
# ---------------------------------------------------------------------------


class TestBuildCoordBatch:
    """Tests for build_coord_batch helper."""

    def test_build_coord_batch_shape(self, default_grid):
        """build_coord_batch returns shape (N, 4) where N = prod(grid_shape)."""
        coords = build_coord_batch(default_grid)
        N = np.prod(default_grid.shape)
        assert coords.shape == (N, 4)

    def test_build_coord_batch_time_coordinate(self, default_grid):
        """Time coordinates default to 0.0, or custom t value."""
        coords_default = build_coord_batch(default_grid)
        assert_allclose(np.array(coords_default[:, 0]), 0.0, atol=0.0)

        coords_custom = build_coord_batch(default_grid, t=1.5)
        assert_allclose(np.array(coords_custom[:, 0]), 1.5, atol=0.0)


# ---------------------------------------------------------------------------
# Grid curvature result shapes
# ---------------------------------------------------------------------------


class TestGridCurvatureShapes:
    """Tests for evaluate_curvature_grid output shapes and types."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.metric = MinkowskiMetric()
        self.grid = GridSpec(
            bounds=[(-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)],
            shape=(4, 4, 2),
        )

    def test_grid_curvature_result_shapes(self):
        """evaluate_curvature_grid returns correct shapes for all fields."""
        result = evaluate_curvature_grid(self.metric, self.grid)
        gs = (4, 4, 2)

        assert isinstance(result, GridCurvatureResult)
        assert result.metric.shape == (*gs, 4, 4)
        assert result.metric_inv.shape == (*gs, 4, 4)
        assert result.christoffel.shape == (*gs, 4, 4, 4)
        assert result.riemann.shape == (*gs, 4, 4, 4, 4)
        assert result.ricci.shape == (*gs, 4, 4)
        assert result.ricci_scalar.shape == gs
        assert result.kretschner.shape == gs
        assert result.ricci_squared.shape == gs
        assert result.weyl_squared.shape == gs

    def test_grid_without_invariants(self):
        """compute_invariants=False returns CurvatureResult without invariant fields."""
        result = evaluate_curvature_grid(
            self.metric, self.grid, compute_invariants=False
        )
        assert isinstance(result, CurvatureResult)
        assert not hasattr(result, "kretschner")
        assert not hasattr(result, "weyl_squared")


# ---------------------------------------------------------------------------
# Batch size consistency
# ---------------------------------------------------------------------------


class TestGridBatching:
    """Chunked batching via batch_size produces identical results to full vmap."""

    def test_grid_batch_size_matches_full_vmap(self):
        """batch_size=32 gives identical results to batch_size=None (full vmap)."""
        metric = AlcubierreMetric()
        grid = GridSpec(
            bounds=[(-3.0, 3.0), (-3.0, 3.0), (-0.5, 0.5)],
            shape=(8, 8, 4),
        )

        result_full = evaluate_curvature_grid(metric, grid, batch_size=None)
        result_chunked = evaluate_curvature_grid(metric, grid, batch_size=32)

        # Compare all 11 fields
        for field_name in GridCurvatureResult._fields:
            full_val = np.array(getattr(result_full, field_name))
            chunked_val = np.array(getattr(result_chunked, field_name))
            assert_allclose(
                chunked_val,
                full_val,
                atol=1e-12,
                err_msg=f"Mismatch in field {field_name}",
            )


# ---------------------------------------------------------------------------
# Minkowski flatness on grid
# ---------------------------------------------------------------------------


class TestGridMinkowskiFlat:
    """Minkowski spacetime: all curvature tensors and invariants are zero on a grid."""

    def test_grid_minkowski_flat(self):
        """All curvature fields zero for Minkowski on a grid."""
        metric = MinkowskiMetric()
        grid = GridSpec(
            bounds=[(-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)],
            shape=(4, 4, 2),
        )
        result = evaluate_curvature_grid(metric, grid)

        # Curvature tensors must be zero
        assert_allclose(np.array(result.riemann), 0.0, atol=1e-12)
        assert_allclose(np.array(result.ricci), 0.0, atol=1e-12)
        assert_allclose(np.array(result.ricci_scalar), 0.0, atol=1e-12)
        assert_allclose(np.array(result.einstein), 0.0, atol=1e-12)
        assert_allclose(np.array(result.stress_energy), 0.0, atol=1e-12)

        # Invariants must be zero
        assert_allclose(np.array(result.kretschner), 0.0, atol=1e-12)
        assert_allclose(np.array(result.ricci_squared), 0.0, atol=1e-12)
        assert_allclose(np.array(result.weyl_squared), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Float64 dtype enforcement
# ---------------------------------------------------------------------------


class TestGridFloat64:
    """All grid output fields must be float64."""

    def test_grid_float64_dtype(self):
        """All output fields from evaluate_curvature_grid are float64."""
        metric = MinkowskiMetric()
        grid = GridSpec(
            bounds=[(-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)],
            shape=(4, 4, 2),
        )
        result = evaluate_curvature_grid(metric, grid)

        for field_name in GridCurvatureResult._fields:
            arr = getattr(result, field_name)
            assert arr.dtype == jnp.float64, (
                f"Field {field_name} has dtype {arr.dtype}, expected float64"
            )


# ---------------------------------------------------------------------------
# Schwarzschild Kretschner field on grid
# ---------------------------------------------------------------------------


class TestGridSchwarzschildKretschner:
    """Schwarzschild Kretschner scalar field matches analytical formula on a grid."""

    def test_grid_schwarzschild_kretschner_field(self):
        """Kretschner field matches K=48*M^2/r_s^6 at each grid point."""
        M = 1.0
        metric = SchwarzschildMetric(M=M)
        # Grid far from origin to avoid singularity
        grid = GridSpec(
            bounds=[(3.0, 7.0), (3.0, 7.0), (-0.5, 0.5)],
            shape=(4, 4, 2),
        )
        result = evaluate_curvature_grid(metric, grid)

        # Compute analytical Kretschner at each grid point
        X, Y, Z = grid.meshgrid
        K_analytical = np.array(kretschner_isotropic(X, Y, Z, M=M))

        assert_allclose(
            np.array(result.kretschner),
            K_analytical,
            rtol=1e-8,
        )
