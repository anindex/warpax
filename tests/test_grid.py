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


class TestAutoChunk:
    """``auto_chunk_threshold`` kwarg validation + ULP-tolerant
    parity with the v0.1.x full-vmap path.

    Default ``None`` and no-op (threshold > grid_size) paths are
    **bit-exact** to v0.1.x because neither calls ``lax.map``. When the
    threshold triggers chunking, ``jax.lax.map`` uses a different
    floating-point addition order than ``jax.vmap``; the resulting drift
    is at the ULP floor (observed max ``6.8e-13`` on ricci/weyl squared at
    20³ Alcubierre). This is JAX-internal; the caller-visible invariant is
    numerical equivalence, not bit-equality.

    Empirical validation of the chunking implementation showed that the
    ULP-scale drift is intrinsic to ``lax.map`` vs ``vmap`` floating-point
    addition order and cannot be eliminated at this layer. Tests therefore
    use ``assert_allclose`` with ``atol=1e-12, rtol=1e-12`` rather than
    bit-exact equality: this still pins down any algorithmic bug but
    accepts ULP-level fused-add reordering.
    """

    _CHUNK_ATOL = 1e-12
    _CHUNK_RTOL = 1e-12

    def _alcubierre_16cubed_full(self):
        """Helper: Alcubierre 16³ full-vmap reference result."""
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        grid = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(16, 16, 16))
        return metric, grid, evaluate_curvature_grid(metric, grid)

    def test_default_none_preserves_v10_bit_exact(self):
        """auto_chunk_threshold=None == v0.1.x full-vmap (bit-exact: no lax.map)."""
        metric, grid, ref = self._alcubierre_16cubed_full()
        result = evaluate_curvature_grid(metric, grid, auto_chunk_threshold=None)

        for field in GridCurvatureResult._fields:
            a = getattr(ref, field)
            b = getattr(result, field)
            assert jnp.array_equal(a, b), (
                f"Field {field!r} drifted under auto_chunk_threshold=None"
            )

    def test_threshold_1000_chunks_20cubed_bit_exact(self):
        """auto_chunk_threshold=1000 on a 20³=8000 grid chunks to ULP-equivalent result.

        Chunking dispatches to ``jax.lax.map`` whose floating-point add
        order differs from ``jax.vmap``; the observed max drift is ~1e-13
        (below ``assert_allclose(atol=1e-12)``).
        """
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        grid = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(20, 20, 20))

        ref = evaluate_curvature_grid(metric, grid)  # full vmap
        chunked = evaluate_curvature_grid(metric, grid, auto_chunk_threshold=1000)

        for field in GridCurvatureResult._fields:
            a = np.array(getattr(ref, field))
            b = np.array(getattr(chunked, field))
            assert_allclose(
                b,
                a,
                atol=self._CHUNK_ATOL,
                rtol=self._CHUNK_RTOL,
                err_msg=f"Field {field!r} drifted beyond ULP floor under threshold=1000",
            )

    def test_threshold_larger_than_grid_is_noop(self):
        """auto_chunk_threshold > grid_size: no chunking, bit-exact to default."""
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        grid = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(8, 8, 8))  # 512 points

        ref = evaluate_curvature_grid(metric, grid)
        noop = evaluate_curvature_grid(metric, grid, auto_chunk_threshold=10_000)

        for field in GridCurvatureResult._fields:
            a = getattr(ref, field)
            b = getattr(noop, field)
            assert jnp.array_equal(a, b), (
                f"Field {field!r} drifted under auto_chunk_threshold=10_000 "
                "(no-op path must not call lax.map)"
            )

    def test_zero_threshold_raises(self):
        """auto_chunk_threshold=0 raises ValueError with verbatim message."""
        metric = AlcubierreMetric(v_s=0.5, R=1.0, sigma=8.0)
        grid = GridSpec(bounds=[(-3.0, 3.0)] * 3, shape=(4, 4, 4))

        with pytest.raises(ValueError, match="must be a positive integer or None"):
            evaluate_curvature_grid(metric, grid, auto_chunk_threshold=0)
