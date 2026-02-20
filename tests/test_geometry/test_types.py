"""Tests for TensorField and GridSpec types."""

import jax
import jax.numpy as jnp
import pytest

from warpax.geometry.types import GridSpec, TensorField


# =========================================================================
# TensorField tests
# =========================================================================


class TestTensorField:
    """Tests for the TensorField Equinox module."""

    def test_tensorfield_creation(self):
        """Create TensorField with known components, verify rank and index_positions."""
        components = jnp.eye(4)
        tf = TensorField(components=components, rank=2, index_positions="dd")
        assert tf.rank == 2
        assert tf.index_positions == "dd"
        assert jnp.array_equal(tf.components, components)

    def test_tensorfield_default_index_positions(self):
        """Create with empty index_positions, verify default is 'd' * rank."""
        components = jnp.zeros((4, 4, 4))
        tf = TensorField(components=components, rank=3)
        assert tf.index_positions == "ddd"

    def test_tensorfield_jit(self):
        """Pass TensorField through jax.jit and verify components survive."""
        components = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        tf = TensorField(components=components, rank=2, index_positions="dd")

        @jax.jit
        def identity(field: TensorField) -> TensorField:
            return field

        result = identity(tf)
        assert jnp.allclose(result.components, components)
        assert result.rank == 2
        assert result.index_positions == "dd"

    def test_tensorfield_float64(self):
        """Verify components dtype is float64."""
        components = jnp.ones((4, 4))
        tf = TensorField(components=components, rank=2)
        assert tf.components.dtype == jnp.float64

    def test_tensorfield_invalid_rank(self):
        """Verify ValueError when index_positions length != rank."""
        with pytest.raises(ValueError, match="index_positions length"):
            TensorField(
                components=jnp.zeros((4, 4)),
                rank=2,
                index_positions="ddd",  # 3 chars for rank-2
            )

    def test_tensorfield_grid_shape(self):
        """Verify grid_shape property for a field on a grid."""
        components = jnp.zeros((10, 10, 4, 4))
        tf = TensorField(components=components, rank=2)
        assert tf.grid_shape == (10, 10)
        assert tf.tensor_shape == (4, 4)


# =========================================================================
# GridSpec tests
# =========================================================================


class TestGridSpec:
    """Tests for the GridSpec Equinox module."""

    def test_gridspec_creation(self):
        """Create GridSpec, verify bounds and shape."""
        grid = GridSpec(
            bounds=[(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)],
            shape=(10, 20, 30),
        )
        assert grid.bounds == [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)]
        assert grid.shape == (10, 20, 30)
        assert grid.ndim == 3

    def test_gridspec_spacing(self):
        """Verify spacing computation matches expected values."""
        grid = GridSpec(
            bounds=[(-1.0, 1.0), (0.0, 4.0), (-3.0, 3.0)],
            shape=(11, 5, 7),
        )
        spacing = grid.spacing
        assert abs(spacing[0] - 0.2) < 1e-14  # 2.0 / 10
        assert abs(spacing[1] - 1.0) < 1e-14  # 4.0 / 4
        assert abs(spacing[2] - 1.0) < 1e-14  # 6.0 / 6

    def test_gridspec_axes(self):
        """Verify axes are jnp arrays with correct dtype (float64) and length."""
        grid = GridSpec(
            bounds=[(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)],
            shape=(5, 10, 15),
        )
        axes = grid.axes
        assert len(axes) == 3
        for i, (ax, n) in enumerate(zip(axes, grid.shape)):
            assert isinstance(ax, jax.Array)
            assert ax.dtype == jnp.float64
            assert ax.shape == (n,)

    def test_gridspec_meshgrid(self):
        """Verify meshgrid returns JAX arrays with correct shapes."""
        grid = GridSpec(
            bounds=[(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)],
            shape=(5, 10, 15),
        )
        mg = grid.meshgrid
        assert len(mg) == 3
        for arr in mg:
            assert isinstance(arr, jax.Array)
            assert arr.shape == (5, 10, 15)
            assert arr.dtype == jnp.float64

    def test_gridspec_coordinate_fields(self):
        """Verify 4D coordinate fields [t=0, x, y, z]."""
        grid = GridSpec(
            bounds=[(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)],
            shape=(5, 10, 15),
        )
        fields = grid.coordinate_fields
        assert len(fields) == 4  # t, x, y, z
        # t should be all zeros
        assert jnp.allclose(fields[0], 0.0)
        # x, y, z should match meshgrid
        mg = grid.meshgrid
        for f, m in zip(fields[1:], mg):
            assert jnp.allclose(f, m)
        # dtype check
        for f in fields:
            assert f.dtype == jnp.float64
