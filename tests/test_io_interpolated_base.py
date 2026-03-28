"""Contract tests for :class:`warpax.io.InterpolatedADMMetric`."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from warpax.geometry import ADMMetric, GridSpec, MetricSpecification
from warpax.io import InterpolatedADMMetric


def _minkowski_grids(Nt: int = 2, N: int = 4):
    """Build constant Minkowski ADM grids for sanity testing."""
    alpha = jnp.ones((Nt, N, N, N))
    beta = jnp.zeros((Nt, N, N, N, 3))
    gamma = jnp.broadcast_to(
        jnp.eye(3)[None, None, None, None, :, :], (Nt, N, N, N, 3, 3)
    )
    spec = GridSpec(bounds=[(-1.0, 1.0)] * 4, shape=(Nt, N, N, N))
    return alpha, beta, gamma, spec


class TestInterpolatedADMMetric:
    """Contract tests for the InterpolatedADMMetric base class."""

    def test_construction_valid_shapes_succeeds(self):
        alpha, beta, gamma, spec = _minkowski_grids()
        m = InterpolatedADMMetric(
            alpha_grid=alpha,
            beta_grid=beta,
            gamma_grid=gamma,
            grid_spec=spec,
            name="minkowski",
            interp_method="cubic",
        )
        assert isinstance(m, ADMMetric)
        assert isinstance(m, MetricSpecification)
        assert m.name() == "minkowski"

    def test_construction_shape_mismatch_raises(self):
        alpha, _, gamma, spec = _minkowski_grids()
        bad_beta = jnp.zeros((3, 4, 4, 4, 3))  # Nt=3 vs alpha's Nt=2
        with pytest.raises(ValueError, match="shape"):
            InterpolatedADMMetric(
                alpha_grid=alpha,
                beta_grid=bad_beta,
                gamma_grid=gamma,
                grid_spec=spec,
            )

    def test_adm_methods_return_correct_shapes_and_dtype(self):
        alpha, beta, gamma, spec = _minkowski_grids()
        m = InterpolatedADMMetric(
            alpha_grid=alpha,
            beta_grid=beta,
            gamma_grid=gamma,
            grid_spec=spec,
        )
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        assert m.lapse(coords).shape == ()
        assert m.lapse(coords).dtype == jnp.float64
        assert m.shift(coords).shape == (3,)
        assert m.shift(coords).dtype == jnp.float64
        assert m.spatial_metric(coords).shape == (3, 3)
        assert m.spatial_metric(coords).dtype == jnp.float64

    def test_jit_compat(self):
        alpha, beta, gamma, spec = _minkowski_grids()
        m = InterpolatedADMMetric(
            alpha_grid=alpha,
            beta_grid=beta,
            gamma_grid=gamma,
            grid_spec=spec,
        )
        jit_fn = jax.jit(lambda c: m(c))
        coords = jnp.array([0.0, 0.0, 0.0, 0.0])
        g = jit_fn(coords)
        assert g.shape == (4, 4)
        assert g.dtype == jnp.float64

    def test_minkowski_round_trip(self):
        alpha, beta, gamma, spec = _minkowski_grids()
        m = InterpolatedADMMetric(
            alpha_grid=alpha,
            beta_grid=beta,
            gamma_grid=gamma,
            grid_spec=spec,
        )
        g = m(jnp.array([0.0, 0.0, 0.0, 0.0]))
        expected = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        npt.assert_allclose(np.asarray(g), np.asarray(expected), atol=1e-12)

    def test_symbolic_raises_not_implemented(self):
        alpha, beta, gamma, spec = _minkowski_grids()
        m = InterpolatedADMMetric(
            alpha_grid=alpha,
            beta_grid=beta,
            gamma_grid=gamma,
            grid_spec=spec,
        )
        with pytest.raises(NotImplementedError, match="closed-form"):
            m.symbolic()

    def test_shape_function_value_raises_not_implemented(self):
        alpha, beta, gamma, spec = _minkowski_grids()
        m = InterpolatedADMMetric(
            alpha_grid=alpha,
            beta_grid=beta,
            gamma_grid=gamma,
            grid_spec=spec,
        )
        with pytest.raises(NotImplementedError, match="not inferable"):
            m.shape_function_value(jnp.array([0.0, 0.0, 0.0, 0.0]))

    def test_out_of_bounds_coords_clip_safely(self):
        alpha, beta, gamma, spec = _minkowski_grids()
        m = InterpolatedADMMetric(
            alpha_grid=alpha,
            beta_grid=beta,
            gamma_grid=gamma,
            grid_spec=spec,
        )
        # Coords outside bounds should clip to bounds, NOT raise, NOT produce NaN.
        coords_oob = jnp.array([100.0, 100.0, 100.0, 100.0])
        g = m(coords_oob)
        assert not jnp.any(jnp.isnan(g))
        assert g.shape == (4, 4)
