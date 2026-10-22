"""Shared numerics: Hermite interpolation kernels, strict-mode grid guard."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from warpax.numerics import (
    assert_uniform_grid,
    cubic_interp,
    hermite_cubic,
    local_cubic_slopes,
    not_a_knot_slopes,
    strict_mode_enabled,
)


def _knots(n, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.sort(jnp.asarray(rng.uniform(0.0, 10.0, n)))
    return x, jnp.sin(x) + 0.1 * x**2


@pytest.mark.parametrize("n", [4, 7, 50])
def test_not_a_knot_slopes_match_scipy(n):
    """The C2 not-a-knot spline is unique, so its knot derivatives must equal scipy's."""
    scipy_interp = pytest.importorskip("scipy.interpolate")
    x, f = _knots(n)
    ref = scipy_interp.CubicSpline(np.asarray(x), np.asarray(f), bc_type="not-a-knot")(
        np.asarray(x), 1
    )
    np.testing.assert_allclose(np.asarray(not_a_knot_slopes(x, f)), ref, rtol=1e-10, atol=1e-12)
    xq = jnp.linspace(float(x[0]), float(x[-1]), 97)
    ours = jax.vmap(lambda q: hermite_cubic(q, x, f, not_a_knot_slopes(x, f)))(xq)
    theirs = scipy_interp.CubicSpline(np.asarray(x), np.asarray(f), bc_type="not-a-knot")(
        np.asarray(xq)
    )
    np.testing.assert_allclose(np.asarray(ours), theirs, rtol=1e-10, atol=1e-12)


def test_small_knot_counts():
    """Two knots give the secant, three knots the interpolating parabola."""
    x = jnp.array([0.0, 1.0])
    f = jnp.array([1.0, 3.0])
    np.testing.assert_allclose(np.asarray(not_a_knot_slopes(x, f)), [2.0, 2.0])
    x = jnp.array([0.0, 1.0, 3.0])
    f = x**2
    np.testing.assert_allclose(np.asarray(not_a_knot_slopes(x, f)), 2.0 * np.asarray(x), atol=1e-12)


def test_local_slopes_average_neighbouring_secants():
    x = jnp.array([0.0, 1.0, 3.0, 4.0])
    f = jnp.array([0.0, 2.0, 2.0, 5.0])
    np.testing.assert_allclose(np.asarray(local_cubic_slopes(x, f)), [2.0, 1.0, 1.5, 3.0])


def test_hermite_interpolates_knots_and_is_nan_outside():
    x, f = _knots(12)
    at_knots = jax.vmap(lambda q: cubic_interp(q, x, f))(x)
    np.testing.assert_allclose(np.asarray(at_knots), np.asarray(f), rtol=1e-13, atol=1e-13)
    assert jnp.isnan(cubic_interp(x[-1] + 1.0, x, f))
    assert jnp.isnan(cubic_interp(x[0] - 1.0, x, f))


def test_hermite_reproduces_cubics_exactly_with_exact_slopes():
    """A cubic Hermite segment with exact end slopes reproduces any cubic polynomial."""
    x = jnp.linspace(-2.0, 2.0, 9)
    f = x**3 - 2.0 * x + 1.0
    fx = 3.0 * x**2 - 2.0
    xq = jnp.linspace(-2.0, 2.0, 41)
    val = jax.vmap(lambda q: hermite_cubic(q, x, f, fx))(xq)
    np.testing.assert_allclose(np.asarray(val), np.asarray(xq**3 - 2.0 * xq + 1.0), atol=1e-12)
    grad = jax.grad(lambda q: hermite_cubic(q, x, f, fx))(0.3)
    np.testing.assert_allclose(float(grad), 3.0 * 0.3**2 - 2.0, atol=1e-12)


def test_cubic_interp_jit_and_grad_through_values():
    x, f = _knots(20)
    val = jax.jit(cubic_interp)(2.2, x, f)
    np.testing.assert_allclose(float(val), float(cubic_interp(2.2, x, f)))
    g = jax.grad(lambda v: cubic_interp(2.2, x, v))(f)
    assert g.shape == f.shape and bool(jnp.all(jnp.isfinite(g)))


def test_uniform_grid_guard(monkeypatch):
    monkeypatch.setenv("WARPAX_STRICT", "1")
    assert strict_mode_enabled()
    assert_uniform_grid(jnp.linspace(0.0, 1.0, 11))
    with pytest.raises(ValueError, match="non-uniform"):
        assert_uniform_grid(jnp.array([0.0, 1.0, 3.0, 4.0]))
    monkeypatch.setenv("WARPAX_STRICT", "0")
    assert not strict_mode_enabled()
    assert_uniform_grid(jnp.array([0.0, 1.0, 3.0, 4.0]))
