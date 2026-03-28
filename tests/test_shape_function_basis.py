"""Tests: ShapeFunction basis differentiability + Alcubierre approximation.

- Per-basis differentiability: ``|jacfwd(sf)(r) - fd(sf)(r)|_max < 1e-7``.
- Alcubierre tanh recovery: 24-knot cubic B-spline approximates ``tanh((r-R)/sigma)``
  within relative error ``< 1e-2``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture(scope="module")
def shape_fns():
    """Instantiate one of each basis (spline, bernstein, gmm)."""
    from warpax.design import ShapeFunction

    r_min, r_max = 0.0, 1.0
    n_knots = 24
    knots = jnp.linspace(r_min, r_max, n_knots)
    # Use a smooth tanh profile - the spline basis recovers this near exactly.
    values = jnp.tanh((knots - 0.5) / 0.1)

    sf_spline = ShapeFunction.spline(knots, values, order=3)
    sf_bernstein = ShapeFunction.bernstein(jnp.linspace(0.0, 1.0, 16))
    sf_gmm = ShapeFunction.gmm(
        means=jnp.array([0.3, 0.5, 0.7]),
        widths=jnp.array([0.1, 0.1, 0.1]),
        amps=jnp.array([0.5, 1.0, 0.5]),
    )
    return sf_spline, sf_bernstein, sf_gmm


class TestShapeFunction:
    """Per-basis differentiability + Alcubierre-tanh approx."""

    @pytest.mark.parametrize("basis_name", ["spline", "bernstein", "gmm"])
    def test_jacfwd_vs_finite_difference(self, shape_fns, basis_name):
        """|jax.jacfwd(sf)(r) - fd(sf)(r)|_max < 1e-7 at 100 random probe points."""
        sf_spline, sf_bernstein, sf_gmm = shape_fns
        sf = {"spline": sf_spline, "bernstein": sf_bernstein, "gmm": sf_gmm}[basis_name]

        key = jax.random.PRNGKey(0)
        probes = jax.random.uniform(key, (100,), minval=0.1, maxval=0.9)
        step = 1e-5

        ad_grads = jax.vmap(jax.jacfwd(sf))(probes)
        fd_grads = jax.vmap(
            lambda r: (sf(r + step) - sf(r - step)) / (2.0 * step)
        )(probes)

        max_err = float(jnp.max(jnp.abs(ad_grads - fd_grads)))
        assert max_err < 1e-7, (
            f"Basis '{basis_name}' E2-3 violation: "
            f"|jacfwd - fd|_max = {max_err:.2e} >= 1e-7"
        )

    def test_alcubierre_tanh_approximation_via_spline(self, shape_fns):
        """24-knot spline should approximate tanh((r-R)/sigma) within 1e-2 rel err."""
        from warpax.design import ShapeFunction

        R = 0.5
        sigma = 0.1
        r_samples = jnp.linspace(0.0, 1.0, 128)
        truth = jnp.tanh((r_samples - R) / sigma)

        knots = jnp.linspace(0.0, 1.0, 24)
        values = jnp.tanh((knots - R) / sigma)
        sf = ShapeFunction.spline(knots, values, order=3)

        pred = jax.vmap(sf)(r_samples)
        rel_err = jnp.max(jnp.abs(pred - truth)) / jnp.max(jnp.abs(truth))
        # Coarse: in tightens to < 1e-4
        assert float(rel_err) < 1e-2, (
            f"24-knot spline rel_err = {float(rel_err):.2e} >= 1e-2"
        )
