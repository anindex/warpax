"""Accuracy and validity of the inward TOV scan."""

import jax
import jax.numpy as jnp
import pytest

from warpax.metrics._tov_scan import integrate_tov_inward

jax.config.update("jax_enable_x64", True)

RHO = 1e-3
R_OUT = 2.0
R_IN = 1.0


def _mass(r):
    return 4.0 * jnp.pi * RHO * r**3 / 3.0


def _exact_pressure(r):
    big_m = _mass(R_OUT)
    s_r = jnp.sqrt(1.0 - 2.0 * _mass(r) / r)
    s_big = jnp.sqrt(1.0 - 2.0 * big_m / R_OUT)
    return RHO * (s_r - s_big) / (3.0 * s_big - s_r)


def _max_error(n, exact_mid):
    r = jnp.linspace(R_OUT, R_IN, n)
    rho = jnp.full_like(r, RHO)
    kw = {}
    if exact_mid:
        r_mid = 0.5 * (r[:-1] + r[1:])
        kw = dict(rho_mid=jnp.full_like(r_mid, RHO), m_mid=_mass(r_mid))
    p = integrate_tov_inward(r, rho, _mass(r), **kw)
    return float(jnp.max(jnp.abs(p - _exact_pressure(r))))


def test_exact_midpoints_fourth_order():
    """Constant rho = 1e-3 interior solution p = rho [sqrt(1-2m/r) - sqrt(1-2M/R)] /
    [3 sqrt(1-2M/R) - sqrt(1-2m/r)] on r in [1, 2]; halving h from 17 to 33 points
    with exact midpoint samples must cut the max error by >= 12 (RK4 gives 16)."""
    ratio = _max_error(17, True) / _max_error(33, True)
    assert ratio >= 12.0, ratio


def test_linear_midpoints_second_order():
    """Same problem with linearly interpolated midpoints of the cubic m(r): the
    O(h^2) coefficient error limits the scheme to a ratio of ~4 (measured
    5.9665e-9 at 17 points vs 1.4918e-9 at 33)."""
    ratio = _max_error(17, False) / _max_error(33, False)
    assert 3.0 < ratio < 5.0, ratio


def test_trapped_input_gives_nan():
    """m = 2 on r in [1, 2] violates r > 2m everywhere, so the result is all NaN
    instead of clipped zeros from a floor."""
    r = jnp.linspace(R_OUT, R_IN, 33)
    p = integrate_tov_inward(r, jnp.full_like(r, 0.01), jnp.full_like(r, 2.0))
    assert bool(jnp.all(jnp.isnan(p)))


@pytest.mark.parametrize("n", [17, 33])
def test_outer_boundary_zero(n):
    """p_r(R_2) = 0 is the boundary condition, so the first sample is exactly zero."""
    r = jnp.linspace(R_OUT, R_IN, n)
    p = integrate_tov_inward(r, jnp.full_like(r, RHO), _mass(r))
    assert float(p[0]) == 0.0
