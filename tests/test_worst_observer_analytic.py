"""Tests for the closed-form Type-I worst observer (Contribution 3)."""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from warpax.energy_conditions.optimization import optimize_wec
from warpax.energy_conditions.worst_observer_analytic import (
    boosted_energy_density,
    worst_observer_typeI,
)

MINK = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
EYE = jnp.eye(4)


def _eigvals(rho, p):
    # Eigenvalues of T^a_b in the rest frame: {-rho, p1, p2, p3}.
    return jnp.array([-rho, p[0], p[1], p[2]])


def test_threshold_rapidity_zeroes_energy():
    """At zeta_th the boosted energy density along the worst axis is exactly 0."""
    rho = 1.0
    p = (-2.0, 0.5, 0.5)  # axis 0 (p=-2) is the most-violating: rho+p = -1
    out = worst_observer_typeI(_eigvals(rho, p), EYE, MINK, condition="wec")
    assert int(out["worst_axis"]) == 1  # eigenvalue index of p=-2 (idx 1 in array)
    assert float(out["p_star"]) == pytest.approx(-2.0)
    zeta_th = float(out["zeta_th"])
    # sinh^2 zeta_th = rho / |rho+p_star| = 1/1 = 1 -> zeta_th = arcsinh(1)
    assert zeta_th == pytest.approx(np.arcsinh(1.0), abs=1e-9)
    e = float(boosted_energy_density(jnp.array(rho), jnp.array(-2.0), jnp.array(zeta_th)))
    assert e == pytest.approx(0.0, abs=1e-9)
    assert float(out["asymptotic_sign"]) < 0  # energy -> -inf


def test_satisfied_gives_infinite_threshold():
    """When NEC holds on every axis, no observer sees negative energy."""
    rho = 1.0
    p = (0.5, 0.5, 0.5)
    out = worst_observer_typeI(_eigvals(rho, p), EYE, MINK, condition="wec")
    assert np.isinf(float(out["zeta_th"]))
    assert float(out["asymptotic_sign"]) > 0


def test_boost_direction_is_worst_eigenvector():
    """The worst spatial boost direction is the eigenvector of the worst axis."""
    rho = 1.0
    p = (0.3, -3.0, 0.2)  # worst axis is the p=-3 one (array index 2)
    out = worst_observer_typeI(_eigvals(rho, p), EYE, MINK, condition="wec")
    assert int(out["worst_axis"]) == 2
    # eigenvectors = identity columns -> direction is e_2 = (0,0,1,0)
    direction = np.asarray(out["boost_direction"])
    np.testing.assert_allclose(np.abs(direction), [0, 0, 1, 0], atol=1e-9)


def test_dec_worst_axis_is_largest_pressure_magnitude():
    rho = 1.0
    p = (0.4, -0.6, 3.0)  # |p| largest at p=3 (array index 3)
    out = worst_observer_typeI(_eigvals(rho, p), EYE, MINK, condition="dec")
    assert int(out["worst_axis"]) == 3
    assert float(out["margin"]) == pytest.approx(rho - 3.0)  # rho - |p_star| = -2


def test_matches_optimizer_on_isotropic_violator():
    """The analytic WEC worst observer agrees with the BFGS optimizer:
    the boosted energy density at the optimizer's rapidity matches the closed
    form, and the optimizer drives the margin negative along the worst axis."""
    rho, p = 1.0, -2.0
    T_mixed = jnp.diag(jnp.array([-rho, p, p, p]))  # T^a_b in Minkowski
    T_ab = MINK @ T_mixed  # lower index back to T_{ab}
    res = optimize_wec(T_ab, MINK, n_starts=8, zeta_max=5.0)
    # Optimizer margin must be negative (WEC violated) and must agree with
    # the closed-form value at zeta_max (the capped extremum).
    closed_at_cap = float(boosted_energy_density(jnp.array(rho), jnp.array(p), jnp.array(5.0)))
    assert float(res.margin) < 0
    # optimizer (capped at zeta_max) must reach the closed-form minimum at the cap
    assert float(res.margin) == pytest.approx(closed_at_cap, rel=1e-6)
