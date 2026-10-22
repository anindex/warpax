"""Tests for the closed-form Type-I worst observer."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from warpax.energy_conditions.optimization import optimize_wec
from warpax.energy_conditions.worst_observer_analytic import (
    boosted_energy_density,
    worst_observer_typeI,
)

from ._helpers import ETA

EYE = jnp.eye(4)


def _eigvals(rho, p):
    """Eigenvalues of T^a_b in the rest frame: {-rho, p1, p2, p3}."""
    return jnp.array([-rho, p[0], p[1], p[2]])


def test_threshold_rapidity_zeroes_energy():
    """At zeta_th the boosted energy density along the worst axis is exactly 0.

    Axis p = -2 is the most violating (rho + p = -1); it sits at eigenvalue index 1,
    and sinh^2 zeta_th = rho / |rho + p_star| = 1 gives zeta_th = arcsinh(1).
    """
    rho = 1.0
    p = (-2.0, 0.5, 0.5)
    out = worst_observer_typeI(_eigvals(rho, p), EYE, ETA, condition="wec")
    assert int(out["worst_axis"]) == 1
    assert float(out["p_star"]) == pytest.approx(-2.0)
    zeta_th = float(out["zeta_th"])
    assert zeta_th == pytest.approx(np.arcsinh(1.0), abs=1e-9)
    e = float(boosted_energy_density(jnp.array(rho), jnp.array(-2.0), jnp.array(zeta_th)))
    assert e == pytest.approx(0.0, abs=1e-9)
    assert float(out["asymptotic_sign"]) < 0, "energy -> -inf"


def test_satisfied_gives_infinite_threshold():
    """When NEC holds on every axis, no observer sees negative energy."""
    rho = 1.0
    p = (0.5, 0.5, 0.5)
    out = worst_observer_typeI(_eigvals(rho, p), EYE, ETA, condition="wec")
    assert np.isinf(float(out["zeta_th"]))
    assert float(out["asymptotic_sign"]) > 0


def test_boost_direction_is_worst_eigenvector():
    """The worst spatial boost direction is the eigenvector of the worst axis.

    The worst axis is p = -3 (array index 2); with identity eigenvectors the
    direction is e_2 = (0, 0, 1, 0).
    """
    rho = 1.0
    p = (0.3, -3.0, 0.2)
    out = worst_observer_typeI(_eigvals(rho, p), EYE, ETA, condition="wec")
    assert int(out["worst_axis"]) == 2
    direction = np.asarray(out["boost_direction"])
    np.testing.assert_allclose(np.abs(direction), [0, 0, 1, 0], atol=1e-9)


def test_dec_worst_axis_is_largest_pressure_magnitude():
    """DEC picks |p| largest, p = 3 at array index 3, with margin rho - |p_star| = -2."""
    rho = 1.0
    p = (0.4, -0.6, 3.0)
    out = worst_observer_typeI(_eigvals(rho, p), EYE, ETA, condition="dec")
    assert int(out["worst_axis"]) == 3
    assert float(out["margin"]) == pytest.approx(rho - 3.0)


def test_matches_optimizer_on_isotropic_violator():
    """The analytic WEC worst observer agrees with the BFGS optimizer:
    the boosted energy density at the optimizer's rapidity matches the closed
    form, and the optimizer drives the margin negative along the worst axis.
    ``T_mixed`` is T^a_b in Minkowski; lowering with eta gives T_ab. The optimizer
    is capped at zeta_max, so its margin must equal the closed form at the cap."""
    rho, p = 1.0, -2.0
    T_mixed = jnp.diag(jnp.array([-rho, p, p, p]))
    T_ab = ETA @ T_mixed
    res = optimize_wec(T_ab, ETA, n_starts=8, zeta_max=5.0)
    closed_at_cap = float(boosted_energy_density(jnp.array(rho), jnp.array(p), jnp.array(5.0)))
    assert float(res.margin) < 0
    assert float(res.margin) == pytest.approx(closed_at_cap, rel=1e-6)


def test_rest_frame_is_unit_and_future_directed_for_badly_scaled_eigenvectors():
    """Columns scaled by ``(1e-8, -1, 1, 1)`` with the timelike column negated give
    ``g(v0, v0) = -1e-16``; normalizing by the column's own causal norm yields
    ``g(e0, e0) = -1`` and the sign flip gives ``e0^0 > 0`` (an absolute
    ``atol = 1e-12`` floor gives ``e0^2 = -1e-4`` and a past-directed ``e0``).
    With ``rho = -1`` the threshold rapidity is 0, so ``worst_observer = e0``."""
    vecs = EYE * jnp.array([-1e-8, -1.0, 1.0, 1.0])
    out = worst_observer_typeI(_eigvals(-1.0, (0.5, 0.5, 0.5)), vecs, ETA, condition="wec")
    assert float(out["zeta_th"]) == 0.0
    for key in ("rest_frame", "worst_observer"):
        u = out[key]
        assert float(u[0]) > 0.0
        assert float(jnp.einsum("a,ab,b->", u, ETA, u)) == pytest.approx(-1.0, abs=1e-10)
