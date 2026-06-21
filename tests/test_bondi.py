r"""Tests for the Bondi radiated-momentum / news extractor (warpax.bondi).

Validates the model-independent no-reactionless-steering flux balance on three
independent spacetimes:

  * Kinnersley photon rocket (accelerating + radiating): the curvature-derived
    flux equals the kinematic recoil ``(-mdot, -m|a|, 0, 0)`` and the dipole is
    gravitational-wave-silent (news proxy ~ 0), an independent re-derivation of the
    flux-balance closure.
  * Schwarzschild (static vacuum): zero radiated flux -> the no-go corollary
    (a static drive cannot self-accelerate).
  * Vaidya monopole (isotropic mass loss): energy radiated, momentum zero ->
    steering is sourced by the l=1 dipole alone.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.bondi import radiated_momentum_flux

_ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


def _make_krt(G0, M0=1.0, M1=0.05, OM=0.5):
    """Exact accelerating Kerr-Schild photon rocket; G0 -> 0 gives Vaidya."""

    def worldline(u):
        sx = jnp.sinh(G0 * u) / G0 if G0 != 0 else u
        xx = (jnp.cosh(G0 * u) - 1.0) / G0 if G0 != 0 else 0.0
        return jnp.array([sx, xx, 0.0, 0.0])

    def m_of(u):
        return M0 + M1 * jnp.sin(OM * u)

    @jax.custom_jvp
    def retarded_u(X):
        def body(u, _):
            z = worldline(u)
            zdot = jax.jacfwd(worldline)(u)
            D = X - z
            r = -(_ETA @ zdot) @ D
            F = -D[0] ** 2 + D[1] ** 2 + D[2] ** 2 + D[3] ** 2
            return u - F / (2.0 * r), None

        u0 = X[0] - jnp.sqrt(X[1] ** 2 + X[2] ** 2 + X[3] ** 2)
        u, _ = jax.lax.scan(body, u0, None, length=12)
        return u

    @retarded_u.defjvp
    def _jvp(primals, tangents):
        (X,), (dX,) = primals, tangents
        u = retarded_u(X)
        z = worldline(u)
        zdot = jax.jacfwd(worldline)(u)
        D = X - z
        r = -(_ETA @ zdot) @ D
        l_dn = _ETA @ (D / r)
        return u, -(l_dn @ dX)

    def metric_fn(X):
        u = retarded_u(X)
        z = worldline(u)
        zdot = jax.jacfwd(worldline)(u)
        D = X - z
        r = -(_ETA @ zdot) @ D
        l_up = D / r
        l_dn = _ETA @ l_up
        return _ETA + (2.0 * m_of(u) / r) * jnp.outer(l_dn, l_dn)

    return metric_fn, m_of


def _krt_cone(r, theta):
    c, s = np.cos(theta), np.sin(theta)
    return jnp.array([r, r * c, r * s, 0.0])


def _rest_static(X):
    return jnp.array([1.0, 0.0, 0.0, 0.0])


def test_krt_flux_equals_recoil_and_is_gw_silent():
    """Kinnersley rocket: flux = kinematic recoil, dipole is GW-silent."""
    G0 = 0.10
    metric_fn, m_of = _make_krt(G0)
    res = radiated_momentum_flux(
        metric_fn, _krt_cone, _rest_static, n_theta=24, radii=(4.0, 8.0), richardson=False
    )
    m0 = float(m_of(0.0))
    mdot0 = float(jax.jacfwd(m_of)(0.0))
    # kinematic dP_B/du = mdot v + m a, with v=(1,0,0,0), a=(0,G0,0,0) at u=0
    kin = np.array([mdot0, m0 * G0, 0.0, 0.0])

    assert res.ricci_max < 1e-8                      # exact solution
    np.testing.assert_allclose(res.flux, -kin, atol=1e-3)  # F = -dP_B/du (closure)
    assert res.psi4_rms < 1e-6                       # Damour dipole: no gravitational news


def test_schwarzschild_static_no_radiated_flux():
    """Static vacuum radiates nothing: the no-reactionless-steering benchmark."""
    schw = SchwarzschildMetric(M=1.0)
    cone = lambda r, th: jnp.array([0.0, r * np.cos(th), r * np.sin(th), 0.0])
    res = radiated_momentum_flux(
        schw.__call__, cone, _rest_static, n_theta=24, radii=(40.0, 80.0, 160.0)
    )
    assert np.linalg.norm(res.flux) < 1e-3           # P_B constant => cannot self-accelerate
    assert res.psi4_rms < 1e-6


def test_vaidya_monopole_radiates_energy_not_momentum():
    """Isotropic mass loss (straight worldline): energy flux = mdot, momentum = 0."""
    metric_fn, m_of = _make_krt(G0=1e-8)             # G0 -> 0: spherical Vaidya
    res = radiated_momentum_flux(
        metric_fn, _krt_cone, _rest_static, n_theta=24, radii=(4.0, 8.0), richardson=False
    )
    mdot0 = float(jax.jacfwd(m_of)(0.0))
    assert res.ricci_max < 1e-8
    assert abs(res.energy_flux + mdot0) < 1e-3       # F^0 = -mdot
    assert abs(res.momentum_flux[0]) < 1e-3          # zero radiated momentum
