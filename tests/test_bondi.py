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

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from warpax.benchmarks.schwarzschild import SchwarzschildMetric
from warpax.bondi import radiated_momentum_flux, weyl_scalars
from warpax.bondi.extract import _psi4_at

from ._helpers import ETA, krt_cone, make_krt


def _rest_static(X):
    return jnp.array([1.0, 0.0, 0.0, 0.0])


def test_krt_flux_equals_recoil_and_is_gw_silent():
    """Kinnersley rocket: flux = kinematic recoil ``-(mdot, m G0, 0, 0)`` at u=0.

    On the Eulerian g-null tetrad (not principal on the Kerr-Schild slice) the
    type-D Coulomb term leaks ``Psi_4 ~ 0.14 (m/r)^2 Psi_2 ~ 0.14 m^3/r^5``, so
    ``r |Psi_4| <~ 0.14 m^3/r^4 = 3.4e-5`` at r=8, m=1 (measured RMS 2.3e-5;
    a principal eta tetrad gives < 1e-6).
    """
    G0 = 0.10
    metric_fn, m_of = make_krt(G0)
    res = radiated_momentum_flux(
        metric_fn, krt_cone, _rest_static, n_theta=24, radii=(4.0, 8.0), richardson=False
    )
    m0 = float(m_of(0.0))
    mdot0 = float(jax.jacfwd(m_of)(0.0))
    kin = np.array([mdot0, m0 * G0, 0.0, 0.0])

    assert res.ricci_max < 1e-8
    np.testing.assert_allclose(res.flux, -kin, atol=1e-3)
    assert res.psi4_rms < 0.15 * m0**3 / 8.0**4


def test_schwarzschild_static_no_radiated_flux():
    """Static vacuum radiates nothing: the no-reactionless-steering benchmark."""
    schw = SchwarzschildMetric(M=1.0)
    cone = lambda r, th: jnp.array([0.0, r * np.cos(th), r * np.sin(th), 0.0])
    res = radiated_momentum_flux(
        schw.__call__, cone, _rest_static, n_theta=24, radii=(40.0, 80.0, 160.0)
    )
    assert np.linalg.norm(res.flux) < 1e-3
    assert res.psi4_rms < 1e-6


def test_vaidya_monopole_radiates_energy_not_momentum():
    """Isotropic mass loss (straight worldline): energy flux = mdot, momentum = 0."""
    metric_fn, m_of = make_krt(G0=1e-8)
    res = radiated_momentum_flux(
        metric_fn, krt_cone, _rest_static, n_theta=24, radii=(4.0, 8.0), richardson=False
    )
    mdot0 = float(jax.jacfwd(m_of)(0.0))
    assert res.ricci_max < 1e-8
    assert abs(res.energy_flux + mdot0) < 1e-3
    assert abs(res.momentum_flux[0]) < 1e-3


def test_constant_curvature_weyl_vanishes_on_g_tetrad():
    """Constant curvature ``R_abcd = g_ac g_bd - g_ad g_bc`` with g = diag(-1,2,3,4) is pure trace.

    Weyl vanishes identically, so every ``|Psi_n|`` must be < 1e-12 on the g-null
    tetrad; an eta tetrad is not g-null and gives ``|Psi_4| = 0.125``.
    """
    g = np.diag([-1.0, 2.0, 3.0, 4.0])
    g_inv = np.linalg.inv(g)
    R = np.einsum("ac,bd->abcd", g, g) - np.einsum("ad,bc->abcd", g, g)
    riemann = np.einsum("ae,ebcd->abcd", g_inv, R)
    ricci = np.einsum("abad->bd", riemann)
    res = SimpleNamespace(
        metric=g,
        metric_inv=g_inv,
        riemann=riemann,
        ricci=ricci,
        ricci_scalar=float(np.einsum("bd,bd->", g_inv, ricci)),
    )
    assert _psi4_at(res, 1.0, 0.0, 1) < 1e-12
    c, s = np.cos(0.7), np.sin(0.7)
    assert np.all(np.abs(weyl_scalars(res, c, s, 1)) < 1e-12)


def test_psi4_rms_is_gauss_legendre_weighted(monkeypatch):
    """``psi4_rms = sqrt(sum_j w_j psi_j^2 / 2)`` with ``psi = cos(theta)`` is exactly 1/sqrt(3).

    Gauss-Legendre integrates ``c^2`` on [-1,1] exactly (2/3, halved to 1/3);
    an unweighted node mean gives sqrt(23/47) = 0.6995.
    """
    monkeypatch.setattr("warpax.bondi.extract._psi4_at", lambda res, c, s, axis: c)
    res = radiated_momentum_flux(
        lambda X: ETA, krt_cone, _rest_static, n_theta=24, radii=(1.0,), richardson=False
    )
    assert res.psi4_rms == pytest.approx(1.0 / np.sqrt(3.0), abs=1e-12)
