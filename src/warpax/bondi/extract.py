r"""Bondi four-momentum radiated-flux extractor at null infinity.

This module extracts, *directly from the spacetime curvature*, the
four-momentum radiated through future null infinity by a localized,
asymptotically-flat source, and the gravitational-news (GW) content of the
radiation.  It is the numerical instrument behind the model-independent
no-reactionless-steering law (T1-GR):

    dP_B^mu/du  =  - (1/4pi) oint [ |N|^2 + 4pi n^2_matter ] lhat^mu dOmega ,

so a confined positive-energy drive can change its Bondi momentum P_B^i ONLY by
radiating four-momentum to scri.  On a non-radiating segment (no news, no matter
flux) P_B^mu is constant: there is no reactionless steering.

What is computed here
---------------------
* **Matter flux** ``F^mu = oint n^2_matter(Omega) lhat^mu dOmega`` where the local
  radiation amplitude seen by the asymptotic rest observer ``u^a`` is

      n^2_matter(Omega) = lim_{r->inf} r^2 (u^a u^b T_{ab}) ,

  with ``T_{ab}`` the curvature-derived stress-energy
  (:func:`warpax.geometry.geometry.compute_curvature_chain`) and ``lhat^mu =
  (1, nhat)`` the outgoing null direction.
* **News proxy** ``psi4_rms``: the RMS of ``|r * Psi_4|`` over the cut, where
  ``Psi_4 = C_{abcd} n^a mbar^b n^c mbar^d`` is the outgoing Weyl scalar built on
  an asymptotic null tetrad.  For a gravitational-wave-silent maneuver (e.g. the
  Damour/Kinnersley dipole photon rocket) this vanishes; nonzero ``psi4_rms``
  flags genuine gravitational radiation.

Conventions match :mod:`warpax.geometry.geometry`: ``riemann`` is
``R^a_{bcd}`` (``[upper, lower, lower, lower]``); ``stress_energy`` is
``T_{ab}`` (geometric units ``G=c=1``); metric signature ``(-+++)``.

.. note::

   The result is the radiated flux on a single outgoing cut at retarded time
   ``u``.  ``F^0 = oint n^2 dOmega`` is (minus) the Bondi mass-loss rate and
   ``F^i`` the radiated momentum; for an exact photon rocket these equal
   ``-dP_B^mu/du`` (the conservation law).  The extractor is **axisymmetric**
   about ``axis`` (the phi integral is done analytically, killing the
   off-axis momentum components); a full-sphere variant is a direct extension.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from warpax.geometry.geometry import compute_curvature_chain

_ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))


class BondiFluxResult(NamedTuple):
    """Radiated four-momentum flux through a cut of null infinity.

    Attributes
    ----------
    flux : (4,) radiated four-momentum flux ``F^mu = oint n^2 lhat^mu dOmega``.
    energy_flux : ``F^0 = oint n^2 dOmega`` (= minus Bondi mass-loss rate).
    momentum_flux : (3,) spatial part ``F^i``.
    n2_samples : matter amplitude ``n^2(theta)`` at the Gauss-Legendre nodes
        (Richardson-extrapolated to ``r -> inf`` when ``richardson`` is set).
    cos_nodes : ``cos(theta)`` Gauss-Legendre nodes.
    psi4_rms : RMS of ``|r * Psi_4|`` over the cut (news / GW-silence proxy).
    ricci_max : max ``|R|`` over the sampled points (exact-solution sanity).
    """

    flux: np.ndarray
    energy_flux: float
    momentum_flux: np.ndarray
    n2_samples: np.ndarray
    cos_nodes: np.ndarray
    psi4_rms: float
    ricci_max: float


def _unit_dir(c: float, s: float, axis: int) -> np.ndarray:
    """Spatial unit vector at polar angle ``theta`` about ``axis`` (phi=0 plane).

    ``nhat`` has component ``cos(theta)`` along ``axis`` and ``sin(theta)`` along
    the next spatial axis (cyclically), with the ``(c, s, 0)`` axis convention
    for ``axis = 1`` (x).
    """
    perp = 1 + (axis % 3)  # next spatial index, cyclic in {1,2,3}
    n = np.zeros(3)
    n[axis - 1] = c
    n[perp - 1] = s
    return n


def _matter_amplitude(
    metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
    X: jnp.ndarray,
    u_rest: jnp.ndarray,
    r_lum: float,
) -> tuple[float, float, object]:
    """``n^2 = r^2 (u.T.u)`` and ``|Ricci|`` at one cone point."""
    res = compute_curvature_chain(metric_fn, X)
    T = res.stress_energy
    n2 = (r_lum ** 2) * (u_rest @ T @ u_rest)
    return float(n2), float(jnp.abs(res.ricci_scalar)), res


def _psi4_at(res, c: float, s: float, axis: int) -> float:
    """Outgoing Weyl scalar ``Psi_4`` on an asymptotic Minkowski null tetrad.

    Uses the full Riemann tensor (``Weyl = Riemann`` near scri where Ricci -> 0).
    Tetrad: ``l = (1, nhat)`` outgoing null, ``n = (1, -nhat)/2`` ingoing null,
    ``m = (0, e_theta + i e_phi)/sqrt2`` transverse complex null leg.
    """
    nhat = _unit_dir(c, s, axis)
    e_theta = _unit_dir(-s, c, axis)              # d nhat / d theta (in-plane transverse)
    # e_phi: the remaining spatial axis, orthogonal to both nhat and e_theta
    perp = 1 + (axis % 3)                         # the axis carrying sin(theta)
    rem = ({0, 1, 2} - {axis - 1, perp - 1}).pop()
    e_phi = np.zeros(3)
    e_phi[rem] = 1.0                              # out-of-plane transverse unit

    nin_up = np.array([1.0, *(-nhat)]) * 0.5      # ingoing null n^a (l.n = -1)
    m_up = np.concatenate([[0.0], (e_theta + 1j * e_phi) / np.sqrt(2.0)])
    mbar = np.conj(m_up)

    g = np.array(res.metric)
    Rmix = np.array(res.riemann)                  # R^a_{bcd}
    # lower first index: R_{abcd} = g_{ae} R^e_{bcd}
    Rdown = np.einsum("ae,ebcd->abcd", g, Rmix)
    # Psi4 = C_{abcd} n^a mbar^b n^c mbar^d  (n here = ingoing null nin_up)
    psi4 = np.einsum("abcd,a,b,c,d->", Rdown, nin_up, mbar, nin_up, mbar)
    return float(np.abs(psi4))


def radiated_momentum_flux(
    metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
    cone_fn: Callable[[float, float], jnp.ndarray],
    rest_frame_fn: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    n_theta: int = 24,
    radii: tuple[float, ...] = (40.0, 80.0, 160.0),
    axis: int = 1,
    richardson: bool = True,
) -> BondiFluxResult:
    """Radiated four-momentum flux through a cut of null infinity.

    Parameters
    ----------
    metric_fn : ``coords (4,) -> g_ab (4,4)`` callable (warpax convention).
    cone_fn : ``(r_lum, theta) -> X (4,)`` spacetime point on the outgoing null
        cut at luminosity radius ``r_lum`` and polar angle ``theta`` about
        ``axis`` (phi = 0 plane).  For a Kerr-Schild photon rocket on the
        ``u* = 0`` cone this is ``X = (r, r*nhat)``.
    rest_frame_fn : ``X -> u^a (4,)`` the asymptotic rest-frame 4-velocity
        (Bondi frame).  Constant ``(1,0,0,0)`` for a momentarily-static source.
    n_theta : Gauss-Legendre nodes in ``cos(theta)``.
    radii : luminosity radii; ``n^2(r)`` is Richardson-extrapolated to
        ``r -> inf`` from the two largest when ``richardson`` is set.
    axis : symmetry axis (1 = x default).
    richardson : extrapolate ``n^2`` in ``1/r`` (set False for exact solutions
        where ``n^2`` is already ``r``-independent).

    Returns
    -------
    BondiFluxResult
    """
    c_nodes, w = np.polynomial.legendre.leggauss(n_theta)
    radii = tuple(float(r) for r in radii)

    n2_by_r = np.zeros((len(radii), n_theta))
    ricci_max = 0.0
    psi4_vals = np.zeros(n_theta)

    for j, c in enumerate(c_nodes):
        s = float(np.sqrt(max(0.0, 1.0 - c * c)))
        for i, r in enumerate(radii):
            X = cone_fn(r, float(np.arccos(np.clip(c, -1.0, 1.0))))
            u_rest = rest_frame_fn(X)
            n2, Rs, res = _matter_amplitude(metric_fn, X, u_rest, r)
            n2_by_r[i, j] = n2
            ricci_max = max(ricci_max, Rs)
            if i == len(radii) - 1:  # news proxy at the largest radius
                psi4_vals[j] = r * _psi4_at(res, float(c), s, axis)

    # Richardson r->inf:  n^2(r) ~ n2_inf + C/r  ->  n2_inf = (r2 n2_2 - r1 n2_1)/(r2 - r1)
    if richardson and len(radii) >= 2:
        r1, r2 = radii[-2], radii[-1]
        n2 = (r2 * n2_by_r[-1] - r1 * n2_by_r[-2]) / (r2 - r1)
    else:
        n2 = n2_by_r[-1]

    # oint f dOmega = 2 pi int_{-1}^{1} f dc  (axisymmetry kills the phi integral).
    F0 = 2 * np.pi * np.sum(w * n2)                 # lhat^0 = 1
    Faxis = 2 * np.pi * np.sum(w * n2 * c_nodes)    # lhat^axis = cos(theta)
    flux = np.zeros(4)
    flux[0] = F0
    flux[axis] = Faxis

    psi4_rms = float(np.sqrt(np.mean(psi4_vals ** 2)))
    return BondiFluxResult(
        flux=flux,
        energy_flux=float(F0),
        momentum_flux=flux[1:].copy(),
        n2_samples=n2,
        cos_nodes=c_nodes,
        psi4_rms=psi4_rms,
        ricci_max=float(ricci_max),
    )
