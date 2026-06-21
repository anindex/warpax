r"""Newman--Penrose Weyl scalars and the peeling-falloff verification.

This module builds the five Newman--Penrose Weyl scalars
:math:`\Psi_0,\dots,\Psi_4` on the asymptotic outgoing null tetrad of
:mod:`warpax.bondi.extract`, and verifies the *peeling property*

.. math::

    \Psi_n(r,\Omega) \;\sim\; r^{-(5-n)} \qquad (n=0,\dots,4)

along outgoing null cones as :math:`r\to\infty`.  Peeling is the precise
asymptotic-falloff hypothesis underlying the model-independent
no-reactionless-steering theorem (T1-GR): it guarantees a smooth future null
infinity with a well-defined Bondi mass aspect, news, and :math:`\ell=0,1`
momentum aspect, so that the Bondi four-momentum :math:`P_B^\mu(u)` and the
flux-balance law are well-defined.  Verifying it numerically on the test
spacetimes confirms that the structure the theorem *assumes* is actually
realized.

What is computed
----------------
* :func:`weyl_scalars` -- the five complex NP scalars at one cone point, built
  from the exact curvature chain (:func:`warpax.geometry.geometry.compute_curvature_chain`)
  by forming the Weyl tensor :math:`C_{abcd}` (Riemann minus its Ricci trace
  parts, so the matter-bearing null-dust cones of Kinnersley/Vaidya are handled
  correctly) and contracting it with the asymptotic null tetrad
  ``l`` (outgoing null), ``n`` (ingoing null), ``m`` (transverse complex null)
  reused from :mod:`~warpax.bondi.extract`.
* :func:`peeling_slopes` -- samples :math:`|\Psi_n|` over a range of luminosity
  radii along an outgoing cone and fits :math:`\log|\Psi_n|` vs :math:`\log r`,
  returning the per-scalar slope.  The peeling prediction is
  ``slope[n] == -(5-n)`` i.e. ``[-5,-4,-3,-2,-1]``.

Robustness note
---------------
For a *gravitational-wave--silent* maneuver (e.g. the Damour/Kinnersley dipole
photon rocket) the radiative scalar :math:`\Psi_4` is identically zero, so a
numerical :math:`\Psi_4` only probes the pipeline floor (:math:`\sim10^{-18}`),
not a physical :math:`1/r` tail.  Such scalars are flagged ``above_floor=False``
(relative to the Coulombic :math:`\Psi_2`) and are *not* slope-fit.  The robust,
universal peeling certificate across all test spacetimes is the Coulombic
:math:`\Psi_2\sim r^{-3}` (slope :math:`-3`); a genuine :math:`\Psi_4\sim r^{-1}`
tail is demonstrated on a deliberately *non-silent* (e.g. :math:`\ell\ge2`)
emission used as a positive control.

Conventions match :mod:`warpax.geometry.geometry` and
:mod:`warpax.bondi.extract`: ``riemann`` is ``R^a_{bcd}`` (``[up,lo,lo,lo]``);
``ricci`` is ``R_{ab}``; metric signature ``(-+++)``; the asymptotic tetrad is
Minkowskian (``l.n = -1``, ``m.mbar = +1``), exact as ``g -> eta`` at large
``r``, which is where peeling is read.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from warpax.geometry.geometry import compute_curvature_chain

from .extract import _unit_dir

# Peeling prediction: Psi_n ~ r^{-(5-n)}  ->  log-log slope -(5-n).
_EXPECTED_SLOPES = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])


class PeelingResult(NamedTuple):
    """Peeling-falloff fit for the five NP Weyl scalars along an outgoing cone.

    Attributes
    ----------
    slopes : (5,) fitted log-log slopes of ``|Psi_n|`` vs ``r`` (NaN where the
        scalar is below the resolution floor and not fit).
    expected : (5,) peeling prediction ``[-5,-4,-3,-2,-1]``.
    slope_err : (5,) RMS residual of each log-log fit (NaN where not fit).
    amplitudes : (5,) ``|Psi_n|`` at the largest sampled radius.
    radii : (k,) luminosity radii sampled.
    psi_by_r : (5, k) ``|Psi_n|(r)`` table.
    theta : polar angle of the sampled cone generator.
    above_floor : (5,) bool -- scalar resolved above the relative noise floor
        (and therefore slope-fit).  ``Psi_2`` (Coulombic) is the reference.
    ricci_max : max ``|R|`` over the sampled radii (exact-solution sanity).
    """

    slopes: np.ndarray
    expected: np.ndarray
    slope_err: np.ndarray
    amplitudes: np.ndarray
    radii: np.ndarray
    psi_by_r: np.ndarray
    theta: float
    above_floor: np.ndarray
    ricci_max: float


def _null_tetrad(c: float, s: float, axis: int):
    """Asymptotic Minkowski null tetrad ``(l, n, m)`` at polar angle ``theta``.

    Matches :func:`warpax.bondi.extract._psi4_at`: ``l = (1, nhat)`` outgoing
    null, ``n = (1, -nhat)/2`` ingoing null (so ``eta(l,n) = -1``), and
    ``m = (0, e_theta + i e_phi)/sqrt2`` the transverse complex null leg
    (``eta(m, mbar) = +1``).
    """
    nhat = _unit_dir(c, s, axis)
    e_theta = _unit_dir(-s, c, axis)            # in-plane transverse (d nhat / d theta)
    perp = 1 + (axis % 3)
    rem = ({0, 1, 2} - {axis - 1, perp - 1}).pop()
    e_phi = np.zeros(3)
    e_phi[rem] = 1.0                            # out-of-plane transverse unit
    l_up = np.array([1.0, *nhat])
    n_up = np.array([1.0, *(-nhat)]) * 0.5
    m_up = np.concatenate([[0.0], (e_theta + 1j * e_phi) / np.sqrt(2.0)])
    return l_up, n_up, m_up


def weyl_scalars(res, c: float, s: float, axis: int = 1) -> np.ndarray:
    r"""Five complex NP Weyl scalars ``[Psi0..Psi4]`` at one cone point.

    Builds the Weyl tensor from the curvature result and contracts with the
    asymptotic null tetrad.  In 4D,

    .. math::
        C_{abcd} = R_{abcd}
          - \tfrac12(g_{ac}R_{bd} - g_{ad}R_{bc} - g_{bc}R_{ad} + g_{bd}R_{ac})
          + \tfrac16 R\,(g_{ac}g_{bd} - g_{ad}g_{bc}),

    with ``R_{abcd} = g_{ae} R^e{}_{bcd}``.  Subtracting the Ricci parts matters
    for the null-dust (Kinnersley/Vaidya) cones where ``R_{ab} != 0``.

    Parameters
    ----------
    res : CurvatureResult from :func:`compute_curvature_chain`.
    c, s : ``cos(theta)``, ``sin(theta)`` of the cone generator.
    axis : symmetry axis (1 = x, matching ``bondi.extract``).

    Returns
    -------
    (5,) complex ``np.ndarray`` ``[Psi0, Psi1, Psi2, Psi3, Psi4]``.
    """
    g = np.array(res.metric)
    Rmix = np.array(res.riemann)                # R^a_{bcd}
    Ric = np.array(res.ricci)                   # R_{ab}
    Rs = float(res.ricci_scalar)

    Rdown = np.einsum("ae,ebcd->abcd", g, Rmix)  # R_{abcd}
    ricci_part = 0.5 * (
        np.einsum("ac,bd->abcd", g, Ric)
        - np.einsum("ad,bc->abcd", g, Ric)
        - np.einsum("bc,ad->abcd", g, Ric)
        + np.einsum("bd,ac->abcd", g, Ric)
    )
    scalar_part = (Rs / 6.0) * (
        np.einsum("ac,bd->abcd", g, g) - np.einsum("ad,bc->abcd", g, g)
    )
    C = Rdown - ricci_part + scalar_part

    l, n, m = _null_tetrad(c, s, axis)
    mb = np.conj(m)
    psi0 = np.einsum("abcd,a,b,c,d->", C, l, m, l, m)
    psi1 = np.einsum("abcd,a,b,c,d->", C, l, n, l, m)
    psi2 = np.einsum("abcd,a,b,c,d->", C, l, m, mb, n)
    psi3 = np.einsum("abcd,a,b,c,d->", C, l, n, mb, n)
    psi4 = np.einsum("abcd,a,b,c,d->", C, n, mb, n, mb)
    return np.array([psi0, psi1, psi2, psi3, psi4], dtype=complex)


def peeling_slopes(
    metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
    cone_fn: Callable[[float, float], jnp.ndarray],
    *,
    theta: float = float(np.pi / 3),
    axis: int = 1,
    radii: tuple[float, ...] = (50.0, 100.0, 200.0, 400.0, 800.0),
    rel_floor: float = 1e-6,
) -> PeelingResult:
    r"""Fit the peeling falloff ``|Psi_n| ~ r^{-(5-n)}`` along an outgoing cone.

    Samples the five NP scalars at the luminosity radii ``radii`` along the cone
    generator at polar angle ``theta`` and least-squares fits ``log|Psi_n|`` vs
    ``log r``.  A scalar is fit only if it is resolved above ``rel_floor`` times
    the Coulombic ``|Psi_2|`` at the same radii (else it is at the numerical
    floor -- e.g. ``Psi_4`` for a GW-silent dipole -- and reported, not fit).

    Returns
    -------
    PeelingResult
    """
    radii_arr = np.array([float(r) for r in radii])
    c = float(np.cos(theta))
    s = float(np.sin(theta))

    psi_by_r = np.zeros((5, radii_arr.size))
    ricci_max = 0.0
    for i, r in enumerate(radii_arr):
        X = cone_fn(float(r), float(theta))
        res = compute_curvature_chain(metric_fn, X)
        psi_by_r[:, i] = np.abs(weyl_scalars(res, c, s, axis))
        ricci_max = max(ricci_max, float(np.abs(res.ricci_scalar)))

    # Resolution gate: a scalar is "above floor" if its largest sampled
    # magnitude exceeds rel_floor times the Coulombic Psi_2 reference.
    psi2_ref = float(np.max(psi_by_r[2])) if np.max(psi_by_r[2]) > 0 else 1.0
    above = np.max(psi_by_r, axis=1) > rel_floor * psi2_ref

    slopes = np.full(5, np.nan)
    errs = np.full(5, np.nan)
    logr = np.log(radii_arr)
    A = np.vstack([logr, np.ones_like(logr)]).T
    for n in range(5):
        y = psi_by_r[n]
        if above[n] and np.all(y > 0):
            ly = np.log(y)
            (slope, b), *_ = np.linalg.lstsq(A, ly, rcond=None)
            slopes[n] = float(slope)
            errs[n] = float(np.sqrt(np.mean((ly - (slope * logr + b)) ** 2)))

    return PeelingResult(
        slopes=slopes,
        expected=_EXPECTED_SLOPES.copy(),
        slope_err=errs,
        amplitudes=psi_by_r[:, -1].copy(),
        radii=radii_arr,
        psi_by_r=psi_by_r,
        theta=float(theta),
        above_floor=above,
        ricci_max=float(ricci_max),
    )
