"""Energy-condition verdicts certified from the METRIC, not from float64 components.

:mod:`.slemma` decides the conditions by a linear matrix inequality in binary64,
and :mod:`.certificate` re-checks a verdict in exact rational arithmetic. But the
exact check is exact *relative to the float64 components it is handed*: it
certifies a statement about a snapshot of ``T_ab``, not about the stress-energy
the metric actually induces. The gap is the curvature chain.

This module closes it at a point, by composing two pieces that already exist:

* :func:`._intervalcurv.eulerian_fields_interval`, a rigorous interval enclosure
  of the whole chain ``g -> Gamma -> Riem -> Ric -> G -> T``, evaluated with
  interval forward-mode AD and directed rounding, returning ``(rho_n, b, S)`` in
  an orthonormal spatial frame.
* :func:`._enclosure._lmi_dual_lower`, the multiplier search with an interval
  ``LDL^T`` acceptance test, which certifies ``M(sigma) >> 0`` for every member of
  the interval matrix.

Run on a degenerate (point) box the first is an enclosure of the true tensor at
that point, so the verdict is about the spacetime.

All four conditions, not two. ``(rho, b, S)`` *is* the tetrad-frame ``That``, so
``Theta`` and ``-(T^2)`` are interval-algebraic rearrangements of what the chain
hands back (:func:`_trace_reversed`, :func:`_minus_t_squared`), and :mod:`.slemma`
gives SEC as the ball condition on ``Theta`` and DEC as the ball condition on
``T`` and on ``-(T^2)``.

The obstruction is on the witness side instead. For a
null ``k`` the metric term drops out, so ``Theta(k,k) = T(k,k)``: the sphere
witness that upper-bounds the NEC carries *no* information about the SEC. The
timelike conditions need a witness from the closed ball, which is what
:func:`_ball_upper` supplies, ``w = 0`` included, since ``rho < 0`` alone is a
weak-energy violation and no direction search is needed to see it.

The verdicts stay one-sided by nature: interval arithmetic refuses to certify
rather than claiming the opposite, so a saturated point comes back
``inconclusive``.

The sign convention for ``b`` differs from :mod:`.slemma` (there ``q = rho - 2b.w
+ ...``, here ``q = rho + 2b.v + ...``). It does not matter, and it does not matter
for the two rearrangements either: the conventions differ by the congruence
``J = diag(1,-1,-1,-1)``, which commutes with ``eta``, leaves ``tr_g(T)``
invariant, and satisfies ``J(-That eta That)J = -(J That J) eta (J That J)``. So
the formulas below hold verbatim in whichever convention they are fed. Do not
"fix" a sign here.
"""

from __future__ import annotations

import math

import mpmath
from mpmath import iv

from ._intervalcurv import eulerian_fields_interval
from .enclosure import _hi, _lmi_dual_lower, _lo, _mid_iv, _objective_interval, _trs_argmin

__all__ = ["certify_point_from_metric"]


def _point_box(point):
    """Degenerate interval box at ``point``, or a rigorous enclosure of it.

    ``float(c)`` is exact when ``c`` is already binary64, which is what every caller
    here passes, and then the singleton box IS the requested point. It is not exact for
    anything carried at higher precision: ``float(mpf(1) + mpf(2)**-53)`` rounds to
    ``1.0``, and a singleton built from it would certify a different point from the one
    asked about while the result still names the original. Widen by one ulp in that
    case, so the box encloses the coordinate rather than replacing it.
    """
    box = []
    for c in point:
        lo = hi = float(c)
        if lo != c:  # c is not representable in binary64; bracket it instead
            lo, hi = math.nextafter(lo, -math.inf), math.nextafter(hi, math.inf)
        box.append(iv.mpf([lo, hi]))
    return box


def _trace_reversed(rho_iv, b_iv, S_iv):
    """``(rho, b, S)`` of ``Theta = T - (1/2) tr_g(T) g``, same ``b`` convention.

    ``tr_g(T) = -rho + tr S`` in the tetrad, and ``Theta_hat = That - (1/2) tr_g(T)
    eta`` with ``eta = diag(-1,1,1,1)``, so the time-time entry gains half the
    trace while the spatial block loses it and ``b`` is untouched.
    """
    tr_s = S_iv[0][0] + S_iv[1][1] + S_iv[2][2]
    half = (rho_iv - tr_s) / 2
    rho_t = (rho_iv + tr_s) / 2
    S_t = [[S_iv[i][j] + (half if i == j else 0) for j in range(3)] for i in range(3)]
    return rho_t, list(b_iv), S_t


def _minus_t_squared(rho_iv, b_iv, S_iv):
    """``(rho, b, S)`` of ``-(T^2)_ab = -T_ac g^cd T_db``, same ``b`` convention.

    With ``That = [[rho, b^T], [b, S]]`` and ``eta = diag(-1, I)``,
    ``That eta That`` has blocks ``-rho^2 + |b|^2``, ``(S - rho I) b`` and
    ``S^2 - b b^T``; negating gives what is returned.
    """

    def _dot3(u, v):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    rho_f = rho_iv * rho_iv - _dot3(b_iv, b_iv)
    Sb = [_dot3(S_iv[i], b_iv) for i in range(3)]
    b_f = [rho_iv * b_iv[i] - Sb[i] for i in range(3)]
    S_f = [
        [b_iv[i] * b_iv[j] - _dot3(S_iv[i], [S_iv[0][j], S_iv[1][j], S_iv[2][j]]) for j in range(3)]
        for i in range(3)
    ]
    return rho_f, b_f, S_f


def _mid_fields(rho_iv, b_iv, S_iv):
    """Float midpoints, used only to PROPOSE a direction to the interval evaluator."""
    return (
        _mid_iv(rho_iv),
        [_mid_iv(c) for c in b_iv],
        [[_mid_iv(S_iv[i][j]) for j in range(3)] for i in range(3)],
    )


def _clamp_to_ball(w, pad: float = 1e-12):
    """Pull a float direction just inside the unit ball so ``|w| <= 1`` certifies.

    A float64 unit vector has ``|w|^2 = 1 + O(ulp)``, and the interval check rounds
    that endpoint UP, so the sphere candidate, the one that matters at a wall
    point, where the minimum sits at ``|w| = 1``, was rejected every time and the
    only surviving candidate was ``w = 0``. That is sound but useless: it turned
    every certified strong-energy violation into ``inconclusive``.

    Shrinking by a relative ``pad`` keeps the candidate admissible and costs ``pad``
    of tightness in a bound that is otherwise exact to the last bit.
    """
    n = math.sqrt(sum(float(c) * float(c) for c in w))
    if n <= 1.0 - pad:
        return tuple(float(c) for c in w)
    s = (1.0 - pad) / n
    return tuple(float(c) * s for c in w)


def _q_at(rho_iv, b_iv, S_iv, w):
    """Interval enclosure of ``q(w) = rho + 2 b.w + w^T S w`` at a fixed float ``w``.

    Unlike :func:`.enclosure._objective_interval` this does NOT normalise: it is the
    ball evaluator, so ``|w| <= 1`` is the caller's obligation and is checked here
    in interval arithmetic rather than assumed from the float that proposed it.
    Returns ``None`` when ``|w| <= 1`` cannot be certified, which only forfeits a
    candidate.
    """
    w = _clamp_to_ball(w)
    wi = [iv.mpf([float(c), float(c)]) for c in w]
    n2 = wi[0] * wi[0] + wi[1] * wi[1] + wi[2] * wi[2]
    if not (_hi(n2) <= 1.0):
        return None
    bdotw = b_iv[0] * wi[0] + b_iv[1] * wi[1] + b_iv[2] * wi[2]
    quad = sum(((S_iv[i][j] * wi[i]) * wi[j] for i in range(3) for j in range(3)), iv.mpf([0, 0]))
    return rho_iv + 2 * bdotw + quad


def _ball_upper(rho_iv, b_iv, S_iv) -> float:
    """Rigorous upper bound on ``min_{|w| <= 1} q(w)``.

    Any admissible ``w`` bounds the minimum from above, so this is sound whatever
    the proposals are; only its tightness depends on them. Three candidates cover
    the cases that matter: ``w = 0`` (which is why ``rho < 0`` alone convicts the
    weak condition, with no direction search at all), the unconstrained stationary
    point ``S w = -b`` when it lands inside the ball, and the sphere minimiser that
    :func:`.enclosure._trs_argmin` already computes, ray-minimised over ``t`` in
    ``[0, 1]`` so the interior of the ball is reachable along that direction too.
    """
    import numpy as np

    rho_m, b_m, S_m = _mid_fields(rho_iv, b_iv, S_iv)
    cands = [(0.0, 0.0, 0.0)]

    S_arr = np.asarray(S_m, dtype=float)
    S_arr = 0.5 * (S_arr + S_arr.T)
    try:
        w_int = np.linalg.solve(S_arr, -np.asarray(b_m, dtype=float))
        if float(np.linalg.norm(w_int)) < 1.0:
            cands.append(tuple(float(c) for c in w_int))
    except np.linalg.LinAlgError:
        pass

    v = _trs_argmin(rho_m, b_m, S_m)
    if v is not None:
        v = np.asarray(v, dtype=float)
        nv = float(np.linalg.norm(v))
        if nv > 0.0:
            v = v / nv
            # q(t v) = rho + 2t (b.v) + t^2 (v^T S v): a scalar quadratic in t.
            lin, quad = float(np.dot(b_m, v)), float(v @ S_arr @ v)
            t = 1.0 if quad <= 0.0 else min(1.0, max(0.0, -lin / quad))
            for tc in {1.0, t}:
                cands.append(tuple(float(tc * c) for c in v))

    best = math.inf
    for w in cands:
        q = _q_at(rho_iv, b_iv, S_iv, w)
        if q is not None:
            best = min(best, _hi(q))
    return best


def certify_point_from_metric(metric_dual_fn, point, *, prec: int = 80) -> dict:
    """Certify the NEC, WEC, SEC and DEC at ``point`` from the metric.

    Every verdict is a statement about the spacetime at ``point``, not about a
    float64 snapshot of its stress-energy: the whole chain ``g -> Gamma -> Riem ->
    Ric -> G -> T`` is enclosed in interval arithmetic, and the acceptance test is
    an interval ``LDL^T``. Nothing here consults a Hawking-Ellis type, an
    eigendecomposition, a rapidity cap or a classification tolerance.

    Parameters
    ----------
    metric_dual_fn
        Callable of four :class:`._intervalad.Dual2` coordinates returning the
        4x4 metric as nested ``Dual2``, the same interface the enclosure
        branch-and-bound uses (see :mod:`.enclosure` for the four constructions).
    point
        ``(t, x, y, z)``.
    prec
        Working precision in bits. ``mpmath.iv`` keeps its own, so both are set.

    Returns
    -------
    dict
        For each of ``nec``, ``wec``, ``sec``, ``dec`` a verdict (``'satisfied'``,
        ``'violated'`` or ``'inconclusive'``) and a ``*_lower``/``*_upper``
        bracket. ``dec_flux_*`` is the ``-(T^2)`` half on its own.

    Notes
    -----
    The ``*_lower`` values are sign-valid bounds, not calibrated margins. The
    multiplier bound returns ``2 lambda_min``, which is the exact null deficit on
    the sphere but on the ball can overstate the true minimum by up to a factor of
    two, because ``x = (1, w)`` has ``|x|^2 = 1 + |w|^2 <= 2``. Positive still means
    satisfied and the verdict is unaffected; do not quote a ball ``*_lower`` as a
    margin. ``dec_lower`` combines two quantities of different degree in ``T`` and
    is sign-only for that reason as well.
    """
    mpmath.mp.prec = prec
    iv.prec = prec

    rho_iv, b_iv, S_iv = eulerian_fields_interval(metric_dual_fn, _point_box(point))
    theta = _trace_reversed(rho_iv, b_iv, S_iv)
    flux = _minus_t_squared(rho_iv, b_iv, S_iv)

    # Null (sphere): the float minimiser only proposes a direction; the value is
    # re-evaluated in interval arithmetic, so a poor direction weakens the bound
    # and cannot invalidate it.
    nec_lower = _lmi_dual_lower(rho_iv, b_iv, S_iv)
    q_iv = _objective_interval(rho_iv, b_iv, S_iv, _trs_argmin(*_mid_fields(rho_iv, b_iv, S_iv)))
    nec_upper = _hi(q_iv) if q_iv is not None else math.inf

    # Timelike (ball): the multiplier is restricted to sigma >= 0, and the witness
    # must come from the ball, on the null cone Theta(k,k) = T(k,k), so the NEC
    # witness says nothing whatever about the SEC.
    wec_lower = _lmi_dual_lower(rho_iv, b_iv, S_iv, sigma_min=0.0)
    wec_upper = _ball_upper(rho_iv, b_iv, S_iv)
    sec_lower = _lmi_dual_lower(*theta, sigma_min=0.0)
    sec_upper = _ball_upper(*theta)
    flux_lower = _lmi_dual_lower(*flux, sigma_min=0.0)
    flux_upper = _ball_upper(*flux)

    def _verdict(lo, hi):
        if lo > 0.0:
            return "satisfied"
        if hi < 0.0:
            return "violated"
        return "inconclusive"

    # DEC needs both inequalities, so it is certified only when both are, and it is
    # refuted as soon as either is: min at both ends gives exactly that.
    dec_lower, dec_upper = min(wec_lower, flux_lower), min(wec_upper, flux_upper)

    return {
        "point": [float(c) for c in point],
        "prec_bits": prec,
        "rho_n": [_lo(rho_iv), _hi(rho_iv)],
        "nec_lower": nec_lower,
        "nec_upper": nec_upper,
        "nec": _verdict(nec_lower, nec_upper),
        "wec_lower": wec_lower,
        "wec_upper": wec_upper,
        "wec": _verdict(wec_lower, wec_upper),
        "sec_lower": sec_lower,
        "sec_upper": sec_upper,
        "sec": _verdict(sec_lower, sec_upper),
        "dec_flux_lower": flux_lower,
        "dec_flux_upper": flux_upper,
        "dec_lower": dec_lower,
        "dec_upper": dec_upper,
        "dec": _verdict(dec_lower, dec_upper),
    }
