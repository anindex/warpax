r"""Certified global enclosures of wall diagnostics by interval branch-and-bound.

Local refinement of a grid-sampled extremum (``warpax.analysis.extrema``) is
*basin-local*: it converges on the extremum of whichever basin the seed lands
in, and cannot by itself exclude a deeper basin that the grid never sampled.
This module supplies the missing global-coverage statement.

Method
------
Three ingredients make a rigorous global bound cheap here.

1. **Axisymmetry.** Every metric in the benchmark family is invariant under
   rotations about the propagation axis, so every *scalar* built covariantly
   from it depends only on ``(x, s)`` with ``s = sqrt(y^2 + z^2)``. Setting
   ``z = 0`` and letting ``y = s >= 0`` therefore sweeps the entire orbit space
   without loss: the search is genuinely two-dimensional, not a 2D slice of a 3D
   problem. Reflection ``z -> -z`` is an isometry fixing that half-plane, which
   is what licenses the reduction.

2. **Interval-valued automatic differentiation.** The whole curvature chain
   ``g -> Gamma -> Riem -> Ric -> G -> T`` is differentiated in *interval*
   forward mode to second order (:mod:`._intervalad`, :mod:`._intervalcurv`), so
   every derivative carries a rigorous enclosure rather than a floating-point
   estimate. Closed-form symbolic differentiation was tried first and abandoned:
   the 3-component Rodal shift, with its ``1/r`` and ``log cosh``, produced
   expressions whose interval evaluation was hopelessly over-wide.

3. **Tight interval evaluation.** The only transcendental in the shape function
   is ``tanh``, and ``tanh(u) = 1 - 2/(exp(2u) + 1)`` contains ``u`` exactly
   once, so its interval evaluation is both rigorous and free of dependency
   overestimation.

The search is Moore-Skelboe branch-and-bound: maintain a priority queue of
boxes ordered by the lower end of the enclosure of the objective, split the most
promising box, and discard boxes that cannot contain the global minimum. On exit

    lower  <=  inf over the domain  <=  upper

with ``lower`` from interval arithmetic over *all* surviving boxes (hence valid
on the continuum, not merely at sample points) and ``upper`` from an actual
evaluated point. The gap is the certified enclosure width.

Objective
---------
We certify the null-energy deficit in the form that is defined at every point
regardless of Hawking-Ellis type. Writing a future null direction as
``k = n + v`` with ``n`` the Eulerian normal and ``v`` a unit spatial vector,

    q(v) = T_ab k^a k^b = rho_n + 2 b.v + v^T S v,
    rho_n = T_ab n^a n^b,   b_i = n^a T_ai,   S_ij = T_ij,

and the pointwise deficit is ``N(x) = min_{|v|=1} q(v)``. A rigorous lower bound
needs no eigen-decomposition:

    N(x) >= rho_n - 2|b| + lambda_min(S),

with ``lambda_min(S)`` bounded below by Gershgorin on the interval matrix. That is
looser than a Weyl bound taken from ``eigvalsh`` of the midpoint, but it is
*rigorous*: an eigensolver carries no certified error bound, and this quantity is
subtracted from a lower bound, where an inward error is not conservative. At a
point where the momentum vanishes (any irrotational shift, e.g. Rodal) ``b = 0``
and the bound is an equality, ``N = rho_n + lambda_min(S) = min_i(rho + p_i)``,
so for those drives this certifies the invariant margin itself.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import mpmath
from mpmath import iv

from . import _intervalad as ad
from ._intervalcurv import eulerian_fields_interval

__all__ = [
    "METRICS",
    "Enclosure",
    "alcubierre_metric",
    "certify_nec_deficit",
    "rodal_metric",
    "shape_interval",
    "tail_bound",
]

# A block of sympy-lambdify interval helpers lived here, left over from an
# earlier symbolic-differentiation attempt that was abandoned (the 3-component
# Rodal shift with 1/r and log cosh blew up). It was unreachable, it called
# ``sp.lambdify`` with no ``sympy`` import, and has been removed. The live
# path is the interval forward-mode AD of ``_intervalad``.


# ---------------------------------------------------------------------------
# rigorous interval constants
# ---------------------------------------------------------------------------
#
# The shape functions carry transcendental *constants*, tanh(sigma R),
# sinh(R sigma), cosh(R sigma). Evaluating them with point mpmath and handing
# the result to ad.constant() wrapped a ROUNDED value as a degenerate interval,
# so their rounding error escaped the enclosure. mpmath.iv exposes only exp, log
# and sqrt, so we build the rest from iv.exp exactly as _intervalad does.


def _c_exp(u):
    return iv.exp(iv.mpf([u, u]))


def _c_tanh(u):
    """Interval enclosure of ``tanh(u)``; ``u`` appears once, so it is tight."""
    return 1 - 2 / (_c_exp(2 * u) + 1)


def _c_sinh(u):
    # e - 1/e is monotone increasing in e, so the naive interval form is exact.
    e = _c_exp(u)
    return (e - 1 / e) / 2


def _c_cosh(u):
    e = _c_exp(u)
    return (e + 1 / e) / 2


# ---------------------------------------------------------------------------
# metrics in the interval-AD algebra
# ---------------------------------------------------------------------------


def _shape(r, R, sigma):
    """tanh top-hat, written so each occurrence of ``r`` is a separate leaf."""
    return (ad.tanh(sigma * (r + R)) - ad.tanh(sigma * (r - R))) / (
        ad.constant(2 * _c_tanh(sigma * R))
    )


def _assemble(beta, conformal=None):
    """ADM metric of a unit-lapse drive with shift ``beta^i``.

    ``conformal`` is the factor ``B`` of a conformally flat slice
    ``gamma_ij = B^2 delta_ij``; ``None`` means the flat-slice case ``B = 1``. Then
    ``g_00 = B^2 |beta|^2 - 1``, ``g_0i = B^2 beta_i`` and ``g_ij = B^2 delta_ij``.
    """
    g = [[ad.constant(0) for _ in range(4)] for _ in range(4)]
    b2 = beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2]
    if conformal is None:
        g[0][0] = b2 - 1
        for i in range(3):
            g[0][i + 1] = beta[i]
            g[i + 1][0] = beta[i]
            g[i + 1][i + 1] = ad.constant(1)
        return g
    B2 = conformal * conformal
    g[0][0] = B2 * b2 - 1
    for i in range(3):
        g[0][i + 1] = B2 * beta[i]
        g[i + 1][0] = g[0][i + 1]
        g[i + 1][i + 1] = B2
    return g


def alcubierre_metric(v_s=0.5, R=1.0, sigma=8.0):
    def fn(t, x, y, z):
        # The bubble translates: x_s(t) = v_s t. Omitting this makes d_t g vanish
        # and silently corrupts the curvature even on the t = 0 slice.
        dx = x - v_s * t
        r = ad.sqrt(dx * dx + y * y + z * z + ad.constant(1e-60))
        f = _shape(r, R, sigma)
        return _assemble([-v_s * f, ad.constant(0), ad.constant(0)])

    return fn


def rodal_metric(v_s=0.5, R=1.0, sigma=8.0):
    """Irrotational Rodal drive, lab-frame standardization (paper Appendix F)."""

    def fn(t, x, y, z):
        dx = x - v_s * t
        r2 = dx * dx + y * y + z * z
        # Two floors, matching metrics/rodal.py exactly: the tight one for the
        # profile values, the coarser one in the direction divisor. Using r_safe
        # for both here certified a bracket for a different metric from the one
        # every other table reports.
        r = ad.sqrt(r2 + ad.constant(1e-60))
        r_div = ad.sqrt(r2 + ad.constant(1e-12))
        F = _shape(r, R, sigma)
        num = r * ad.constant(2 * sigma * _c_sinh(R * sigma)) + ad.constant(_c_cosh(R * sigma)) * (
            ad.log(ad.cosh(sigma * (r - R))) - ad.log(ad.cosh(sigma * (r + R)))
        )
        G = 1 - num / (r * ad.constant(2 * sigma * _c_sinh(R * sigma)))
        n = [dx / r_div, y / r_div, z / r_div]
        beta = [-v_s * (G * (1 if i == 0 else 0) + (F - G) * n[0] * n[i]) for i in range(3)]
        return _assemble(beta)

    return fn


def natario_metric(v_s=0.5, R=1.0, sigma=8.0):
    """Natario zero-expansion drive, lab-frame standardization (paper Appendix F).

    ``n(r) = (1 - f(r))/2`` with ``f`` the tanh top-hat, and
    ``dn/dr = -sigma (sech^2(sigma(r+R)) - sech^2(sigma(r-R))) / (4 tanh(sigma R))``.
    The Cartesian shift is the one that makes ``div beta = 0`` exactly; the interval
    form below is a transcription of ``metrics/natario.py``, checked against it
    pointwise in ``tests/test_enclosure.py``.

    Expect wider brackets than Alcubierre: the shift carries ``r`` in several factors
    and a ``1/r``, and interval dependency grows with the number of occurrences.  The
    form below therefore shares subexpressions rather than recomputing them.  Both
    ``f`` and ``dn`` are built from the *same* two ``tanh`` nodes via
    ``sech^2(u) = 1 - tanh^2(u)``, and ``dn/r`` is formed once as ``q``, which drops
    ``r`` from eight occurrences to three.  This is an algebraic identity, so point
    values are unchanged (``tests/test_enclosure.py`` pins them against
    ``metrics/natario.py``); it is also cheaper, two ``tanh`` and no ``cosh``.
    """

    def fn(t, x, y, z):
        dx = x - v_s * t
        r = ad.sqrt(dx * dx + y * y + z * z + ad.constant(1e-60))
        tp = ad.tanh(sigma * (r + R))
        tm = ad.tanh(sigma * (r - R))
        norm = ad.constant(2 * _c_tanh(sigma * R))
        n = (1 - (tp - tm) / norm) / 2
        # dn/dr = -sigma (sech^2(sigma(r+R)) - sech^2(sigma(r-R))) / (4 tanh(sigma R))
        #       =  sigma (tp^2 - tm^2)            / (4 tanh(sigma R))
        dn = (tp * tp - tm * tm) * ad.constant(iv.mpf([sigma, sigma]) / (4 * _c_tanh(sigma * R)))
        q = dn / r
        beta = [
            -v_s * (2 * n + q * (y * y + z * z)),
            v_s * q * dx * y,
            v_s * q * dx * z,
        ]
        return _assemble(beta)

    return fn


def van_den_broeck_metric(v_s=0.5, R=1.0, sigma=8.0, R_tilde=1.0, alpha_vdb=0.5, sigma_B=8.0):
    """Van den Broeck volume-modified drive: Alcubierre shift on a conformal slice.

    ``B(r) = 1 + alpha f_B(r; R_tilde, sigma_B)`` with ``f_B`` the same tanh top-hat,
    and ``gamma_ij = B^2 delta_ij``. Transcription of ``metrics/van_den_broeck.py``,
    checked against it pointwise in ``tests/test_enclosure.py``.
    """

    def fn(t, x, y, z):
        dx = x - v_s * t
        r = ad.sqrt(dx * dx + y * y + z * z + ad.constant(1e-60))
        f = _shape(r, R, sigma)
        B = 1 + alpha_vdb * _shape(r, R_tilde, sigma_B)
        return _assemble([-v_s * f, ad.constant(0), ad.constant(0)], conformal=B)

    return fn


def shape_interval(R=1.0, sigma=8.0):
    """Interval enclosure of the tanh top-hat over a 2D box, for the wall mask."""

    def fn(bx, by):
        # The radius is built in interval arithmetic and handed to iv.sqrt, rather
        # than assembled from round-to-nearest math.sqrt calls and installed as
        # endpoints. The old form could return an interval that did not contain the
        # true shape value, at x = 0.950030236708907, s = 0.5338202611110032 it
        # returned [0.19220380596722561, 0.19220380596722562] around a true
        # 0.19220380596722557. This function decides which boxes are discarded by
        # the wall mask, so a non-enclosure here can drop a box that contains the
        # minimiser and silently raise the certified lower bound.
        def _abs_range(c):
            lo_c, hi_c = mpmath.mpf(c.a), mpmath.mpf(c.b)
            if lo_c >= 0:
                lo_a, hi_a = lo_c, hi_c
            elif hi_c <= 0:
                lo_a, hi_a = -hi_c, -lo_c
            else:
                lo_a, hi_a = mpmath.mpf(0), max(-lo_c, hi_c)
            return iv.mpf([lo_a, lo_a]), iv.mpf([hi_a, hi_a])

        xlo, xhi = _abs_range(bx)
        ylo, yhi = _abs_range(by)
        r2 = iv.mpf([mpmath.mpf((xlo * xlo + ylo * ylo).a), mpmath.mpf((xhi * xhi + yhi * yhi).b)])
        r = iv.sqrt(r2)
        th = lambda u: 1 - 2 / (iv.exp(2 * u) + 1)
        return (th(sigma * (r + R)) - th(sigma * (r - R))) / (2 * _c_tanh(sigma * R))

    return fn


METRICS = {
    "Alcubierre": alcubierre_metric,
    "Rodal": rodal_metric,
    "Natario": natario_metric,
    "VanDenBroeck": van_den_broeck_metric,
}


# ---------------------------------------------------------------------------
# branch and bound
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _Box:
    key: float
    lo: tuple = field(compare=False)
    hi: tuple = field(compare=False)


@dataclass
class Enclosure:
    """Result of a certified global search.

    ``lower <= inf_{domain} objective <= upper`` holds on the continuum.
    """

    lower: float
    upper: float
    argmin: tuple
    n_boxes: int
    n_evaluations: int
    tail_lower: float | None = None

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float, atol: float = 0.0) -> bool:
        """Whether a previously reported extremum lies inside the enclosure."""
        return (self.lower - atol) <= value <= (self.upper + atol)


def _lo(c) -> float:
    """Lower endpoint of an interval, rounded DOWN into binary64.

    float(mpmath.mpf(x)) rounds to NEAREST, which is the wrong direction for an
    endpoint that is used conservatively, and every use here is conservative:

      * the LDL pivot positivity test accepts a certified-positive-definite matrix
        on ``_lo(pivot) > 0``, so a lower endpoint rounded UP can accept a pivot
        that is actually non-positive and certify a bound that does not hold;
      * the Gershgorin bound subtracts a radius from ``_lo(S_ii)``, and an inward
        error there raises a LOWER bound above the truth;
      * the wall mask discards a box on ``_lo(f) > wall_hi``.

    Rounding outward can only widen an interval, so it is unconditionally safe: it
    can make a bound looser, never unsound. The defect is sub-ulp and moves no
    published bracket, but "sound except in the last bit" is not a certificate.
    """
    v = float(mpmath.mpf(c.a))
    return v if v == -math.inf else math.nextafter(v, -math.inf)


def _hi(c) -> float:
    """Upper endpoint of an interval, rounded UP into binary64. See :func:`_lo`."""
    v = float(mpmath.mpf(c.b))
    return v if v == math.inf else math.nextafter(v, math.inf)


def _mid_iv(c) -> float:
    return 0.5 * (_lo(c) + _hi(c))


def _rad_iv(c) -> float:
    return 0.5 * (_hi(c) - _lo(c))


def _mag(c) -> float:
    """Largest absolute value attained on the interval."""
    return max(abs(_lo(c)), abs(_hi(c)))


# Converting mpmath endpoints to double rounds to nearest, which could shave a
# sub-ulp sliver off an enclosure. Widening every bound by this relative amount
# restores outward rounding with a large margin: even at the 53-bit floor the
# per-operation defect is ~1e-16 relative, four orders inside this pad. (An
# earlier comment justified the pad by "60-bit precision", which was not in
# force, mpmath.iv keeps its own precision and mp.prec alone did not set it.)
_OUTWARD = 1e-12


# --- the S-lemma dual bound -------------------------------------------------
#
# The decoupled bound  N >= rho_lo - 2|b|_max + lambda_min_lo(S)  is valid but it
# does NOT converge. Shrinking the box to a point leaves the residual
#
#     [rho - 2|b| + lambda_min(S)]  -  min_{|v|=1} (rho + 2 b.v + v^T S v)  <  0,
#
# because it charges the worst case of the linear and quadratic terms in
# DIFFERENT directions v; the two agree only when b happens to be parallel to the
# lowest eigenvector of S. Measured on the Alcubierre wall the residual is 0.065
# at a box of half-width 1e-7, so the branch-and-bound bracket cannot close at any
# budget. That is what kept the Natario entry uninformative.
#
# The replacement uses the same duality the pointwise certificate uses. With
#
#     That = [[rho, b^T], [b, S]],   eta = diag(-1, 1, 1, 1),
#     M(s) = That + s eta,
#
# one has  min_{|v|=1} q(v) = 2 max_s lambda_min(M(s))  exactly, so for ANY fixed
# multiplier s,  N >= 2 lambda_min(M(s)). Optimality of s costs tightness only,
# never validity. Over a box the interval matrix M(s) is bounded below by a
# verified LDL^T: if every pivot of  M(s) - t I  computed in interval arithmetic
# is strictly positive, then every real member is positive definite, so
# lambda_min >= t pointwise on the whole box. Both are rigorous, so the two bounds
# can simply be maxed.

_ETA4 = ((-1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _that_interval(rho_iv, b_iv, S_iv):
    """``That = [[rho, b^T], [b, S]]`` as a 4x4 symmetric interval matrix."""
    M = [[None] * 4 for _ in range(4)]
    M[0][0] = rho_iv
    for i in range(3):
        M[0][i + 1] = b_iv[i]
        M[i + 1][0] = b_iv[i]
        for j in range(3):
            M[i + 1][j + 1] = S_iv[i][j]
    return M


def _ldl_positive_definite(M, shift) -> bool:
    """Certify ``M - shift*I >> 0`` for every member of the interval matrix ``M``.

    Interval LDL^T without pivoting. Real arithmetic on any member reproduces the
    same recurrence, and every real intermediate lies in the corresponding
    interval, so strictly positive interval pivots certify positive definiteness
    of every member. Returns ``False`` when the interval arithmetic is merely
    inconclusive, a refusal to certify, never a claim of indefiniteness.
    """
    n = 4
    A = [[M[i][j] - (shift if i == j else 0) for j in range(n)] for i in range(n)]
    for k in range(n):
        pivot = A[k][k]
        if not (_lo(pivot) > 0.0):
            return False
        for i in range(k + 1, n):
            factor = A[i][k] / pivot
            for j in range(i, n):
                A[i][j] = A[i][j] - factor * A[k][j]
                A[j][i] = A[i][j]
    return True


def _lmi_dual_lower(rho_iv, b_iv, S_iv, *, sigma_min: float = -math.inf) -> float:
    """Rigorous lower bound on ``min_{|v|=1} rho + 2 b.v + v^T S v`` over the box.

    Returns ``-inf`` when nothing can be certified, so the caller can fall back.
    ``sigma_min`` floors the multiplier search: ``0.0`` restricts it to the
    non-negative multipliers the weak energy condition requires, which is what
    :mod:`.interval_lmi` uses to decide the ball rather than the sphere.
    """
    import numpy as np

    That = _that_interval(rho_iv, b_iv, S_iv)
    mid = np.array([[_mid_iv(That[i][j]) for j in range(4)] for i in range(4)])
    mid = 0.5 * (mid + mid.T)
    eta = np.array(_ETA4)
    # The clamp at 1.0 is the opposite of what slemma.noise_floor does, and that
    # asymmetry is deliberate. There the bracket must collapse with the tensor or a
    # relative floor convicts Minkowski; here the bracket only PROPOSES a multiplier
    # and the interval LDL below is the acceptance test, so an over-wide bracket
    # costs residual, never soundness. Below unit scale it makes the search absolute
    # and the returned bound looser, a near-vacuum point comes back inconclusive
    # rather than certified, which is the honest answer at a saturated point anyway.
    # ponytail: fix by dividing the interval matrix by an exact binary64 upper bound
    # on max|That| (PSD is scale-invariant) if a census ever reports inconclusives.
    scale = max(1.0, float(np.max(np.abs(mid))))

    # Maximise the concave s -> lambda_min(mid + s eta). Float only: the
    # multiplier is a free parameter, and a poor choice weakens the bound
    # without invalidating it.
    def lam(s):
        return float(np.linalg.eigvalsh(mid + s * eta)[0])

    lo_s, hi_s = max(-2.0 * scale, sigma_min), 2.0 * scale
    if not (hi_s > lo_s):
        return -math.inf
    for _ in range(80):
        m1 = lo_s + (hi_s - lo_s) / 3.0
        m2 = hi_s - (hi_s - lo_s) / 3.0
        if lam(m1) < lam(m2):
            lo_s = m1
        else:
            hi_s = m2
    s_star = 0.5 * (lo_s + hi_s)

    # Weyl: lambda_min moves by at most the spectral norm of the perturbation,
    # which the Frobenius norm of the radius matrix bounds. Used only to place the
    # first trial value; the acceptance test is the interval LDL below.
    rad_f = math.sqrt(sum(_rad_iv(That[i][j]) ** 2 for i in range(4) for j in range(4)))
    M_iv = [[That[i][j] + s_star * _ETA4[i][j] for j in range(4)] for i in range(4)]

    t_hi = lam(s_star) - rad_f
    if not math.isfinite(t_hi):
        return -math.inf
    # A pivot of size epsilon cannot be certified positive in interval arithmetic,
    # so the certifiable t sits strictly below lambda_min. Bracket it by doubling,
    # then bisect: the loop below spends ~50 LDL factorisations of a 4x4, and it
    # matters, a fixed geometric back-off instead of a bisection left 0.03 of
    # slack on the Alcubierre wall, which is the whole residual gap.
    offset = max(rad_f, 1e-12 * scale)
    t_lo = None
    for _ in range(60):
        cand = t_hi - offset
        if cand < -1e8 * scale:
            break
        if _ldl_positive_definite(M_iv, cand):
            t_lo = cand
            break
        offset *= 2.0
    if t_lo is None:
        return -math.inf
    for _ in range(40):
        mid_t = 0.5 * (t_lo + t_hi)
        if _ldl_positive_definite(M_iv, mid_t):
            t_lo = mid_t
        else:
            t_hi = mid_t
    return 2.0 * t_lo


def _trs_argmin(rho: float, b, S):
    """Minimizing direction of ``rho + 2 b.v + v^T S v`` over ``|v| = 1``.

    The classical secular-equation solution: at the global minimizer
    ``(S - mu I) v = -b`` with ``mu <= lambda_min(S)``, and ``|v| = 1`` fixes
    ``mu`` as the unique root of ``sum_i (b~_i/(lam_i - mu))^2 = 1`` on
    ``(-inf, lambda_min)``. The degenerate ("hard") case, where ``b`` has no
    component along the lowest eigenvector, is handled by placing the missing
    norm in that eigenspace.

    Returns the direction, not the value: the value is then re-evaluated in
    interval arithmetic (:func:`_objective_interval`), which is what makes the
    resulting upper bound rigorous. A suboptimal direction only weakens the
    bound; it cannot invalidate it. Returns ``None`` if no direction is found.
    """
    import numpy as np

    S = np.asarray(S, dtype=float)
    S = 0.5 * (S + S.T)
    b = np.asarray(b, dtype=float)
    w, V = np.linalg.eigh(S)
    bt = V.T @ b

    # Hard case: b has no component along the WHOLE lowest eigenspace, not merely
    # along the first eigenvector returned. With S = diag(0, 0, 2) and b = (0,1,0)
    # the lowest eigenvalue is degenerate, bt[0] = 0 but bt[1] != 0, so testing
    # bt[0] alone took the hard branch and returned v = (1,0,0) with q = 0 where
    # the true minimiser is v = (0,-1,0) with q = -2.
    low = np.abs(w - w[0]) <= 1e-14 * max(1.0, abs(w[0]))
    if np.max(np.abs(bt[low])) < 1e-14:
        rest = np.zeros(3)
        nz = np.abs(w - w[0]) > 1e-14
        rest[nz] = -bt[nz] / (w[nz] - w[0])
        n2 = float(rest @ rest)
        if n2 <= 1.0:
            v = rest.copy()
            v[0] = math.sqrt(max(0.0, 1.0 - n2))
            v = V @ v
            nv = float(np.linalg.norm(v))
            return (v / nv) if nv > 0 else None

    def norm_at(mu):
        d = w - mu
        if np.any(np.abs(d) < 1e-300):
            return np.inf
        return float(np.sqrt(np.sum((bt / d) ** 2)))

    # ``norm_at`` increases monotonically as mu rises towards lambda_min, so the
    # root of norm_at(mu) = 1 lies BELOW any mu where the norm exceeds one. The
    # previous update moved the wrong endpoint and converged to the bracket top:
    # for S = diag(0,1,2), b = (1,1,0.2) it returned -2.4653 against the exact
    # -2.4972.
    lo, hi = w[0] - float(np.linalg.norm(b)) - 1.0, w[0] - 1e-15
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_at(mid) > 1.0:
            hi = mid
        else:
            lo = mid
    d = w - hi
    if np.any(np.abs(d) < 1e-300):
        return None
    v = V @ (-bt / d)
    nv = float(np.linalg.norm(v))
    return (v / nv) if nv > 0 else None


def _trs_argmin_lmi(rho: float, b, S):
    """Minimising direction recovered from the LMI multiplier, not the secular root.

    At the optimal multiplier ``s*`` the semidefinite relaxation is tight and its
    optimal solution is rank one, so the eigenvector ``x`` of ``M(s*)`` for the
    least eigenvalue has the form ``x0 (1, v)`` with ``v`` the minimiser. This
    agrees with the dual value by construction, which the secular solver does not:
    on the Alcubierre wall ``_trs_argmin`` returns a direction worth ``-0.2765``
    where the true minimum is ``-0.3070``, and that 0.03 was the entire residual
    width of the bracket at a point-sized box.

    Returns ``None`` in the hard case ``x0 ~ 0``, where the caller falls back.
    """
    import numpy as np

    S = np.asarray(S, dtype=float)
    S = 0.5 * (S + S.T)
    b = np.asarray(b, dtype=float)
    M0 = np.zeros((4, 4))
    M0[0, 0] = rho
    M0[0, 1:] = b
    M0[1:, 0] = b
    M0[1:, 1:] = S
    eta = np.array(_ETA4)
    scale = max(1.0, float(np.max(np.abs(M0))))

    def lam(s):
        return float(np.linalg.eigvalsh(M0 + s * eta)[0])

    lo, hi = -2.0 * scale, 2.0 * scale
    for _ in range(120):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if lam(m1) < lam(m2):
            lo = m1
        else:
            hi = m2
    s_star = 0.5 * (lo + hi)
    w, V = np.linalg.eigh(M0 + s_star * eta)
    x = V[:, 0]
    if abs(x[0]) < 1e-12:
        return None
    v = x[1:] / x[0]
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        return None
    return v / n


def _objective_interval(rho_iv, b_iv, S_iv, v):
    """Interval enclosure of ``q(v/|v|) = rho + 2 b.vhat + vhat^T S vhat``.

    ``v`` arrives as float64 components, which are exact binary rationals, so the
    only inexactness is in the arithmetic, all of it done in interval form
    here, including the normalisation ``|v|``. The upper endpoint of the result
    is therefore a rigorous upper bound on ``N`` at this point, and hence on the
    infimum over the whole domain.
    """
    if v is None:
        return None
    vi = [iv.mpf([float(c), float(c)]) for c in v]
    n2 = vi[0] * vi[0] + vi[1] * vi[1] + vi[2] * vi[2]
    if _lo(n2) <= 0.0:
        return None
    nrm = iv.sqrt(n2)
    u = [c / nrm for c in vi]
    bdotv = b_iv[0] * u[0] + b_iv[1] * u[1] + b_iv[2] * u[2]
    quad = sum((S_iv[i][j] * u[i]) * u[j] for i in range(3) for j in range(3))
    return rho_iv + 2 * bdotv + quad


def _make_objective(metric_fn, shape_fn, wall_lo=0.1, wall_hi=0.9):
    """Bounding function over a 2D box in ``(x, s)`` of the reduced half-plane."""

    def _eval(bx, by, jet=False):
        box = [iv.mpf([0, 0]), bx, by, iv.mpf([0, 0])]
        return eulerian_fields_interval(metric_fn, box, jet=jet)

    def _centered(comp_jet, comp_ctr, dx, dy):
        """Mean-value enclosure ``f(c) + f'(X).(X - c)`` of one component.

        ``comp_jet`` carries the gradient over the box, ``comp_ctr`` is the exact
        value at the centre (a degenerate-box evaluation), and ``dx``/``dy`` are the
        intervals ``[x] - c_x`` and ``[y] - c_y``. Rigorous by the mean value theorem
        on a box, which is convex; the chain being differentiable there is certified
        by the box evaluation having succeeded.
        """
        return comp_ctr + comp_jet.d[1] * dx + comp_jet.d[2] * dy

    def bounds(bx, by):
        """Rigorous enclosure of the NEC deficit ``N`` over the box, plus ``f``.

        Two evaluations. The *centre* evaluation is on a degenerate box, so it is
        an exact pointwise value of ``N`` and hence a valid upper bound on the
        infimum, an achieved value, not an interval artefact. The *box*
        evaluation supplies a rigorous lower bound

            N >= rho_lo - 2 |b|_max + lambda_min_lo(S),

        with ``lambda_min_lo`` from Weyl's inequality applied to the interval
        matrix: the least eigenvalue of a symmetric matrix moves by at most the
        spectral norm of the perturbation, which the Frobenius norm of the radius
        matrix bounds from above.
        """
        # The shape enclosure never depends on the curvature evaluation, so
        # compute it first: it must stay available even when the metric chain
        # fails on a wide box, or such a box could never be discarded by the
        # wall mask and would pin the certified lower bound at -inf forever.
        try:
            f_iv = shape_fn(bx, by)
        except (ZeroDivisionError, ValueError, OverflowError):
            f_iv = iv.mpf([0, 1])
        if _hi(f_iv) < wall_lo or _lo(f_iv) > wall_hi:
            return math.inf, math.inf, f_iv
        # Run the box evaluation over the jet ring when we can: it returns the same
        # interval values, plus the gradients the centered bound below needs. The
        # plain ring stays as a fallback because the jet ring refuses one case the
        # plain one tolerates (an ``abs`` whose argument straddles zero), and a wide
        # box is exactly where that can happen.
        jet_fields = None
        try:
            jet_fields = _eval(bx, by, jet=True)
            rho, bvec, S = (
                jet_fields[0].v,
                [c.v for c in jet_fields[1]],
                [[jet_fields[2][i][j].v for j in range(3)] for i in range(3)],
            )
        except (ZeroDivisionError, ValueError, OverflowError, NotImplementedError):
            jet_fields = None
            try:
                rho, bvec, S = _eval(bx, by)
            except (ZeroDivisionError, ValueError, OverflowError, NotImplementedError):
                return -math.inf, math.inf, f_iv

        parts = [rho, *bvec, *[S[i][j] for i in range(3) for j in range(3)]]
        if not all(math.isfinite(_lo(c)) and math.isfinite(_hi(c)) for c in parts):
            # A box straddling the coordinate axis r = 0 encloses the removable
            # 1/r of the spherical form. Report it as unbounded so that
            # branch-and-bound splits it rather than trusting it; the sub-boxes
            # leave the wall band and are then discarded by the mask test.
            return -math.inf, math.inf, f_iv

        def _decoupled(rho_, b_, S_):
            """Rigorous version of ``N >= rho_lo - 2|b|_max + lambda_min_lo(S)``.

            The inequality is valid; an earlier implementation of it was not. It
            converted interval endpoints to binary64 with round-to-nearest, took the
            radius with the same rounding, and read ``lambda_min`` off
            ``numpy.linalg.eigvalsh`` of the midpoint matrix, which carries no
            certified error bound. Every one of those errors can move the result
            *upward*, and since this value is max'd against the verified LMI bound an
            upward error is not conservative, it is an unsound lower bound that
            wins. A constructed example at coefficient scale 1e100 returned a
            "lower bound" six times the true deficit and above an achieved upper
            bound.

            Everything below is therefore done in interval arithmetic, with
            ``lambda_min`` bounded by Gershgorin (rigorous for a symmetric interval
            matrix, and needing no eigensolver), and the single conversion to
            binary64 rounds outward.
            """
            # lambda_min(S) >= min_i ( S_ii - sum_{j != i} |S_ij| ), Gershgorin.
            lam_iv = None
            for i in range(3):
                row = iv.mpf([0, 0])
                for j in range(3):
                    if j != i:
                        row = row + abs(S_[i][j])
                cand = S_[i][i] - row
                lam_iv = (
                    cand
                    if lam_iv is None
                    else iv.mpf(
                        [
                            min(mpmath.mpf(lam_iv.a), mpmath.mpf(cand.a)),
                            min(mpmath.mpf(lam_iv.b), mpmath.mpf(cand.b)),
                        ]
                    )
                )
            b2 = iv.mpf([0, 0])
            for c in b_:
                b2 = b2 + c * c
            # |b|^2 is a sum of squares and cannot be negative, but its interval can
            # dip below zero when the working precision is low enough that the squares
            # round outward past it, and then iv.sqrt raises ComplexResult and the
            # whole enclosure dies. That happened whenever certify_nec_deficit ran
            # without another caller having already raised iv.prec, since it is global
            # mutable state; the tests only passed because an earlier test in the same
            # process had set it. Clamping the LOWER endpoint up to zero is outward-safe
            # here: it leaves the upper endpoint alone, and it is the upper endpoint of
            # 2 sqrt(b2) that this bound subtracts.
            if mpmath.mpf(b2.a) < 0:
                b2 = iv.mpf([0, mpmath.mpf(b2.b)])
            bnd = rho_ - 2 * iv.sqrt(b2) + lam_iv
            return _lo(bnd)

        # Two independently valid lower bounds; the max of them is valid too. The
        # decoupled one is retained because it survives boxes on which the LDL
        # declines to certify, and it is occasionally the better of the two on very
        # wide boxes where the midpoint carries no information.
        lower = max(_decoupled(rho, bvec, S), _lmi_dual_lower(rho, bvec, S))

        # Centre evaluation. It is a degenerate box, so it carries no dependency at
        # all: it supplies the exact pointwise value used both as the achieved upper
        # bound below and as the expansion point of the centered lower bound here.
        cx, cy = _mid_iv(bx), _mid_iv(by)
        ctr = None
        try:
            ctr = _eval(iv.mpf([cx, cx]), iv.mpf([cy, cy]))
        except (ZeroDivisionError, ValueError, OverflowError, NotImplementedError):
            pass

        # Centered (mean-value) lower bound. Naive interval evaluation of the
        # curvature chain overestimates by a factor that is constant in the box
        # width, measured at 148x on the Alcubierre wall, because the same
        # derivative enters many cancelling terms. The mean-value form replaces that
        # O(h) excess with O(h^2) and is what lets the branch and bound close at a
        # realistic box budget. It is taken only as a max against the bounds above,
        # so it can tighten the result and never loosen or invalidate it.
        if jet_fields is not None and ctr is not None:
            dx = bx - iv.mpf([cx, cx])
            dy = by - iv.mpf([cy, cy])
            rho_k = _centered(jet_fields[0], ctr[0], dx, dy)
            b_k = [_centered(jet_fields[1][i], ctr[1][i], dx, dy) for i in range(3)]
            S_k = [
                [_centered(jet_fields[2][i][j], ctr[2][i][j], dx, dy) for j in range(3)]
                for i in range(3)
            ]
            parts_k = [rho_k, *b_k, *[S_k[i][j] for i in range(3) for j in range(3)]]
            # On a very wide box the centered form can be looser than the naive one
            # and large enough to overflow the squaring in the decoupled bound. It is
            # a tightening only, so any failure here just leaves ``lower`` as it was.
            try:
                if all(math.isfinite(_lo(c)) and math.isfinite(_hi(c)) for c in parts_k):
                    lower = max(
                        lower, _decoupled(rho_k, b_k, S_k), _lmi_dual_lower(rho_k, b_k, S_k)
                    )
            except (OverflowError, ValueError, ZeroDivisionError):
                pass

        # Achieved value at the box centre (degenerate box => a single point).
        #
        # The direction is found in floating point, but the *value* is then
        # evaluated in interval arithmetic and its upper endpoint taken. That is
        # what makes this a rigorous upper bound on the infimum: for ANY unit v
        # at ANY point, q(v) >= N(point) >= inf N. Optimality of v only affects
        # how tight the bound is, never whether it is valid. Previously the value
        # came from a float trust-region solve on the *midpoints* of separately
        # rounded rho, b and S, a synthetic tensor that need not be T at that
        # point, and a float result with no outward rounding.
        upper = math.inf
        f_mid = shape_fn(iv.mpf([cx, cx]), iv.mpf([cy, cy]))
        if ctr is not None and _lo(f_mid) >= wall_lo and _hi(f_mid) <= wall_hi:
            try:
                rho_c, b_c, S_c = ctr
                rho_m = _mid_iv(rho_c)
                b_m = [_mid_iv(c) for c in b_c]
                S_m = [[_mid_iv(S_c[i][j]) for j in range(3)] for i in range(3)]
                # Two candidate directions, both evaluated rigorously; the smaller
                # upper endpoint wins. Neither can invalidate the bound, any unit
                # direction gives a valid upper bound on the infimum, so this is
                # purely a tightening.
                for v_star in (_trs_argmin_lmi(rho_m, b_m, S_m), _trs_argmin(rho_m, b_m, S_m)):
                    if v_star is None:
                        continue
                    q_iv = _objective_interval(rho_c, b_c, S_c, v_star)
                    if q_iv is not None and math.isfinite(_hi(q_iv)):
                        upper = min(upper, _hi(q_iv))
            except (ZeroDivisionError, ValueError, OverflowError, NotImplementedError):
                pass

        scale = max(1.0, abs(lower), abs(upper) if math.isfinite(upper) else 1.0)
        return (
            lower - _OUTWARD * scale,
            upper + (_OUTWARD * scale if math.isfinite(upper) else 0.0),
            f_iv,
        )

    return bounds


def certify_nec_deficit(
    metric_fn,
    shape_fn,
    x_range,
    s_range,
    *,
    wall_mask=(0.1, 0.9),
    max_boxes=20000,
    tol=1e-3,
    prec=60,
):
    """Certified global minimum of the null-energy deficit over a 2D domain.

    Parameters
    ----------
    metric_fn
        The metric in the :mod:`._intervalad` ``Dual2`` algebra: takes four
        seeded duals ``(t, x, y, z)`` and returns a nested-list 4x4. See
        :func:`alcubierre_metric` and :func:`rodal_metric` for templates.
    shape_fn
        Interval enclosure of the shape function over a 2-D box, used only to
        decide whether a box can meet ``wall_mask``. See :func:`shape_interval`.
    x_range, s_range
        Axial and cylindrical-radial extent of the search box. ``s >= 0``.
    wall_mask
        Only boxes whose shape-function enclosure meets this band are searched,
        matching the wall restriction used throughout the paper. A box is kept
        unless its ``f``-enclosure lies entirely outside the band, so no wall
        point is discarded, including points on the curved mask boundary.
    tol
        Target enclosure width.
    """
    # Both must be set: mpmath.iv keeps its OWN precision, so setting only
    # mp.prec left every interval operation at the 53-bit default and made
    # this argument inert.
    mpmath.mp.prec = prec
    iv.prec = prec
    bounds = _make_objective(metric_fn, shape_fn, wall_mask[0], wall_mask[1])

    def box_bounds(lo, hi):
        bx = iv.mpf([lo[0], hi[0]])
        by = iv.mpf([lo[1], hi[1]])
        return bounds(bx, by)

    lo0 = (x_range[0], s_range[0])
    hi0 = (x_range[1], s_range[1])
    l0, u0, f0 = box_bounds(lo0, hi0)
    heap = [_Box(l0, lo0, hi0)]

    def _inside_band(f_iv):
        """True only when the whole box is provably inside the wall band.

        This distinction is what keeps ``best_upper`` a valid upper bound for the
        *band-restricted* problem. The objective enclosure of a box that merely
        STRADDLES the band bounds the objective over the whole box, out-of-band
        points included, so its upper end can sit below the band-restricted
        minimum. Using it as best_upper then prunes, through ``l > best_upper``,
        exactly the boxes that contain that minimum, and the returned lower bound
        comes out above the truth, an invalid certificate that no test on the
        four paper metrics would notice, because there the wall band and the deep
        minimum coincide.
        """
        return _lo(f_iv) >= wall_mask[0] and _hi(f_iv) <= wall_mask[1]

    best_upper = u0 if _inside_band(f0) else math.inf
    best_pt = tuple((a + b) / 2 for a, b in zip(lo0, hi0, strict=True))
    n_eval = 1

    # +2, not +0: each pass evaluates both halves, so testing before the split
    # let the budget overshoot by one (120000 -> 120001 evaluations reported).
    while heap and n_eval + 2 <= max_boxes:
        box = heapq.heappop(heap)
        if box.key > best_upper:
            continue  # cannot contain the global minimum
        if best_upper - box.key < tol:
            heapq.heappush(heap, box)
            break
        # split the longer side
        widths = [box.hi[i] - box.lo[i] for i in range(2)]
        ax = 0 if widths[0] >= widths[1] else 1
        mid = (box.lo[ax] + box.hi[ax]) / 2
        for half in range(2):
            lo = list(box.lo)
            hi = list(box.hi)
            if half == 0:
                hi[ax] = mid
            else:
                lo[ax] = mid
            l, u, f_iv = box_bounds(tuple(lo), tuple(hi))
            n_eval += 1
            # wall restriction: discard only if provably outside the band
            if _hi(f_iv) < wall_mask[0] or _lo(f_iv) > wall_mask[1]:
                continue
            if math.isfinite(u) and u < best_upper and _inside_band(f_iv):
                best_upper = u
                best_pt = tuple((a + b) / 2 for a, b in zip(lo, hi, strict=True))
            if l <= best_upper:
                heapq.heappush(heap, _Box(l, tuple(lo), tuple(hi)))

    # Boxes are pushed under the best_upper of the moment, so the heap can retain
    # boxes that a later, smaller best_upper would have rejected. The minimum over
    # the heap keys is still a valid lower bound, every discarded box was
    # discarded because its own rigorous lower bound exceeded an achieved value,
    # but the bracket must be reported as an ordered pair, so clamp rather than
    # emit lower > upper on a budget-exhausted run.
    certified_lower = min([b.key for b in heap], default=best_upper)
    certified_lower = min(certified_lower, best_upper)
    return Enclosure(
        lower=float(certified_lower),
        upper=float(best_upper),
        argmin=best_pt,
        n_boxes=len(heap),
        n_evaluations=n_eval,
    )


def tail_bound(metric_fn, shape_fn, x_outer, s_outer, prec=60, wall_mask=(0.1, 0.9)):
    """Certify that the exterior holds no wall point at all, in one evaluation.

    The search box is finite, so something has to be said about the region outside
    it. What is said here is *not* a deficit comparison, and an earlier version of
    this docstring and of the appendix claimed it was: "the tanh tail puts the
    deficit orders of magnitude above the interior minimum". The objective built by
    :func:`_make_objective` carries the same wall mask as the search, so on the
    exterior it takes the masked branch and never computes a deficit; the returned
    number was ``+inf``, and the comparison ``tail > upper`` that consumed it was
    vacuously true.

    The honest statement is stronger and needs no comparison. The wall band is a
    condition on the shape function alone, and ``f`` depends on the coordinates only
    through ``r = |x - x_s(t)|``, decaying like ``exp(-2 sigma (r - R))``. One
    interval evaluation of ``f`` on the exterior therefore certifies that its
    enclosure lies entirely outside ``[wall_lo, wall_hi]``, so the exterior contains
    no wall point in any direction, including ``x < -x_outer[0]``, which the old
    annulus never touched, since ``r`` does not distinguish the two sides.

    The exterior is UNBOUNDED, and a box evaluation is not. ``x_outer``/``s_outer``
    only fix the inner radius ``r_min``; the bound is then taken on all of
    ``r >= r_min`` using monotonicity, which the top hat has exactly: with
    ``f = [tanh(s(r+R)) - tanh(s(r-R))] / (2 tanh(sR))``,
    ``df/dr = s[sech^2(s(r+R)) - sech^2(s(r-R))] / (2 tanh(sR)) < 0`` for every
    ``r > 0``, because ``|r+R| > |r-R|`` there. So ``f`` on ``[r_min, inf)`` is
    enclosed by ``[0, f(r_min)]``. Evaluating the box alone certified only the
    annulus ``r_min <= r <= |corner|``, while the appendix claimed ``r >= 3``.

    Returns ``(f_lo, f_hi, excluded)``: the certified enclosure of the shape function
    on the exterior, and whether it is provably disjoint from the wall band.
    """
    # Both must be set: mpmath.iv keeps its OWN precision, so setting only
    # mp.prec left every interval operation at the 53-bit default and made
    # this argument inert.
    mpmath.mp.prec = prec
    iv.prec = prec
    # min |x| over the interval, which is 0 when it straddles the origin.
    # min(|endpoint|) is not that, and returned a certified exclusion for a
    # slab containing the wall.
    if x_outer[0] <= 0.0 <= x_outer[1]:
        r_min = 0.0
    else:
        r_min = min(abs(x_outer[0]), abs(x_outer[1]))
    if s_outer[0] > 0:
        r_min = math.hypot(r_min, s_outer[0])
    f_at_rmin = shape_fn(iv.mpf([r_min, r_min]), iv.mpf([0.0, 0.0]))
    f_lo, f_hi = 0.0, _hi(f_at_rmin)
    return f_lo, f_hi, bool(f_hi < wall_mask[0] or f_lo > wall_mask[1])
