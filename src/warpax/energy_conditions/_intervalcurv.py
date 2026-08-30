r"""Rigorous interval enclosure of the Eulerian stress-energy decomposition.

Given a metric written in the small interval-AD algebra of
:mod:`warpax.energy_conditions._intervalad`, this reproduces exactly the chain
``g -> Gamma -> Riemann -> Ricci -> G_ab -> T_ab`` that ``warpax.geometry``
evaluates with JAX, but in interval arithmetic, so every output is a rigorous
enclosure over the box rather than a floating-point value at a point.

The output is the Eulerian decomposition ``(rho_n, b_i, S_ij)`` used by the
certified null-deficit objective, in an ORTHONORMAL spatial frame.

The distinction matters. ``T(n, d_i)`` and ``T(d_i, d_j)`` are components
in the coordinate basis, and the null deficit ``min_{|v|=1} q(v)`` means the
minimum over vectors of unit length *in the spatial metric*. The two coincide only
when ``gamma_ij = delta_ij``. For a conformally flat slice ``gamma_ij = B^2
delta_ij``, the Van den Broeck branch, they do not, and the difference reverses
the sign: at ``v_s=0.5, R=1, sigma=8, R_tilde=1, alpha=1`` and the point
``(0,1,0,0)`` the coordinate objective gives ``-0.0350`` and the orthonormal one
``+0.0706``, so the coordinate version certifies a null-energy violation at a point
where the condition holds.

The fix is basis-independent: Cholesky-factor ``gamma = L L^T`` and push the
decomposition through ``L^{-1}``. Writing ``v = L^{-T} u`` makes
``gamma(v,v) = |u|^2``, so
``b_hat = L^{-1} b`` and ``S_hat = L^{-1} S L^{-T}`` are the components against
which the unit-sphere minimisation is the right one. For a flat slice ``L`` is the
identity and nothing changes.
"""

from __future__ import annotations

import mpmath
from mpmath import iv

from . import _intervalad as ad
from . import _jet

_N = 4
_ZERO = iv.mpf([0, 0])

# ``_jet.sqrt`` is ``iv.sqrt`` on a bare interval and the chain rule on a jet, so the
# chain below runs unchanged over either scalar ring.
_sqrt, _add, _mul, _sub, _div = (
    _jet.sqrt, _jet.add, _jet.mul, _jet.sub, _jet.div)


def _fold(terms):
    """Sum a generator, ring-agnostically (``sum`` would put ``_ZERO`` on the left)."""
    out = _ZERO
    for t in terms:
        out = _add(out, t)
    return out


def _mid(c) -> float:
    """Midpoint of an interval as a plain float (used only for pivot choice)."""
    return 0.5 * (float(mpmath.mpf(c.a)) + float(mpmath.mpf(c.b)))


def _inv4(m):
    """Interval inverse of a 4x4 matrix by Gauss-Jordan with partial pivoting.

    Pivoting is on the interval midpoint; correctness does not depend on the
    choice, only tightness does. A pivot interval straddling zero makes the
    enclosure unbounded, which the caller treats as "cannot bound this box".
    """
    n = _N
    a = [[m[i][j] for j in range(n)] + [iv.mpf([1, 1]) if i == k else iv.mpf([0, 0]) for k in range(n)]
         for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(_mid(a[r][col])))
        pv = a[piv][col]
        if float(mpmath.mpf(pv.a)) <= 0 <= float(mpmath.mpf(pv.b)):
            raise ZeroDivisionError("interval pivot contains zero")
        a[col], a[piv] = a[piv], a[col]
        d = a[col][col]
        a[col] = [_div(e, d) for e in a[col]]
        for r in range(n):
            if r == col:
                continue
            f = a[r][col]
            # No zero-factor shortcut. Testing only the VALUE endpoints skips the
            # elimination for a jet whose value is zero but whose derivative is not,
            # which drops that derivative on the floor; and a genuinely nonzero
            # magnitude below the binary64 range reads as zero, which would narrow
            # the enclosure below the truth. A 4x4 inverse is far too small for the
            # saving to be worth either hazard.
            a[r] = [_sub(a[r][k], _mul(f, a[col][k])) for k in range(2 * n)]
    return [[a[i][n + j] for j in range(n)] for i in range(n)]


def _chol3(gam):
    """Interval Cholesky ``gamma = L L^T`` for a 3x3 spatial metric.

    Raises ``ValueError`` when a pivot interval reaches zero, i.e. when the box is
    too wide for the factorisation to be certified. The caller treats that as
    "cannot bound this box" and subdivides, which is the correct response.
    """
    L = [[_ZERO] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i + 1):
            s = gam[i][j]
            for k in range(j):
                s = _sub(s, _mul(L[i][k], L[j][k]))
            if i == j:
                if float(mpmath.mpf(s.a)) <= 0.0:
                    raise ValueError("spatial metric not certified positive definite")
                L[i][j] = _sqrt(s)
            else:
                L[i][j] = _div(s, L[j][j])
    return L


def _fwd_solve3(L, rhs):
    """Solve ``L y = rhs`` for lower-triangular ``L`` (interval entries)."""
    y = [_ZERO] * 3
    for i in range(3):
        s = rhs[i]
        for k in range(i):
            s = _sub(s, _mul(L[i][k], y[k]))
        y[i] = _div(s, L[i][i])
    return y


def eulerian_fields_interval(metric_dual_fn, box, jet=False):
    """Return interval enclosures of ``(rho_n, b_i, S_ij)`` over ``box``.

    Parameters
    ----------
    metric_dual_fn
        Callable taking the four seeded :class:`~._intervalad.Dual2` coordinates
        and returning the 4x4 metric as a nested list of ``Dual2``.
    box
        Four intervals enclosing ``(t, x, y, z)`` over the region of interest.
    jet
        When true, run the whole chain over :class:`~._jet.Jet` instead of over bare
        intervals, so each returned quantity carries its gradient with respect to the
        box coordinates as well as its value. That gradient is what the centered
        bound of :mod:`.enclosure` needs; the values are identical either way.
    """
    seed = ad.variable_jet if jet else ad.variable
    coords = [seed(box[i], i) for i in range(_N)]
    gd = metric_dual_fn(*coords)

    g = [[gd[i][j].v for j in range(_N)] for i in range(_N)]
    dg = [[[gd[i][j].g[k] for k in range(_N)] for j in range(_N)] for i in range(_N)]
    ddg = [
        [[[gd[i][j].h[k][l] for l in range(_N)] for k in range(_N)] for j in range(_N)]
        for i in range(_N)
    ]
    gi = _inv4(g)

    # The bracket dg_bdc + dg_cdb - dg_bcd is free of the contracted index a and
    # of e, so it is built once instead of 16 times per entry.
    S1 = [[[_ZERO] * _N for _ in range(_N)] for _ in range(_N)]
    for b in range(_N):
        for c in range(_N):
            for d in range(_N):
                S1[b][c][d] = _sub(_add(dg[b][d][c], dg[c][d][b]), dg[b][c][d])

    # Christoffel symbols and their first derivatives.
    Gam = [[[_ZERO] * _N for _ in range(_N)] for _ in range(_N)]
    for a in range(_N):
        for b in range(_N):
            for c in range(_N):
                s = _ZERO
                for d in range(_N):
                    s = _add(s, _mul(gi[a][d], S1[b][c][d]))
                Gam[a][b][c] = _div(s, 2)

    # d(g^{ad})/dx_e = -g^{af} (dg_{fh}/dx_e) g^{hd}
    dgi = [[[_ZERO] * _N for _ in range(_N)] for _ in range(_N)]
    for a in range(_N):
        for d in range(_N):
            for e in range(_N):
                s = _ZERO
                for f in range(_N):
                    for h in range(_N):
                        s = _add(s, _mul(_mul(gi[a][f], dg[f][h][e]), gi[h][d]))
                dgi[a][d][e] = -s

    # Same bracket one derivative up; free of a.
    S2 = [[[[_ZERO] * _N for _ in range(_N)] for _ in range(_N)] for _ in range(_N)]
    for b in range(_N):
        for c in range(_N):
            for d in range(_N):
                for e in range(_N):
                    S2[b][c][d][e] = _sub(
                        _add(ddg[b][d][c][e], ddg[c][d][b][e]), ddg[b][c][d][e]
                    )

    dGam = [[[[_ZERO] * _N for _ in range(_N)] for _ in range(_N)] for _ in range(_N)]
    for a in range(_N):
        for b in range(_N):
            for c in range(_N):
                for e in range(_N):
                    s = _ZERO
                    for d in range(_N):
                        s = _add(s, _mul(dgi[a][d][e], S1[b][c][d]))
                        s = _add(s, _mul(gi[a][d], S2[b][c][d][e]))
                    dGam[a][b][c][e] = _div(s, 2)

    # Ricci tensor R_bc = d_a Gam^a_bc - d_c Gam^a_ba + Gam^a_af Gam^f_bc
    #                     - Gam^a_cf Gam^f_ba
    Ric = [[_ZERO] * _N for _ in range(_N)]
    for b in range(_N):
        for c in range(_N):
            s = _ZERO
            for a in range(_N):
                s = _sub(_add(s, dGam[a][b][c][a]), dGam[a][b][a][c])
                for f in range(_N):
                    s = _sub(_add(s, _mul(Gam[a][a][f], Gam[f][b][c])),
                             _mul(Gam[a][c][f], Gam[f][b][a]))
            Ric[b][c] = s

    Rs = _ZERO
    for a in range(_N):
        for b in range(_N):
            Rs = _add(Rs, _mul(gi[a][b], Ric[a][b]))

    eight_pi = 8 * iv.pi
    T = [
        [_div(_sub(Ric[a][b], _div(_mul(Rs, g[a][b]), 2)), eight_pi)
         for b in range(_N)]
        for a in range(_N)
    ]

    # Eulerian normal, derived from the metric rather than from a shift-sign
    # convention: n_a = (-alpha, 0, 0, 0), so n^a = g^{ab} n_b, normalized by
    # |n_a n^a|^{1/2}. This mirrors the reference implementation in
    # ``warpax.energy_conditions.frame_free`` exactly and is convention-free.
    n_low = [iv.mpf([-1, -1]), _ZERO, _ZERO, _ZERO]
    n_raw = [_fold(_mul(gi[a][b], n_low[b]) for b in range(_N)) for a in range(_N)]
    norm2 = _fold(_mul(n_low[a], n_raw[a]) for a in range(_N))
    inv_norm = 1 / _sqrt(abs(norm2))
    n_up = [_mul(c, inv_norm) for c in n_raw]

    rho_n = _ZERO
    for a in range(_N):
        for b in range(_N):
            rho_n = _add(rho_n, _mul(_mul(T[a][b], n_up[a]), n_up[b]))

    b_cov = []
    for i in range(3):
        s = _ZERO
        for a in range(_N):
            s = _add(s, _mul(n_up[a], T[a][i + 1]))
        b_cov.append(s)

    S_cov = [[T[i + 1][j + 1] for j in range(3)] for i in range(3)]

    # Push both into an orthonormal spatial frame: gamma = L L^T, then
    # b_hat = L^{-1} b and S_hat = L^{-1} S L^{-T}. Identity work for a flat slice.
    gam = [[g[i + 1][j + 1] for j in range(3)] for i in range(3)]
    L = _chol3(gam)
    b_vec = _fwd_solve3(L, b_cov)
    # M = L^{-1} S  (column by column), then S_hat = L^{-1} M^T, using S symmetric.
    M = [_fwd_solve3(L, [S_cov[i][j] for i in range(3)]) for j in range(3)]   # M[j] = column j
    S = [_fwd_solve3(L, [M[j][i] for j in range(3)]) for i in range(3)]
    return rho_n, b_vec, S
