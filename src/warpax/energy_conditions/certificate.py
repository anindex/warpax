"""Exact rational certificates for the LMI verdicts, with no floating point.

:mod:`.slemma` decides every energy condition over every observer from a 4x4 linear
matrix inequality, which is exact as mathematics but is searched in binary64. This
module turns each verdict into an object a reader can check by hand in exact
arithmetic: a rational multiplier for a satisfied condition, a rational observer (or
observer pair) for a violated one. Nothing here has a tolerance.

**No tetrad is needed.** The S-lemma is usually written in an orthonormal frame,
where ``M(sigma) = That + sigma eta`` must be positive semidefinite. But
``That = e T e^T`` and ``eta = e g e^T`` for the tetrad ``e``, so
``M(sigma) = e (T + sigma g) e^T`` is *congruent* to the coordinate-basis matrix
``T_ab + sigma g_ab``, and congruence preserves inertia. Positive semidefiniteness of
a symmetric bilinear form is therefore a basis-free statement, and the certificate can
be written directly in the coordinate components the pipeline already has -- which is
what makes it exactly checkable, since an orthonormal tetrad is not a rational object.

Certificate of **satisfaction** -- a rational ``sigma``:

    NEC   exists sigma in Q          with T + sigma g  PSD
    WEC   exists sigma in Q, >= 0    with T + sigma g  PSD
    SEC   exists sigma in Q, >= 0    with Theta + sigma g  PSD,  Theta = T - (1/2) tr_g(T) g
    DEC   the WEC certificate, together with one for  -T g^{-1} T

Certificate of **violation** -- a rational observer, from the dual side. The dual of
``max_sigma lambda_min`` is a positive semidefinite ``X`` with ``<T, X> < 0``, subject
to the sign constraint on ``<g, X>`` that the multiplier's range imposes:

    WEC/SEC/DEC   a single causal ``u``: ``g(u,u) <= 0`` and ``T(u,u) < 0``. Then
                  ``X = u u^T`` has ``<g, X> <= 0``, so ``<T + sigma g, X> < 0`` for
                  every ``sigma >= 0`` and no admissible multiplier can work.

    NEC           the multiplier is free in sign, so ``<g, X>`` must vanish *exactly*
                  and a single vector will not generally do it over Q -- a rational
                  Lorentzian form need not represent zero. A pair does: pick rational
                  ``k, l`` with ``g(k,k)`` and ``g(l,l)`` of opposite sign and set
                  ``alpha = |g(l,l)|``, ``beta = |g(k,k)|``. Then
                  ``X = alpha k k^T + beta l l^T`` is PSD with ``<g, X> = 0`` exactly,
                  and ``alpha T(k,k) + beta T(l,l) < 0`` rules out every real sigma at
                  once. Three lines of rational arithmetic to check.

The tensor itself is taken as exact: a binary64 float *is* a rational number, and
``Fraction(x)`` is its exact value. The certificate therefore says nothing about
discretization error in ``T`` -- that is the job of the interval branch and bound in
:mod:`.enclosure` -- and everything about the decision made from it. Those are the two
separate questions a "tolerance-dependent" verdict conflates.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Any, Sequence

import numpy as np

Mat = list[list[Fraction]]
Vec = list[Fraction]

_CONDITIONS = ("nec", "wec", "sec", "dec")

# Denominator ladder for rounding the float multiplier back to Q. A certificate with a
# small denominator is one a reader can retype; the ladder stops at the first that
# verifies, so saturated points are the only ones that reach the end.
_DENOMINATORS = (2, 4, 8, 16, 64, 256, 1024, 10**4, 10**6, 10**9, 10**12)


def to_exact(M: Any) -> Mat:
    """Exact rational copy of a matrix, preserving inputs that are already exact.

    A binary64 float *is* a rational number, so ``Fraction(x)`` is its exact value and
    that conversion loses nothing. What did lose something was going through
    ``np.asarray(M, dtype=float)`` first: an input already carried as ``Fraction`` or
    ``int`` -- or at any precision above binary64 -- was rounded to binary64 before the
    exact arithmetic began, so every determinant and PSD test downstream was exact only
    relative to that snapshot, not to the tensor handed in. Exact inputs now pass
    through untouched.
    """
    rows = M.tolist() if hasattr(M, "tolist") else M
    return [[x if isinstance(x, Fraction) else Fraction(x) for x in row] for row in rows]


def _det(M: Mat) -> Fraction:
    """Exact determinant by fraction-free Gaussian elimination (Bareiss)."""
    n = len(M)
    A = [row[:] for row in M]
    sign = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    A[k], A[i] = A[i], A[k]
                    sign = -sign
                    break
            else:
                return Fraction(0)
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = A[i][j] - A[i][k] * A[k][j] / A[k][k]
            A[i][k] = Fraction(0)
    out = Fraction(sign)
    for k in range(n):
        out *= A[k][k]
    return out


def _minor_sums(M: Mat) -> list[Fraction]:
    """``[e_1, ..., e_n]``, the sums of the ``k x k`` principal minors of ``M``."""
    n = len(M)
    return [
        sum(
            (_det([[M[i][j] for j in idx] for i in idx]) for idx in combinations(range(n), k)),
            Fraction(0),
        )
        for k in range(1, n + 1)
    ]


def is_psd_exact(M: Mat) -> bool:
    """Exactly decide ``M >= 0`` for a rational symmetric matrix, with no square roots.

    A real symmetric matrix has real eigenvalues, and they are all non-negative exactly
    when every elementary symmetric function of them is non-negative -- because
    ``p(-t) = t^n + e_1 t^(n-1) + ... + e_n`` is then strictly positive for ``t > 0``
    unless every ``e_k`` vanishes, in which case every eigenvalue is zero. Each ``e_k``
    is the sum of the ``k x k`` principal minors, so this is 15 determinants for a 4x4
    and needs no pivoting strategy, unlike an ``LDL^T`` that must cope with a singular
    positive semidefinite matrix.

    Symmetry is a precondition, not a check: on a nonsymmetric argument the principal
    minors are not the elementary symmetric functions of anything real, and the answer
    is meaningless rather than wrong. :func:`verify` is where that precondition is
    enforced, because that is the one entry point an untrusted matrix arrives through.
    """
    return all(e_k >= 0 for e_k in _minor_sums(M))


def _is_symmetric(M: Mat) -> bool:
    n = len(M)
    return len(M) > 0 and all(len(row) == n for row in M) and all(
        M[i][j] == M[j][i] for i in range(n) for j in range(i + 1, n)
    )


def _inertia(M: Mat) -> tuple[int, int, int]:
    """``(n_pos, n_neg, n_zero)`` eigenvalue counts of a rational symmetric ``M``.

    The characteristic polynomial ``det(tI - M) = t^n - e_1 t^(n-1) + ... + (-1)^n e_n``
    has only real roots, and for a real-rooted polynomial Descartes' rule is an equality
    rather than a bound, so the sign changes of the coefficient sequence count the
    positive roots exactly and the sign changes of ``p(-t)`` count the negative ones.
    No square roots, no eigensolver, no tolerance.
    """
    n = len(M)
    coeffs = [Fraction(1)]
    for k, e_k in enumerate(_minor_sums(M), start=1):
        coeffs.append((-1) ** k * e_k)

    def changes(seq: list[Fraction]) -> int:
        nz = [c for c in seq if c != 0]
        return sum(1 for a, b in zip(nz, nz[1:]) if (a > 0) != (b > 0))

    n_pos = changes(coeffs)
    n_neg = changes([c * (-1) ** i for i, c in enumerate(coeffs)])
    return n_pos, n_neg, n - n_pos - n_neg


def _add(A: Mat, B: Mat, s: Fraction) -> Mat:
    return [[A[i][j] + s * B[i][j] for j in range(len(A))] for i in range(len(A))]


def _quad(M: Mat, v: Vec) -> Fraction:
    return sum(v[i] * M[i][j] * v[j] for i in range(len(v)) for j in range(len(v)))


def _inverse(M: Mat) -> Mat:
    """Exact inverse by Gauss-Jordan over Q."""
    n = len(M)
    A = [row[:] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(M)]
    for k in range(n):
        if A[k][k] == 0:
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    A[k], A[i] = A[i], A[k]
                    break
            else:
                raise ValueError("singular metric")
        p = A[k][k]
        A[k] = [x / p for x in A[k]]
        for i in range(n):
            if i != k and A[i][k] != 0:
                f = A[i][k]
                A[i] = [a - f * b for a, b in zip(A[i], A[k])]
    return [row[n:] for row in A]


def condition_matrix(T: Mat, g: Mat, condition: str) -> Mat:
    """The symmetric form whose PSD-ness the multiplier certifies, for one condition.

    NEC and WEC test ``T`` itself; SEC tests the trace-reversed ``Theta``; the DEC
    tests ``-T g^{-1} T``, whose non-negativity on the causal cone is exactly the
    statement that the flux ``J^a = -T^a{}_b u^b`` is causal. The DEC needs the WEC
    as well, which :func:`certify` handles by requiring both.
    """
    if condition in ("nec", "wec"):
        return T
    ginv = _inverse(g)
    if condition == "sec":
        tr = sum(ginv[a][b] * T[a][b] for a in range(4) for b in range(4))
        return _add(T, g, -tr / 2)
    if condition == "dec":
        TgT = [[sum(T[a][c] * ginv[c][d] * T[d][b] for c in range(4) for d in range(4))
                for b in range(4)] for a in range(4)]
        return [[-x for x in row] for row in TgT]
    raise ValueError(f"unknown condition {condition!r}")


def _sigma_is_admissible(sigma: Fraction, condition: str) -> bool:
    """The null cone is an equality constraint, so its multiplier is free in sign."""
    return True if condition == "nec" else sigma >= 0


def _required_lmis(condition: str) -> tuple[str, ...]:
    """Which LMIs a satisfaction proof of ``condition`` must exhibit, all of them."""
    return ("wec", "dec") if condition == "dec" else (condition,)


def _admissible_bindings(condition: str) -> frozenset[str]:
    """Which single LMIs a violation of ``condition`` may legitimately bind on.

    DEC implies WEC, so breaking either half breaks the DEC. Nothing else
    substitutes for the condition claimed.
    """
    return frozenset(_required_lmis(condition))


def find_multiplier(
    T: Mat, g: Mat, condition: str, sigma_hint: float
) -> Fraction | None:
    """A rational ``sigma`` proving ``condition`` holds, or ``None`` if none is found.

    The float search supplies the hint; this rounds it down the denominator ladder and
    also tries the midpoint of the exactly-bracketed feasible interval, which is what
    succeeds when the hint sits near an endpoint. Returning ``None`` is not a proof of
    violation -- it means the point is saturated (the feasible set is a single, possibly
    irrational, multiplier) and the violation side must be certified instead.
    """
    A = condition_matrix(T, g, condition)

    def ok(s: Fraction) -> bool:
        return _sigma_is_admissible(s, condition) and is_psd_exact(_add(A, g, s))

    # At a saturated point the feasible set is the single multiplier the tensor forces,
    # and no rounding of the float search will land on it: on the Type-II locus
    # rho = S_par = 1, |j| = 1, p_t = 0.2 the SEC saturates at sigma = fl(0.2) exactly,
    # while the ternary search returns 0.19999999256728812 and limit_denominator turns
    # that into 1/5 -- a different rational, and infeasible. But fl(0.2) is right there
    # in the tensor, as tr_g(T)/2 and as T_22, so the exact entries and the half-trace
    # are tried before any rounding. Every candidate is only a guess; is_psd_exact
    # decides.
    cands: list[Fraction] = [Fraction(sigma_hint)]
    ginv = _inverse(g)
    half_trace = sum(ginv[a][b] * T[a][b] for a in range(4) for b in range(4)) / 2
    cands.extend((half_trace, -half_trace))
    for M in (A, T):
        for i in range(4):
            for j in range(4):
                cands.extend((M[i][j], -M[i][j]))
                if g[i][i] != 0:
                    cands.append(-M[i][j] / g[i][i])
    for d in _DENOMINATORS:
        cands.append(Fraction(sigma_hint).limit_denominator(d))
    if condition != "nec":
        cands = [Fraction(0) if s < 0 else s for s in cands]
    for s in cands:
        if ok(s):
            return s

    # Bracket the feasible interval exactly and take its midpoint. The set
    # {sigma : A + sigma g >= 0} is convex, hence an interval, so any feasible probe
    # can be widened by bisection until both endpoints are pinned to a fine grid.
    seed = next((s for s in cands if ok(s)), None)
    if seed is None:
        scale = max((abs(A[i][j]) for i in range(4) for j in range(4)), default=Fraction(1))
        step = scale if scale > 0 else Fraction(1)
        for n in range(1, 400):
            for s in (Fraction(n, 200) * step, -Fraction(n, 200) * step):
                if ok(s):
                    seed = s
                    break
            if seed is not None:
                break
    if seed is None:
        return None
    lo = hi = seed
    step = max(abs(seed), Fraction(1))
    while ok(lo - step):
        lo -= step
    while ok(hi + step):
        hi += step
    for _ in range(60):
        step /= 2
        while ok(lo - step):
            lo -= step
        while ok(hi + step):
            hi += step
    mid = (lo + hi) / 2
    if not _sigma_is_admissible(mid, condition):
        mid = max(mid, Fraction(0))
    return mid if ok(mid) else seed


def _repair_causal(u: Vec, g: Mat) -> Vec | None:
    """Shrink the spatial part of ``u`` until ``g(u,u) <= 0`` exactly.

    Rounding a float observer to Q can push a marginally timelike vector outside the
    cone. Scaling the spatial components toward the time axis moves ``g(u,u)`` down
    monotonically and leaves the violation intact when it is strict.
    """
    for _ in range(200):
        if _quad(g, u) <= 0:
            return u
        u = [u[0]] + [x * Fraction(15, 16) for x in u[1:]]
    return None


def find_violating_observer(T: Mat, g: Mat, condition: str, u_hint: Sequence[float]):
    """A rational witness that ``condition`` fails, or ``None``.

    For WEC/SEC/DEC one causal ``u`` suffices: ``X = u u^T`` is PSD with
    ``<g, X> <= 0``, so ``<A + sigma g, X> = A(u,u) + sigma g(u,u) <= A(u,u) < 0`` for
    every admissible ``sigma >= 0``, and no multiplier can make ``A + sigma g`` PSD.

    For the NEC the multiplier is free in sign, so the witness must annihilate ``g``
    exactly. Returns ``(k, l, alpha, beta)`` with ``alpha g(k,k) + beta g(l,l) = 0``
    and ``alpha T(k,k) + beta T(l,l) < 0``.
    """
    A = condition_matrix(T, g, condition)
    for d in _DENOMINATORS:
        u = [Fraction(float(x)).limit_denominator(d) for x in u_hint]
        if all(x == 0 for x in u):
            continue
        if condition != "nec":
            v = _repair_causal(u, g)
            if v is not None and _quad(A, v) < 0:
                return v
            continue
        gk = _quad(g, u)
        if gk == 0:
            if _quad(A, u) < 0:
                return (u, [Fraction(0)] * 4, Fraction(1), Fraction(0))
            continue
        # A partner of opposite g-sign. One of the four coordinate axes always is:
        # g is Lorentzian, so it has a timelike and a spacelike direction among them
        # unless the basis is null-aligned, in which case the sum of two axes works.
        partners: list[Vec] = []
        for i in range(4):
            e = [Fraction(0)] * 4
            e[i] = Fraction(1)
            partners.append(e)
        for i in range(4):
            for j in range(i + 1, 4):
                e = [Fraction(0)] * 4
                e[i] = e[j] = Fraction(1)
                partners.append(e)
        for l in partners:
            gl = _quad(g, l)
            if gl == 0 or (gl > 0) == (gk > 0):
                continue
            alpha, beta = abs(gl), abs(gk)
            assert alpha * gk + beta * gl == 0
            if alpha * _quad(A, u) + beta * _quad(A, l) < 0:
                return (u, l, alpha, beta)
    return None


def certify(
    T_ab: Any,
    g_ab: Any,
    condition: str,
    *,
    sigma_hint: float | None = None,
    observer_hint: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Exact certificate for one condition at one point.

    ``kind`` is ``"satisfied"``, ``"violated"`` or ``"saturated"``. The first two carry
    a proof and :func:`verify` checks it. The third carries none: it means neither
    search found one, which is the expected outcome where the feasible multiplier set
    is a single, possibly irrational, value, but is also what a search failure looks
    like. :func:`verify` returns ``False`` on it for exactly that reason -- it reports
    whether an object is a valid proof, and "saturated" is the absence of one.
    """
    if condition not in _CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    T, g = to_exact(T_ab), to_exact(g_ab)

    # The DEC needs both halves; report the binding one.
    conds = ("wec", "dec") if condition == "dec" else (condition,)
    sigmas: dict[str, Fraction] = {}
    for c in conds:
        hint = sigma_hint if sigma_hint is not None else _sigma_hint(T_ab, g_ab, c)
        s = find_multiplier(T, g, c, hint)
        if s is None:
            break
        sigmas[c] = s
    else:
        return {
            "condition": condition,
            "kind": "satisfied",
            "sigma": {k: [v.numerator, v.denominator] for k, v in sigmas.items()},
        }

    for c in conds:
        # The hint must come from the form that is actually being violated. Seeding the
        # DEC search with a witness minimizing T over the ball found nothing whenever
        # the WEC held and only the flux failed -- which is the entire DEC-specific case.
        hint = observer_hint if observer_hint is not None else _observer_hint(T, g, c)
        w = find_violating_observer(T, g, c, hint)
        if w is not None:
            if c == "nec":
                k, l, alpha, beta = w
                return {
                    "condition": condition,
                    "kind": "violated",
                    "binding": c,
                    "witness_pair": [[[x.numerator, x.denominator] for x in k],
                                     [[x.numerator, x.denominator] for x in l]],
                    "weights": [[alpha.numerator, alpha.denominator],
                                [beta.numerator, beta.denominator]],
                }
            return {
                "condition": condition,
                "kind": "violated",
                "binding": c,
                "witness": [[x.numerator, x.denominator] for x in w],
            }
    return {"condition": condition, "kind": "saturated"}


def verify(cert: dict[str, Any], T_ab: Any, g_ab: Any) -> bool:
    """Re-check a certificate from scratch, using nothing but exact arithmetic.

    This is what an auditor runs. It shares no code path with :func:`certify` beyond the
    rational primitives, and it never consults a float.

    An auditor supplies the tensor as well as the certificate, so both are untrusted and
    both are checked. Without the shape checks below a forged certificate verifies: for
    the nonsymmetric ``T = [[1,10,0,0],[-1,1,0,0],[0,0,1,0],[0,0,0,1]]`` with
    ``g = diag(-1,1,1,1)`` the principal-minor sums are ``4, 16, 24, 11``, all positive,
    so ``sigma = 0`` passes :func:`is_psd_exact` and a "satisfied" NEC certificate
    returns True -- while the null vector ``k = (1,-1,0,0)`` gives ``T(k,k) = -7``. The
    S-lemma also needs ``g`` to be the metric it claims to be: a Euclidean or degenerate
    ``g`` has no null cone for the multiplier to be free over.
    """
    T, g = to_exact(T_ab), to_exact(g_ab)
    if not _is_symmetric(T) or not _is_symmetric(g) or len(T) != len(g):
        return False
    if _inertia(g) != (len(g) - 1, 1, 0):
        return False
    cond = cert["condition"]
    if cond not in _CONDITIONS:
        return False
    if cert["kind"] == "satisfied":
        # Bind the proof to the claim: a "dec" certificate carrying only the
        # "nec" multiplier used to verify True.
        if set(cert["sigma"]) != set(_required_lmis(cond)):
            return False
        for c, (num, den) in cert["sigma"].items():
            s = Fraction(num, den)
            if not _sigma_is_admissible(s, c):
                return False
            if not is_psd_exact(_add(condition_matrix(T, g, c), g, s)):
                return False
        return True
    if cert["kind"] == "violated":
        # Same, with the implication running the other way.
        if cert["binding"] not in _admissible_bindings(cond):
            return False
        A = condition_matrix(T, g, cert["binding"])
        if cert["binding"] == "nec":
            k, l = ([Fraction(n, d) for n, d in v] for v in cert["witness_pair"])
            (an, ad), (bn, bd) = cert["weights"]
            alpha, beta = Fraction(an, ad), Fraction(bn, bd)
            if alpha <= 0 or beta < 0:
                return False
            if alpha * _quad(g, k) + beta * _quad(g, l) != 0:
                return False
            return alpha * _quad(A, k) + beta * _quad(A, l) < 0
        u = [Fraction(n, d) for n, d in cert["witness"]]
        return _quad(g, u) <= 0 and _quad(A, u) < 0
    # "saturated" is the ABSENCE of a certificate, not a certified verdict, and
    # verify() answers only one question: is this object a valid proof? Returning
    # True here made a forged {"kind": "saturated"} verify against a tensor with a
    # null deficit of -2, because no arithmetic was performed at all. It also let a
    # search failure -- certify() falls through to "saturated" whenever BOTH searches
    # come up empty, which is not the same as the feasible multiplier set being a
    # single point -- present itself as a checked result.
    return False


# --------------------------------------------------------------------------
# Float hints. These only seed the exact search; nothing they return is trusted.
# --------------------------------------------------------------------------


def _to_float(M: Mat) -> np.ndarray:
    return np.array([[float(x) for x in row] for row in M], dtype=float)


def _sigma_hint(T_ab: Any, g_ab: Any, condition: str) -> float:
    """Where the float search puts the optimal multiplier, for the given condition."""
    import jax.numpy as jnp

    from .slemma import _lmi_margin, tetrad_components

    T, g = to_exact(T_ab), to_exact(g_ab)
    A = jnp.asarray(_to_float(condition_matrix(T, g, condition)))
    gj = jnp.asarray(_to_float(g))
    lo = jnp.asarray(-jnp.inf) if condition == "nec" else jnp.zeros(())
    sigma, _ = _lmi_margin(tetrad_components(A, gj), lo)
    return float(sigma)


def _observer_hint(T: Mat, g: Mat, condition: str) -> list[float]:
    """A float observer minimizing the *condition's* form, lifted to coordinates.

    ``witness_observer`` searches the closed unit ball of boost vectors; for the NEC we
    want the sphere instead, and a ball minimizer sitting in the interior is useless
    there, so the null case falls back to the deepest direction of the spatial block.
    """
    import jax.numpy as jnp

    from .observer import compute_orthonormal_tetrad
    from .slemma import tetrad_components, witness_observer

    A = jnp.asarray(_to_float(condition_matrix(T, g, condition)))
    gj = jnp.asarray(_to_float(g))
    w = np.asarray(witness_observer(A, gj), dtype=float)
    if not np.all(np.isfinite(w)) or (condition == "nec" and np.linalg.norm(w) < 1e-12):
        A_hat = np.asarray(tetrad_components(A, gj))
        b, S = -A_hat[0, 1:], A_hat[1:, 1:]
        # argmin over the sphere of rho - 2 b.s + s^T S s, to leading order.
        evals, evecs = np.linalg.eigh(S)
        cand = [evecs[:, 0], -evecs[:, 0]]
        if np.linalg.norm(b) > 0:
            cand.append(b / np.linalg.norm(b))
        w = min(cand, key=lambda s: float(A_hat[0, 0] - 2 * b @ s + s @ (S @ s)))
    if condition == "nec":
        n = np.linalg.norm(w)
        w = w / n if n > 0 else np.array([1.0, 0.0, 0.0])
    e = np.asarray(compute_orthonormal_tetrad(gj))  # e[I, a] = e_I^a
    return list(np.concatenate([[1.0], w]) @ e)
