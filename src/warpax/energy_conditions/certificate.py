"""Sufficient rational certificates for pointwise energy conditions.

Candidate searches use floating-point hints and may return no certificate.
Accepted certificates are checked with exact rational arithmetic in coordinate
components. They establish a condition or its violation for the supplied tensor
and metric; they do not bound errors in those inputs or spatial variation.
``Fraction(x)`` preserves the exact value of a supplied binary64 number.

For a rational Lorentzian metric ``g`` and symmetric stress tensor ``T``, a
satisfaction certificate supplies multipliers making these matrices positive
semidefinite (PSD):

    NEC: T + sigma g, with any rational sigma
    WEC: T + sigma g, with rational sigma >= 0
    SEC: Theta + sigma g, with rational sigma >= 0
    DEC: both T + sigma_wec g and -T g^{-1} T + sigma_dec g,
         with rational sigma_wec, sigma_dec >= 0

Here ``Theta = T - (1/2) tr_g(T) g``. Congruence preserves PSD, so checking these
coordinate matrices requires no orthonormal tetrad.

A WEC, SEC, or DEC violation certificate gives a rational causal vector ``u``
with ``A(u,u) < 0`` for the relevant form ``A``. The NEC certificate gives
rational vectors ``k, l`` and weights ``alpha > 0, beta >= 0`` such that

    alpha g(k,k) + beta g(l,l) = 0,
    alpha T(k,k) + beta T(l,l) < 0.

The associated PSD matrix ``X = alpha k k^T + beta l l^T`` excludes every
admissible NEC multiplier. A directly rational null witness is the case
``beta = 0``. These searches are sufficient and need not find every certificate.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations, pairwise
from typing import Any

import numpy as np

Mat = list[list[Fraction]]
Vec = list[Fraction]

_CONDITIONS = ("nec", "wec", "sec", "dec")

# Try small rational denominators first; accept candidates only after exact checks.
_DENOMINATORS = (2, 4, 8, 16, 64, 256, 1024, 10**4, 10**6, 10**9, 10**12)


def to_exact(M: Any) -> Mat:
    """Copy matrix entries to exact rational values without an intermediate float.

    Existing ``Fraction`` values are retained. Other entries are passed to
    ``Fraction`` directly, preserving supported integers and binary floats.
    """
    rows = M.tolist() if hasattr(M, "tolist") else M
    return [[x if isinstance(x, Fraction) else Fraction(x) for x in row] for row in rows]


def _det(M: Mat) -> Fraction:
    """Exact determinant by Gaussian elimination with rational arithmetic."""
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
    """Test PSD for a rational symmetric matrix using principal-minor sums.

    For symmetric ``M``, the coefficients of ``det(tI + M)`` are the
    sums of its principal minors. If all coefficients are nonnegative,
    this polynomial is positive for ``t > 0``, excluding negative
    eigenvalues of ``M``. The converse follows from nonnegative eigenvalues.

    This uses 15 principal minors for a 4x4 matrix. Symmetry is a precondition
    here; :func:`verify` checks it before using this test.
    """
    return all(e_k >= 0 for e_k in _minor_sums(M))


def _is_symmetric(M: Mat) -> bool:
    n = len(M)
    return (
        len(M) > 0
        and all(len(row) == n for row in M)
        and all(M[i][j] == M[j][i] for i in range(n) for j in range(i + 1, n))
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
        return sum(1 for a, b in pairwise(nz) if (a > 0) != (b > 0))

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
                A[i] = [a - f * b for a, b in zip(A[i], A[k], strict=True)]
    return [row[n:] for row in A]


def condition_matrix(T: Mat, g: Mat, condition: str) -> Mat:
    """Return the symmetric form tested by the selected LMI.

    NEC and WEC use ``T``; SEC uses its trace reversal. DEC uses
    ``-T g^{-1} T``, whose nonnegativity on the causal cone makes the energy
    flux causal or zero. Full DEC satisfaction also requires WEC, which
    :func:`certify` checks separately.
    """
    if condition in ("nec", "wec"):
        return T
    ginv = _inverse(g)
    if condition == "sec":
        tr = sum(ginv[a][b] * T[a][b] for a in range(4) for b in range(4))
        return _add(T, g, -tr / 2)
    if condition == "dec":
        TgT = [
            [
                sum(T[a][c] * ginv[c][d] * T[d][b] for c in range(4) for d in range(4))
                for b in range(4)
            ]
            for a in range(4)
        ]
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


def find_multiplier(T: Mat, g: Mat, condition: str, sigma_hint: float) -> Fraction | None:
    """Find a sufficient rational multiplier for the selected LMI, or ``None``.

    Candidates come from the floating-point hint, tensor entries, and a finite
    rational search. Feasible candidates are checked exactly. A failed search
    is inconclusive; it proves neither violation nor saturation. For
    ``condition="dec"``, this tests flux causality only; full DEC also needs
    a WEC certificate.
    """
    A = condition_matrix(T, g, condition)

    def ok(s: Fraction) -> bool:
        return _sigma_is_admissible(s, condition) and is_psd_exact(_add(A, g, s))

    # Tensor entries and the half-trace can supply exact boundary multipliers
    # that rounded hints miss. Every candidate still requires the PSD check.
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
    """Try spatial rescalings of ``u`` until ``g(u,u) <= 0`` holds exactly.

    This finite heuristic need not find a causal vector in general
    coordinates. Its caller separately checks the negative contraction.
    """
    for _ in range(200):
        if _quad(g, u) <= 0:
            return u
        u = [u[0]] + [x * Fraction(15, 16) for x in u[1:]]
    return None


def find_violating_observer(T: Mat, g: Mat, condition: str, u_hint: Sequence[float]):
    """Find a sufficient rational violation witness from a hint, or ``None``.

    WEC and SEC use a causal vector ``u`` with ``A(u,u) < 0`` for the
    relevant form. The DEC search uses ``A = -T g^{-1} T`` to test flux
    causality; :func:`certify` also searches for a WEC violation.

    For NEC, returns ``(k, l, alpha, beta)`` with nonnegative weights,
    ``alpha g(k,k) + beta g(l,l) = 0``, and
    ``alpha T(k,k) + beta T(l,l) < 0``. The finite candidate search may miss
    a witness, so ``None`` is inconclusive.
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
        # Try coordinate axes and pairwise sums for an opposite-sign partner.
        # This finite set need not contain one for a general Lorentzian metric.
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
    """Search for an exact certificate for one condition at one point.

    ``kind="satisfied"`` or ``"violated"`` carries a certificate that
    :func:`verify` can check. The legacy value ``kind="saturated"`` means
    neither search found a certificate. It does not establish saturation
    or any energy-condition verdict, and :func:`verify` rejects it.
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
        # A DEC witness may violate either the WEC form or the flux form.
        hint = observer_hint if observer_hint is not None else _observer_hint(T, g, c)
        w = find_violating_observer(T, g, c, hint)
        if w is not None:
            if c == "nec":
                k, l, alpha, beta = w
                return {
                    "condition": condition,
                    "kind": "violated",
                    "binding": c,
                    "witness_pair": [
                        [[x.numerator, x.denominator] for x in k],
                        [[x.numerator, x.denominator] for x in l],
                    ],
                    "weights": [
                        [alpha.numerator, alpha.denominator],
                        [beta.numerator, beta.denominator],
                    ],
                }
            return {
                "condition": condition,
                "kind": "violated",
                "binding": c,
                "witness": [[x.numerator, x.denominator] for x in w],
            }
    return {"condition": condition, "kind": "saturated"}


def verify(cert: dict[str, Any], T_ab: Any, g_ab: Any) -> bool:
    """Check a certificate using exact arithmetic on the supplied inputs.

    Checks tensor symmetry, Lorentzian metric inertia, condition bindings,
    and the required PSD or witness inequalities. Candidate-search hints
    are not used. ``"saturated"`` is not a certificate and returns ``False``.
    The certificate dictionary must follow the schema returned by
    :func:`certify`; malformed entries may raise an exception.
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
        # Require exactly the multipliers needed for the claimed condition.
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
    # Search failure provides no certificate, including under the legacy name.
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
    """Construct a numerical witness candidate in coordinate components.

    Search the relevant quadratic form on the closed unit velocity ball.
    If the search fails, try the least-eigenvalue spatial directions and
    the momentum direction. NEC candidates are projected onto the unit
    sphere. No global minimum or violation is guaranteed by this hint.
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
        # Try spatial eigenvectors and the momentum direction on the sphere.
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
