"""Exact all-observer energy conditions as one 4x4 linear matrix inequality.

This decides the NEC, WEC, SEC and DEC over *every* observer at a point, with no
rapidity cap, no optimizer, no eigen-decomposition of ``T^a_b`` and no
classification tolerance. It is therefore independent of the Hawking-Ellis type:
it decides Type II and Type III points, where the eigenvalue route in
:mod:`.eigenvalue_checks` has no rest frame to work with and
:func:`.frame_free._exact_margins` previously returned NaN.

The construction. Let ``{n, e_i}`` be an orthonormal tetrad with ``n`` the unit
slice normal. Every future timelike observer is ``u = gamma (n + w)`` with
``|w| < 1``, and every future null direction is ``k = c (n + s)`` with ``|s| = 1``
and ``c > 0``. Since

    T_ab u^a u^b = gamma^2 q(w),   q(w) = rho - 2 b.w + w^T S w,   gamma^2 > 0,

the *sign* is decided by ``q`` on a compact set -- the closed unit ball for the
timelike conditions, the unit sphere for the null one. No cap is needed, and none
is used. Here ``rho = T(n,n)``, ``b_i = -T(n,e_i)`` is the momentum density and
``S_ij = T(e_i,e_j)``.

By the S-lemma (Yakubovich; exact for a single quadratic constraint, and the
Slater point is ``w = 0``, where ``1 - |w|^2 = 1 > 0`` -- note this needs nothing
of ``rho``, so it holds precisely at the exotic points of interest),

    q(w) >= 0 on the ball  <=>  exists sigma >= 0 with  M(sigma) >= 0 (PSD),

where ``M(sigma)`` is nothing but the tetrad-frame component matrix of the tensor
``T_ab + sigma g_ab``:

    M(sigma) = That + sigma * eta = [[rho - sigma, -b^T], [-b, S + sigma I]].

For the sphere (the NEC) the constraint is an equality; the S-procedure is still
exact because ``1 - |w|^2`` takes both signs, and the multiplier is free in sign.

Three of the four conditions are then the *same* primitive on a different tensor:

    NEC(T) : sphere form on T
    WEC(T) : ball form on T
    SEC(T) : ball form on Theta = T - (1/2) tr_g(T) g   (SEC is WEC for Theta)
    DEC(T) : ball form on T *and* on -T^2, where (T^2)_ab = T_ac g^cd T_db.

DEC needs the second inequality because ``J^a = -T^a{}_b u^b`` is causal exactly
when ``(T^2)(u,u) <= 0``. Future-directedness is then automatic, not a separate
test: ``T(u,u) = -J.u``, so a causal ``J`` with WEC satisfied is future-directed.

Because ``M`` is affine in ``sigma``, ``lambda_min(M(sigma))`` is *concave*, so
the certificate search is a one-dimensional concave maximization -- a ternary
search over 4x4 symmetric eigenvalues. No SDP solver and no new dependency.

Scope. This is a *pointwise* statement, ``for all x, exists sigma(x)``. It does
not lift to a spatial box: one interval LMI would prove the strictly stronger
``exists sigma, for all x``, which already fails for the saturated family
``rho = a(x) > 0, b = 0, S = -a(x) I`` where ``sigma = a(x)`` is forced, and
subdivision does not help. Global coverage over a domain stays with the
Moore-Skelboe branch and bound in :mod:`.enclosure`, which brackets
``min_{|w|<=1} q`` directly. Two tools, two jobs; do not conflate them.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .observer import compute_orthonormal_tetrad

# eta in the orthonormal tetrad frame; M(sigma) = That + sigma * ETA.
_ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

# Ternary search steps. Each step shrinks the bracket by 2/3, so n steps give
# (2/3)^n; 80 takes a bracket of width ~1 to ~1e-14, and the margin error is
# second order in the bracket at a smooth maximum. Measured against 120 steps over
# 3000 random tensors: NEC/WEC/SEC agree to 5.3e-15 and DEC to 3.2e-13, both far
# under noise_floor, at two thirds of the cost. Each step is two 4x4 eigensolves
# and this search is the dominant cost of any grid that carries non-Type-I points.
_TERNARY_STEPS = 80

# Relative floor below which a negative margin is noise rather than a violation.
# Set by the residual bracket and the eigvalsh error on M(sigma); see noise_floor.
_NOISE_REL = 1e-12

# Absolute floor. Zero: with the bracket unclamped it collapses to the single
# point 0 at zero tensor scale, where lam_min(0) = 0 exactly, so no absolute
# term is needed. A nonzero one dominates below scale ~1e-6 and reported a
# violation of 100% of its own tensor scale as inconclusive.
_NOISE_ABS = 0.0

# Projected-gradient steps for the violating-observer search in witness_observer.
_DESCENT_STEPS = 400


def tetrad_components(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
) -> Float[Array, "4 4"]:
    """Component matrix ``That_IJ = T_ab e_I^a e_J^b`` in an orthonormal tetrad.

    ``That[0,0] = rho``, ``That[0,i] = -b_i``, ``That[i,j] = S_ij``.
    """
    e = compute_orthonormal_tetrad(g_ab)  # e[I, a] = e_I^a
    return e @ T_ab @ e.T


def _lmi_margin(
    T_hat: Float[Array, "4 4"],
    sigma_lo: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Maximize the concave ``sigma -> lambda_min(That + sigma eta)``.

    ``sigma_lo`` is the lower end of the admissible multiplier range: ``0`` for
    the ball (timelike) conditions, ``-inf`` in effect for the sphere (null) one,
    which we realize as a symmetric bracket.

    Returns ``(sigma_star, margin)``.

    Two distinct effects blunt this number, and the contract has to survive both.
    Ternary search *under*-estimates a concave maximum, which biases the result
    downward; and ``lambda_min`` comes from an eigensolver with backward error of
    order ``eps * ||M||``, which is unbiased and can push it either way. So the
    verdict is two-sided against :func:`noise_floor`, not one-sided against zero:
    ``> +floor`` says satisfied, ``< -floor`` says violated, in between says nothing.

    An earlier version of this docstring claimed ``margin >= 0`` certifies
    satisfaction outright. It does not -- the eigensolver error alone can lift a
    marginally violating tensor above zero -- and the exact-arithmetic escape in
    :mod:`.certificate` exists precisely for the band where no float64 search can
    decide. On the exact vacuum ``T = 0`` the true maximum is ``0``; with the bracket
    scaling to the tensor the search now returns exactly ``0`` there, but on a
    saturated tensor it will still land inside the floor, and that is the honest
    answer rather than a defect.

    The maximum is always attained at finite ``sigma``, so there is no
    optimizer-at-infinity case to guard against: ``lambda_min(That + sigma eta)``
    is bounded above by ``That_00 - sigma`` and by ``That_ii + sigma``, hence tends
    to ``-inf`` in both directions.
    """
    # The bracket must contain the argmax whether or not the LMI is feasible.
    # The feasible-set argument (M(sigma) PSD forces every diagonal entry
    # non-negative, so -max_i|S_ii| <= sigma <= rho) says nothing on a violated
    # point, where no sigma is PSD.
    #
    # Unconditionally: lam_min(M(sigma)) <= That_00 - sigma <= scale - sigma and
    # <= That_ii + sigma <= scale + sigma, while the maximum is at least
    # lam_min(That) >= -4 scale by Gershgorin. Hence |sigma*| <= 5 scale.
    #
    # The scale is NOT clamped at 1, so the bracket collapses with the tensor:
    # to the single point 0 on vacuum, where lam_min(0) = 0 exactly. Clamping
    # made the residual absolute and convicted Minkowski of violating all four
    # conditions.
    scale = _tensor_scale(T_hat)
    lo = jnp.maximum(sigma_lo, -5.0 * scale)
    hi = 5.0 * scale

    def lam_min(s):
        return jnp.linalg.eigvalsh(T_hat + s * _ETA)[0]

    def step(_, bounds):
        a, b = bounds
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        take_upper = lam_min(m1) < lam_min(m2)
        return (jnp.where(take_upper, m1, a), jnp.where(take_upper, b, m2))

    lo, hi = jax.lax.fori_loop(0, _TERNARY_STEPS, step, (lo, hi))
    sigma = 0.5 * (lo + hi)
    return sigma, lam_min(sigma)


def _trace_reversed(T_hat: Float[Array, "4 4"]) -> Float[Array, "4 4"]:
    """``Theta = T - (1/2) tr_g(T) g`` in tetrad components."""
    trace = -T_hat[0, 0] + T_hat[1, 1] + T_hat[2, 2] + T_hat[3, 3]
    return T_hat - 0.5 * trace * _ETA


def _minus_T_squared(T_hat: Float[Array, "4 4"]) -> Float[Array, "4 4"]:
    """``-(T^2)_ab = -T_ac g^cd T_db`` in tetrad components."""
    return -(T_hat @ _ETA @ T_hat)


def _tensor_scale(T_hat: Float[Array, "4 4"]) -> Float[Array, ""]:
    """``max |That_IJ|`` -- the scale the noise floor and the flux margin share."""
    return jnp.max(jnp.abs(T_hat))


def _flux_margin_linear(
    flux: Float[Array, ""], scale: Float[Array, ""]
) -> Float[Array, ""]:
    """Put the ``-T^2`` ball margin back in the units of ``T``, by ``flux / |T|``.

    The flux half of the DEC is the ball margin of ``-T^2``, so it is homogeneous
    of degree *two* in the tensor while the other three margins are degree one.
    Combining them with a bare ``min`` gave a number whose scaling changed with
    which constraint binds: on ``T = diag(1, 2, 0, 0)`` scaled by ``c``, the WEC
    margin ran 0.5, 1, 5 while the raw flux ran -1.5, -6, -150. Signs were
    unaffected -- both vanish at the same tensors, so no verdict ever moved -- but a
    reported "DEC margin" that is quadratic at some points and linear at others is
    not a margin, and a ranking or scaling fit that crosses the switch compares
    different powers of the same tensor.

    Dividing by the scale is monotone at fixed ``T`` and vanishes exactly where
    ``flux`` does, so it changes neither the verdict nor the argmin -- only the
    units. It is preferred to ``sgn(flux) sqrt(|flux|)``, which is also degree one
    but has unbounded derivative at zero: an absolute error ``1e-12 scale^2`` in
    ``flux`` would become ``1e-6 scale`` in the margin, six orders above the floor
    the other three conditions answer to. Under the division the error is
    ``1e-12 scale``, so one floor covers all four.
    """
    return flux / jnp.where(scale > 0.0, scale, 1.0)


def noise_floor(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    *,
    condition: str = "nec",
) -> Float[Array, ""]:
    """Magnitude below which an LMI margin cannot be read as a violation.

    Two effects set it: the residual ternary bracket, and the ``eigvalsh`` error
    on ``M(sigma)``, both relative to the scale of the tensor.

    All four conditions now answer to the *same* relative floor. The flux half of
    the DEC feeds ``-T^2`` to the same search, so its absolute error is
    ``1e-12 scale^2``; :func:`_flux_margin_linear` divides by ``scale`` before the
    ``min``, which brings that back to ``1e-12 scale`` alongside the other three.
    The floor used to carry a ``scale**2`` branch to cover the undivided flux, and
    that branch was wrong in both directions -- too tight below unit scale, too
    loose above it, since the DEC margin was quadratic only where the flux bound.

    Use as ``margin < -noise_floor(...)`` to decide violation. At saturation, where
    the multiplier is forced to a single value and the true margin is exactly zero,
    no float64 search can do better; the exact ``LDL^T`` check on a rational
    ``sigma`` is the escape hatch (see the certificate).
    """
    if condition not in ("nec", "wec", "sec", "dec"):
        raise ValueError(f"unknown condition {condition!r}")
    T_hat = tetrad_components(T_ab, g_ab)
    # Clamping the scale at 1 made the floor ABSOLUTE below unit scale, which
    # breaks covariance under T -> c T: on T = 1e-7 diag(1,2,0,0) the DEC fails by
    # 100% of its own scale (margin -1.5e-14) and was reported inconclusive
    # against a 1e-12 floor. The floor is now relative, plus a small absolute term
    # that covers the residual ternary bracket at exact vacuum -- where the true
    # maximum is 0, the search returns about -6e-22, and a purely relative floor
    # would convict Minkowski of violating all four conditions.
    return _NOISE_REL * _tensor_scale(T_hat) + _NOISE_ABS


def null_deficit(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
) -> Float[Array, ""]:
    """Worst null contraction at Eulerian normalization, at any algebraic type.

    Returns ``min { T_ab k^a k^b : k null, -g(k, n) = 1 }``, i.e. the minimum of
    ``q(s) = rho - 2 b.s + s^T S s`` over the unit sphere. This is the
    type-independent replacement for the rest-frame quantity ``min_i (rho + p_i)``,
    which exists only at Type I, and for the momentum-plane witness
    ``rho + S_par - 2|j|``, which only probes one direction.

    It needs no separate solver. Lifting ``q`` to the null cone of ``eta`` sends
    ``s`` with ``|s| = 1`` to ``x = (1, s)/sqrt(2)``, and the rank-one extreme
    points of the corresponding semidefinite program give

        max_sigma lambda_min(That + sigma eta) = (1/2) min_{|s|=1} q(s),

    so the deficit is exactly twice the NEC margin already computed by
    :func:`certify_point`. Verified against a dense null-cone scan over random
    tensors in ``tests/test_slemma.py``.
    """
    T_hat = tetrad_components(T_ab, g_ab)
    _, nec = _lmi_margin(T_hat, jnp.asarray(-jnp.inf, dtype=T_hat.dtype))
    return 2.0 * nec


def certify_point(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
) -> dict[str, Float[Array, ""]]:
    """Cap-free all-observer margins for NEC/WEC/SEC/DEC at one point.

    Every margin is the optimal LMI value, valid at every Hawking-Ellis type with
    no classification and no rapidity cap.

    The decision it supports is *two*-sided and neither side is exact in binary64:
    ``> +noise_floor(...)`` says the condition holds for every observer,
    ``< -noise_floor(...)`` says some observer sees it fail, and in between the answer
    is inconclusive. Thresholding the satisfied side at zero instead -- which this
    docstring used to do -- is not sound: ``lambda_min`` from a symmetric eigensolver
    carries a backward error of order ``eps * ||M||``, so a computed margin can sit
    just above zero when the true one is just below, and "certified" would then be
    claimed for a tensor that marginally violates. The floor is what makes the claim
    honest, and it is deliberately symmetric.

    Where a verdict is actually needed inside the floor, the escape is exact rather
    than tighter: :mod:`.certificate` emits a rational multiplier verified by an exact
    ``LDL^T``, or a rational violating observer verified by exact evaluation. Neither
    consults a float, and both are what the word "certified" should be reserved for.

    The returned numbers are certification margins, not the observed extrema of
    the contraction; only their sign carries the exact statement. For the NEC the
    extremum is available too, as exactly twice this margin: see
    :func:`null_deficit`.
    """
    T_hat = tetrad_components(T_ab, g_ab)
    zero = jnp.zeros((), dtype=T_hat.dtype)
    neg_inf = jnp.asarray(-jnp.inf, dtype=T_hat.dtype)

    _, nec = _lmi_margin(T_hat, neg_inf)
    _, wec = _lmi_margin(T_hat, zero)
    _, sec = _lmi_margin(_trace_reversed(T_hat), zero)
    _, flux = _lmi_margin(_minus_T_squared(T_hat), zero)
    flux = _flux_margin_linear(flux, _tensor_scale(T_hat))
    return {"nec": nec, "wec": wec, "sec": sec, "dec": jnp.minimum(wec, flux)}


def witness_observer(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
) -> Float[Array, "3"]:
    """A boost 3-vector ``w`` with ``|w| <= 1`` and ``q(w) < 0``, when the WEC fails.

    The previous implementation read ``w = v_spatial / v_0`` off the eigenvector of
    the most negative eigenvalue of ``M(sigma_star)``, justified by the claim that
    ``q(w) - sigma (1 - |w|^2) = lambda_min |v|^2 / v_0^2 < 0`` forces ``q(w) < 0``.
    It does not: ``q(w) = sigma (1 - |w|^2) + lambda_min |v|^2 / v_0^2`` and the
    first term is ``>= 0`` for ``sigma >= 0`` and ``|w| <= 1``, so the sign is
    decided only when ``|lambda_min| >= sigma``, which nothing guarantees. It also
    breaks whenever ``lambda_min`` is repeated, because ``eigh`` then returns an
    arbitrary basis of the eigenspace: for ``That = diag(1, -3, 10, 10)`` the
    optimum is ``sigma = 2``, ``M = diag(-1, -1, 12, 12)``, the routine picked
    ``e_0`` and returned ``w = 0`` with ``q(0) = +1`` -- presented as a violating
    observer, while ``w = (-0.999, 0, 0)`` gives ``q = -1.994``.

    Minimize ``q`` directly instead. This does not need the minimum to be global,
    and it had better not: a quadratic on a ball can carry a local minimizer that
    is not global, e.g. ``q = -x^2 + x + y^2 + z^2`` has a strict local minimum at
    ``w = +e_0`` with ``q = 0`` against the global ``-2`` at ``w = -e_0``. The
    search is one-sided. Any ``w`` with ``|w| <= 1`` and ``q(w) < 0`` exhibits an
    observer and refutes the condition whatever its optimality status, while
    satisfaction is certified by a multiplier and is never inferred from a failed
    search; a descent that stalls at a local minimizer returns NaN, which is a
    refusal and not a false verdict. Two starts cover the degenerate corners: the
    lowest eigenvector of ``S`` (which is the answer when ``b = 0``, where a
    gradient start at the origin would stall) and the momentum direction.

    Returns NaN when no violating observer is found, which is the certified
    outcome when the WEC holds.
    """
    T_hat = tetrad_components(T_ab, g_ab)
    rho, b, S = T_hat[0, 0], -T_hat[0, 1:], T_hat[1:, 1:]

    def q(w):
        return rho - 2.0 * (b @ w) + w @ (S @ w)

    def project(w):  # onto the closed unit ball
        n = jnp.linalg.norm(w)
        return jnp.where(n > 1.0, w / jnp.where(n > 0, n, 1.0), w)

    # Gradient of q is 2(S w - b), Lipschitz constant 2||S||; step 1/(2||S||).
    s_norm = jnp.maximum(jnp.max(jnp.abs(jnp.linalg.eigvalsh(S))), 1e-12)
    step = 0.5 / s_norm

    def descend(w0):
        def body(_, w):
            return project(w - step * 2.0 * (S @ w - b))
        return jax.lax.fori_loop(0, _DESCENT_STEPS, body, project(w0))

    evals_S, evecs_S = jnp.linalg.eigh(S)
    b_norm = jnp.linalg.norm(b)
    starts = jnp.stack([
        evecs_S[:, 0],
        -evecs_S[:, 0],
        b / jnp.where(b_norm > 1e-30, b_norm, 1.0),
    ])
    cands = jax.vmap(descend)(starts)
    vals = jax.vmap(q)(cands)
    best = cands[jnp.argmin(vals)]
    return jnp.where(jnp.min(vals) < 0.0, best, jnp.nan)
