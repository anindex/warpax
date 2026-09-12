"""All-observer energy conditions as pointwise linear matrix inequalities.

This module numerically optimizes 4x4 linear matrix inequalities (LMIs) without
an observer rapidity cap or Hawking-Ellis classification. The reductions are
exact in mathematics; floating-point margins use :func:`noise_floor` and are
inconclusive near zero. Exact sufficient certificates are available separately
in :mod:`.certificate`.

Let ``{n, e_i}`` be an Eulerian orthonormal tetrad. Timelike observers have
``u = gamma (n + w)``, ``|w| < 1``, and normalized null directions have
``k = n + s``, ``|s| = 1``. With ``rho = T(n,n)``, ``b_i = -T(n,e_i)``, and
``S_ij = T(e_i,e_j)``,

    T(u,u) = gamma^2 q(w),   q(w) = rho - 2 b.w + w^T S w.

Continuity reduces the timelike sign condition to nonnegativity on the closed
unit ball. The S-lemma applies because ``1 - |w|^2 > 0`` at ``w = 0``:

    q >= 0 on the ball  <=>  That + sigma eta >= 0 for some sigma >= 0,
    eta = diag(-1, 1, 1, 1).

For NEC, the homogeneous equality constraint ``x^T eta x = 0`` is indefinite.
The equality S-lemma gives the same PSD test with a multiplier of either sign
(Xia, Wang, and Sheu, Mathematical Programming 156, 513-547, 2016).

The four conditions use these forms:

    NEC: sphere form on T
    WEC: ball form on T
    SEC: ball form on Theta = T - (1/2) tr_g(T) g
    DEC: ball forms on both T and -T g^{-1} T.

The second DEC form makes ``J^a = -T^a{}_b u^b`` causal or zero. WEC then fixes
its future orientation when nonzero. The minimum eigenvalue of each affine
matrix is concave in ``sigma``; a finite ternary search approximates its maximum.

These are pointwise tests. A common multiplier over a spatial box is a stronger
requirement and can fail even when every point satisfies the condition.
Spatial enclosures require the separate bounds in :mod:`.enclosure`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .observer import compute_orthonormal_tetrad

# eta in the orthonormal tetrad frame; M(sigma) = That + sigma * ETA.
_ETA = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

# Each ternary step shrinks the multiplier bracket by 2/3.
_TERNARY_STEPS = 80

# Relative tolerance for inconclusive margins on either side of zero.
_NOISE_REL = 1e-12

# A zero absolute floor preserves positive-rescaling covariance.
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
    """Approximate ``max_sigma lambda_min(That + sigma eta)`` by ternary search.

    ``sigma_lo=0`` selects the ball constraint; ``sigma_lo=-inf`` selects
    the sphere constraint. Returns the candidate ``(sigma_star, margin)``.

    The search can underestimate the maximum, and the symmetric eigensolver
    can err in either direction. Apply :func:`noise_floor` symmetrically;
    values near zero are inconclusive. This is a numerical estimate, not
    an exact certificate.

    The maximum occurs at finite ``sigma``: diagonal Rayleigh quotients
    bound it above by ``That_00 - sigma`` and ``That_ii + sigma``, which
    tend to ``-inf`` in the respective directions.
    """
    # The bracket must hold the argmax whether or not the LMI is feasible, so it comes
    # from an unconditional bound: lam_min(M(sigma)) <= scale -+ sigma while the maximum
    # is at least lam_min(That) >= -4 scale by Gershgorin, hence |sigma*| <= 5 scale.
    # Not clamped at 1, so the bracket collapses to the point 0 on vacuum.
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
    """``max |That_IJ|``, the scale the noise floor and the flux margin share."""
    return jnp.max(jnp.abs(T_hat))


def _flux_margin_linear(flux: Float[Array, ""], scale: Float[Array, ""]) -> Float[Array, ""]:
    """Divide the quadratic flux margin by ``max |That_IJ|``.

    The resulting margin scales linearly with positive rescalings of ``T``,
    as do the WEC, SEC, and NEC margins. Its sign is unchanged for nonzero
    ``T``. At vacuum the denominator is replaced by one and the result is zero.
    The scale depends on the chosen tetrad; this is not an invariant severity.
    """
    return flux / jnp.where(scale > 0.0, scale, 1.0)


def noise_floor(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
    *,
    condition: str = "nec",
) -> Float[Array, ""]:
    """Return the numerical decision tolerance for an LMI margin.

    The floor is ``1e-12 * max |That_IJ|`` for all four conditions. It allows
    for the residual multiplier bracket and symmetric eigensolver error;
    it is a numerical policy, not an interval error bound. The DEC flux
    margin is divided by the tensor scale before comparison with this floor.

    Margins above ``+floor`` indicate satisfaction numerically, margins below
    ``-floor`` indicate violation, and the intervening band is inconclusive.
    For :func:`null_deficit`, double this floor because that function returns
    twice the NEC LMI margin. :mod:`.certificate` can sometimes resolve a
    marginal case with an exact sufficient certificate.
    """
    if condition not in ("nec", "wec", "sec", "dec"):
        raise ValueError(f"unknown condition {condition!r}")
    T_hat = tetrad_components(T_ab, g_ab)
    # Keep the scale unclamped so the tolerance rescales with the tensor.
    return _NOISE_REL * _tensor_scale(T_hat) + _NOISE_ABS


def null_deficit(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
) -> Float[Array, ""]:
    """Estimate the least null contraction at Eulerian normalization.

    The target is ``min {T(k,k): g(k,k)=0, -g(k,n)=1}``, or equivalently
    ``min_{|s|=1} q(s)``. This definition applies at every algebraic type and
    is generally different from the Type-I eigenframe slack
    ``min_i(rho + p_i)`` or a single momentum-direction witness.

    In exact arithmetic the equality S-lemma gives

        min_{|s|=1} q(s) = 2 max_sigma lambda_min(That + sigma eta).

    The implementation returns twice the numerical NEC LMI margin. Its
    numerical tolerance is therefore twice :func:`noise_floor`.
    """
    T_hat = tetrad_components(T_ab, g_ab)
    _, nec = _lmi_margin(T_hat, jnp.asarray(-jnp.inf, dtype=T_hat.dtype))
    return 2.0 * nec


def certify_point(
    T_ab: Float[Array, "4 4"],
    g_ab: Float[Array, "4 4"],
) -> dict[str, Float[Array, ""]]:
    """Compute numerical cap-free NEC, WEC, SEC, and DEC margins at one point.

    NEC, WEC, and SEC use the optimized LMI values. DEC is the smaller of
    the WEC margin and the flux LMI margin divided by ``max |That_IJ|``.
    The metric must be Lorentzian with spacelike coordinate slices.

    Compare each margin with both signs of :func:`noise_floor`: values
    above the floor indicate numerical satisfaction, values below its
    negative indicate numerical violation, and the middle band is
    inconclusive. The tolerance does not turn floating-point estimates
    into rigorous error bounds. :mod:`.certificate` searches separately
    for sufficient certificates checked by exact arithmetic.

    These margins are sign diagnostics, not general observer extrema.
    The NEC exception is :func:`null_deficit`, which is twice its LMI value
    at the specified Eulerian normalization.
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
    """Search for a causal velocity ``w`` with ``|w| <= 1`` and ``q(w) < 0``.

    Projected gradient descent uses three starts: both signs of the least
    spatial eigenvector and the momentum direction. A returned candidate
    has negative contraction in floating-point arithmetic; an interior
    velocity represents a timelike observer and a boundary velocity a null
    direction. Check numerical tolerances or an exact certificate before
    assigning a verdict near zero.

    Returns NaN if no negative candidate is found. The search need not find
    a global minimum, so NaN proves neither satisfaction nor saturation.
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
    starts = jnp.stack(
        [
            evecs_S[:, 0],
            -evecs_S[:, 0],
            b / jnp.where(b_norm > 1e-30, b_norm, 1.0),
        ]
    )
    cands = jax.vmap(descend)(starts)
    vals = jax.vmap(q)(cands)
    best = cands[jnp.argmin(vals)]
    return jnp.where(jnp.min(vals) < 0.0, best, jnp.nan)
