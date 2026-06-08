"""Observer parameterizations for energy condition verification.

Boost 3-vector ``w in R^3`` with ``zeta = |w|`` and direction
``w / |w|`` (``w = 0`` is the Eulerian observer; optional smooth cap
``zeta_max * tanh(|w| / zeta_max)``); also a ``(zeta, theta, phi)``
rapidity-angle form. Null directions use stereographic projection
from ``R^2`` to ``S^2`` to avoid polar singularities. JIT- and
vmap-safe via ``jnp.where``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _gram_schmidt_step(
    v: Float[Array, "4"],
    tetrad_rows: Float[Array, "4 4"],
    signature: Float[Array, "4"],
    g_ab: Float[Array, "4 4"],
) -> Float[Array, "4"]:
    """Subtract metric-orthogonal projections of ``v`` onto ``tetrad_rows``.

    ``signature[i] = -1`` for the timelike row, ``+1`` for spacelike rows.
    Rows with sentinel value (signature == 0) are inert no-ops.
    """
    gv = g_ab @ v
    coeffs = (tetrad_rows @ gv) * signature
    return v - coeffs @ tetrad_rows


def _select_first_nondegenerate(
    candidates: Float[Array, "K 4"],
    norm_sqs: Float[Array, "K"],
    *,
    threshold: float = 1e-14,
) -> tuple[Float[Array, "4"], Float[Array, ""]]:
    """Pick the first candidate whose ``|v^T g v|`` is above ``threshold``.

    Branchless fallback that keeps the function vmap-safe at degenerate
    spatial bases (e.g. when the primary axis already lies in the
    timelike direction's plane).
    """
    ok = norm_sqs > threshold
    idx = jnp.argmax(ok)
    return candidates[idx], norm_sqs[idx]


def compute_orthonormal_tetrad(g_ab: Float[Array, "4 4"]) -> Float[Array, "4 4"]:
    """Orthonormal tetrad ``{e_0, e_1, e_2, e_3}`` from the metric at a point.

    ADM-motivated construction: ``e_0`` is the unit normal to spatial
    slices, ``e_1, e_2, e_3`` are obtained by Gram-Schmidt on the spatial
    basis ``{x, y, z}`` using ``g_{ab}`` as the inner product. If any
    candidate spatial basis vector is degenerate (numerically aligned
    with already-fixed tetrad rows), the next axis is tried.

    Returns
    -------
    Float[Array, "4 4"]
        Tetrad with ``tetrad[I, a] = e_I^a`` (row ``I`` is the ``I``-th
        tetrad vector, column ``a`` is the coordinate component). Well-defined
        at all warp speeds: the slice normal ``e_0`` stays timelike even where
        the coordinate-time direction ``g_{00}`` turns spacelike.
    """
    g_inv = jnp.linalg.inv(g_ab)

    # e_0 is the future-pointing unit normal to the spatial slices. For a
    # spacelike foliation (positive-definite gamma) ``g^{00} = -1/alpha^2 < 0``
    # holds at *all* warp speeds -- even superluminally, where the coordinate
    # time direction g_{00} turns spacelike, the slice normal stays timelike.
    # The tetrad is therefore well-defined for every v_s (see the superluminal
    # orthonormality sentinel in tests/test_ec_observer_and_solvers.py).
    alpha = 1.0 / jnp.sqrt(jnp.maximum(-g_inv[0, 0], 1e-30))
    beta_up = -g_inv[0, 1:4] / g_inv[0, 0]
    e0 = jnp.array([1.0 / alpha, -beta_up[0] / alpha,
                    -beta_up[1] / alpha, -beta_up[2] / alpha])

    spatial_basis = jnp.eye(4)[1:]
    tetrad = jnp.zeros((4, 4)).at[0].set(e0)

    # Each spatial slot tries its primary axis first, then falls back to
    # the other spatial axes if the projection collapses.
    fallback_order = (
        (0, 1, 2),
        (1, 0, 2),
        (2, 0, 1),
    )

    for slot, axes in enumerate(fallback_order, start=1):
        # The signature for the current set of already-built rows:
        # row 0 (timelike) is -1, built spatial rows are +1, the rest +1
        # but inert because their entries are zero.
        signature = jnp.array([-1.0] + [1.0] * 3)

        candidates = []
        norm_sqs = []
        for axis in axes:
            v = spatial_basis[axis]
            v = _gram_schmidt_step(v, tetrad, signature, g_ab)
            candidates.append(v)
            norm_sqs.append(jnp.dot(g_ab @ v, v))
        candidates = jnp.stack(candidates)
        norm_sqs = jnp.stack(norm_sqs)

        v_sel, norm_sq_sel = _select_first_nondegenerate(candidates, norm_sqs)
        # Floor radicand: autodiff-safe at norm_sq -> 0.
        norm = jnp.sqrt(jnp.abs(norm_sq_sel) + 1e-60)
        tetrad = tetrad.at[slot].set(v_sel / norm)

    return tetrad


def timelike_from_rapidity(
    zeta: Float[Array, ""],
    theta: Float[Array, ""],
    phi: Float[Array, ""],
    tetrad: Float[Array, "4 4"],
) -> Float[Array, "4"]:
    """Construct a unit timelike 4-vector from rapidity parameters.

    u^a = cosh(zeta) e_0^a + sinh(zeta) [sin(theta)cos(phi) e_1
          + sin(theta)sin(phi) e_2 + cos(theta) e_3]

    Parameters
    ----------
    zeta : Float[Array, ""]
        Rapidity zeta in [0, inf). Zero = comoving with Eulerian observer.
    theta : Float[Array, ""]
        Polar angle theta in [0, pi].
    phi : Float[Array, ""]
        Azimuthal angle phi in [0, 2*pi).
    tetrad : Float[Array, "4 4"]
        Orthonormal tetrad, shape (4, 4).

    Returns
    -------
    Float[Array, "4"]
        Unit timelike 4-vector u^a, shape (4,).
    """
    n = (
        jnp.sin(theta) * jnp.cos(phi) * tetrad[1]
        + jnp.sin(theta) * jnp.sin(phi) * tetrad[2]
        + jnp.cos(theta) * tetrad[3]
    )
    return jnp.cosh(zeta) * tetrad[0] + jnp.sinh(zeta) * n


def null_from_angles(
    theta: Float[Array, ""],
    phi: Float[Array, ""],
    tetrad: Float[Array, "4 4"],
) -> Float[Array, "4"]:
    """Construct a null 4-vector from angular parameters.

    k^a = e_0^a + sin(theta)cos(phi) e_1 + sin(theta)sin(phi) e_2
          + cos(theta) e_3

    Parameters
    ----------
    theta : Float[Array, ""]
        Polar angle theta in [0, pi].
    phi : Float[Array, ""]
        Azimuthal angle phi in [0, 2*pi).
    tetrad : Float[Array, "4 4"]
        Orthonormal tetrad, shape (4, 4).

    Returns
    -------
    Float[Array, "4"]
        Null 4-vector k^a, shape (4,).
    """
    return (
        tetrad[0]
        + jnp.sin(theta) * jnp.cos(phi) * tetrad[1]
        + jnp.sin(theta) * jnp.sin(phi) * tetrad[2]
        + jnp.cos(theta) * tetrad[3]
    )


def timelike_from_boost_vector(
    w: Float[Array, "3"],
    tetrad: Float[Array, "4 4"],
    zeta_max: Float[Array, ""] | None = None,
) -> Float[Array, "4"]:
    """Construct a unit timelike 4-vector from an unconstrained boost 3-vector.

    u^a = cosh(zeta) e_0^a + sinh(zeta) s^a

    where ``zeta = |w|`` (or ``zeta_max * tanh(|w| / zeta_max)`` when a
    rapidity cap is supplied), and ``s^a`` is the unit spatial direction
    obtained by projecting ``w / |w|`` into the orthonormal tetrad frame.

    When ``w = 0``, this returns the Eulerian observer ``e_0`` exactly,
    since ``cosh(0) = 1`` and ``sinh(0) = 0``.

    Parameters
    ----------
    w : Float[Array, "3"]
        Unconstrained boost 3-vector in the tetrad spatial frame.
    tetrad : Float[Array, "4 4"]
        Orthonormal tetrad, shape (4, 4).
    zeta_max : Float[Array, ""] or None
        If provided, cap rapidity smoothly via ``zeta_max * tanh(|w| / zeta_max)``.

    Returns
    -------
    Float[Array, "4"]
        Unit timelike 4-vector u^a, shape (4,).
    """
    eps = 1e-12
    norm = jnp.sqrt(jnp.dot(w, w) + eps**2)

    if zeta_max is not None:
        zeta = zeta_max * jnp.tanh(norm / zeta_max)
    else:
        zeta = norm

    # w=0 picks an arbitrary direction; sinh(0)=0 cancels the contribution.
    s_hat = w / norm
    s = s_hat[0] * tetrad[1] + s_hat[1] * tetrad[2] + s_hat[2] * tetrad[3]

    return jnp.cosh(zeta) * tetrad[0] + jnp.sinh(zeta) * s


def null_from_stereo(
    w: Float[Array, "2"],
    tetrad: Float[Array, "4 4"],
) -> Float[Array, "4"]:
    """Construct a null 4-vector via stereographic projection from R^2 to S^2.

    Maps ``w in R^2`` to a direction on S^2 via:
        n = (2 w_1, 2 w_2, 1 - |w|^2) / (1 + |w|^2)

    then ``k^a = e_0^a + n^i e_i^a``.

    ``w = 0`` maps to the north pole (``e_3`` direction).
    ``|w| -> inf`` approaches the south pole (``-e_3``).
    This covers all of S^2 smoothly without polar coordinate singularities.

    Parameters
    ----------
    w : Float[Array, "2"]
        Unconstrained 2-vector for stereographic projection.
    tetrad : Float[Array, "4 4"]
        Orthonormal tetrad, shape (4, 4).

    Returns
    -------
    Float[Array, "4"]
        Null 4-vector k^a, shape (4,).
    """
    r_sq = jnp.dot(w, w)
    denom = 1.0 + r_sq
    n1 = 2.0 * w[0] / denom
    n2 = 2.0 * w[1] / denom
    n3 = (1.0 - r_sq) / denom

    s = n1 * tetrad[1] + n2 * tetrad[2] + n3 * tetrad[3]
    return tetrad[0] + s


def boost_vector_to_params(
    w: Float[Array, "3"],
    zeta_max: Float[Array, ""] | None = None,
) -> Float[Array, "3"]:
    """Convert boost 3-vector to (zeta, theta, phi) for reporting.

    Parameters
    ----------
    w : Float[Array, "3"]
        Boost 3-vector.
    zeta_max : Float[Array, ""] or None
        Rapidity cap (same as used in ``timelike_from_boost_vector``).

    Returns
    -------
    Float[Array, "3"]
        Array ``[zeta, theta, phi]``.
    """
    eps = 1e-12
    norm = jnp.sqrt(jnp.dot(w, w) + eps**2)
    if zeta_max is not None:
        zeta = zeta_max * jnp.tanh(norm / zeta_max)
    else:
        zeta = norm

    s_hat = w / norm
    theta = jnp.arccos(jnp.clip(s_hat[2], -1.0, 1.0))
    phi = jnp.arctan2(s_hat[1], s_hat[0]) % (2.0 * jnp.pi)

    return jnp.array([zeta, theta, phi])


def stereo_to_params(
    w: Float[Array, "2"],
) -> Float[Array, "3"]:
    """Convert stereographic 2-vector to (0, theta, phi) for reporting.

    Parameters
    ----------
    w : Float[Array, "2"]
        Stereographic 2-vector.

    Returns
    -------
    Float[Array, "3"]
        Array ``[0, theta, phi]`` (zeta = 0 for null vectors).
    """
    r_sq = jnp.dot(w, w)
    denom = 1.0 + r_sq
    n3 = (1.0 - r_sq) / denom
    n1 = 2.0 * w[0] / denom
    n2 = 2.0 * w[1] / denom
    theta = jnp.arccos(jnp.clip(n3, -1.0, 1.0))
    phi = jnp.arctan2(n2, n1) % (2.0 * jnp.pi)
    return jnp.array([jnp.float64(0.0), theta, phi])


def bounded_param(
    raw: Float[Array, ""],
    lower: Float[Array, ""],
    upper: Float[Array, ""],
) -> Float[Array, ""]:
    """Map raw unconstrained parameter to bounded domain via sigmoid.

    result = lower + (upper - lower) * sigmoid(raw)

    Used by optimization to enforce box constraints with
    Optimistix BFGS, which does not natively support bounded minimization.

    Parameters
    ----------
    raw : Float[Array, ""]
        Unconstrained parameter in (-inf, inf).
    lower : Float[Array, ""]
        Lower bound of the target interval.
    upper : Float[Array, ""]
        Upper bound of the target interval.

    Returns
    -------
    Float[Array, ""]
        Value in (lower, upper).
    """
    return lower + (upper - lower) * jax.nn.sigmoid(raw)


def unbounded_param(
    val: Float[Array, ""],
    lower: Float[Array, ""],
    upper: Float[Array, ""],
) -> Float[Array, ""]:
    """Inverse sigmoid: map bounded value to unconstrained domain.

    raw = log((val - lower) / (upper - val))

    Used to initialize raw optimization parameters from physical values.

    Parameters
    ----------
    val : Float[Array, ""]
        Value in (lower, upper).
    lower : Float[Array, ""]
        Lower bound of the interval.
    upper : Float[Array, ""]
        Upper bound of the interval.

    Returns
    -------
    Float[Array, ""]
        Unconstrained parameter in (-inf, inf).
    """
    eps = 1e-30
    val = jnp.clip(val, lower + eps, upper - eps)
    return jnp.log((val - lower) / (upper - val))
