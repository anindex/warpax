"""Tetrads and observer parameterizations for energy-condition evaluation.

Timelike observers use rapidity angles or a spatial rapidity vector with an
optional smooth cap. Rapidity is relative to the supplied tetrad's timelike
leg; the tetrad constructor uses the Eulerian slice normal. Null directions
use angles or a stereographic chart on the sphere with one pole excluded.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

# Absolute floors make the tetrad depend on the coordinate scale.
_DEGENERATE_RTOL = 1e-12


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
    rtol: float = _DEGENERATE_RTOL,
) -> tuple[Float[Array, "4"], Float[Array, ""]]:
    """Select the first candidate with positive norm above a relative floor.

    The floor is ``rtol * max |norm_sqs|``. Returns NaN if none qualifies.
    The branchless selection supports ``vmap``.
    """
    scale = jnp.max(jnp.abs(norm_sqs))
    ok = norm_sqs > rtol * scale
    idx = jnp.argmax(ok)
    any_ok = jnp.any(ok)
    return (
        jnp.where(any_ok, candidates[idx], jnp.nan),
        jnp.where(any_ok, norm_sqs[idx], jnp.nan),
    )


def compute_orthonormal_tetrad(g_ab: Float[Array, "4 4"]) -> Float[Array, "4 4"]:
    """Construct an Eulerian orthonormal tetrad for a Lorentzian metric.

    Requires spacelike coordinate slices, so ``g^{00} < 0``. The timelike
    leg is the future slice normal; spatial legs use metric Gram-Schmidt
    with coordinate-axis fallbacks. Nonspacelike slices or unresolved
    spatial degeneracy produce NaN entries.

    Returns
    -------
    Float[Array, "4 4"]
        ``tetrad[I, a] = e_I^a``. Normalization holds to numerical accuracy.
        The slice normal remains timelike at any warp speed under the stated
        hypotheses, including where the coordinate-time direction is spacelike.
    """
    g_inv = jnp.linalg.inv(g_ab)

    # Spacelike slices require g^{00} < 0. An absolute floor would rescale e_0.
    neg_g00 = -g_inv[0, 0]
    alpha = 1.0 / jnp.sqrt(jnp.where(neg_g00 > 0.0, neg_g00, jnp.nan))
    beta_up = -g_inv[0, 1:4] / g_inv[0, 0]
    e0 = jnp.array([1.0 / alpha, -beta_up[0] / alpha, -beta_up[1] / alpha, -beta_up[2] / alpha])

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
        # Selection requires a positive radicand; abs() would normalize a
        # timelike candidate as spacelike.
        norm = jnp.sqrt(norm_sq_sel)
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
        Rapidity in [0, inf) relative to the tetrad's timelike leg ``e_0``.
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
        Null 4-vector with ``-g(k, e_0)=1``, to numerical accuracy.
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
    """Construct a timelike vector from a spatial rapidity vector.

    Uses ``u = cosh(zeta) e_0 + sinh(zeta) s``. The radial variable is
    ``r = sqrt(|w|^2 + 1e-24)``; ``s = (w^i/r) e_i`` and ``zeta = r``, or
    ``zeta = zeta_max * tanh(r/zeta_max)`` when a cap is supplied. This
    regularization makes the map smooth at zero; normalization is accurate
    up to the regularization and floating-point errors.

    Parameters
    ----------
    w : Float[Array, "3"]
        Unconstrained spatial rapidity vector in the supplied tetrad.
    tetrad : Float[Array, "4 4"]
        Orthonormal tetrad; its timelike leg fixes the rapidity reference.
    zeta_max : Float[Array, ""] or None
        Positive smooth rapidity cap relative to that timelike leg.

    Returns
    -------
    Float[Array, "4"]
        Timelike vector, approximately unit normalized. At ``w=0`` it agrees
        with ``e_0`` to floating-point accuracy.
    """
    eps = 1e-12
    norm = jnp.sqrt(jnp.dot(w, w) + eps**2)

    if zeta_max is not None:
        zeta = zeta_max * jnp.tanh(norm / zeta_max)
    else:
        zeta = norm

    # At w=0 the spatial contribution vanishes.
    s_hat = w / norm
    s = s_hat[0] * tetrad[1] + s_hat[1] * tetrad[2] + s_hat[2] * tetrad[3]

    return jnp.cosh(zeta) * tetrad[0] + jnp.sinh(zeta) * s


def null_from_stereo(
    w: Float[Array, "2"],
    tetrad: Float[Array, "4 4"],
) -> Float[Array, "4"]:
    """Construct a tetrad-normalized null vector in a stereographic chart.

    Maps ``w in R^2`` to

        n = (2 w_1, 2 w_2, 1 - |w|^2) / (1 + |w|^2),
        k = e_0 + n^i e_i.

    The chart covers the unit sphere except its south pole. ``w=0`` gives
    the north pole; ``|w| -> inf`` approaches the south pole. Normalization
    is ``-g(k, e_0)=1`` relative to the supplied tetrad.

    Parameters
    ----------
    w : Float[Array, "2"]
        Unconstrained stereographic coordinates.
    tetrad : Float[Array, "4 4"]
        Orthonormal tetrad.

    Returns
    -------
    Float[Array, "4"]
        Future null vector to numerical accuracy.
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
