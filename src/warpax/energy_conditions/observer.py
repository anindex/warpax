"""Observer parameterization for energy condition verification (pure JAX).

Two parameterizations are provided:

1. **3-vector (preferred):**  An unconstrained boost vector ``w in R^3``
   where ``zeta = |w|`` and the spatial direction is ``w / |w|``.
   When ``w = 0``, this returns the Eulerian observer exactly.  A smooth
   rapidity cap ``zeta_max * tanh(|w| / zeta_max)`` is available.

2. **Rapidity-angle (legacy):**  ``(zeta, theta, phi)`` with explicit
   angular parameterization on S^2.  Retained for backward compatibility.

For null vectors, stereographic projection from ``R^2`` to ``S^2``
avoids the polar singularities of ``(theta, phi)``.

All functions are JIT-compilable and vmappable. No Python if/else on
traced values uses jnp.where for all conditional logic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def compute_orthonormal_tetrad(g_ab: Float[Array, "4 4"]) -> Float[Array, "4 4"]:
    """Construct an orthonormal tetrad {e_0, e_1, e_2, e_3} from the metric at a point.

    Uses ADM-motivated construction:
        e_0 = n^a (unit normal to spatial slices)
        e_1, e_2, e_3 from Gram-Schmidt on spatial metric

    The Gram-Schmidt loop is unrolled (only 3 spatial basis vectors).
    Degenerate basis vector fallback uses jnp.where (no Python if/else).

    Parameters
    ----------
    g_ab : Float[Array, "4 4"]
        4x4 metric tensor at a single point.

    Returns
    -------
    Float[Array, "4 4"]
        Tetrad e^a_I of shape (4, 4), where I labels the tetrad vector
        and a labels the coordinate index.
        tetrad[I, a] = e_I^a (I-th tetrad vector, a-th component).
    """
    # Invert the metric
    g_inv = jnp.linalg.inv(g_ab)

    # ADM decomposition: normal vector to t=const slices
    # n^a = (1/alpha, -beta^i/alpha), where alpha = (-g^{00})^{-1/2}
    alpha = 1.0 / jnp.sqrt(-g_inv[0, 0])
    beta_up = jnp.array([-g_inv[0, i] / g_inv[0, 0] for i in range(1, 4)])

    # e_0^a = n^a = (1/alpha, -beta^i/alpha)
    e0 = jnp.zeros(4)
    e0 = e0.at[0].set(1.0 / alpha)
    e0 = e0.at[1:].set(-beta_up / alpha)

    # Candidate spatial basis: coordinate vectors delta^a_{i+1}
    spatial_basis = jnp.eye(4)[1:]  # (3, 4)

    # Unrolled Gram-Schmidt for 3 spatial basis vectors
    tetrad = jnp.zeros((4, 4))
    tetrad = tetrad.at[0].set(e0)

    # --- Spatial vector 1 (i=0) ---
    v0 = spatial_basis[0]
    # Remove component along e_0 (timelike, sign = -1)
    inner_00 = jnp.dot(g_ab @ tetrad[0], v0)
    v0 = v0 - (inner_00 / (-1.0)) * tetrad[0]

    norm_sq_0 = jnp.dot(g_ab @ v0, v0)
    # Fallback: try other basis vectors if degenerate
    v0_alt1 = spatial_basis[1]
    inner_00_alt1 = jnp.dot(g_ab @ tetrad[0], v0_alt1)
    v0_alt1 = v0_alt1 - (inner_00_alt1 / (-1.0)) * tetrad[0]
    norm_sq_0_alt1 = jnp.dot(g_ab @ v0_alt1, v0_alt1)

    v0_alt2 = spatial_basis[2]
    inner_00_alt2 = jnp.dot(g_ab @ tetrad[0], v0_alt2)
    v0_alt2 = v0_alt2 - (inner_00_alt2 / (-1.0)) * tetrad[0]
    norm_sq_0_alt2 = jnp.dot(g_ab @ v0_alt2, v0_alt2)

    # Select best candidate via jnp.where
    use_alt1_0 = norm_sq_0 < 1e-14
    use_alt2_0 = use_alt1_0 & (norm_sq_0_alt1 < 1e-14)
    v0 = jnp.where(use_alt2_0, v0_alt2, jnp.where(use_alt1_0, v0_alt1, v0))
    norm_sq_0 = jnp.where(use_alt2_0, norm_sq_0_alt2,
                           jnp.where(use_alt1_0, norm_sq_0_alt1, norm_sq_0))

    e1 = v0 / jnp.sqrt(jnp.abs(norm_sq_0))
    tetrad = tetrad.at[1].set(e1)

    # --- Spatial vector 2 (i=1) ---
    v1 = spatial_basis[1]
    # Remove component along e_0 (timelike)
    inner_10 = jnp.dot(g_ab @ tetrad[0], v1)
    v1 = v1 - (inner_10 / (-1.0)) * tetrad[0]
    # Remove component along e_1 (spacelike, sign = +1)
    inner_11 = jnp.dot(g_ab @ tetrad[1], v1)
    v1 = v1 - (inner_11 / 1.0) * tetrad[1]

    norm_sq_1 = jnp.dot(g_ab @ v1, v1)
    # Fallback: try other basis vectors
    v1_alt0 = spatial_basis[0]
    inner_10_alt0 = jnp.dot(g_ab @ tetrad[0], v1_alt0)
    v1_alt0 = v1_alt0 - (inner_10_alt0 / (-1.0)) * tetrad[0]
    inner_11_alt0 = jnp.dot(g_ab @ tetrad[1], v1_alt0)
    v1_alt0 = v1_alt0 - (inner_11_alt0 / 1.0) * tetrad[1]
    norm_sq_1_alt0 = jnp.dot(g_ab @ v1_alt0, v1_alt0)

    v1_alt2 = spatial_basis[2]
    inner_10_alt2 = jnp.dot(g_ab @ tetrad[0], v1_alt2)
    v1_alt2 = v1_alt2 - (inner_10_alt2 / (-1.0)) * tetrad[0]
    inner_11_alt2 = jnp.dot(g_ab @ tetrad[1], v1_alt2)
    v1_alt2 = v1_alt2 - (inner_11_alt2 / 1.0) * tetrad[1]
    norm_sq_1_alt2 = jnp.dot(g_ab @ v1_alt2, v1_alt2)

    use_alt0_1 = norm_sq_1 < 1e-14
    use_alt2_1 = use_alt0_1 & (norm_sq_1_alt0 < 1e-14)
    v1 = jnp.where(use_alt2_1, v1_alt2, jnp.where(use_alt0_1, v1_alt0, v1))
    norm_sq_1 = jnp.where(use_alt2_1, norm_sq_1_alt2,
                           jnp.where(use_alt0_1, norm_sq_1_alt0, norm_sq_1))

    e2 = v1 / jnp.sqrt(jnp.abs(norm_sq_1))
    tetrad = tetrad.at[2].set(e2)

    # --- Spatial vector 3 (i=2) ---
    v2 = spatial_basis[2]
    # Remove component along e_0 (timelike)
    inner_20 = jnp.dot(g_ab @ tetrad[0], v2)
    v2 = v2 - (inner_20 / (-1.0)) * tetrad[0]
    # Remove component along e_1 (spacelike)
    inner_21 = jnp.dot(g_ab @ tetrad[1], v2)
    v2 = v2 - (inner_21 / 1.0) * tetrad[1]
    # Remove component along e_2 (spacelike)
    inner_22 = jnp.dot(g_ab @ tetrad[2], v2)
    v2 = v2 - (inner_22 / 1.0) * tetrad[2]

    norm_sq_2 = jnp.dot(g_ab @ v2, v2)
    # Fallback: try other basis vectors
    v2_alt0 = spatial_basis[0]
    inner_20_alt0 = jnp.dot(g_ab @ tetrad[0], v2_alt0)
    v2_alt0 = v2_alt0 - (inner_20_alt0 / (-1.0)) * tetrad[0]
    inner_21_alt0 = jnp.dot(g_ab @ tetrad[1], v2_alt0)
    v2_alt0 = v2_alt0 - (inner_21_alt0 / 1.0) * tetrad[1]
    inner_22_alt0 = jnp.dot(g_ab @ tetrad[2], v2_alt0)
    v2_alt0 = v2_alt0 - (inner_22_alt0 / 1.0) * tetrad[2]
    norm_sq_2_alt0 = jnp.dot(g_ab @ v2_alt0, v2_alt0)

    v2_alt1 = spatial_basis[1]
    inner_20_alt1 = jnp.dot(g_ab @ tetrad[0], v2_alt1)
    v2_alt1 = v2_alt1 - (inner_20_alt1 / (-1.0)) * tetrad[0]
    inner_21_alt1 = jnp.dot(g_ab @ tetrad[1], v2_alt1)
    v2_alt1 = v2_alt1 - (inner_21_alt1 / 1.0) * tetrad[1]
    inner_22_alt1 = jnp.dot(g_ab @ tetrad[2], v2_alt1)
    v2_alt1 = v2_alt1 - (inner_22_alt1 / 1.0) * tetrad[2]
    norm_sq_2_alt1 = jnp.dot(g_ab @ v2_alt1, v2_alt1)

    use_alt0_2 = norm_sq_2 < 1e-14
    use_alt1_2 = use_alt0_2 & (norm_sq_2_alt0 < 1e-14)
    v2 = jnp.where(use_alt1_2, v2_alt1, jnp.where(use_alt0_2, v2_alt0, v2))
    norm_sq_2 = jnp.where(use_alt1_2, norm_sq_2_alt1,
                           jnp.where(use_alt0_2, norm_sq_2_alt0, norm_sq_2))

    e3 = v2 / jnp.sqrt(jnp.abs(norm_sq_2))
    tetrad = tetrad.at[3].set(e3)

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

    # Smooth rapidity cap
    if zeta_max is not None:
        zeta = zeta_max * jnp.tanh(norm / zeta_max)
    else:
        zeta = norm

    # Smooth unit direction (w=0 gives arbitrary direction, but sinh(0)=0 zeroes it)
    s_hat = w / norm

    # Spatial direction in tetrad frame
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
    return jnp.log((val - lower) / (upper - val))
