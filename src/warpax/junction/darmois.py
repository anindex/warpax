"""Darmois / Israel junction-condition checker.

Citations:

- Darmois, G. (1927). "Les équations de la gravitation Einsteinienne."
  *Mémorial des Sciences Mathématiques* 25.
- Israel, W. (1966). "Singular hypersurfaces and thin shells in general
  relativity." *Il Nuovo Cimento* B 44, 1-14.
- Poisson, E. (2004). *A Relativist's Toolkit*, §3.7.

Implements the pointwise discontinuity diagnostics:

1. First fundamental form: ``h_{ab} = g_{ab} - epsilon n_a n_b`` where
   ``n_a`` is the unit normal covector and ``epsilon = g(n, n)`` is its
   sign (``+1`` for spacelike normals, ``-1`` for timelike normals).
2. Second fundamental form (extrinsic curvature):
   ``K_{ab} = h^c_a h^d_b \\nabla_c n_d`` using the covariant derivative
   ``\\nabla_c n_d = \\partial_c n_d - \\Gamma^\\lambda_{cd} n_\\lambda``.

Both are evaluated at user-supplied probe points on either side of the
boundary; the discontinuities ``[[h]] = h_in - h_out`` and
``[[K]] = K_in - K_out`` are compared against the tolerance ``tol``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, jaxtyped

from ..geometry.geometry import christoffel_symbols
from ..geometry.metric import MetricSpecification


class DarmoisResult(NamedTuple):
    """Darmois junction discontinuity result.

    Attributes
    ----------
    first_form_discontinuity : Float[Array, ""]
        ``|[[h_{ab}]]|_max`` - maximum component of the induced metric
        jump across the boundary hypersurface (absolute).
    second_form_discontinuity : Float[Array, ""]
        ``|[[K_{ab}]]|_max`` - maximum component of the extrinsic
        curvature jump across the boundary hypersurface (absolute).
        Directly proportional to the surface stress
        ``|S_{ab}|`` modulo trace terms, so it is a physically
        meaningful comparand for ``tol``.
    physical : Bool[Array, ""]
        Traced boolean (``first_form_disc < tol`` and
        ``second_form_disc < tol``). Cast with ``bool(result.physical)``
        outside JIT/vmap if a host-side decision is required.
    """

    first_form_discontinuity: Float[Array, ""]
    second_form_discontinuity: Float[Array, ""]
    physical: Bool[Array, ""]


def _unit_normal(
    metric: MetricSpecification,
    boundary_fn: Callable[[Float[Array, "4"]], Float[Array, ""]],
    coords: Float[Array, "4"],
) -> tuple[Float[Array, "4"], Float[Array, "4"], Float[Array, ""]]:
    """Return ``(n_covec, n_vec, epsilon)`` for the boundary level set.

    The unit normal is built from the gradient of ``boundary_fn``
    rescaled by ``sqrt(|g(n, n)|)``; ``epsilon = sign(g(n, n))`` is
    ``+1`` for a spacelike normal (timelike hypersurface) and ``-1`` for
    a timelike normal (spacelike hypersurface). A tiny ``1e-30`` floor
    on the norm prevents division by zero on null boundaries.
    """
    n_covec_raw = jax.jacfwd(boundary_fn)(coords)
    g = metric(coords)
    g_inv = jnp.linalg.inv(g)
    n_vec_raw = g_inv @ n_covec_raw
    n_norm_sq = jnp.dot(n_vec_raw, n_covec_raw)
    epsilon = jnp.where(n_norm_sq >= 0.0, 1.0, -1.0)
    scale = jnp.sqrt(jnp.abs(n_norm_sq) + 1e-30)
    return n_covec_raw / scale, n_vec_raw / scale, epsilon


def _induced_and_extrinsic(
    metric: MetricSpecification,
    boundary_fn: Callable[[Float[Array, "4"]], Float[Array, ""]],
    coords: Float[Array, "4"],
) -> tuple[Float[Array, "4 4"], Float[Array, "4 4"], Float[Array, ""]]:
    """Compute induced metric ``h_{ab}``, extrinsic curvature ``K_{ab}``, and ``epsilon``.

    Uses the Gauss-Codazzi decomposition with a unit normal that may be
    timelike or spacelike. The first fundamental form is
    ``h_{ab} = g_{ab} - epsilon n_a n_b`` and the second is
    ``K_{ab} = h^c_a h^d_b \\nabla_c n_d`` with the covariant derivative
    ``\\nabla_c n_d = \\partial_c n_d - \\Gamma^\\lambda_{cd} n_\\lambda``.
    """
    n_covec, _n_vec, epsilon = _unit_normal(metric, boundary_fn, coords)
    g = metric(coords)

    h_ab = g - epsilon * jnp.outer(n_covec, n_covec)

    def _n_covec_fn(c):
        n, _, _ = _unit_normal(metric, boundary_fn, c)
        return n

    dn = jax.jacfwd(_n_covec_fn)(coords)
    gamma = christoffel_symbols(metric, coords)
    grad_n = dn - jnp.einsum("lcd,l->cd", gamma, n_covec)

    g_inv = jnp.linalg.inv(g)
    h_mixed = h_ab @ g_inv
    K_ab = h_mixed @ grad_n @ h_mixed.T

    return h_ab, K_ab, epsilon


@jaxtyped(typechecker=beartype)
def darmois(
    metric: MetricSpecification,
    boundary_fn: Callable[[Float[Array, "4"]], Float[Array, ""]],
    probe_coords_inside: Float[Array, "4"] | None = None,
    probe_coords_outside: Float[Array, "4"] | None = None,
    tol: float = 1e-8,
) -> DarmoisResult:
    """Check Darmois / Israel junction conditions at a boundary hypersurface.

    Parameters
    ----------
    metric : MetricSpecification
        The spacetime metric being tested.
    boundary_fn : Callable[[Float[Array, "4"]], Float[Array, ""]]
        Level-set function whose zero defines the boundary hypersurface
        ``Sigma = { x : boundary_fn(x) = 0 }``. The unit normal covector
        ``n_a = d(boundary_fn) / d x^a`` is computed via ``jax.jacfwd``.
    probe_coords_inside : Float[Array, "4"], optional
        Sample point just inside the boundary (default:
        ``(0, 0.9, 0, 0)`` - appropriate for a unit-radius boundary).
    probe_coords_outside : Float[Array, "4"], optional
        Sample point just outside the boundary (default:
        ``(0, 1.1, 0, 0)``).
    tol : float
        Physicality tolerance. Default ``1e-8``.

    Returns
    -------
    DarmoisResult
        NamedTuple with ``first_form_discontinuity``,
        ``second_form_discontinuity``, ``physical`` fields.

    Notes
    -----
    Pointwise probe test (not a full Sigma integral). Smooth metrics
    (Alcubierre-family) give discontinuities ``O(probe_separation^2)``
    and pass at any reasonable ``tol``; genuine shell metrics (WarpShell
    at the shell boundary) yield finite ``[[K]]``.
    """
    if probe_coords_inside is None:
        probe_coords_inside = jnp.array([0.0, 0.9, 0.0, 0.0])
    if probe_coords_outside is None:
        probe_coords_outside = jnp.array([0.0, 1.1, 0.0, 0.0])

    h_in, K_in, _ = _induced_and_extrinsic(metric, boundary_fn, probe_coords_inside)
    h_out, K_out, _ = _induced_and_extrinsic(metric, boundary_fn, probe_coords_outside)

    first_form_disc = jnp.max(jnp.abs(h_in - h_out))
    second_form_disc = jnp.max(jnp.abs(K_in - K_out))

    physical = (first_form_disc < tol) & (second_form_disc < tol)

    return DarmoisResult(
        first_form_discontinuity=first_form_disc,
        second_form_discontinuity=second_form_disc,
        physical=physical,
    )


@jaxtyped(typechecker=beartype)
def surface_stress_energy(
    metric: MetricSpecification,
    boundary_fn: Callable[[Float[Array, "4"]], Float[Array, ""]],
    probe_coords_inside: Float[Array, "4"],
    probe_coords_outside: Float[Array, "4"],
) -> Float[Array, "4 4"]:
    """Compute surface stress-energy ``S_{ab}`` via Israel junction conditions.

    Uses the proper two-sided jump formulation

    .. math::
        S_{ab} = -\\frac{\\epsilon}{8\\pi}\\bigl([K_{ab}] - [K] h_{ab}\\bigr),

    where ``[K_{ab}] = K^+_{ab} - K^-_{ab}`` is the jump of extrinsic
    curvature across ``Sigma``, ``[K] = h^{ab} [K_{ab}]`` is its trace
    with respect to the (pseudo-)inverse induced metric, ``h_{ab}`` is
    the averaged induced metric, and ``epsilon = g(n, n)`` is the sign
    of the unit normal (``+1`` timelike hypersurface, ``-1`` spacelike).

    The sign convention follows Israel (1966): the normal points from
    inside (-) to outside (+). The jump is ``[X] = X^+ - X^-``.

    Parameters
    ----------
    metric : MetricSpecification
    boundary_fn : level-set function defining Sigma
    probe_coords_inside : point on the (-) side of Sigma
    probe_coords_outside : point on the (+) side of Sigma

    Returns
    -------
    Float[Array, "4 4"]
        Surface stress-energy tensor (full 4x4, tangential to Sigma).
    """
    h_in, K_in, eps_in = _induced_and_extrinsic(
        metric, boundary_fn, probe_coords_inside
    )
    h_out, K_out, eps_out = _induced_and_extrinsic(
        metric, boundary_fn, probe_coords_outside
    )

    delta_K = K_out - K_in
    h_avg = 0.5 * (h_in + h_out)
    epsilon = 0.5 * (eps_in + eps_out)

    # Trace of the jump on the induced metric (h^{ab} is a 3D inverse;
    # use the Moore-Penrose pseudo-inverse for the 4x4 representation
    # because h_ab has a null direction along n^a).
    h_inv_avg = jnp.linalg.pinv(h_avg)
    delta_K_trace = jnp.einsum("ab,ab->", h_inv_avg, delta_K)

    S_ab = -(epsilon / (8.0 * jnp.pi)) * (delta_K - delta_K_trace * h_avg)
    return S_ab

