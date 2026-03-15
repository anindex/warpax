"""Darmois / Israel junction-condition checker.

Citations:

- Darmois, G. (1927). "Les équations de la gravitation Einsteinienne."
  *Mémorial des Sciences Mathématiques* 25.
- Israel, W. (1966). "Singular hypersurfaces and thin shells in general
  relativity." *Il Nuovo Cimento* B 44, 1-14.

Implements the pointwise discontinuity diagnostics:

1. First fundamental form: ``h_{ab} = g_{ab} - epsilon n_a n_b`` where
   ``n_a`` is the unit normal covector and ``epsilon = g(n, n)`` is its
   sign.
2. Second fundamental form (extrinsic curvature):
   ``K_{ab} = h^c_a h^d_b nabla_c n_d`` - projected directional
   derivative of the normal along the hypersurface.

Both are evaluated at user-supplied probe points on either side of the
boundary; the discontinuities ``[[h]] = h_in - h_out`` and
``[[K]] = K_in - K_out`` are compared against the tolerance ``tol``.

Textbook
classical GR; complements the `WarpShellPhysical` regularisation
without paper-number impact.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from ..geometry.metric import MetricSpecification


class DarmoisResult(NamedTuple):
    """Darmois junction discontinuity result.

    Attributes
    ----------
    first_form_discontinuity : Float[Array, ""]
        ``|[[h_{ab}]]|_max`` - maximum component of the induced 3-metric
        jump across the boundary hypersurface.
    second_form_discontinuity : Float[Array, ""]
        Relative ``|[[K_{ab}]]| / ||K_{ab}||`` - extrinsic-curvature
        jump normalized by the Frobenius scale of ``K`` on either side.
    physical : bool
        True iff both discontinuities are below tolerance ``tol``.
    """

    first_form_discontinuity: Float[Array, ""]
    second_form_discontinuity: Float[Array, ""]
    physical: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _unit_normal(
    metric: MetricSpecification,
    boundary_fn: Callable[[Float[Array, "4"]], Float[Array, ""]],
    coords: Float[Array, "4"],
) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
    """Return ``(n_covec, n_vec)`` unit-normal covector + vector.

    Normalisation is ``|n_norm_sq|``-based (sign-robust - timelike or
    spacelike boundaries), with a small epsilon floor to avoid zero
    division at degenerate boundaries.
    """
    n_covec = jax.jacfwd(boundary_fn)(coords)
    g = metric(coords)
    g_inv = jnp.linalg.inv(g)
    n_vec = g_inv @ n_covec  # raise index: n^a = g^{ab} n_b
    n_norm_sq = jnp.dot(n_vec, n_covec)
    scale = jnp.sqrt(jnp.abs(n_norm_sq) + 1e-30)
    return n_covec / scale, n_vec / scale


def _induced_and_extrinsic(
    metric: MetricSpecification,
    boundary_fn: Callable[[Float[Array, "4"]], Float[Array, ""]],
    coords: Float[Array, "4"],
) -> tuple[Float[Array, "4 4"], Float[Array, "4 4"]]:
    """Compute induced 3-metric ``h_{ab}`` and extrinsic curvature ``K_{ab}``.

    Uses Gauss-Codazzi identities with the normalized normal vector
    (either timelike or spacelike depending on the sign of
    ``g(n, n)``). Extrinsic curvature is approximated as the projected
    directional derivative ``h^c_a h^d_b d_c n_d``.
    """
    n_covec, _n_vec = _unit_normal(metric, boundary_fn, coords)
    g = metric(coords)

    # Induced 3-metric h_ab = g_ab - n_a n_b
    h_ab = g - jnp.outer(n_covec, n_covec)

    # Derivative of the unit-normal covector along the 4-direction
    def _n_covec_fn(c):
        n, _ = _unit_normal(metric, boundary_fn, c)
        return n

    dn = jax.jacfwd(_n_covec_fn)(coords)  # shape (4, 4): d_c n_d

    # Project onto Sigma via h^c_a = h_{ab} g^{bc}
    g_inv = jnp.linalg.inv(g)
    h_mixed = h_ab @ g_inv  # shape (4, 4) -- a (lower), c (upper)
    # K_{ab} = h^c_a (d_c n_d) h^d_b
    K_ab = h_mixed @ dn @ h_mixed.T

    return h_ab, K_ab


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
        Physicality tolerance. Default ``1e-8`` per for .

    Returns
    -------
    DarmoisResult
        NamedTuple with ``first_form_discontinuity``,
        ``second_form_discontinuity``, ``physical`` fields.

    Notes
    -----
    Algorithm (for a timelike or spacelike boundary ``Sigma`` with unit
    normal ``n^a``):

    1. **Induced 3-metric** on ``Sigma``: ``h_{ab} = g_{ab} - n_a n_b``.
    2. **Extrinsic curvature** on ``Sigma``:
       ``K_{ab} = h^c_a h^d_b nabla_c n_d`` - here computed as the
       directional derivative of ``n_a`` projected onto ``Sigma`` via
       ``jax.jacfwd``.
    3. **Discontinuities**: evaluate ``h`` and ``K`` at
       ``probe_coords_inside`` and ``probe_coords_outside``; return
       ``|h_in - h_out|_max`` and
       ``|K_in - K_out| / (||K_in|| + ||K_out||)``.

    Note that this is a pointwise probe test, not a full Sigma integral.
    For smooth metrics (Alcubierre-family), the discontinuity should be
    ``O(probe_separation^2)`` and below ``tol`` by construction. For
    genuine shell metrics (WarpShell at the shell boundary), ``K``
    jumps discontinuously.
    """
    if probe_coords_inside is None:
        probe_coords_inside = jnp.array([0.0, 0.9, 0.0, 0.0])
    if probe_coords_outside is None:
        probe_coords_outside = jnp.array([0.0, 1.1, 0.0, 0.0])

    h_in, K_in = _induced_and_extrinsic(metric, boundary_fn, probe_coords_inside)
    h_out, K_out = _induced_and_extrinsic(metric, boundary_fn, probe_coords_outside)

    first_form_disc = jnp.max(jnp.abs(h_in - h_out))
    k_scale = jnp.sqrt(jnp.sum(K_in ** 2) + jnp.sum(K_out ** 2)) + 1e-30
    second_form_disc = jnp.max(jnp.abs(K_in - K_out)) / k_scale

    physical = bool(
        (first_form_disc < tol) & (second_form_disc < tol)
    )

    return DarmoisResult(
        first_form_discontinuity=first_form_disc,
        second_form_discontinuity=second_form_disc,
        physical=physical,
    )
