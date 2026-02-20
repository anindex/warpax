"""Grid evaluation: lift pointwise curvature chain to 3D spatial grids.

Uses the flatten-vmap-reshape pattern to efficiently evaluate the curvature
chain (and optionally curvature invariants) at every point on a spatial grid.

For large grids (e.g. 100^3 = 1M points), use the ``batch_size`` parameter
to process the grid in chunks via ``jax.lax.map``, trading peak parallelism
for bounded memory usage.

The key insight: invariants are computed INSIDE the vmapped function so that
the full Riemann tensor at each point does not need to persist across the
batch only the resulting scalars survive into the output.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from .geometry import CurvatureResult, compute_curvature_chain
from .invariants import kretschner_scalar, ricci_squared, weyl_squared
from .types import GridSpec


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class GridCurvatureResult(NamedTuple):
    """All curvature tensors and invariants over a spatial grid.

    Extends CurvatureResult with three scalar invariant fields.
    Each field has shape ``(*grid_shape, ...)`` where the trailing
    dimensions are the tensor indices (e.g. ``(4, 4)`` for the metric).
    """
    metric: Float[Array, "..."]
    metric_inv: Float[Array, "..."]
    christoffel: Float[Array, "..."]
    riemann: Float[Array, "..."]
    ricci: Float[Array, "..."]
    ricci_scalar: Float[Array, "..."]
    einstein: Float[Array, "..."]
    stress_energy: Float[Array, "..."]
    kretschner: Float[Array, "..."]
    ricci_squared: Float[Array, "..."]
    weyl_squared: Float[Array, "..."]


# ---------------------------------------------------------------------------
# Coordinate batch construction
# ---------------------------------------------------------------------------


def build_coord_batch(
    grid_spec: GridSpec,
    t: float = 0.0,
) -> Float[Array, "N 4"]:
    """Build a flat batch of spacetime 4-vectors from a GridSpec.

    Parameters
    ----------
    grid_spec : GridSpec
        Grid specification with bounds and shape.
    t : float, optional
        Time coordinate for the constant-time slice (default 0.0).

    Returns
    -------
    Float[Array, "N 4"]
        Flattened coordinate array of shape ``(N, 4)`` where
        ``N = Nx * Ny * Nz``.  Each row is ``(t, x, y, z)``.
    """
    X, Y, Z = grid_spec.meshgrid  # each (Nx, Ny, Nz)
    T = jnp.full_like(X, t)
    coords_4d = jnp.stack([T, X, Y, Z], axis=-1)  # (Nx, Ny, Nz, 4)
    return coords_4d.reshape(-1, 4)  # (N, 4)


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------


def evaluate_curvature_grid(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    grid_spec: GridSpec,
    *,
    batch_size: int | None = None,
    compute_invariants: bool = True,
    t: float = 0.0,
) -> GridCurvatureResult | CurvatureResult:
    """Evaluate the curvature chain (and optionally invariants) over a 3D grid.

    Uses the flatten-vmap-reshape pattern: flatten the 3D grid of spacetime
    coordinates into a 1D batch of shape ``(N, 4)``, apply the pointwise
    function via ``jax.vmap`` (or chunked ``jax.lax.map``), then reshape
    results back to ``(*grid_shape, ...)``.

    Parameters
    ----------
    metric_fn : callable
        A MetricSpecification or any callable mapping coords ``(4,)`` to
        metric tensor ``(4, 4)``.
    grid_spec : GridSpec
        Grid specification with bounds and shape.
    batch_size : int or None, optional
        If None (default), use full ``jax.vmap`` over all grid points.
        If int, use ``jax.lax.map`` with ``batch_size`` for memory-safe
        chunked processing of large grids.
    compute_invariants : bool, optional
        If True (default), compute Kretschner, Ricci-squared, and
        Weyl-squared inside the vmap and return a ``GridCurvatureResult``.
        If False, return a ``CurvatureResult`` only (no invariants).
    t : float, optional
        Time coordinate for the constant-time slice (default 0.0).
        Passed through to ``build_coord_batch``.

    Returns
    -------
    GridCurvatureResult or CurvatureResult
        ``GridCurvatureResult`` if ``compute_invariants=True``, otherwise
        ``CurvatureResult``.  All fields have shape ``(*grid_shape, ...)``.
    """
    coords_flat = build_coord_batch(grid_spec, t=t)  # (N, 4)
    grid_shape = grid_spec.shape

    # Define the per-point function.
    # The `if compute_invariants` branch is resolved at Python trace time
    # (not JAX runtime) since compute_invariants is a plain Python bool.
    if compute_invariants:
        def point_fn(c: Float[Array, "4"]) -> GridCurvatureResult:
            result = compute_curvature_chain(metric_fn, c)
            K = kretschner_scalar(result.riemann, result.metric, result.metric_inv)
            R2 = ricci_squared(result.ricci, result.metric_inv)
            W2 = weyl_squared(K, R2, result.ricci_scalar)
            return GridCurvatureResult(
                metric=result.metric,
                metric_inv=result.metric_inv,
                christoffel=result.christoffel,
                riemann=result.riemann,
                ricci=result.ricci,
                ricci_scalar=result.ricci_scalar,
                einstein=result.einstein,
                stress_energy=result.stress_energy,
                kretschner=K,
                ricci_squared=R2,
                weyl_squared=W2,
            )
    else:
        def point_fn(c: Float[Array, "4"]) -> CurvatureResult:
            return compute_curvature_chain(metric_fn, c)

    # Evaluate over the batch
    if batch_size is not None:
        results_flat = lax.map(point_fn, coords_flat, batch_size=batch_size)
    else:
        results_flat = jax.vmap(point_fn)(coords_flat)

    # Reshape all output fields from (N, ...) to (*grid_shape, ...)
    return jax.tree.map(
        lambda a: a.reshape(*grid_shape, *a.shape[1:]),
        results_flat,
    )
