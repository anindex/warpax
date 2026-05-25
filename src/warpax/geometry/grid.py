"""Grid evaluation: lift pointwise curvature chain to 3D spatial grids.

Uses the flatten-vmap-reshape pattern to evaluate the curvature chain
(and optional invariants) at every point on a spatial grid.

For large grids (e.g. 100^3 = 1M points), use ``batch_size`` or
``auto_chunk_threshold`` to process the grid in chunks via ``jax.lax.map``,
trading peak parallelism for bounded memory usage.

Invariants are computed inside the vmapped function so the full Riemann
tensor at each point does not need to persist across the batch -- only the
resulting scalars survive into the output.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from .geometry import CurvatureResult, compute_curvature_chain
from .invariants import kretschmann_scalar, ricci_squared, weyl_squared
from .types import GridSpec


class GridCurvatureResult(NamedTuple):
    """All curvature tensors and invariants over a spatial grid.

    Extends CurvatureResult with three scalar invariant fields. Each field
    has shape ``(*grid_shape, ...)`` where the trailing dimensions are the
    tensor indices (e.g. ``(4, 4)`` for the metric).
    """
    metric: Float[Array, "..."]
    metric_inv: Float[Array, "..."]
    christoffel: Float[Array, "..."]
    riemann: Float[Array, "..."]
    ricci: Float[Array, "..."]
    ricci_scalar: Float[Array, "..."]
    einstein: Float[Array, "..."]
    stress_energy: Float[Array, "..."]
    kretschmann: Float[Array, "..."]
    ricci_squared: Float[Array, "..."]
    weyl_squared: Float[Array, "..."]


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
        ``N = Nx * Ny * Nz``. Each row is ``(t, x, y, z)``.
    """
    X, Y, Z = grid_spec.meshgrid  # each (Nx, Ny, Nz)
    T = jnp.full_like(X, t)
    coords_4d = jnp.stack([T, X, Y, Z], axis=-1)  # (Nx, Ny, Nz, 4)
    return coords_4d.reshape(-1, 4)  # (N, 4)


def _make_point_fn(metric_fn, compute_invariants: bool) -> Callable:
    """Build the per-point function (with or without invariants)."""
    if compute_invariants:
        def point_fn(c: Float[Array, "4"]) -> GridCurvatureResult:
            result = compute_curvature_chain(metric_fn, c)
            K = kretschmann_scalar(result.riemann, result.metric, result.metric_inv)
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
                kretschmann=K,
                ricci_squared=R2,
                weyl_squared=W2,
            )
    else:
        def point_fn(c: Float[Array, "4"]) -> CurvatureResult:
            return compute_curvature_chain(metric_fn, c)
    return point_fn


@eqx.filter_jit
def evaluate_curvature_grid(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    grid_spec: GridSpec,
    *,
    batch_size: int | None = None,
    compute_invariants: bool = True,
    t: float = 0.0,
    auto_chunk_threshold: int | None = None,
) -> GridCurvatureResult | CurvatureResult:
    """Evaluate the curvature chain (and optionally invariants) over a 3D grid.

    Flattens the 3D grid of spacetime coordinates into a 1D batch of shape
    ``(N, 4)``, applies the pointwise function via ``jax.vmap`` (or chunked
    ``jax.lax.map``), then reshapes results back to ``(*grid_shape, ...)``.

    Parameters
    ----------
    metric_fn : callable
        Any callable mapping coords ``(4,)`` to metric tensor ``(4, 4)``.
    grid_spec : GridSpec
        Grid specification with bounds and shape.
    batch_size : int or None, optional
        If ``None`` (default), use full ``jax.vmap`` over all grid points.
        If int, use ``jax.lax.map`` with this ``batch_size`` for memory-safe
        chunked processing. Takes priority over ``auto_chunk_threshold``.
    compute_invariants : bool, optional
        If True (default), compute Kretschmann, Ricci-squared, and
        Weyl-squared invariants and return a ``GridCurvatureResult``.
        If False, return a ``CurvatureResult`` only.
    t : float, optional
        Time coordinate for the constant-time slice (default 0.0).
    auto_chunk_threshold : int or None, optional
        If set and ``prod(grid_spec.shape) > auto_chunk_threshold``,
        dispatches to ``jax.lax.map(..., batch_size=auto_chunk_threshold)``.
        Ignored when ``batch_size`` is set (explicit beats automatic).
        Must be a positive integer when not ``None``.

    Returns
    -------
    GridCurvatureResult or CurvatureResult
        ``GridCurvatureResult`` if ``compute_invariants=True``, otherwise
        ``CurvatureResult``. All fields have shape ``(*grid_shape, ...)``.
    """
    if auto_chunk_threshold is not None and auto_chunk_threshold <= 0:
        raise ValueError(
            "auto_chunk_threshold must be a positive integer or None, "
            f"got {auto_chunk_threshold}"
        )

    coords_flat = build_coord_batch(grid_spec, t=t)
    grid_shape = grid_spec.shape

    point_fn = _make_point_fn(metric_fn, compute_invariants)

    n_points = int(coords_flat.shape[0])
    if batch_size is not None:
        results_flat = lax.map(point_fn, coords_flat, batch_size=batch_size)
    elif auto_chunk_threshold is not None and n_points > auto_chunk_threshold:
        results_flat = lax.map(point_fn, coords_flat, batch_size=auto_chunk_threshold)
    else:
        results_flat = jax.vmap(point_fn)(coords_flat)

    return jax.tree.map(
        lambda a: a.reshape(*grid_shape, *a.shape[1:]),
        results_flat,
    )
