"""Grid evaluation: lift pointwise curvature chain to 3D spatial grids.

Uses the flatten-vmap-reshape pattern to evaluate the curvature chain
(and optional invariants) at every point on a spatial grid.

For large grids (e.g. 100^3 = 1M points), use ``batch_size`` or
``auto_chunk_threshold`` to process the grid in chunks via ``jax.lax.map``,
trading peak parallelism for bounded memory usage.

Invariants are computed inside the vmapped function so the full Riemann
tensor at each point does not need to persist across the batch, only the
resulting scalars survive into the output.
"""

from collections.abc import Callable
from typing import NamedTuple

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


@eqx.filter_jit
def _build_coord_batch(grid_spec: GridSpec, t: Float[Array, ""]) -> Float[Array, "N 4"]:
    """Compiled coordinate builder; ``t`` is a traced scalar, the grid is static."""
    X, Y, Z = grid_spec.meshgrid
    T = jnp.full_like(X, t)
    return jnp.stack([T, X, Y, Z], axis=-1).reshape(-1, 4)


def build_coord_batch(
    grid_spec: GridSpec,
    t: float | Float[Array, ""] = 0.0,
) -> Float[Array, "N 4"]:
    """Build a flat batch of spacetime 4-vectors from a GridSpec.

    Compiled so uniform-grid coordinates are constant-folded by XLA exactly as
    they were when the whole grid evaluation was one jitted function. ``t``
    is passed as a float64 array so the compiled builder is cached per grid
    spec, not per time value.

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
    return _build_coord_batch(grid_spec, jnp.asarray(t, dtype=jnp.float64))


def _make_point_fn(metric_fn, compute_invariants: bool) -> Callable:
    """Build the per-point function (with or without invariants)."""
    if compute_invariants:

        def point_fn(c: Float[Array, "4"]) -> GridCurvatureResult:
            result = compute_curvature_chain(metric_fn, c)
            K = kretschmann_scalar(result.riemann, result.metric, result.metric_inv)
            R2 = ricci_squared(result.ricci, result.metric_inv)
            W2 = weyl_squared(
                result.riemann, result.ricci, result.ricci_scalar, result.metric, result.metric_inv
            )
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
def evaluate_curvature_points(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "N 4"],
    *,
    batch_size: int | None = None,
    compute_invariants: bool = True,
) -> GridCurvatureResult | CurvatureResult:
    """Evaluate the curvature chain on an explicit batch of 4-coordinates.

    :func:`evaluate_curvature_grid` requires a Cartesian-product grid, which
    cannot express a curvilinear sampling. An axisymmetric ``(r, mu)`` sweep,
    for instance, needs ``x = r*mu`` and ``y = r*sqrt(1-mu^2)``, not a product
    of per-axis coordinate arrays. This is the underlying primitive; the grid
    entry point flattens onto it.

    Parameters
    ----------
    coords : ``(N, 4)`` array of ``(t, x, y, z)`` rows.

    Returns
    -------
    Fields of shape ``(N, ...)``.
    """
    point_fn = _make_point_fn(metric_fn, compute_invariants)
    if batch_size is not None:
        return lax.map(point_fn, coords, batch_size=batch_size)
    return jax.vmap(point_fn)(coords)


@eqx.filter_jit
def _stress_energy_points(
    metric_fn: Callable[[Float[Array, "4"]], Float[Array, "4 4"]],
    coords: Float[Array, "N 4"],
    *,
    batch_size: int | None = None,
) -> tuple[Float[Array, "N 4 4"], Float[Array, "N 4 4"], Float[Array, "N 4 4"]]:
    """``(stress_energy, metric, metric_inv)`` on a coordinate batch.

    The selection happens inside the compiled function, so the Riemann and
    Christoffel fields never leave the trace (21 GiB at 192^3).
    """

    def point_fn(c):
        r = compute_curvature_chain(metric_fn, c)
        return r.stress_energy, r.metric, r.metric_inv

    if batch_size is not None:
        return lax.map(point_fn, coords, batch_size=batch_size)
    return jax.vmap(point_fn)(coords)


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

    Builds the ``(N, 4)`` coordinate batch and hands it to the compiled
    :func:`evaluate_curvature_points`, so ``t`` and the grid bounds are data
    to the curvature kernel: its compiled cache is keyed only on the metric
    statics, ``N``, ``batch_size`` and ``compute_invariants``. Results are reshaped back to
    ``(*grid_shape, ...)``.

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
            f"auto_chunk_threshold must be a positive integer or None, got {auto_chunk_threshold}"
        )

    coords_flat = build_coord_batch(grid_spec, t=t)
    if (
        batch_size is None
        and auto_chunk_threshold is not None
        and coords_flat.shape[0] > auto_chunk_threshold
    ):
        batch_size = auto_chunk_threshold
    results_flat = evaluate_curvature_points(
        metric_fn, coords_flat, batch_size=batch_size, compute_invariants=compute_invariants
    )
    return jax.tree.map(
        lambda a: a.reshape(*grid_spec.shape, *a.shape[1:]),
        results_flat,
    )
