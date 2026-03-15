"""2-level AMR wall-refined grid .

Returns a :class:`RefinedGrid` NamedTuple containing:

- ``base``: coarse :class:`GridSpec` over the full bounds
- ``fine``: fine :class:`GridSpec` over the masked sub-region
- ``mask``: boolean array (coarse shape) selecting which base cells are
  covered by the fine patch

Scope (per :file:`):

- 2-level only (coarse + fine pair).
- True dynamic tree-based AMR  deferred.

Composition: Downstream consumers (``verify_grid``, ``curvature_chain``)
evaluate ``.base`` and ``.fine`` independently, then blend via ``.mask``
(coarse cells in mask override base result with upsampled fine result).
The blending glue lives in the existing
``energy_conditions.verifier`` path - this module only generates the
coarse/fine pair.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from warpax.geometry import GridSpec
from warpax.geometry.metric import MetricSpecification

__all__ = ["RefinedGrid", "wall_refined"]


class RefinedGrid(NamedTuple):
    """2-level AMR composite: coarse base + fine patch + selection mask.

    Attributes
    ----------
    base : GridSpec
        Coarse grid over the full bounds.
    fine : GridSpec
        Fine grid over the masked sub-region.
    mask : Bool[Array, "*base.shape"]
        Which base cells are covered by ``fine`` (used by downstream blender).
    """

    base: GridSpec
    fine: GridSpec
    mask: Bool[Array, "..."]


def _meshgrid_flat(
    bounds: tuple[tuple[float, float], ...], shape: tuple[int, ...]
) -> Float[Array, "N 4"]:
    """Return flattened spatial coords shape ``(N, 4)`` for a given bounds+shape."""
    axes = [jnp.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, shape)]
    grids = jnp.meshgrid(*axes, indexing="ij")
    t_axis = jnp.zeros_like(grids[0])
    stacked = jnp.stack(
        [t_axis.ravel()] + [g.ravel() for g in grids], axis=-1
    )
    return stacked


def _bounds_from_mask(
    mask_flat: Bool[Array, "N"],
    coords_flat: Float[Array, "N 4"],
    default_bounds: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Compute tight axis-aligned bounding box around the masked subregion."""
    if not bool(jnp.any(mask_flat)):
        return default_bounds
    selected = coords_flat[mask_flat]
    new_bounds: list[tuple[float, float]] = []
    for axis_idx in range(1, 4):  # skip t
        lo = float(jnp.min(selected[:, axis_idx]))
        hi = float(jnp.max(selected[:, axis_idx]))
        # Guard against degenerate bounds
        if hi == lo:
            base_lo, base_hi = default_bounds[axis_idx - 1]
            span = 0.05 * (base_hi - base_lo)
            lo, hi = lo - span, hi + span
        new_bounds.append((lo, hi))
    return tuple(new_bounds)


def wall_refined(
    metric: MetricSpecification,
    bounds: tuple[tuple[float, float], ...],
    shape: tuple[int, ...],
    refine_where: Callable[[Float[Array, "N 4"]], Bool[Array, "N"]],
    refine_factor: int = 2,
) -> RefinedGrid:
    """Build a 2-level AMR wall-refined grid.

    Parameters
    ----------
    metric : MetricSpecification
        Source metric (passed through for downstream use; not required here).
    bounds : tuple of (lo, hi) pairs
        Spatial bounds for the coarse base grid.
    shape : tuple of int
        Coarse base shape.
    refine_where : callable
        Mask function ``(coords_flat: (N, 4)) -> (N,) bool``.
    refine_factor : int, default 2
        Fine grid resolution multiplier per axis.

    Returns
    -------
    RefinedGrid
        NamedTuple with ``.base``, ``.fine``, ``.mask``.
    """
    del metric  # accepted for symmetry with wall_clustered; unused here
    base = GridSpec(bounds=list(bounds), shape=shape)

    coords_flat = _meshgrid_flat(bounds, shape)
    mask_flat = refine_where(coords_flat)
    mask = mask_flat.reshape(shape)

    fine_bounds = _bounds_from_mask(mask_flat, coords_flat, bounds)
    fine_shape = tuple(n * refine_factor for n in shape)

    fine = GridSpec(bounds=list(fine_bounds), shape=fine_shape)

    return RefinedGrid(base=base, fine=fine, mask=mask)
