"""JAX-native tensor field and grid types as Equinox modules.

Layout convention: (*grid_shape, 4, 4, ...) grid indices first, tensor indices trailing.
This enables efficient vectorized operations and einsum contractions.

These types are registered as JAX pytrees via ``eqx.Module``, making them
compatible with ``jax.jit``, ``jax.vmap``, and other JAX transformations.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

# ---------------------------------------------------------------------------
# TensorField
# ---------------------------------------------------------------------------


class TensorField(eqx.Module):
    """A tensor field defined on a computational grid (or at a single point).

    Parameters
    ----------
    components : Float[Array, "..."]
        Array of shape ``(*grid_shape, 4, 4, ...)`` with *rank* trailing
        tensor indices.
    rank : int
        Number of tensor indices (e.g. 2 for a (0,2)-tensor).
    index_positions : str
        String of ``'u'`` (upper / contravariant) and ``'d'`` (lower /
        covariant) describing each index.  E.g. ``'dd'`` for *g_{ab}*,
        ``'udd'`` for Gamma^a_{bc}.  Defaults to all-lower (``'d' * rank``).
    """

    components: Float[Array, "..."]
    rank: int = eqx.field(static=True)
    index_positions: str = eqx.field(static=True, default="")

    def __check_init__(self) -> None:
        """Validate and default *index_positions* after frozen init."""
        if not self.index_positions:
            # Module is frozen; use object.__setattr__ for the default.
            object.__setattr__(self, "index_positions", "d" * self.rank)
        if len(self.index_positions) != self.rank:
            raise ValueError(
                f"index_positions length {len(self.index_positions)} != rank {self.rank}"
            )

    # derived properties --------------------------------------------------

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Shape of the spatial grid dimensions."""
        return (
            self.components.shape[: -self.rank]
            if self.rank > 0
            else self.components.shape
        )

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        """Shape of the trailing tensor index dimensions."""
        return self.components.shape[-self.rank :] if self.rank > 0 else ()


# ---------------------------------------------------------------------------
# GridSpec
# ---------------------------------------------------------------------------


class GridSpec(eqx.Module):
    """Specification for a computational grid in 3D spatial coordinates.

    All fields are static (pure metadata, no dynamic JAX arrays), so a
    ``GridSpec`` can be used freely as a closed-over constant inside
    ``jax.jit``-compiled functions.

    Parameters
    ----------
    bounds : list of (float, float)
        ``(min, max)`` for each spatial dimension ``[x, y, z]``.
    shape : tuple of int
        Number of grid points in each dimension ``(Nx, Ny, Nz)``.
    """

    bounds: list = eqx.field(static=True)
    shape: tuple = eqx.field(static=True)

    # derived properties --------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.shape)

    @property
    def spacing(self) -> list[float]:
        """Grid spacing in each dimension."""
        return [
            (b[1] - b[0]) / (n - 1) if n > 1 else 0.0
            for b, n in zip(self.bounds, self.shape)
        ]

    @property
    def axes(self) -> list[jnp.ndarray]:
        """1-D coordinate arrays for each dimension (JAX float64)."""
        return [
            jnp.linspace(b[0], b[1], n)
            for b, n in zip(self.bounds, self.shape)
        ]

    @property
    def meshgrid(self) -> list[jnp.ndarray]:
        """3-D meshgrid arrays (``indexing='ij'``, JAX float64)."""
        return list(jnp.meshgrid(*self.axes, indexing="ij"))

    @property
    def coordinate_fields(self) -> list[jnp.ndarray]:
        """4-D coordinate fields ``[t=0, x, y, z]`` on the grid."""
        grids = self.meshgrid
        t = jnp.zeros_like(grids[0])
        return [t, *grids]
