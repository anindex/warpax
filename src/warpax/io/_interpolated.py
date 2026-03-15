"""InterpolatedADMMetric - ADM 3+1 metric backed by interpolated 4D grids.

Shared base class for the three external-metric loaders:

- :func:`warpax.io.load_warpfactory` 
- :func:`warpax.io.load_einfield` 
- :func:`warpax.io.load_cactus_slice` 

Option C architecture per
Private module; consumers should import from :mod:`warpax.io`.
"""
from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from warpax.geometry import ADMMetric, GridSpec

from ._interpolation import (
    _interpolate_scalar,
    _interpolate_tensor,
    _interpolate_vector,
)

__all__ = ["InterpolatedADMMetric"]


class InterpolatedADMMetric(ADMMetric):
    """ADM 3+1 metric defined by interpolated grids on a regular 4D spacetime lattice.

    Parameters
    ----------
    alpha_grid : Float[Array, "Nt Nx Ny Nz"]
        Lapse function on a regular 4D grid.
    beta_grid : Float[Array, "Nt Nx Ny Nz 3"]
        Shift vector components on the same grid.
    gamma_grid : Float[Array, "Nt Nx Ny Nz 3 3"]
        Spatial metric on the same grid.
    grid_spec : GridSpec
        4D bounds + shape (static; JIT cache key per ).
    name : str, default "interpolated"
        Human-readable name (typically derived from source filename by loader).
    interp_method : {"linear", "cubic"}, default "cubic"
        Interpolation scheme. Default "cubic" for Christoffel autodiff
        smoothness. Internally both map to ``order=1`` in
        ``jax.scipy.ndimage.map_coordinates`` (cubic is preserved as a
        forward-compatible option for when JAX adds native cubic support).

    Notes
    -----
    - :meth:`symbolic` raises :class:`NotImplementedError` (no closed-form
      representation for externally-loaded metrics).
    - :meth:`shape_function_value` raises :class:`NotImplementedError`
      (shape function is not inferable from grid data).
    - Out-of-bounds coordinates clip to grid edges via
      ``map_coordinates(mode="nearest")`` - no NaN, no raise; JIT-safe.
    """

    alpha_grid: Float[Array, "Nt Nx Ny Nz"]
    beta_grid: Float[Array, "Nt Nx Ny Nz 3"]
    gamma_grid: Float[Array, "Nt Nx Ny Nz 3 3"]
    grid_spec: GridSpec = eqx.field(static=True)
    _name: str = eqx.field(static=True)
    interp_method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        alpha_grid: Float[Array, "Nt Nx Ny Nz"],
        beta_grid: Float[Array, "Nt Nx Ny Nz 3"],
        gamma_grid: Float[Array, "Nt Nx Ny Nz 3 3"],
        grid_spec: GridSpec,
        name: str = "interpolated",
        interp_method: str = "cubic",
    ) -> None:
        self.alpha_grid = alpha_grid
        self.beta_grid = beta_grid
        self.gamma_grid = gamma_grid
        self.grid_spec = grid_spec
        self._name = name
        self.interp_method = interp_method

    def __check_init__(self) -> None:
        """Validate shape consistency across the three grids."""
        if self.alpha_grid.shape[:4] != self.beta_grid.shape[:4]:
            raise ValueError(
                f"alpha_grid shape {self.alpha_grid.shape[:4]} does not match "
                f"beta_grid shape prefix {self.beta_grid.shape[:4]}"
            )
        if self.alpha_grid.shape[:4] != self.gamma_grid.shape[:4]:
            raise ValueError(
                f"alpha_grid shape {self.alpha_grid.shape[:4]} does not match "
                f"gamma_grid shape prefix {self.gamma_grid.shape[:4]}"
            )
        if self.beta_grid.shape[-1] != 3:
            raise ValueError(
                f"beta_grid must have trailing dim 3; got shape "
                f"{self.beta_grid.shape}"
            )
        if self.gamma_grid.shape[-2:] != (3, 3):
            raise ValueError(
                f"gamma_grid must have trailing dims (3, 3); got shape "
                f"{self.gamma_grid.shape}"
            )
        if self.interp_method not in ("linear", "cubic"):
            raise ValueError(
                f"interp_method must be 'linear' or 'cubic'; got "
                f"{self.interp_method!r}"
            )

    def lapse(self, coords: Float[Array, "4"]) -> Float[Array, ""]:
        """Lapse :math:`\\alpha(t, \\vec{x})` via grid interpolation."""
        return _interpolate_scalar(
            self.alpha_grid, coords, self.grid_spec, self.interp_method
        )

    def shift(self, coords: Float[Array, "4"]) -> Float[Array, "3"]:
        """Shift 3-vector :math:`\\beta^i(t, \\vec{x})` via grid interpolation."""
        return _interpolate_vector(
            self.beta_grid, coords, self.grid_spec, self.interp_method
        )

    def spatial_metric(self, coords: Float[Array, "4"]) -> Float[Array, "3 3"]:
        """Spatial metric :math:`\\gamma_{ij}(t, \\vec{x})` via grid interpolation."""
        return _interpolate_tensor(
            self.gamma_grid, coords, self.grid_spec, self.interp_method
        )

    def symbolic(self):  # type: ignore[override]
        """Raise :class:`NotImplementedError` - no closed form for external data."""
        raise NotImplementedError(
            "InterpolatedADMMetric has no closed-form symbolic representation. "
            "Cross-validation against SymPy is unavailable for "
            "externally-loaded metrics."
        )

    def shape_function_value(
        self, coords: Float[Array, "4"]
    ) -> Float[Array, ""]:
        """Raise :class:`NotImplementedError` - shape function not inferable."""
        raise NotImplementedError(
            "Shape function for externally-loaded metric is not inferable "
            "from grid data. Wall-restricted diagnostics require manual f(r) "
            "specification."
        )

    def name(self) -> str:
        """Human-readable metric name (typically derived from source filename)."""
        return self._name
