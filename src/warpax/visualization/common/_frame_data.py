"""FrameData: frozen snapshot bridging JAX computation to visualization backends.

FrameData is an Equinox module that holds pure NumPy arrays (never JAX arrays)
alongside static metadata describing the metric, velocity, grid, and rendering
hints. It is the boundary object: everything upstream is JAX, everything
downstream is NumPy/Manim.

Notes:
- Scalar fields stored as ``dict[str, np.ndarray]`` for extensibility.
- HE type stored as integer-valued float64 array (key ``"he_type"``).
- Rendering hints (colormaps, clim, isosurface values) computed at freeze time.
- No serialization FrameData is ephemeral (recomputed each run).
- All evaluation is eager (no lazy computation).
"""
from __future__ import annotations

import numpy as np

import equinox as eqx


class FrameData(eqx.Module):
    """Frozen snapshot of JAX computation results for visualization rendering.

    All array fields are NumPy (not JAX) arrays. This is the boundary
    between JAX computation and visualization rendering no JAX tracing
    occurs downstream of FrameData construction.

    Parameters
    ----------
    x, y, z : np.ndarray
        Spatial coordinate arrays, each with shape ``grid_shape``.
    scalar_fields : dict[str, np.ndarray]
        Named scalar fields, each with shape ``grid_shape``.
        Extensible: new fields are added by key as needed.
    metric_name : str
        Name of the warp metric (e.g. ``"alcubierre"``).
    v_s : float
        Warp bubble velocity parameter.
    grid_shape : tuple[int, ...]
        Shape of the spatial grid ``(Nx, Ny, Nz)``.
    t : float
        Time coordinate of the snapshot (default 0.0).
    colormaps : dict[str, str]
        Suggested colormap name per scalar field.
    clim : dict[str, tuple[float, float]]
        Suggested color limits ``(vmin, vmax)`` per scalar field.
    isosurface_values : dict[str, list[float]]
        Suggested isosurface threshold values per scalar field.
    """

    # Spatial coordinates: three 3D arrays from GridSpec.meshgrid
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    # Scalar fields: name -> (Nx, Ny, Nz) NumPy array
    scalar_fields: dict[str, np.ndarray]

    # Metric identity metadata (static)
    metric_name: str = eqx.field(static=True)
    v_s: float = eqx.field(static=True)
    grid_shape: tuple[int, ...] = eqx.field(static=True)
    t: float = eqx.field(static=True, default=0.0)

    # Rendering hints (static)
    colormaps: dict[str, str] = eqx.field(static=True, default_factory=dict)
    clim: dict[str, tuple[float, float]] = eqx.field(static=True, default_factory=dict)
    isosurface_values: dict[str, list[float]] = eqx.field(static=True, default_factory=dict)

    def __check_init__(self) -> None:
        """Validate arrays after frozen init."""
        import jax as _jax

        # Validate coordinate shapes
        for name, arr in [("x", self.x), ("y", self.y), ("z", self.z)]:
            if arr.shape != self.grid_shape:
                raise ValueError(
                    f"Coordinate '{name}' shape {arr.shape} != grid_shape {self.grid_shape}"
                )
            if isinstance(arr, _jax.Array):
                raise TypeError(
                    f"Coordinate '{name}' is a jax.Array must be np.ndarray. "
                    "Use np.asarray() to convert."
                )

        # Validate scalar fields
        for field_name, arr in self.scalar_fields.items():
            if arr.shape != self.grid_shape:
                raise ValueError(
                    f"Scalar field '{field_name}' shape {arr.shape} "
                    f"!= grid_shape {self.grid_shape}"
                )
            if isinstance(arr, _jax.Array):
                raise TypeError(
                    f"Scalar field '{field_name}' is a jax.Array must be np.ndarray. "
                    "Use np.asarray() to convert."
                )

    @property
    def field_names(self) -> list[str]:
        """Names of all scalar fields."""
        return list(self.scalar_fields.keys())

    def with_clim(
        self,
        field: str,
        clim: tuple[float, float],
    ) -> "FrameData":
        """Return a shallow copy with a single ``clim`` entry overridden.

        Parameters
        ----------
        field : str
            Scalar field name whose color limits should be updated.
        clim : tuple[float, float]
            New ``(vmin, vmax)`` color limits for *field*.

        Returns
        -------
        FrameData
            New FrameData identical to *self* except for the updated
            ``clim[field]`` entry.
        """
        new_clim = dict(self.clim)
        new_clim[field] = clim
        return FrameData(
            x=self.x,
            y=self.y,
            z=self.z,
            scalar_fields=self.scalar_fields,
            metric_name=self.metric_name,
            v_s=self.v_s,
            grid_shape=self.grid_shape,
            t=self.t,
            colormaps=self.colormaps,
            clim=new_clim,
            isosurface_values=self.isosurface_values,
        )

