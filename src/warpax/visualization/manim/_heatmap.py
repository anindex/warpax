"""FrameData-to-Manim flat colored slab (equatorial heatmap).

Creates a Manim ``Surface`` whose vertices are first lifted to ``z = value`` so
``set_fill_by_value`` can bake per-vertex colors, then (when ``flat=True``)
collapsed to a genuinely flat plane and positioned at ``z_offset``. The result
is a color-mapped slab: its colour encodes the scalar, its height
carries no information (it is not a value relief).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from manim import Surface, ThreeDAxes

from warpax.visualization.manim._image_utils import bilinear_sampler

if TYPE_CHECKING:
    from warpax.visualization.common._frame_data import FrameData


def _build_colorscale(
    vmin: float,
    vmax: float,
    colormap: str,
) -> list[tuple[str, float]]:
    """Build a 5-stop colorscale for ``set_fill_by_value``.

    Parameters
    ----------
    vmin, vmax : float
        Data limits.
    colormap : str
        ``"nec_depth"`` (one-sided violation depth), ``"inferno"``
        (sequential), or the default ``"RdBu_r"`` (diverging).
    """
    if colormap == "nec_depth":
        # One-sided sequential "violation depth": bright blue (deepest, vmin)
        # -> dark (marginal, vmax=0). Honest for a strictly-non-positive field
        # (no diverging "satisfied" half that the data never reaches).
        span = (vmax - vmin) if vmax > vmin else 1.0
        return [
            ("#CFE0FF", vmin),
            ("#7FA0FF", vmin + span * 0.25),
            ("#3B4CC0", vmin + span * 0.5),
            ("#27306E", vmin + span * 0.75),
            ("#15151F", vmax),
        ]
    if colormap == "inferno":
        # Sequential dark-to-bright
        span = vmax - vmin
        return [
            ("#000004", vmin),
            ("#420A68", vmin + span * 0.25),
            ("#932567", vmin + span * 0.5),
            ("#DD513A", vmin + span * 0.75),
            ("#FCA50A", vmax),
        ]
    else:
        # Default: RdBu_r diverging (5 stops)
        if vmax <= 0:
            return [
                ("#2166AC", vmin),
                ("#67A9CF", vmin * 0.5),
                ("#D1E5F0", vmin * 0.15),
                ("#E8EFF5", (vmin + vmax) / 2),
                ("#F7F7F7", vmax),
            ]
        elif vmin >= 0:
            return [
                ("#F7F7F7", vmin),
                ("#FDDBC7", vmax * 0.15),
                ("#EF8A62", vmax * 0.5),
                ("#D6604D", vmax * 0.75),
                ("#B2182B", vmax),
            ]
        else:
            return [
                ("#2166AC", vmin),
                ("#67A9CF", vmin * 0.4),
                ("#F7F7F7", 0.0),
                ("#EF8A62", vmax * 0.4),
                ("#B2182B", vmax),
            ]


def framedata_to_heatmap(
    frame: FrameData,
    color_field: str,
    axes: ThreeDAxes,
    *,
    slice_idx: int | None = None,
    z_offset: float = 0.0,
    resolution: tuple[int, int] | None = None,
    colormap: str = "RdBu_r",
    flat: bool = True,
) -> Surface:
    """Convert a FrameData equatorial slice to a flat colored slab (heatmap).

    The surface is constructed with z = field_value so that
    ``set_fill_by_value`` can map the scalar data to colors via ``axis=2``.
    When ``flat`` is true (default) the value-relief is then collapsed to a
    genuinely flat plane positioned at ``z_offset`` -- the colours are already
    baked, so the slab encodes the scalar by colour alone and its height carries
    no information. With ``flat=False`` the legacy value-relief is shifted to
    ``z_offset`` instead.

    Parameters
    ----------
    frame : FrameData
        Frozen snapshot with spatial coordinates and scalar fields.
    color_field : str
        Key into ``frame.scalar_fields`` for the color-mapping field.
    axes : ThreeDAxes
        Manim axes used for coordinate conversion and color mapping.
    slice_idx : int, optional
        Index along the z-axis for the equatorial slice. Defaults to
        ``frame.grid_shape[2] // 2`` (z-midplane).
    z_offset : float, optional
        Visual z-position of the heatmap plane (default 0.0).
    resolution : tuple[int, int], optional
        Manim surface resolution ``(u_res, v_res)``. Defaults to
        ``(min(Nx-1, 32), min(Ny-1, 32))``.
    colormap : str, optional
        Color scale name: ``"RdBu_r"`` (default diverging blue-red),
        ``"nec_depth"`` (one-sided violation depth for EC fields), or
        ``"inferno"`` (sequential).

    Returns
    -------
    Surface
        Flat Manim Surface colored by the scalar field, positioned at
        *z_offset*.
    """
    if slice_idx is None:
        slice_idx = frame.grid_shape[2] // 2

    # Extract 2D equatorial slice
    x_2d = frame.x[:, :, slice_idx]
    y_2d = frame.y[:, :, slice_idx]
    color_2d = frame.scalar_fields[color_field][:, :, slice_idx]

    # 1D coordinate vectors
    x_1d = x_2d[:, 0]
    y_1d = y_2d[0, :]

    # Pure-numpy bilinear sampler (scipy's RegularGridInterpolator segfaults
    # under the repeated pointwise calls a 3D movie render makes on Python 3.14)
    sample = bilinear_sampler(x_1d, y_1d, color_2d)

    # Determine resolution
    if resolution is None:
        resolution = (min(len(x_1d) - 1, 32), min(len(y_1d) - 1, 32))

    u_min, u_max = float(x_1d[0]), float(x_1d[-1])
    v_min, v_max = float(y_1d[0]), float(y_1d[-1])

    # Color limits from FrameData hints or data
    clim = frame.clim.get(color_field)
    if clim is not None:
        vmin, vmax = clim
    else:
        vmin = float(np.nanmin(color_2d))
        vmax = float(np.nanmax(color_2d))

    # Parametric function: z encodes field value for color mapping.
    # Clip to [vmin, vmax] so extreme outliers don't push vertices
    # off-screen (e.g. WEC margin at high rapidity where cosh²(ζ)
    # amplifies values by orders of magnitude).
    _clip_lo, _clip_hi = vmin, vmax

    def param_func(u: float, v: float) -> np.ndarray:
        z_val = max(_clip_lo, min(_clip_hi, sample(u, v)))
        return axes.c2p(u, v, z_val)

    surface = Surface(
        param_func,
        u_range=(u_min, u_max),
        v_range=(v_min, v_max),
        resolution=resolution,
        fill_opacity=0.85,
    )

    # Build colorscale based on requested colormap
    colorscale = _build_colorscale(vmin, vmax, colormap)

    surface.set_fill_by_value(axes=axes, colorscale=colorscale, axis=2)

    if flat:
        # Colours are baked; collapse the value-relief to a genuinely flat slab
        # so the height no longer (mis)reads as a second encoding of the field.
        # Set the scene-z of every vertex to the target plane directly -- a
        # division-free flatten. ``stretch_to_fit_depth`` would divide by the
        # relief depth, which collapses to ~0 as the field flattens (e.g. the
        # rampdown tail), yielding inf coordinates that crash the Cairo renderer.
        target_z = float(axes.c2p(0.0, 0.0, z_offset)[2])
        surface.apply_function(
            lambda p: np.array([p[0], p[1], target_z], dtype=float)
        )
    elif z_offset != 0.0:
        # Legacy value-relief: shift along the z-axis in scene coordinates.
        origin = axes.c2p(0, 0, 0)
        target = axes.c2p(0, 0, z_offset)
        shift_vec = np.array(target) - np.array(origin)
        surface.shift(shift_vec)

    return surface
