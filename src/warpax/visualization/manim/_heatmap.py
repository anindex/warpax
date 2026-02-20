"""FrameData-to-Manim flat colored surface (equatorial heatmap).

Creates a Manim ``Surface`` where the z-coordinate encodes the scalar
field value for color mapping via ``set_fill_by_value()``, then shifts
the surface to a desired visual position.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from manim import Surface, ThreeDAxes
from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from warpax.visualization.common._frame_data import FrameData


# ---------------------------------------------------------------------------
# Colorscale builders
# ---------------------------------------------------------------------------

def _build_colorscale(
    vmin: float,
    vmax: float,
    colormap: str,
) -> list[tuple[str, float]]:
    """Build a 5-stop colorscale for ``set_fill_by_value()``.

    Parameters
    ----------
    vmin, vmax : float
        Data limits.
    colormap : str
        ``"RdBu_r"`` (default diverging), ``"RdYlGn"`` (EC fields:
        red=violated, green=satisfied), or ``"inferno"`` (sequential).
    """
    if colormap == "RdYlGn":
        # Diverging red-yellow-green for energy conditions
        # Red = violated (negative), Green = satisfied (positive)
        if vmax <= 0:
            return [
                ("#A50026", vmin),
                ("#D73027", vmin * 0.5),
                ("#F46D43", vmin * 0.2),
                ("#FDAE61", (vmin + vmax) / 2),
                ("#FFFFBF", vmax),
            ]
        elif vmin >= 0:
            return [
                ("#FFFFBF", vmin),
                ("#A6D96A", vmax * 0.2),
                ("#66BD63", vmax * 0.5),
                ("#1A9850", vmax * 0.75),
                ("#006837", vmax),
            ]
        else:
            return [
                ("#A50026", vmin),
                ("#F46D43", vmin * 0.4),
                ("#FFFFBF", 0.0),
                ("#66BD63", vmax * 0.4),
                ("#006837", vmax),
            ]
    elif colormap == "inferno":
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
) -> Surface:
    """Convert a FrameData equatorial slice to a flat colored surface (heatmap).

    The surface is constructed with z = field_value so that
    ``set_fill_by_value()`` can map the scalar data to colors via
    ``axis=2``.  The surface is then visually shifted to ``z_offset``.

    Parameters
    ----------
    frame : FrameData
        Frozen snapshot with spatial coordinates and scalar fields.
    color_field : str
        Key into ``frame.scalar_fields`` for the color-mapping field.
    axes : ThreeDAxes
        Manim axes used for coordinate conversion and color mapping.
    slice_idx : int, optional
        Index along the z-axis for the equatorial slice.  Defaults to
        ``frame.grid_shape[2] // 2`` (z-midplane).
    z_offset : float, optional
        Visual z-position of the heatmap plane (default 0.0).
    resolution : tuple[int, int], optional
        Manim surface resolution ``(u_res, v_res)``.  Defaults to
        ``(min(Nx-1, 32), min(Ny-1, 32))``.
    colormap : str, optional
        Color scale name: ``"RdBu_r"`` (default diverging blue-red),
        ``"RdYlGn"`` (diverging red-yellow-green for EC fields), or
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

    # Build interpolator
    interp = RegularGridInterpolator(
        (x_1d, y_1d),
        color_2d,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

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
        z_val = float(interp(np.array([[u, v]])))
        z_val = max(_clip_lo, min(_clip_hi, z_val))
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

    # Shift to desired visual z-position
    if z_offset != 0.0:
        # Compute the shift in scene coordinates: axes.c2p gives us the
        # offset direction along the z-axis.
        origin = axes.c2p(0, 0, 0)
        target = axes.c2p(0, 0, z_offset)
        shift_vec = np.array(target) - np.array(origin)
        surface.shift(shift_vec)

    return surface
