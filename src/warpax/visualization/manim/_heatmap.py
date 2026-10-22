"""FrameData-to-Manim flat colored slab (equatorial heatmap).

Creates a Manim ``Surface`` whose vertices are first lifted to ``z = value`` so
``set_fill_by_value`` can bake per-vertex colors, then collapsed to a flat
plane positioned at ``z_offset``. The result is a color-mapped slab: its
color encodes the scalar, its height carries no information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from manim import Surface, ThreeDAxes

from warpax.visualization.manim._image_utils import NEC_DEPTH_COLORS, bilinear_sampler, color_stops
from warpax.visualization.manim._surface import symlog_height

if TYPE_CHECKING:
    from warpax.visualization.common._frame_data import FrameData

_INFERNO_COLORS = ("#000004", "#420A68", "#932567", "#DD513A", "#FCA50A")


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
        return color_stops(NEC_DEPTH_COLORS, vmin, vmax if vmax > vmin else vmin + 1.0)
    if colormap == "inferno":
        return color_stops(_INFERNO_COLORS, vmin, vmax)
    if vmax <= 0:
        return color_stops(("#2166AC", "#67A9CF", "#D1E5F0", "#E8EFF5", "#F7F7F7"), vmin, vmax)
    if vmin >= 0:
        return color_stops(("#F7F7F7", "#FDDBC7", "#EF8A62", "#D6604D", "#B2182B"), vmin, vmax)
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
    linthresh: float | None = None,
) -> Surface:
    """Convert a FrameData equatorial slice to a flat colored slab (heatmap).

    The surface is constructed with z = field_value so that
    ``set_fill_by_value`` can map the scalar data to colors via ``axis=2``;
    the relief is then collapsed to a flat plane at ``z_offset`` with the
    colors already baked. Field values are clipped to ``[vmin, vmax]`` so
    outliers (such as the WEC margin at high rapidity) cannot push vertices
    off-screen.

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
    linthresh : float, optional
        If given, the color ramp is :func:`symlog_height` of the field
        rather than the field itself. Use it when one color scale has to
        serve frames spanning more than about a decade, where a linear ramp
        leaves every frame but the most extreme one a single flat color.

    Returns
    -------
    Surface
        Flat Manim Surface colored by the scalar field, positioned at
        *z_offset*.
    """
    if slice_idx is None:
        slice_idx = frame.grid_shape[2] // 2

    x_2d = frame.x[:, :, slice_idx]
    y_2d = frame.y[:, :, slice_idx]
    color_2d = np.asarray(frame.scalar_fields[color_field][:, :, slice_idx])
    if linthresh is not None:
        color_2d = symlog_height(color_2d, linthresh)

    x_1d = x_2d[:, 0]
    y_1d = y_2d[0, :]
    sample = bilinear_sampler(x_1d, y_1d, color_2d)

    if resolution is None:
        resolution = (min(len(x_1d) - 1, 32), min(len(y_1d) - 1, 32))

    u_min, u_max = float(x_1d[0]), float(x_1d[-1])
    v_min, v_max = float(y_1d[0]), float(y_1d[-1])

    clim = frame.clim.get(color_field)
    if clim is not None:
        vmin, vmax = clim
        if linthresh is not None:
            vmin, vmax = (float(v) for v in symlog_height(np.array([vmin, vmax]), linthresh))
    else:
        vmin = float(np.nanmin(color_2d))
        vmax = float(np.nanmax(color_2d))

    def param_func(u: float, v: float) -> np.ndarray:
        z_val = max(vmin, min(vmax, sample(u, v)))
        return axes.c2p(u, v, z_val)

    surface = Surface(
        param_func,
        u_range=(u_min, u_max),
        v_range=(v_min, v_max),
        resolution=resolution,
        fill_opacity=0.85,
    )
    surface.set_fill_by_value(axes=axes, colorscale=_build_colorscale(vmin, vmax, colormap), axis=2)

    target_z = float(axes.c2p(0.0, 0.0, z_offset)[2])
    surface.apply_function(lambda p: np.array([p[0], p[1], target_z], dtype=float))
    return surface
