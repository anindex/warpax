"""FrameData-to-Manim Surface bridge for 2+1D embedding diagrams.

Converts an equatorial slice of a FrameData scalar field into a Manim
``Surface`` where the z-coordinate encodes the field value (warped
embedding diagram).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from manim import Surface, ThreeDAxes
from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from warpax.visualization.common._frame_data import FrameData


def framedata_to_surface(
    frame: FrameData,
    warp_field: str,
    axes: ThreeDAxes,
    *,
    slice_idx: int | None = None,
    exaggeration: float | None = None,
    resolution: tuple[int, int] | None = None,
) -> Surface:
    """Convert a FrameData equatorial slice to a Manim Surface (embedding diagram).

    Parameters
    ----------
    frame : FrameData
        Frozen snapshot with spatial coordinates and scalar fields.
    warp_field : str
        Key into ``frame.scalar_fields`` for the z-displacement field.
    axes : ThreeDAxes
        Manim axes used for coordinate conversion and color mapping.
    slice_idx : int, optional
        Index along the z-axis for the equatorial slice.  Defaults to
        ``frame.grid_shape[2] // 2`` (z-midplane).
    exaggeration : float, optional
        Vertical exaggeration factor. If *None*, auto-computed as
        ``0.3 * extent / max(|warp|)``.
    resolution : tuple[int, int], optional
        Manim surface resolution ``(u_res, v_res)``.  Defaults to
        ``(min(Nx-1, 32), min(Ny-1, 32))``.

    Returns
    -------
    Surface
        Manim Surface with z = warp_field * exaggeration, colored by z-value.
    """
    if slice_idx is None:
        slice_idx = frame.grid_shape[2] // 2

    # Extract 2D equatorial slice
    x_2d = frame.x[:, :, slice_idx]  # (Nx, Ny)
    y_2d = frame.y[:, :, slice_idx]
    warp_2d = frame.scalar_fields[warp_field][:, :, slice_idx]

    # 1D coordinate vectors for the interpolator
    x_1d = x_2d[:, 0]
    y_1d = y_2d[0, :]

    # Build interpolator from discrete grid
    interp = RegularGridInterpolator(
        (x_1d, y_1d),
        warp_2d,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # Auto-scale exaggeration if not provided
    max_warp = float(np.max(np.abs(warp_2d)))
    extent = max(float(x_1d[-1] - x_1d[0]), float(y_1d[-1] - y_1d[0]))
    if exaggeration is None:
        exaggeration = 0.3 * extent / max(max_warp, 1e-15)

    # Determine resolution
    if resolution is None:
        resolution = (min(len(x_1d) - 1, 32), min(len(y_1d) - 1, 32))

    u_min, u_max = float(x_1d[0]), float(x_1d[-1])
    v_min, v_max = float(y_1d[0]), float(y_1d[-1])

    # Parametric function: (u, v) -> axes.c2p(u, v, z * exaggeration)
    _exag = exaggeration  # capture for closure

    def param_func(u: float, v: float) -> np.ndarray:
        z_val = float(interp(np.array([[u, v]])))
        return axes.c2p(u, v, z_val * _exag)

    surface = Surface(
        param_func,
        u_range=(u_min, u_max),
        v_range=(v_min, v_max),
        resolution=resolution,
        fill_opacity=0.15,
    )
    # Add visible wireframe grid lines so curvature evolution is readable
    # through the translucent fill
    surface.set_stroke(width=1.0, opacity=0.6)

    # Color by z-axis value (field magnitude)
    clim = frame.clim.get(warp_field)
    if clim is not None:
        vmin, vmax = clim
    else:
        vmin = float(np.nanmin(warp_2d))
        vmax = float(np.nanmax(warp_2d))

    # 5-stop RdBu_r colorscale for smoother gradients
    vmin_scaled = vmin * _exag
    vmax_scaled = vmax * _exag

    if vmax_scaled <= 0:
        # All non-positive: blue tones (cool end of RdBu_r)
        colorscale = [
            ("#2166AC", vmin_scaled),
            ("#67A9CF", vmin_scaled * 0.5),
            ("#D1E5F0", vmin_scaled * 0.15),
            ("#E8EFF5", (vmin_scaled + vmax_scaled) / 2),
            ("#F7F7F7", vmax_scaled),
        ]
    elif vmin_scaled >= 0:
        # All non-negative: red tones (warm end of RdBu_r)
        colorscale = [
            ("#F7F7F7", vmin_scaled),
            ("#FDDBC7", vmax_scaled * 0.15),
            ("#EF8A62", vmax_scaled * 0.5),
            ("#D6604D", vmax_scaled * 0.75),
            ("#B2182B", vmax_scaled),
        ]
    else:
        # Diverging around zero 5 stops for smooth gradient
        colorscale = [
            ("#2166AC", vmin_scaled),
            ("#67A9CF", vmin_scaled * 0.4),
            ("#F7F7F7", 0.0),
            ("#EF8A62", vmax_scaled * 0.4),
            ("#B2182B", vmax_scaled),
        ]

    surface.set_fill_by_value(axes=axes, colorscale=colorscale, axis=2)

    return surface
