"""FrameData-to-Manim Surface bridge for 2+1D embedding diagrams.

Converts an equatorial slice of a FrameData scalar field into a Manim
``Surface`` where the z-coordinate encodes the field value (warped
embedding diagram).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from manim import Surface, ThreeDAxes

from warpax.visualization.manim._image_utils import bilinear_sampler

if TYPE_CHECKING:
    from warpax.visualization.common._frame_data import FrameData


def symlog_height(values: np.ndarray, linthresh: float) -> np.ndarray:
    """Signed logarithmic height map, linear below *linthresh*.

    ``sign(v) * log1p(|v| / linthresh)``. Monotone and sign-preserving, so
    the ordering of the field is untouched while a dynamic range of several
    decades compresses into one a single height axis can display. Without it
    a parameter sweep whose field grows by 10^3 renders flat everywhere but
    at its own maximum.
    """
    v = np.asarray(values, dtype=float)
    return np.sign(v) * np.log1p(np.abs(v) / linthresh)


def auto_linthresh(max_abs: float, decades: float = 3.0) -> float:
    """Linear threshold placing *decades* of the field above the linear knee."""
    return max_abs / 10.0**decades if max_abs > 0.0 else 1.0


def framedata_to_surface(
    frame: FrameData,
    warp_field: str,
    axes: ThreeDAxes,
    *,
    slice_idx: int | None = None,
    exaggeration: float | None = None,
    resolution: tuple[int, int] | None = None,
    linthresh: float | None = None,
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
        Index along the z-axis for the equatorial slice. Defaults to
        ``frame.grid_shape[2] // 2`` (z-midplane).
    exaggeration : float, optional
        Vertical exaggeration factor. If *None*, auto-computed as
        ``0.3 * extent / max(|warp|)``.
    resolution : tuple[int, int], optional
        Manim surface resolution ``(u_res, v_res)``. Defaults to
        ``(min(Nx-1, 32), min(Ny-1, 32))``.
    linthresh : float, optional
        If given, the height is :func:`symlog_height` of the field rather
        than the field itself. Use it whenever one *exaggeration* has to
        serve frames spanning more than about a decade.

    Returns
    -------
    Surface
        Manim Surface with z = warp_field * exaggeration, colored by z-value.
        With *linthresh* set, z = symlog_height(warp_field) * exaggeration.
    """
    if slice_idx is None:
        slice_idx = frame.grid_shape[2] // 2

    # Extract 2D equatorial slice
    x_2d = frame.x[:, :, slice_idx]  # (Nx, Ny)
    y_2d = frame.y[:, :, slice_idx]
    warp_2d = np.asarray(frame.scalar_fields[warp_field][:, :, slice_idx])
    if linthresh is not None:
        warp_2d = symlog_height(warp_2d, linthresh)

    # 1D coordinate vectors for the interpolator
    x_1d = x_2d[:, 0]
    y_1d = y_2d[0, :]

    # Pure-numpy bilinear sampler (scipy's RegularGridInterpolator segfaults
    # under the repeated pointwise calls a 3D movie render makes on Python 3.14)
    sample = bilinear_sampler(x_1d, y_1d, warp_2d)

    # Auto-scale exaggeration if not provided
    max_warp = float(np.max(np.abs(warp_2d)))
    extent = max(float(x_1d[-1] - x_1d[0]), float(y_1d[-1] - y_1d[0]))
    if exaggeration is None:
        # Guard near-flat fields: the naive 0.3*extent/eps would explode the
        # surface off-axis. A flat field gets a neutral factor instead.
        exaggeration = 0.3 * extent / max_warp if max_warp > 1e-9 * max(extent, 1.0) else 1.0

    # Determine resolution
    if resolution is None:
        resolution = (min(len(x_1d) - 1, 32), min(len(y_1d) - 1, 32))

    u_min, u_max = float(x_1d[0]), float(x_1d[-1])
    v_min, v_max = float(y_1d[0]), float(y_1d[-1])

    # Parametric function: (u, v) -> axes.c2p(u, v, z * exaggeration)
    _exag = exaggeration  # capture for closure

    def param_func(u: float, v: float) -> np.ndarray:
        return axes.c2p(u, v, sample(u, v) * _exag)

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
        if linthresh is not None:
            vmin, vmax = (float(v) for v in symlog_height(np.array([vmin, vmax]), linthresh))
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
