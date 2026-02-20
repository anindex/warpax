"""Shared image utilities for Manim 2D scenes.

Provides functions to convert FrameData equatorial slices into RGBA uint8
arrays using matplotlib colormaps with SymLogNorm normalization, and to
extract contour paths for overlaying on heatmap scenes.

These utilities are the foundation for all 2D ``ImageMobject``-based Manim
scenes, replacing the slower ``Surface`` + ``set_fill_by_value`` pipeline
for equatorial-plane visualizations.

"""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom as _ndimage_zoom

if TYPE_CHECKING:
    from warpax.visualization.common._frame_data import FrameData


def compute_symlog_clim(
    frames: list[FrameData],
    field_name: str,
    linthresh: float = 1.0,
) -> tuple[float, float, float]:
    """Pre-compute global symmetric color limits across all frames.

    Iterates over all FrameData snapshots, extracts the equatorial slice
    (z-midplane) for *field_name*, and computes a symmetric range
    ``(-max_abs, max_abs)`` where ``max_abs = max(|vmin|, |vmax|)``
    across every frame.  This prevents per-frame flicker.

    Parameters
    ----------
    frames : list[FrameData]
        Sequence of FrameData snapshots.
    field_name : str
        Key into ``frame.scalar_fields``.
    linthresh : float
        Linear threshold for ``SymLogNorm``.  If the global maximum
        absolute value is smaller than *linthresh*, linear scaling is
        used instead (returned linthresh is set to the max absolute
        value to signal this).

    Returns
    -------
    tuple[float, float, float]
        ``(vmin, vmax, linthresh)`` symmetric color limits and the
        (possibly adjusted) linear threshold.
    """
    global_min = np.inf
    global_max = -np.inf

    for frame in frames:
        if field_name not in frame.scalar_fields:
            continue
        mid_z = frame.grid_shape[2] // 2
        data_2d = frame.scalar_fields[field_name][:, :, mid_z]
        frame_min = float(np.nanmin(data_2d))
        frame_max = float(np.nanmax(data_2d))
        if frame_min < global_min:
            global_min = frame_min
        if frame_max > global_max:
            global_max = frame_max

    # Fallback if no valid data found
    if np.isinf(global_min) or np.isinf(global_max):
        return (-1.0, 1.0, linthresh)

    max_abs = max(abs(global_min), abs(global_max))

    # If all values are near zero, use a minimal linear range
    if max_abs < linthresh:
        effective_linthresh = max(max_abs, 1e-15)
        return (-effective_linthresh, effective_linthresh, effective_linthresh)

    return (-max_abs, max_abs, linthresh)


def frame_to_rgba(
    frame: FrameData,
    field_name: str,
    global_clim: tuple[float, float, float],
    cmap_name: str = "RdBu_r",
    linthresh: float | None = None,
) -> np.ndarray:
    """Convert a FrameData equatorial slice to an RGBA uint8 array.

    Extracts the z-midplane of the named scalar field, applies
    ``matplotlib.colors.SymLogNorm`` with the provided global color
    limits, maps through the specified colormap, and returns a
    ``(Nx, Ny, 4)`` uint8 array suitable for ``ImageMobject``.

    Parameters
    ----------
    frame : FrameData
        Frozen snapshot containing spatial coordinates and scalar fields.
    field_name : str
        Key into ``frame.scalar_fields``.
    global_clim : tuple[float, float, float]
        ``(vmin, vmax, linthresh)`` as returned by
        :func:`compute_symlog_clim`.
    cmap_name : str
        Matplotlib colormap name (default ``"RdBu_r"``).
    linthresh : float or None
        Override the linear threshold from *global_clim*.  If ``None``,
        the third element of *global_clim* is used.

    Returns
    -------
    np.ndarray
        RGBA array with shape ``(Nx, Ny, 4)`` and dtype ``uint8``.
    """
    mid_z = frame.grid_shape[2] // 2
    data_2d = frame.scalar_fields[field_name][:, :, mid_z]

    vmin, vmax = global_clim[0], global_clim[1]
    if linthresh is None:
        linthresh = global_clim[2]

    # Edge case: uniform data
    if abs(vmax - vmin) < 1e-15:
        gray = np.full((*data_2d.shape, 4), 128, dtype=np.uint8)
        gray[:, :, 3] = 255  # fully opaque
        return gray

    norm = mcolors.SymLogNorm(
        linthresh=linthresh,
        vmin=vmin,
        vmax=vmax,
    )

    cmap = mcm.get_cmap(cmap_name)
    rgba_float = cmap(norm(data_2d))  # (Nx, Ny, 4) float in [0, 1]
    return (rgba_float * 255).astype(np.uint8)


def extract_zero_contour(
    data_2d: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    level: float = 0.0,
    scene_width: float | None = None,
) -> list[np.ndarray]:
    """Extract contour path vertices at a given iso-level from 2D data.

    Uses ``matplotlib.pyplot.contour()`` to find iso-level curves, then
    extracts path vertices.  Optionally rescales from data coordinates
    to Manim scene coordinates.

    Parameters
    ----------
    data_2d : np.ndarray
        2D scalar field with shape ``(Nx, Ny)``.
    x_range : tuple[float, float]
        ``(x_min, x_max)`` spatial extent.
    y_range : tuple[float, float]
        ``(y_min, y_max)`` spatial extent.
    level : float
        Iso-level to extract (default 0.0).
    scene_width : float or None
        If provided, rescale contour coordinates from data space to
        ``[-scene_width/2, scene_width/2]`` for both axes.  If ``None``,
        coordinates are returned in data space.

    Returns
    -------
    list[np.ndarray]
        List of ``(N, 2)`` arrays, one per contour segment.  Empty list
        if no contour found at the requested level.
    """
    # Bicubic upsample for smooth marching-squares paths (30×30 -> 240×240).
    # Replace NaN first - bicubic interpolation propagates even a single
    # NaN across the entire upsampled array.
    upsample_factor = 8
    data_2d = np.nan_to_num(np.asarray(data_2d, dtype=float), nan=0.0)
    data_up = _ndimage_zoom(data_2d, upsample_factor, order=3)

    fig, ax = plt.subplots()
    x_coords = np.linspace(x_range[0], x_range[1], data_up.shape[1])
    y_coords = np.linspace(y_range[0], y_range[1], data_up.shape[0])

    cs = ax.contour(x_coords, y_coords, data_up, levels=[level])
    plt.close(fig)

    paths: list[np.ndarray] = []
    # Matplotlib >= 3.8 deprecates cs.collections; use cs.get_paths() if
    # available, otherwise fall back to the legacy interface.
    if hasattr(cs, "get_paths"):
        raw_paths = cs.get_paths()
    else:
        raw_paths = []
        for collection in cs.collections:
            raw_paths.extend(collection.get_paths())

    for path in raw_paths:
        # Split path at MOVETO codes to avoid cross-bubble bridge artifacts
        codes = path.codes
        all_verts = path.vertices.copy()
        if codes is not None:
            segments: list[np.ndarray] = []
            start = 0
            for k in range(1, len(codes)):
                if codes[k] == 1:  # MOVETO
                    seg = all_verts[start:k]
                    if len(seg) >= 3:
                        segments.append(seg)
                    start = k
            seg = all_verts[start:]
            if len(seg) >= 3:
                segments.append(seg)
        else:
            segments = [all_verts] if len(all_verts) >= 3 else []

        for verts in segments:
            # Filter degenerate segments (too short arc-length)
            diffs = np.diff(verts, axis=0)
            arc_length = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))
            data_diag = np.sqrt(
                (x_range[1] - x_range[0]) ** 2 + (y_range[1] - y_range[0]) ** 2
            )
            if arc_length < data_diag * 0.02:
                continue

            if scene_width is not None:
                # Rescale x
                x_min, x_max = x_range
                verts[:, 0] = (
                    (verts[:, 0] - x_min) / (x_max - x_min) - 0.5
                ) * scene_width
                # Rescale y
                y_min, y_max = y_range
                verts[:, 1] = (
                    (verts[:, 1] - y_min) / (y_max - y_min) - 0.5
                ) * scene_width
            paths.append(verts)

    return paths


def extract_bubble_contour(
    frame: FrameData,
    shape_fn_field: str = "shape_function",
    level: float = 0.5,
    scene_width: float | None = None,
) -> list[np.ndarray]:
    """Extract the bubble wall contour from a FrameData snapshot.

    Convenience wrapper around :func:`extract_zero_contour` that looks
    for the shape function field ``f`` in the FrameData.  If the field
    is not present, falls back to a circular approximation at the bubble
    radius inferred from the coordinate range.

    Parameters
    ----------
    frame : FrameData
        Frozen snapshot with spatial coordinates and scalar fields.
    shape_fn_field : str
        Key for the shape function in ``frame.scalar_fields``
        (default ``"shape_function"``).
    level : float
        Iso-level defining the bubble wall (default 0.5).
    scene_width : float or None
        If provided, rescale to Manim scene coordinates.

    Returns
    -------
    list[np.ndarray]
        List of ``(N, 2)`` arrays defining the bubble wall contour.
        If using circular fallback, returns a single contour with 64
        vertices.
    """
    mid_z = frame.grid_shape[2] // 2

    if shape_fn_field in frame.scalar_fields:
        data_2d = frame.scalar_fields[shape_fn_field][:, :, mid_z]
        x_1d = frame.x[:, 0, 0]
        y_1d = frame.y[0, :, 0]
        x_range = (float(x_1d[0]), float(x_1d[-1]))
        y_range = (float(y_1d[0]), float(y_1d[-1]))
        return extract_zero_contour(
            data_2d, x_range, y_range, level=level, scene_width=scene_width
        )

    # Fallback: smooth circular contour at R ~ half the coordinate range
    # Use 128 points for a smoother circle
    x_1d = frame.x[:, 0, 0]
    x_extent = float(x_1d[-1] - x_1d[0])
    radius = x_extent / 4.0  # heuristic: bubble at ~quarter extent

    theta = np.linspace(0, 2 * np.pi, 128)
    circle = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

    if scene_width is not None:
        # Rescale from data coordinates to scene coordinates
        x_range_half = x_extent / 2.0
        circle = circle / x_range_half * (scene_width / 2.0)

    return [circle]
