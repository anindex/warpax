"""Image utilities for Manim 2D scenes: FrameData slice -> RGBA + contour extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as _mpl

_mpl.use("Agg")  # offscreen contour extraction only; never an interactive backend
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom as _ndimage_zoom

if TYPE_CHECKING:
    from warpax.visualization.common._frame_data import FrameData


# Sequential "violation depth" colormap for one-sided (<= 0) fields such as the
# NEC margin: dark (= marginal, 0) -> bright blue (= deepest violation). Using a
# sequential ramp on a strictly-non-positive field avoids a diverging scale that
# would imply (and waste colour range on) a never-reached "satisfied" half.
_NEC_DEPTH = mcolors.LinearSegmentedColormap.from_list(
    "nec_depth",
    # cmap position 0 (= vmin, deepest) -> bright; position 1 (= 0) -> dark.
    ["#CFE0FF", "#7FA0FF", "#3B4CC0", "#27306E", "#15151F"],
    N=256,
)
try:
    _mpl.colormaps.register(_NEC_DEPTH, name="nec_depth")
except ValueError:
    pass  # already registered


def bilinear_sampler(x_1d, y_1d, values):
    """Return ``f(u, v) -> z`` bilinearly interpolating *values* on a grid.

    Pure-numpy replacement for ``scipy.interpolate.RegularGridInterpolator``
    used pointwise by the Manim ``Surface`` param functions. scipy's compiled
    ``_find_indices`` crashes (SIGILL) under the hundreds of thousands of
    repeated scalar calls a 3D movie render makes on Python 3.14; this uses
    only arithmetic and array indexing. The warpax grids are uniform
    (``linspace`` bounds), so direct index arithmetic is exact -- no search.

    Parameters
    ----------
    x_1d, y_1d : array-like
        Strictly increasing, uniformly spaced grid coordinates (axes 0 and 1
        of *values*).
    values : np.ndarray
        2D field of shape ``(len(x_1d), len(y_1d))``.
    """
    xs = np.asarray(x_1d, dtype=float)
    ys = np.asarray(y_1d, dtype=float)
    vals = np.asarray(values, dtype=float)
    nx, ny = len(xs), len(ys)
    x0, xN = float(xs[0]), float(xs[-1])
    y0, yN = float(ys[0]), float(ys[-1])
    dx = (xN - x0) or 1.0
    dy = (yN - y0) or 1.0

    def f(u: float, v: float) -> float:
        fu = (min(max(float(u), x0), xN) - x0) / dx * (nx - 1)
        fv = (min(max(float(v), y0), yN) - y0) / dy * (ny - 1)
        i = min(int(fu), nx - 2)
        j = min(int(fv), ny - 2)
        tx = fu - i
        ty = fv - j
        return float(
            (1.0 - tx) * (1.0 - ty) * vals[i, j]
            + tx * (1.0 - ty) * vals[i + 1, j]
            + (1.0 - tx) * ty * vals[i, j + 1]
            + tx * ty * vals[i + 1, j + 1]
        )

    return f


def colorbar_gradient(cmap_name: str, width_px: int = 256, height_px: int = 14) -> np.ndarray:
    """RGBA gradient strip sampled uniformly from a colormap (for a legend bar)."""
    cmap = _mpl.colormaps[cmap_name]
    row = (cmap(np.linspace(0.0, 1.0, width_px)) * 255).astype(np.uint8)  # (W, 4)
    return np.repeat(row[None, :, :], height_px, axis=0)  # (H, W, 4)


def colorbar_tick_fractions(
    vmin: float,
    vmax: float,
    linthresh: float,
) -> list[tuple[float, float]]:
    """(value, fraction-along-bar) ticks honouring the SymLogNorm mapping.

    The bar shows the colormap uniformly in norm space, so a data value sits at
    ``fraction = SymLogNorm(value)``. Ticks are placed at the physically
    meaningful values vmin, -linthresh, 0, +linthresh, vmax (those in range),
    so the legend reads correctly even though the scale is symmetric-log.
    """
    if vmax <= vmin:
        return [(vmin, 0.0), (vmax, 1.0)]
    norm = mcolors.SymLogNorm(linthresh=max(linthresh, 1e-30), vmin=vmin, vmax=vmax)
    candidates = [vmin, -linthresh, 0.0, linthresh, vmax]
    ticks: list[tuple[float, float]] = []
    seen: set[float] = set()
    for v in candidates:
        if v < vmin - 1e-30 or v > vmax + 1e-30:
            continue
        key = round(v, 12)
        if key in seen:
            continue
        seen.add(key)
        ticks.append((float(v), float(np.clip(norm(v), 0.0, 1.0))))
    return ticks


def compute_symlog_clim(
    frames: list[FrameData],
    field_name: str,
    linthresh: float = 1.0,
    one_sided: bool = False,
) -> tuple[float, float, float]:
    """Pre-compute global symmetric color limits across all frames.

    Iterates over all FrameData snapshots, extracts the equatorial slice
    (z-midplane) for *field_name*, and computes a symmetric range
    ``(-max_abs, max_abs)`` where ``max_abs = max(|vmin|, |vmax|)``
    across every frame. This prevents per-frame flicker.

    Parameters
    ----------
    frames : list[FrameData]
        Sequence of FrameData snapshots.
    field_name : str
        Key into ``frame.scalar_fields``.
    linthresh : float
        Linear threshold for ``SymLogNorm``. If the global maximum
        absolute value is smaller than *linthresh*, linear scaling is
        used instead (returned linthresh is set to the max absolute
        value to signal this).

    Returns
    -------
    tuple[float, float, float]
        ``(vmin, vmax, linthresh)`` symmetric color limits and the
        (possibly adjusted) linear threshold.
    """
    # Pool the finite slice data across all frames and use robust percentile
    # limits. Raw min/max let a single near-singular grid point (e.g. the
    # Eulerian NEC can blow up to ~-1e13 near the bubble centre) hijack the
    # whole scale and wash the structure out -- the 0.5/99.5 percentiles reject
    # such blow-ups while keeping the genuine wall-region extremes.
    pooled: list = []
    for frame in frames:
        if field_name not in frame.scalar_fields:
            continue
        mid_z = frame.grid_shape[2] // 2
        arr = np.asarray(frame.scalar_fields[field_name][:, :, mid_z]).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            pooled.append(arr)

    # Fallback if no valid data found
    if not pooled:
        return (-1.0, 1.0, linthresh)

    combined = np.concatenate(pooled)
    global_min = float(np.percentile(combined, 0.5))
    global_max = float(np.percentile(combined, 99.5))

    # One-sided fields (e.g. NEC margin, <= 0 everywhere): map the colormap
    # onto the actual [vmin, 0] range so the full colour resolution encodes
    # violation depth, instead of wasting half a diverging scale on a
    # never-reached positive ("satisfied") half. Falls back to symmetric if
    # the data is not actually one-sided.
    if one_sided and global_max <= 1e-12 and global_min < 0.0:
        eff_lt = min(linthresh, abs(global_min))
        eff_lt = max(eff_lt, 1e-15)
        return (global_min, 0.0, eff_lt)

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
        Override the linear threshold from *global_clim*. If ``None``,
        the third element of *global_clim* is used.

    Returns
    -------
    np.ndarray
        RGBA array with shape ``(Nx, Ny, 4)`` and dtype ``uint8``.
    """
    mid_z = frame.grid_shape[2] // 2
    # Canonical screen orientation: physical x -> horizontal (right), physical
    # y -> vertical (up). Slices are (Nx, Ny) with ij indexing (axis 0 = x);
    # ImageMobject renders rows top->bottom and columns left->right, so
    # transpose (rows<-y, cols<-x) then flip rows so +y points up.
    data_2d = np.asarray(frame.scalar_fields[field_name][:, :, mid_z]).T[::-1, :]

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

    cmap = _mpl.colormaps[cmap_name]
    rgba_float = cmap(norm(data_2d))  # (Ny, Nx, 4) float in [0, 1]
    return (rgba_float * 255).astype(np.uint8)


def _process_contour_segments(
    segments: list[np.ndarray],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    data_diag: float,
    scene_width: float | None,
    scene_height: float | None,
) -> list[np.ndarray]:
    """Filter degenerate segments and optionally rescale to scene coordinates."""
    paths: list[np.ndarray] = []
    for raw in segments:
        verts = np.asarray(raw, dtype=float)
        if len(verts) < 3:
            continue
        # Drop too-short segments (marching-squares speckle).
        diffs = np.diff(verts, axis=0)
        arc_length = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))
        if arc_length < data_diag * 0.02:
            continue
        verts = verts.copy()
        if scene_width is not None:
            sh = scene_height if scene_height is not None else scene_width
            x_min, x_max = x_range
            verts[:, 0] = ((verts[:, 0] - x_min) / (x_max - x_min) - 0.5) * scene_width
            # Independent height avoids stretch when w != h.
            y_min, y_max = y_range
            verts[:, 1] = ((verts[:, 1] - y_min) / (y_max - y_min) - 0.5) * sh
        paths.append(verts)
    return paths


def extract_contours(
    data_2d: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    levels: list[float],
    *,
    scene_width: float | None = None,
    scene_height: float | None = None,
    upsample_order: int = 1,
) -> list[tuple[float, list[np.ndarray]]]:
    """Extract contour path vertices at multiple iso-levels from 2D data.

    Uses ``matplotlib.pyplot.contour`` to trace iso-level curves, grouping the
    resulting segments per requested level (via ``ContourSet.allsegs``). This
    lets a scene draw a graded family of contours (e.g. several NEC violation-
    depth lines) instead of a single boundary, for a clearer topographic read.

    Parameters
    ----------
    data_2d : np.ndarray
        2D scalar field with shape ``(Nx, Ny)`` (ij indexing, axis 0 = x).
    x_range, y_range : tuple[float, float]
        ``(min, max)`` spatial extent along x and y.
    levels : list[float]
        Iso-levels to extract. Out-of-range levels yield empty path lists.
    scene_width, scene_height : float or None
        If *scene_width* is given, rescale contour coordinates from data space
        to ``[-scene_width/2, scene_width/2]`` (and height likewise). If
        ``None``, coordinates are returned in data space.
    upsample_order : int
        Spline order for the 8x upsample before marching squares. Default
        ``1`` (bilinear): cubic (``3``) overshoots and rings around the
        ``NaN -> 0`` fill and the sharp bubble wall, producing spurious
        contour wiggles.

    Returns
    -------
    list[tuple[float, list[np.ndarray]]]
        ``(level, paths)`` pairs, one per level actually traced (in ascending
        order). Each ``paths`` is a list of ``(N, 2)`` segment arrays.
    """
    # Upsample for smooth marching-squares paths (30x30 -> 240x240). Replace
    # NaN first - interpolation propagates even a single NaN across the array.
    upsample_factor = 8
    data_2d = np.nan_to_num(np.asarray(data_2d, dtype=float), nan=0.0)
    # Canonical orientation: contour X = physical x, Y = physical y. Slices are
    # (Nx, Ny) [ij indexing]; matplotlib.contour wants Z[row=y, col=x], so
    # transpose before sampling (keeps contours aligned with frame_to_rgba).
    data_up = _ndimage_zoom(data_2d.T, upsample_factor, order=upsample_order)

    sorted_levels = sorted(float(lv) for lv in levels)
    fig, ax = plt.subplots()
    x_coords = np.linspace(x_range[0], x_range[1], data_up.shape[1])  # len Nx*f
    y_coords = np.linspace(y_range[0], y_range[1], data_up.shape[0])  # len Ny*f
    cs = ax.contour(x_coords, y_coords, data_up, levels=sorted_levels)
    plt.close(fig)

    data_diag = np.sqrt((x_range[1] - x_range[0]) ** 2 + (y_range[1] - y_range[0]) ** 2)

    # ``allsegs`` groups segments per level (aligned with ``cs.levels``), so
    # distinct contours are already separated - no MOVETO bridge splitting.
    out: list[tuple[float, list[np.ndarray]]] = []
    for lvl, segs in zip(np.asarray(cs.levels, dtype=float), cs.allsegs):
        paths = _process_contour_segments(
            list(segs),
            x_range,
            y_range,
            data_diag,
            scene_width,
            scene_height,
        )
        out.append((float(lvl), paths))
    return out


def extract_zero_contour(
    data_2d: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    level: float = 0.0,
    scene_width: float | None = None,
    scene_height: float | None = None,
) -> list[np.ndarray]:
    """Single-level convenience wrapper around :func:`extract_contours`.

    Returns the list of ``(N, 2)`` segment arrays at *level* (empty if the
    level is not crossed). Kept for backward compatibility; new callers that
    want a graded family should use :func:`extract_contours` directly.
    """
    results = extract_contours(
        data_2d,
        x_range,
        y_range,
        [level],
        scene_width=scene_width,
        scene_height=scene_height,
    )
    return results[0][1] if results else []


def extract_bubble_contour(
    frame: FrameData,
    shape_fn_field: str = "shape_function",
    level: float = 0.5,
    scene_width: float | None = None,
    scene_height: float | None = None,
) -> list[np.ndarray]:
    """Extract the bubble wall contour from a FrameData snapshot.

    Convenience wrapper around :func:`extract_zero_contour` that looks
    for the shape function field ``f`` in the FrameData. If the field
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
        # Clip f to [0, 1] so the f = 0.5 wall is robust: the analytic shape
        # function lives in [0, 1], and clipping prevents any interpolation
        # overshoot from spawning a stray 0.5 crossing outside the wall.
        data_2d = np.clip(np.asarray(frame.scalar_fields[shape_fn_field][:, :, mid_z]), 0.0, 1.0)
        x_1d = frame.x[:, 0, 0]
        y_1d = frame.y[0, :, 0]
        x_range = (float(x_1d[0]), float(x_1d[-1]))
        y_range = (float(y_1d[0]), float(y_1d[-1]))
        return extract_zero_contour(
            data_2d,
            x_range,
            y_range,
            level=level,
            scene_width=scene_width,
            scene_height=scene_height,
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
