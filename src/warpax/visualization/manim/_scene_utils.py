"""Shared Manim scene utilities for warp drive animations.

Provides reusable helpers used by all four showcase scenes:
- Title card animation
- Live parameter display (Variable wrapping)
- Global color limit pre-computation (prevents flickering)
- 3blue1brown color palette
- Metric defining equation formatter
- Violation status indicator (red/green dot)
- Scalar field blending for field transitions
- Auto-exaggeration and axes construction for FrameData sequences

"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from manim import (
    DEGREES,
    DOWN,
    RIGHT,
    DL,
    DR,
    UL,
    UR,
    WHITE,
    YELLOW,
    Dot,
    FadeOut,
    MathTex,
    ThreeDAxes,
    Variable,
    VGroup,
    Write,
)

if TYPE_CHECKING:
    from manim import ThreeDScene

    from warpax.visualization.common._frame_data import FrameData

# ---------------------------------------------------------------------------
# 1. 3blue1brown color constants
# ---------------------------------------------------------------------------

COLORS_3B1B: dict[str, str] = {
    "background": "#1C1C1C",
    "surface_blue": "#58C4DD",
    "surface_yellow": "#FFFF00",
    "surface_green": "#83C167",
    "text_white": WHITE,
    "param_yellow": YELLOW,
    "violation_red": "#FF4444",
    "safe_green": "#44FF44",
}


# ---------------------------------------------------------------------------
# 2. Title card
# ---------------------------------------------------------------------------


def play_title_card(
    scene: ThreeDScene,
    metric_name: str,
    params_dict: dict[str, str],
    run_time: float = 2.5,
) -> None:
    """Animate a title card with metric name and parameters.

    Parameters
    ----------
    scene : ThreeDScene
        The active Manim scene.
    metric_name : str
        Display name (e.g. ``"Alcubierre Bubble Collapse"``).
    params_dict : dict[str, str]
        Parameter key-value pairs shown as subtitle.
    run_time : float
        Total duration including write, hold, and fadeout.
    """
    title = MathTex(
        r"\text{" + metric_name + r"}",
        font_size=64,
    )
    subtitle_parts = [f"{k} = {v}" for k, v in params_dict.items()]
    subtitle = MathTex(
        r",\;\;".join(subtitle_parts),
        font_size=42,
    )
    group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
    scene.add_fixed_in_frame_mobjects(group)
    scene.play(Write(group), run_time=1.0)
    scene.wait(1.5)
    scene.play(FadeOut(group), run_time=0.5)
    scene.remove_fixed_in_frame_mobjects(group)


# ---------------------------------------------------------------------------
# 3. Live parameter display
# ---------------------------------------------------------------------------


def make_parameter_display(
    label_tex: str,
    initial_value: float,
    num_decimal_places: int = 2,
    position: str = "UL",
) -> Variable:
    """Create a Manim Variable for a live-updating parameter counter.

    Parameters
    ----------
    label_tex : str
        LaTeX string for the label (e.g. ``r"v_s"``).
    initial_value : float
        Starting value.
    num_decimal_places : int
        Decimal places for the displayed number.
    position : str
        Corner position: ``"UL"``, ``"UR"``, ``"DL"``, ``"DR"``.

    Returns
    -------
    Variable
        Manim Variable mobject. Caller must call
        ``scene.add_fixed_in_frame_mobjects(var)`` and attach updaters.
    """
    var = Variable(
        initial_value,
        MathTex(label_tex, font_size=36),
        num_decimal_places=num_decimal_places,
    )
    var.label.set_color(WHITE)
    var.value.set_color(YELLOW)
    var.value.font_size = 36

    corner_map = {"UL": UL, "UR": UR, "DL": DL, "DR": DR}
    corner = corner_map.get(position, UL)
    var.to_corner(corner)

    return var


# ---------------------------------------------------------------------------
# 4. Global color limits
# ---------------------------------------------------------------------------


def compute_global_clim(
    frames: list[FrameData],
    field_name: str,
    *,
    percentile: float | None = None,
) -> tuple[float, float]:
    """Pre-compute global (vmin, vmax) across all frames for a scalar field.

    Parameters
    ----------
    frames : list[FrameData]
        Sequence of FrameData snapshots.
    field_name : str
        Key into ``frame.scalar_fields``.
    percentile : float, optional
        If given, use the *percentile*-th and (100-percentile)-th
        percentiles of the **actual data** across all frames instead
        of raw min/max from pre-stored clim.  Useful for fields with
        extreme outliers (e.g. WEC margin at high rapidity).  Typical
        value: 2.0 (clips the most extreme 2 % on each tail).

    Returns
    -------
    tuple[float, float]
        ``(vmin, vmax)`` global color limits.
    """
    import numpy as _np

    if percentile is not None:
        all_vals: list = []
        for f in frames:
            if field_name in f.scalar_fields:
                arr = _np.asarray(f.scalar_fields[field_name]).ravel()
                arr = arr[_np.isfinite(arr)]
                all_vals.append(arr)
        if not all_vals:
            return (0.0, 1.0)
        combined = _np.concatenate(all_vals)
        vmin = float(_np.percentile(combined, percentile))
        vmax = float(_np.percentile(combined, 100.0 - percentile))
        # Ensure symmetric around zero for diverging colormaps
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        if abs(vmax - vmin) < 1e-15:
            vmax = vmin + 1.0
        return (vmin, vmax)

    vmins = []
    vmaxs = []
    for f in frames:
        if field_name in f.scalar_fields:
            clim = f.clim.get(field_name)
            if clim is not None:
                vmins.append(clim[0])
                vmaxs.append(clim[1])
    if not vmins:
        return (0.0, 1.0)
    vmin = min(vmins)
    vmax = max(vmaxs)
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1.0
    return (vmin, vmax)


# ---------------------------------------------------------------------------
# 5. Axes for frame sequences
# ---------------------------------------------------------------------------


def make_axes_for_frames(
    frames: list[FrameData],
    field_name: str,
    coord_range: tuple[float, float] = (-3, 3),
    z_headroom: float = 1.3,
) -> ThreeDAxes:
    """Build ThreeDAxes sized for a pre-computed frame sequence.

    Parameters
    ----------
    frames : list[FrameData]
        All frames in the animation.
    field_name : str
        Scalar field driving z-displacement.
    coord_range : tuple[float, float]
        Spatial coordinate range for x and y.
    z_headroom : float
        Multiplicative headroom on z-extent.

    Returns
    -------
    ThreeDAxes
        Axes with computed z-range.
    """
    exag = compute_auto_exaggeration(frames, field_name)
    vmin, vmax = compute_global_clim(frames, field_name)
    max_abs = max(abs(vmin), abs(vmax))
    z_extent = max_abs * exag * z_headroom

    lo, hi = coord_range
    return ThreeDAxes(
        x_range=[lo, hi, 1],
        y_range=[lo, hi, 1],
        z_range=[-z_extent, z_extent, z_extent / 2] if z_extent > 0 else [-1, 1, 0.5],
        x_length=6,
        y_length=6,
        z_length=4,
    )


# ---------------------------------------------------------------------------
# 6. Auto-exaggeration
# ---------------------------------------------------------------------------


def compute_auto_exaggeration(
    frames: list[FrameData],
    field_name: str,
) -> float:
    """Compute the exaggeration factor for embedding surfaces.

    Uses ``0.3 * extent / max_warp`` where ``max_warp`` is the global
    maximum absolute value of the equatorial slice across all frames.

    Parameters
    ----------
    frames : list[FrameData]
        All frames in the animation.
    field_name : str
        Scalar field used for z-displacement.

    Returns
    -------
    float
        Exaggeration factor.
    """
    max_warp = 0.0
    extent = 6.0  # default (-3, 3) range
    for f in frames:
        if field_name not in f.scalar_fields:
            continue
        mid_z = f.grid_shape[2] // 2
        eq_slice = f.scalar_fields[field_name][:, :, mid_z]
        frame_max = float(np.max(np.abs(eq_slice)))
        if frame_max > max_warp:
            max_warp = frame_max
        # Update extent from coordinates
        x_1d = f.x[:, 0, 0]
        extent = float(x_1d[-1] - x_1d[0])
    return 0.3 * extent / max(max_warp, 1e-15)


# ---------------------------------------------------------------------------
# 7. Metric defining equation formatter
# ---------------------------------------------------------------------------

_METRIC_EQUATIONS: dict[str, str] = {
    "Alcubierre": r"ds^2 = -dt^2 + (dx - v_s f \, dt)^2 + dy^2 + dz^2",
    "Lentz": r"ds^2 = -dt^2 + (dx - v_s \hat{X} \, dt)^2 + dy^2 + dz^2",
    "Natario": r"ds^2 = -dt^2 + (dx^i - v_s n^i \, dt)^2",
}


def format_metric_equation(metric_name: str) -> MathTex:
    """Return a MathTex mobject with the defining line element.

    Parameters
    ----------
    metric_name : str
        Metric name (e.g. ``"Alcubierre"``).

    Returns
    -------
    MathTex
        LaTeX-rendered line element formula.
    """
    tex_str = _METRIC_EQUATIONS.get(metric_name, r"\text{" + metric_name + r"}")
    return MathTex(tex_str, font_size=36, color=WHITE)


# ---------------------------------------------------------------------------
# 8. Violation status indicator
# ---------------------------------------------------------------------------


def make_violation_indicator(
    field_name: str = "nec_margin_sweep",
) -> VGroup:
    """Create a violation status indicator (colored dot + EC label).

    Parameters
    ----------
    field_name : str
        Scalar field name for violation checking.

    Returns
    -------
    VGroup
        Group containing a colored Dot and a MathTex label.
        Caller must call ``add_fixed_in_frame_mobjects`` and attach an
        updater that reads the current frame's data.

    Notes
    -----
    Updater logic (implemented by caller):

    - ``energy_density``: dot stays WHITE (neutral)
    - Fields containing ``"nec"`` or ``"wec"``: check if ``min(field) < 0``.
      RED (#FF4444) for violation, GREEN (#44FF44) for safe.
    """
    dot = Dot(radius=0.12, color="#44FF44")

    # Choose label based on field name
    if "nec" in field_name:
        label_tex = r"\text{NEC}"
    elif "wec" in field_name:
        label_tex = r"\text{WEC}"
    elif "energy" in field_name:
        label_tex = r"T_{00}"
    else:
        label_tex = r"\text{EC}"

    label = MathTex(label_tex, font_size=32, color=WHITE)
    group = VGroup(dot, label).arrange(RIGHT, buff=0.15)

    return group


# ---------------------------------------------------------------------------
# 9. Scalar field blending
# ---------------------------------------------------------------------------


def blend_fields(
    frame: FrameData,
    field_a: str,
    field_b: str,
    t: float,
) -> FrameData:
    """Blend two scalar fields from a FrameData object.

    Parameters
    ----------
    frame : FrameData
        Source frame containing at least *field_a*.
    field_a : str
        Name of the first field (t=0 extreme).
    field_b : str
        Name of the second field (t=1 extreme).
    t : float
        Blend factor: 0.0 = pure field_a, 1.0 = pure field_b.

    Returns
    -------
    FrameData
        Shallow copy with an additional ``"_blended"`` field and blended
        ``clim`` entry. If *field_b* is not present, returns *frame*
        unmodified.
    """
    if field_b not in frame.scalar_fields:
        return frame

    arr_a = frame.scalar_fields[field_a]
    arr_b = frame.scalar_fields[field_b]
    blended = (1.0 - t) * arr_a + t * arr_b

    clim_a = frame.clim.get(field_a, (0.0, 1.0))
    clim_b = frame.clim.get(field_b, (0.0, 1.0))
    blended_clim = (
        (1.0 - t) * clim_a[0] + t * clim_b[0],
        (1.0 - t) * clim_a[1] + t * clim_b[1],
    )

    # Build a modified copy FrameData is an eqx.Module (frozen),
    # so we reconstruct with updated dicts.
    from warpax.visualization.common._frame_data import FrameData as FD

    new_fields = dict(frame.scalar_fields)
    new_fields["_blended"] = blended

    new_clim = dict(frame.clim)
    new_clim["_blended"] = blended_clim

    new_colormaps = dict(frame.colormaps)
    new_colormaps["_blended"] = "RdBu_r"

    return FD(
        x=frame.x,
        y=frame.y,
        z=frame.z,
        scalar_fields=new_fields,
        metric_name=frame.metric_name,
        v_s=frame.v_s,
        grid_shape=frame.grid_shape,
        t=frame.t,
        colormaps=new_colormaps,
        clim=new_clim,
    )
