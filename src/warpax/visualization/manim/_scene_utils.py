"""Shared Manim scene utilities: title card, params, colormaps, 3b1b palette, FrameData helpers.

In ThreeDScene, ``DecimalNumber.set_value`` recreates internal submobjects that
lose fixed-in-frame registration. Scene updaters use ``become`` instead so
geometry is replaced in-place and registration is preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from manim import (
    DOWN,
    RIGHT,
    UP,
    DL,
    DR,
    UL,
    UR,
    WHITE,
    YELLOW,
    Dot,
    FadeOut,
    Group,
    ImageMobject,
    Line,
    MathTex,
    Text,
    ThreeDAxes,
    Variable,
    VGroup,
    Write,
)

if TYPE_CHECKING:
    from manim import ThreeDScene

    from warpax.visualization.common._frame_data import FrameData


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


def cap_recursion_for_3d_render(limit: int = 600) -> None:
    """Cap the Python recursion limit before a 3D Cairo render.

    Manim's 3D Cairo render path recurses deeply (camera projection /
    fill-by-value over many sub-surfaces). Under Python 3.14's C-stack
    accounting the default limit (1000) lets that recursion overflow the
    default 8 MB main-thread stack and segfault inside a numpy C call
    (``get_view_from_index``) instead of raising a catchable ``RecursionError``.

    The render's genuine recursion depth is < 250, so capping at *limit* keeps a
    wide safety margin while ensuring the runaway path recovers before the C
    stack is exhausted. Process-global but only invoked from a render, so it
    does not perturb library callers. No-op if the limit is already lower.

    The render scripts additionally raise the OS stack (see
    ``scripts/render_all_scenes.py``); this guard makes a bare
    ``manim render <file> <Scene>`` safe on its own.
    """
    import sys

    if sys.getrecursionlimit() > limit:
        sys.setrecursionlimit(limit)


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
        Display name (e.g. ``"Alcubierre Velocity Sweep"``).
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
        of raw min/max from pre-stored clim. Useful for fields with
        extreme outliers (e.g. WEC margin at high rapidity). Typical
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
    # Guard near-flat fields: the naive 0.3*extent/eps would explode the
    # surface off-axis. A flat field gets a neutral factor instead.
    if max_warp <= 1e-9 * max(extent, 1.0):
        return 1.0
    return 0.3 * extent / max_warp


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


def make_colorbar_legend(
    cmap_name: str,
    vmin: float,
    vmax: float,
    linthresh: float,
    title: str,
    *,
    bar_width: float = 1.7,
    bar_height: float = 0.16,
) -> Group:
    """Quantitative colorbar: the real colormap as a gradient + numeric ticks.

    Replaces the hand-typed 5-swatch "-"/"+" strips. The bar IS
    the colormap (sampled uniformly in norm space) and the tick labels state the
    actual value->colour mapping at the SymLog-significant points (vmin,
    +/-linthresh, 0, vmax), so a reader can recover magnitudes despite the
    symmetric-log compression. Returns a ``Group`` (mixes an ImageMobject bar
    with VMobject ticks); position it with ``.to_corner(...)``.
    """
    from warpax.visualization.manim._image_utils import (
        colorbar_gradient,
        colorbar_tick_fractions,
    )

    bar = ImageMobject(colorbar_gradient(cmap_name))
    bar.height = bar_height
    bar.width = bar_width

    def _fmt(v: float) -> str:
        if abs(v) < 1e-30:
            return "0"
        a = abs(v)
        return f"{v:.2g}" if 1e-2 <= a < 1e3 else f"{v:.0e}"

    marks = VGroup()
    for value, frac in colorbar_tick_fractions(vmin, vmax, linthresh):
        x = (frac - 0.5) * bar_width
        tick = Line(
            [x, -bar_height / 2, 0],
            [x, -bar_height / 2 - 0.05, 0],
            stroke_width=1.2,
            color=WHITE,
        )
        lbl = MathTex(_fmt(value), font_size=16, color=WHITE)
        lbl.next_to(tick, DOWN, buff=0.03)
        marks.add(VGroup(tick, lbl))

    title_mob = Text(title, font_size=15, color=WHITE, weight="LIGHT")
    title_mob.next_to(bar, UP, buff=0.06)

    return Group(title_mob, bar, marks)


def make_conventions_caption(extra: str = "") -> Text:
    """Compact GR-conventions footer for a publication-grade figure.

    States the unit/signature/slice conventions a relativist needs to read the
    figure unambiguously: geometric units, metric signature, the rendered
    spatial slice, and that each frame is a static metric (a parameter sweep,
    not a time evolution). Caller positions it (e.g. ``cap.to_edge(DOWN)``).

    Parameters
    ----------
    extra : str
        Optional trailing clause appended after a separator.
    """
    base = (
        "G = c = 1   ·   signature (−,+,+,+)   ·   T_ab = G_ab/(8π)   ·   "
        "z = 0 slice   ·   static metric per frame   ·   [ρ, T_ab] = 1/length²"
    )
    text = base if not extra else f"{base}   ·   {extra}"
    cap = Text(text, font_size=13, color=WHITE, weight="LIGHT")
    cap.set_opacity(0.6)
    return cap


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
    elif field_name == "T_00_covariant":
        label_tex = r"T_{00}"
    elif "energy" in field_name:
        label_tex = r"\rho_{\rm Eul}"
    else:
        label_tex = r"\text{EC}"

    label = MathTex(label_tex, font_size=32, color=WHITE)
    group = VGroup(dot, label).arrange(RIGHT, buff=0.15)

    return group
