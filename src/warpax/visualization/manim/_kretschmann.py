"""KretschmannInvariant2D: observer-independent curvature invariant heatmap.

The Kretschmann scalar ``K = R_{abcd} R^{abcd}`` is a coordinate- and
observer-independent measure of spacetime curvature (the tidal magnitude a dust
particle would feel), unlike the frame-dependent energy density and energy
conditions. For the Alcubierre bubble it is non-negative and sharply peaked on
the wall, so a log (SymLog) sequential scale reads best.

Usage: manim render -ql --format mp4 \\
    src/warpax/visualization/manim/_kretschmann.py KretschmannInvariant2D
"""

from __future__ import annotations

import numpy as np
import matplotlib as _mpl
from matplotlib.colors import LinearSegmentedColormap
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    UL,
    UR,
    WHITE,
    YELLOW,
    config,
    BackgroundRectangle,
    FadeIn,
    Group,
    ImageMobject,
    MathTex,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    linear,
)

from warpax.visualization.manim._image_utils import (
    compute_symlog_clim,
    extract_bubble_contour,
    frame_to_rgba,
)
from warpax.visualization.manim._scene_utils import COLORS_3B1B

# Dark-midpoint diverging colormap for the *signed* Kretschmann scalar (it is
# sign-indefinite in Lorentzian signature): blue (negative) -> dark -> red.
_DARK_DIVERGE_K = LinearSegmentedColormap.from_list(
    "dark_diverge_k",
    ["#3B4CC0", "#1A1A2E", "#B40426"],
    N=256,
)
try:
    _mpl.colormaps.register(_DARK_DIVERGE_K, name="dark_diverge_k")
except ValueError:
    pass  # already registered


def _contour_to_vmobject(
    paths: list[np.ndarray],
    center: np.ndarray,
    color: str = "#58C4DD",
    stroke_width: float = 2.5,
) -> VGroup:
    """Convert contour path vertices to Manim VMobjects offset to *center*."""
    group = VGroup()
    for verts in paths:
        if len(verts) < 2:
            continue
        vmob = VMobject(color=color, stroke_width=stroke_width)
        points_3d = [np.array([v[0], v[1], 0.0]) + center for v in verts]
        vmob.set_points_as_corners(points_3d)
        group.add(vmob)
    return group


class KretschmannInvariant2D(Scene):
    """2D heatmap of the Kretschmann invariant K = R_abcd R^abcd (v_s sweep).

    Observer-independent curvature, log-sequential colormap, the f = 0.5 bubble
    wall overlay, and a units annotation ([K] = 1/length^4). Complements the
    observer-dependent energy-condition scenes by showing the true invariant
    tidal magnitude.
    """

    def construct(self) -> None:
        n_frames = 30
        v_start = 0.1
        v_end = 0.99
        field_name = "kretschmann"
        # K = R_abcd R^abcd is sign-indefinite in Lorentzian signature (for the
        # Alcubierre bubble it dips strongly negative on the wall), so use a
        # diverging dark-midpoint scale centred at 0, not a one-sided ramp.
        cmap_name = "dark_diverge_k"
        animation_duration = 12.0
        hold_duration = 1.0

        img_height = config.frame_height - 2.2

        self.camera.background_color = COLORS_3B1B["background"]

        from warpax.benchmarks import AlcubierreMetric
        from warpax.geometry import GridSpec
        from warpax.visualization.common._physics import (
            _shape_function_grid,
            build_frame_sequence,
            make_velocity_sweep,
        )

        grid_spec = GridSpec(
            bounds=[(-3, 3), (-3, 3), (-3, 3)],
            shape=(30, 30, 30),
        )

        metric = AlcubierreMetric(v_s=v_start)
        v_values = make_velocity_sweep(v_start, v_end, n_frames)

        print("Precomputing curvature frames...")
        frames = build_frame_sequence(
            metric,
            grid_spec,
            v_s_values=v_values,
            compute_invariants=True,
            progress=True,
        )
        # Attach the real shape function so the f = 0.5 wall overlay is genuine.
        for frame, v_s in zip(frames, v_values):
            metric_v = AlcubierreMetric(v_s=float(v_s))
            f_grid = _shape_function_grid(metric_v, grid_spec, 0.0)
            if f_grid is not None:
                frame.scalar_fields["shape_function"] = f_grid
        n_frames = len(frames)

        # Global color limits across the sweep: robust symmetric SymLog limits
        # (0.5/99.5 percentiles) so the strong wall spike of either sign does not
        # hijack the scale. K is signed, so the range is centred at 0.
        global_clim = compute_symlog_clim(frames, field_name)

        print("Pre-rendering RGBA frames...")
        rgba_frames = [
            frame_to_rgba(f, field_name, global_clim, cmap_name=cmap_name) for f in frames
        ]

        bubble_contour_paths: list[list[np.ndarray]] = []
        for frame in frames:
            bubble_contour_paths.append(
                extract_bubble_contour(frame, level=0.5, scene_width=img_height)
            )

        title_text = Text(
            "Kretschmann Curvature Invariant",
            font_size=28,
            color=WHITE,
            weight="LIGHT",
        )
        invariant = MathTex(
            r"\mathcal{K} = R_{abcd}\,R^{abcd}\quad"
            r"(\text{observer-independent; sign-indefinite})",
            font_size=24,
            color=WHITE,
        )
        invariant.set_opacity(0.9)
        header = VGroup(title_text, invariant).arrange(DOWN, buff=0.12)
        header.to_edge(UP, buff=0.15)
        self.add(header)

        frame_idx = ValueTracker(0)
        heatmap_center = DOWN * 0.3

        def _make_heatmap():
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_frames - 1))
            img = ImageMobject(rgba_frames[idx])
            img.height = img_height
            img.move_to(heatmap_center)
            return img

        heatmap = always_redraw(_make_heatmap)

        def _make_bubble_contour():
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_frames - 1))
            return _contour_to_vmobject(bubble_contour_paths[idx], heatmap_center)

        bubble_contour = always_redraw(_make_bubble_contour)

        vs_mathtex = [MathTex(f"{f.v_s:.2f}", font_size=30, color=YELLOW) for f in frames]

        def _make_param():
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_frames - 1))
            row = VGroup(
                MathTex(r"v_s", font_size=30, color=WHITE),
                MathTex(r"=", font_size=30, color=WHITE),
                vs_mathtex[idx].copy(),
            ).arrange(RIGHT, buff=0.08)
            row.to_corner(UL, buff=0.35)
            return row

        param_display = always_redraw(_make_param)

        from warpax.visualization.manim._scene_utils import make_colorbar_legend

        color_legend = make_colorbar_legend(
            cmap_name,
            global_clim[0],
            global_clim[1],
            global_clim[2],
            "K = R_abcd R^abcd  (signed; [K] = 1/length⁴)",
        )
        color_legend.to_corner(UR, buff=0.3).shift(DOWN * 0.7)
        color_legend_group = Group(
            BackgroundRectangle(
                color_legend,
                fill_color=COLORS_3B1B["background"],
                fill_opacity=0.8,
                buff=0.1,
            ),
            color_legend,
        )

        solid_sample = VMobject(color="#58C4DD", stroke_width=2.5)
        solid_sample.set_points_as_corners([np.array([-0.3, 0, 0]), np.array([0.3, 0, 0])])
        solid_row = VGroup(
            solid_sample,
            Text("Bubble wall (f = 0.5)", font_size=14, color="#58C4DD", weight="LIGHT"),
        ).arrange(RIGHT, buff=0.12)
        solid_row.to_corner(DOWN + LEFT, buff=0.25)
        solid_row_group = VGroup(
            BackgroundRectangle(
                solid_row,
                fill_color=COLORS_3B1B["background"],
                fill_opacity=0.8,
                buff=0.1,
            ),
            solid_row,
        )

        self.add(heatmap, bubble_contour, param_display)
        self.add(color_legend_group, solid_row_group)

        from warpax.visualization.manim._scene_utils import (
            make_conventions_caption,
        )

        caption = make_conventions_caption()
        if caption.width > 7.0:
            caption.scale_to_fit_width(7.0)
        caption.to_edge(DOWN, buff=0.04)
        self.add(
            VGroup(
                BackgroundRectangle(
                    caption,
                    fill_color=COLORS_3B1B["background"],
                    fill_opacity=0.8,
                    buff=0.05,
                ),
                caption,
            )
        )

        self.play(FadeIn(heatmap), run_time=0.5)
        self.play(
            frame_idx.animate.set_value(n_frames - 1),
            run_time=animation_duration,
            rate_func=linear,
        )
        self.wait(hold_duration)
