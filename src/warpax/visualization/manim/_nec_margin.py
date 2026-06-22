"""NECMargin2D: observer-robust NEC margin heatmap with contour overlays.

2D z=0 slice of the worst-case null contraction min_k T_ab k^a k^b over a dense
null sphere, with the null rays normalized to the local Eulerian frame
(k·n_Eul = −1) so the depth is comparable across the grid. The margin is <= 0
everywhere for Alcubierre (0 in flat space, negative in the wall).

Usage: manim render -ql --format mp4 \\
    src/warpax/visualization/manim/_nec_margin.py NECMargin2D
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
    DashedVMobject,
    Dot,
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
    extract_contours,
    frame_to_rgba,
)
from warpax.visualization.manim._scene_utils import COLORS_3B1B


_DARK_DIVERGE = LinearSegmentedColormap.from_list(
    "dark_diverge_hc",
    ["#3B4CC0", "#1A1A2E", "#B40426"],
    N=256,
)
try:
    _mpl.colormaps.register(_DARK_DIVERGE, name="dark_diverge_hc")
except ValueError:
    pass  # already registered


def _contour_to_vmobject(
    paths: list[np.ndarray],
    center: np.ndarray,
    color: str = WHITE,
    stroke_width: float = 2.0,
    dashed: bool = False,
) -> VGroup:
    """Convert contour path vertices to Manim VMobjects offset to *center*."""
    group = VGroup()
    for verts in paths:
        if len(verts) < 2:
            continue
        vmob = VMobject(color=color, stroke_width=stroke_width)
        points_3d = [np.array([v[0], v[1], 0.0]) + center for v in verts]
        vmob.set_points_as_corners(points_3d)
        if dashed:
            vmob = DashedVMobject(vmob, num_dashes=30, dashed_ratio=0.5)
        group.add(vmob)
    return group


class NECMargin2D(Scene):
    """2D heatmap of observer-robust NEC margins with bubble contour overlay.

    One-sided violation-depth colormap (margin <= 0), graded NEC-depth contours,
    the f = 0.5 bubble wall, and the null-normalization convention k·n_Eul = −1.
    Static header (no title animation).
    """

    def construct(self) -> None:
        n_physics_frames = 30
        v_start = 0.1
        v_end = 0.99
        field_name = "nec_margin_sweep"
        # NEC margin is <= 0 everywhere; use a one-sided sequential ramp so the
        # full colour range encodes violation depth (no implied "satisfied" half).
        cmap_name = "nec_depth"
        animation_duration = 12.0
        hold_duration = 1.0

        img_height = config.frame_height - 2.2
        img_width = img_height

        self.camera.background_color = COLORS_3B1B["background"]

        from warpax.benchmarks import AlcubierreMetric
        from warpax.geometry import GridSpec
        from warpax.visualization.common._physics import (
            build_ec_frame_sequence,
            make_velocity_sweep,
        )

        grid_spec = GridSpec(
            bounds=[(-3, 3), (-3, 3), (-3, 3)],
            shape=(30, 30, 30),
        )

        metric = AlcubierreMetric(v_s=v_start)
        v_values = make_velocity_sweep(v_start, v_end, n_physics_frames)

        print("Precomputing EC frames...")
        frames = build_ec_frame_sequence(
            metric,
            grid_spec,
            v_s_values=v_values,
            progress=True,
        )
        n_frames = len(frames)

        print("Computing global color limits...")
        vmin, vmax, linthresh = compute_symlog_clim(
            frames,
            field_name,
            one_sided=True,
        )
        global_clim = (vmin, vmax, linthresh)

        print("Pre-rendering RGBA frames...")
        rgba_frames = [
            frame_to_rgba(f, field_name, global_clim, cmap_name=cmap_name) for f in frames
        ]

        print("Extracting contours...")
        # Graded NEC violation-depth contours at fixed *global* fractions of the
        # deepest margin, so the lines are stable across the sweep and read as a
        # topographic depth map instead of one boundary. Deepest line is drawn
        # brightest/thickest, shallow lines fainter/thinner.
        global_nec_min = min(
            float(np.nanmin(f.scalar_fields[field_name][:, :, f.grid_shape[2] // 2]))
            for f in frames
        )
        level_fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
        contour_levels = sorted(global_nec_min * fr for fr in level_fracs)
        n_levels = len(contour_levels)
        # ``contour_levels`` ascends (most-negative/deepest first); style index 0
        # is the deepest line.
        level_styles = [
            {
                "stroke_width": 1.8 - 0.8 * (i / max(n_levels - 1, 1)),
                "opacity": 1.0 - 0.5 * (i / max(n_levels - 1, 1)),
            }
            for i in range(n_levels)
        ]

        nec_contour_sets: list[list[tuple[float, list[np.ndarray]]]] = []
        bubble_contour_paths: list[list[np.ndarray]] = []

        for frame in frames:
            mid_z = frame.grid_shape[2] // 2
            data_2d = frame.scalar_fields[field_name][:, :, mid_z]
            x_1d = frame.x[:, 0, 0]
            y_1d = frame.y[0, :, 0]
            x_range = (float(x_1d[0]), float(x_1d[-1]))
            y_range = (float(y_1d[0]), float(y_1d[-1]))

            nec_contour_sets.append(
                extract_contours(
                    data_2d,
                    x_range,
                    y_range,
                    contour_levels,
                    scene_width=img_width,
                )
            )
            bc = extract_bubble_contour(
                frame,
                level=0.5,
                scene_width=img_width,
            )
            bubble_contour_paths.append(bc)

        title_text = Text(
            "Observer-Robust NEC Margin",
            font_size=28,
            color=WHITE,
            weight="LIGHT",
        )
        from warpax.visualization.manim._scene_utils import format_metric_equation

        equation = format_metric_equation("Alcubierre")
        equation.scale(0.6)
        # Null-vector normalization makes the depth well-posed (the min over null
        # directions of T_ab k^a k^b is only defined up to k's frequency).
        normalization = MathTex(
            r"\min_{k}\, T_{ab}\,k^a k^b,\quad k\cdot n_{\rm Eul} = -1",
            font_size=26,
            color=WHITE,
        )
        normalization.set_opacity(0.85)
        header = VGroup(title_text, equation, normalization).arrange(DOWN, buff=0.1)
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

        def _make_nec_contours():
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_frames - 1))
            group = VGroup()
            for (_lvl, paths), style in zip(nec_contour_sets[idx], level_styles):
                for verts in paths:
                    if len(verts) < 2:
                        continue
                    vmob = VMobject(
                        stroke_color=WHITE,
                        stroke_width=style["stroke_width"],
                        stroke_opacity=style["opacity"],
                    )
                    points_3d = [np.array([v[0], v[1], 0.0]) + heatmap_center for v in verts]
                    vmob.set_points_as_corners(points_3d)
                    group.add(DashedVMobject(vmob, num_dashes=40, dashed_ratio=0.5))
            return group

        nec_contours = always_redraw(_make_nec_contours)

        def _make_bubble_contour():
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_frames - 1))
            return _contour_to_vmobject(
                bubble_contour_paths[idx],
                heatmap_center,
                color="#58C4DD",
                stroke_width=2.5,
                dashed=False,
            )

        bubble_contour = always_redraw(_make_bubble_contour)

        center_dot = Dot(point=heatmap_center, radius=0.08, color=WHITE)

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
            vmin,
            vmax,
            linthresh,
            "NEC margin (negative = violation; 0 = marginal)",
        )
        color_legend.to_corner(UR, buff=0.3).shift(DOWN * 0.7)
        color_legend_bg = BackgroundRectangle(
            color_legend,
            fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8,
            buff=0.1,
        )
        color_legend_group = Group(color_legend_bg, color_legend)

        dash_sample = DashedVMobject(
            VMobject(color=WHITE, stroke_width=2.0).set_points_as_corners(
                [np.array([-0.3, 0, 0]), np.array([0.3, 0, 0])]
            ),
            num_dashes=6,
        )
        dash_label = Text(
            "NEC depth contours (0.1–0.9 × deepest)",
            font_size=14,
            color=WHITE,
            weight="LIGHT",
        )
        dash_row = VGroup(dash_sample, dash_label).arrange(RIGHT, buff=0.12)

        solid_sample = VMobject(color="#58C4DD", stroke_width=2.5)
        solid_sample.set_points_as_corners([np.array([-0.3, 0, 0]), np.array([0.3, 0, 0])])
        solid_label = Text(
            "Bubble wall (f = 0.5)",
            font_size=14,
            color="#58C4DD",
            weight="LIGHT",
        )
        solid_row = VGroup(solid_sample, solid_label).arrange(RIGHT, buff=0.12)

        dot_sample = Dot(radius=0.06, color=WHITE)
        dot_label = Text(
            "Bubble center",
            font_size=14,
            color=WHITE,
            weight="LIGHT",
        )
        dot_row = VGroup(dot_sample, dot_label).arrange(RIGHT, buff=0.12)

        annotation = VGroup(dash_row, solid_row, dot_row).arrange(
            DOWN,
            buff=0.1,
            aligned_edge=LEFT,
        )
        annotation.to_corner(DOWN + LEFT, buff=0.25)
        annotation_bg = BackgroundRectangle(
            annotation,
            fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8,
            buff=0.1,
        )
        annotation_group = VGroup(annotation_bg, annotation)

        def _make_violation():
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_frames - 1))
            _mz = frames[idx].grid_shape[2] // 2
            nec_min = float(np.min(frames[idx].scalar_fields[field_name][:, :, _mz]))
            color = COLORS_3B1B["violation_red"] if nec_min < 0 else COLORS_3B1B["safe_green"]
            d = Dot(radius=0.1, color=color)
            lbl = MathTex(r"\text{NEC}", font_size=28, color=WHITE)
            row = VGroup(d, lbl).arrange(RIGHT, buff=0.1)
            row.to_corner(DOWN + RIGHT, buff=0.3)
            return row

        violation = always_redraw(_make_violation)

        self.add(heatmap, nec_contours, bubble_contour, center_dot)
        self.add(param_display, color_legend_group, annotation_group, violation)

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
