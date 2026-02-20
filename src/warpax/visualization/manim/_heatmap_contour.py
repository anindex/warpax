"""ECHeatmapContour: 2D Manim Scene showing EC margin heatmap with contour overlays.

Renders a full-screen equatorial-plane heatmap of observer-robust NEC margins
for the Alcubierre metric, with overlays:

1. **NEC ≈ 0 contour** (near-zero isoline) as a white dashed path - shows
   the boundary where NEC transitions from negligible to significant
   violation.
2. **Bubble wall contour** (shape function f = 0.5) as a cyan outline -
   shows the physical extent of the warp bubble.
3. **Moving center dot** - white dot tracking the bubble center.

RGBA arrays are precomputed with global fixed color limits (SymLogNorm)
to prevent per-frame flicker.

Layout
------
- **Top centre**: title + metric equation (static)
- **Upper-left**: live v_s parameter display
- **Upper-right**: 5-stop color legend bar
- **Lower-left**: contour annotation legend
- **Lower-right**: NEC violation indicator dot

Usage::

    manim render -ql --format mp4 \\
        src/warpax/visualization/manim/_heatmap_contour.py ECHeatmapContour
"""
from __future__ import annotations

import numpy as np
import matplotlib.cm as _mcm
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
    ImageMobject,
    MathTex,
    Rectangle,
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
    extract_zero_contour,
    frame_to_rgba,
)
from warpax.visualization.manim._scene_utils import COLORS_3B1B

# ---------------------------------------------------------------------------
# Dark-midpoint diverging colormap: blue -> dark navy -> red
# ---------------------------------------------------------------------------
_DARK_DIVERGE = LinearSegmentedColormap.from_list(
    "dark_diverge_hc", ["#3B4CC0", "#1A1A2E", "#B40426"], N=256,
)
try:
    _mcm.register_cmap(name="dark_diverge_hc", cmap=_DARK_DIVERGE)
except (ValueError, AttributeError):
    try:
        import matplotlib as _mpl
        _mpl.colormaps.register(_DARK_DIVERGE, name="dark_diverge_hc")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


class ECHeatmapContour(Scene):
    """2D heatmap of observer-robust NEC margins with bubble contour overlay.

    Dark-midpoint colormap, color legend bars, parameter
    display, contour annotation legend.  Static header (no title animation).
    """

    def construct(self) -> None:
        # ==================================================================
        # Configuration
        # ==================================================================
        n_physics_frames = 30
        v_start = 0.1
        v_end = 0.99
        field_name = "nec_margin_sweep"
        cmap_name = "dark_diverge_hc"
        animation_duration = 12.0
        hold_duration = 1.0

        img_height = config.frame_height - 2.2
        img_width = img_height

        self.camera.background_color = COLORS_3B1B["background"]

        # ==================================================================
        # Step 1: Precompute all FrameData via velocity sweep
        # ==================================================================
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
            metric, grid_spec,
            v_s_values=v_values,
            progress=True,
        )
        n_frames = len(frames)

        # ==================================================================
        # Step 2: Pre-render all RGBA arrays with global fixed clim
        # ==================================================================
        print("Computing global color limits...")
        vmin, vmax, linthresh = compute_symlog_clim(frames, field_name)
        global_clim = (vmin, vmax, linthresh)

        print("Pre-rendering RGBA frames...")
        rgba_frames = [
            frame_to_rgba(f, field_name, global_clim, cmap_name=cmap_name)
            for f in frames
        ]

        # ==================================================================
        # Step 3: Pre-extract contours for each frame
        # ==================================================================
        print("Extracting contours...")
        zero_contour_paths: list[list[np.ndarray]] = []
        bubble_contour_paths: list[list[np.ndarray]] = []

        for frame in frames:
            mid_z = frame.grid_shape[2] // 2
            data_2d = frame.scalar_fields[field_name][:, :, mid_z]
            x_1d = frame.x[:, 0, 0]
            y_1d = frame.y[0, :, 0]
            x_range = (float(x_1d[0]), float(x_1d[-1]))
            y_range = (float(y_1d[0]), float(y_1d[-1]))

            # Near-zero contour - observer-swept margins are ≤ 0, so
            # use a small negative threshold for a meaningful boundary.
            nec_min = float(np.min(data_2d))
            nec_threshold = nec_min * 1e-2 if nec_min < 0 else -1e-6
            zc = extract_zero_contour(
                data_2d, x_range, y_range,
                level=nec_threshold, scene_width=img_width,
            )
            zero_contour_paths.append(zc)

            bc = extract_bubble_contour(
                frame, level=0.5, scene_width=img_width,
            )
            bubble_contour_paths.append(bc)

        # ==================================================================
        # Step 4: Static header - title + equation
        # ==================================================================
        title_text = Text(
            "Observer-Robust NEC Margin",
            font_size=28, color=WHITE, weight="LIGHT",
        )
        from warpax.visualization.manim._scene_utils import format_metric_equation
        equation = format_metric_equation("Alcubierre")
        equation.scale(0.6)
        header = VGroup(title_text, equation).arrange(DOWN, buff=0.1)
        header.to_edge(UP, buff=0.15)
        self.add(header)

        # ==================================================================
        # Step 5: Build scene layout - heatmap + contours
        # ==================================================================
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

        def _make_zero_contour():
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_frames - 1))
            return _contour_to_vmobject(
                zero_contour_paths[idx], heatmap_center,
                color=WHITE, stroke_width=2.5, dashed=True,
            )

        zero_contour = always_redraw(_make_zero_contour)

        def _make_bubble_contour():
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_frames - 1))
            return _contour_to_vmobject(
                bubble_contour_paths[idx], heatmap_center,
                color="#58C4DD", stroke_width=2.5, dashed=False,
            )

        bubble_contour = always_redraw(_make_bubble_contour)

        center_dot = Dot(point=heatmap_center, radius=0.08, color=WHITE)

        # ==================================================================
        # Step 6: Parameter display - upper-left
        # ==================================================================
        vs_mathtex = [
            MathTex(f"{f.v_s:.2f}", font_size=30, color=YELLOW)
            for f in frames
        ]

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

        # ==================================================================
        # Step 7: Color legend - upper-right
        # ==================================================================
        bg_colors = ["#3B4CC0", "#2A3377", "#1A1A2E", "#672015", "#B40426"]
        bg_strips = VGroup(*[
            Rectangle(
                width=0.24, height=0.08,
                fill_color=c, fill_opacity=0.95,
                stroke_width=0.3, stroke_color=WHITE,
            ) for c in bg_colors
        ]).arrange(RIGHT, buff=0)
        bg_title = Text(
            "NEC margin", font_size=14, color=WHITE, weight="LIGHT",
        )
        bg_lo = MathTex(r"-", font_size=16, color="#3B4CC0")
        bg_hi = MathTex(r"+", font_size=16, color="#B40426")
        bg_bar_row = VGroup(bg_lo, bg_strips, bg_hi).arrange(RIGHT, buff=0.05)
        color_legend = VGroup(bg_title, bg_bar_row).arrange(
            DOWN, buff=0.06, aligned_edge=LEFT,
        )
        color_legend.to_corner(UR, buff=0.3)
        color_legend_bg = BackgroundRectangle(
            color_legend, fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8, buff=0.1,
        )
        color_legend_group = VGroup(color_legend_bg, color_legend)

        # ==================================================================
        # Step 8: Contour annotation - lower-left
        # ==================================================================
        dash_sample = DashedVMobject(
            VMobject(color=WHITE, stroke_width=2.0).set_points_as_corners(
                [np.array([-0.3, 0, 0]), np.array([0.3, 0, 0])]
            ),
            num_dashes=6,
        )
        dash_label = Text(
            "NEC ≈ 0 boundary", font_size=14, color=WHITE, weight="LIGHT",
        )
        dash_row = VGroup(dash_sample, dash_label).arrange(RIGHT, buff=0.12)

        solid_sample = VMobject(color="#58C4DD", stroke_width=2.5)
        solid_sample.set_points_as_corners(
            [np.array([-0.3, 0, 0]), np.array([0.3, 0, 0])]
        )
        solid_label = Text(
            "Bubble wall (f = 0.5)",
            font_size=14, color="#58C4DD", weight="LIGHT",
        )
        solid_row = VGroup(solid_sample, solid_label).arrange(RIGHT, buff=0.12)

        dot_sample = Dot(radius=0.06, color=WHITE)
        dot_label = Text(
            "Bubble center", font_size=14, color=WHITE, weight="LIGHT",
        )
        dot_row = VGroup(dot_sample, dot_label).arrange(RIGHT, buff=0.12)

        annotation = VGroup(dash_row, solid_row, dot_row).arrange(
            DOWN, buff=0.1, aligned_edge=LEFT,
        )
        annotation.to_corner(DOWN + LEFT, buff=0.25)
        annotation_bg = BackgroundRectangle(
            annotation, fill_color=COLORS_3B1B["background"],
            fill_opacity=0.8, buff=0.1,
        )
        annotation_group = VGroup(annotation_bg, annotation)

        # ==================================================================
        # Step 9: Violation indicator - lower-right
        # ==================================================================
        def _make_violation():
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_frames - 1))
            nec_min = float(np.min(frames[idx].scalar_fields[field_name]))
            color = (COLORS_3B1B["violation_red"] if nec_min < 0
                     else COLORS_3B1B["safe_green"])
            d = Dot(radius=0.1, color=color)
            lbl = MathTex(r"\text{NEC}", font_size=28, color=WHITE)
            row = VGroup(d, lbl).arrange(RIGHT, buff=0.1)
            row.to_corner(DOWN + RIGHT, buff=0.3)
            return row

        violation = always_redraw(_make_violation)

        # ==================================================================
        # Step 10: Assemble and animate
        # ==================================================================
        self.add(heatmap, zero_contour, bubble_contour, center_dot)
        self.add(param_display, color_legend_group, annotation_group, violation)

        self.play(FadeIn(heatmap), run_time=0.5)
        self.play(
            frame_idx.animate.set_value(n_frames - 1),
            run_time=animation_duration,
            rate_func=linear,
        )
        self.wait(hold_duration)
