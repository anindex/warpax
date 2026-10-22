"""KretschmannInvariant2D: observer-independent curvature invariant heatmap.

The Kretschmann scalar ``K = R_{abcd} R^{abcd}`` is a coordinate- and
observer-independent measure of spacetime curvature.
For the Alcubierre bubble it is sign-indefinite (it dips strongly
negative on the wall), so a diverging (SymLog) scale centered at 0 reads best.

Usage: manim render -ql --format mp4 \\
    src/warpax/visualization/manim/_kretschmann.py KretschmannInvariant2D
"""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    UR,
    WHITE,
    YELLOW,
    BackgroundRectangle,
    FadeIn,
    Group,
    MathTex,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    config,
    linear,
    logger,
)

from warpax.visualization.manim._image_utils import (
    compute_symlog_clim,
    extract_bubble_contour,
    frame_to_rgba,
)
from warpax.visualization.manim._scene_utils import (
    COLORS_3B1B,
    _contour_to_vmobject,
    frame_getter,
    make_frame_image,
    make_velocity_row,
)


class KretschmannInvariant2D(Scene):
    """2D heatmap of the Kretschmann invariant K = R_abcd R^abcd (v_s sweep).

    Observer-independent curvature on the signed ``dark_diverge`` SymLog scale
    centered at 0 (K is sign-indefinite and dips strongly negative on the wall),
    the f = 0.5 bubble wall overlay, and a units annotation ([K] = 1/length^4).
    The color limits are the 0.5/99.5 percentiles across the sweep so the wall
    spike of either sign cannot hijack the scale.
    """

    def construct(self) -> None:
        n_frames = 30
        v_start = 0.1
        v_end = 0.99
        field_name = "kretschmann"
        cmap_name = "dark_diverge"
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
            bounds=[(-3, 3), (-3, 3), (0, 0)],
            shape=(241, 241, 1),
        )

        metric = AlcubierreMetric(v_s=v_start)
        v_values = make_velocity_sweep(v_start, v_end, n_frames)

        logger.info("Precomputing curvature frames...")
        frames = build_frame_sequence(
            metric,
            grid_spec,
            v_s_values=v_values,
            compute_invariants=True,
            batch_size=2048,
            progress=True,
        )
        for frame, v_s in zip(frames, v_values, strict=True):
            metric_v = AlcubierreMetric(v_s=float(v_s))
            f_grid = _shape_function_grid(metric_v, grid_spec, 0.0)
            if f_grid is not None:
                frame.scalar_fields["shape_function"] = f_grid
        n_frames = len(frames)

        global_clim = compute_symlog_clim(frames, field_name)

        logger.info("Pre-rendering RGBA frames...")
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

        heatmap = always_redraw(
            make_frame_image(rgba_frames, frame_idx, n_frames, img_height, heatmap_center)
        )

        at_frame = frame_getter(frame_idx, n_frames)

        def _make_bubble_contour():
            return _contour_to_vmobject(
                bubble_contour_paths[at_frame()], heatmap_center, color="#58C4DD", stroke_width=2.5
            )

        bubble_contour = always_redraw(_make_bubble_contour)

        vs_mathtex = [MathTex(f"{f.v_s:.2f}", font_size=30, color=YELLOW) for f in frames]

        param_display = always_redraw(make_velocity_row(vs_mathtex, frame_idx, n_frames))

        from warpax.visualization.manim._scene_utils import make_colorbar_legend

        color_legend = make_colorbar_legend(
            cmap_name,
            global_clim[0],
            global_clim[1],
            global_clim[2],
            "Signed curvature (units: length⁻⁴)",
            bar_width=2.8,
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
