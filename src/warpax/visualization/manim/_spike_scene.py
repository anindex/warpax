"""AlcubierreSpike: Manim scene for warp drive geometry.

Renders both a 2+1D embedding diagram (warped surface z=f(x,y)) and an
equatorial heatmap of the Alcubierre energy density.

Dark #1C1C1C background, RdYlGn 5-stop color legend
(fixed-in-frame), parameter display, metric equation.

Usage::

    manim render -qm src/warpax/visualization/manim/_spike_scene.py AlcubierreSpike
"""
from __future__ import annotations

import numpy as np
from manim import (
    DEGREES,
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UR,
    WHITE,
    YELLOW,
    MathTex,
    Rectangle,
    Text,
    ThreeDAxes,
    ThreeDScene,
    VGroup,
)

from warpax.visualization.manim._heatmap import framedata_to_heatmap
from warpax.visualization.manim._scene_utils import (
    COLORS_3B1B,
    format_metric_equation,
)
from warpax.visualization.manim._surface import framedata_to_surface


class AlcubierreSpike(ThreeDScene):
    """Alcubierre warp bubble embedding + heatmap.

    Dark background, color legend, parameter display, metric equation.
    """

    def construct(self) -> None:
        self.camera.background_color = COLORS_3B1B["background"]

        # 1. Build FrameData
        frame = self._build_frame()

        # 2. Camera
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)

        # 3. ThreeDAxes
        ed = frame.scalar_fields["energy_density"]
        mid_z = frame.grid_shape[2] // 2
        ed_slice = ed[:, :, mid_z]
        max_abs = float(np.max(np.abs(ed_slice)))
        extent = 6.0
        auto_exag = 0.3 * extent / max(max_abs, 1e-15)
        z_extent = max_abs * auto_exag * 1.3

        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-z_extent, z_extent, z_extent / 2],
            x_length=6, y_length=6, z_length=4,
        )

        # 4. Embedding surface
        embedding = framedata_to_surface(
            frame, "energy_density", axes, resolution=(24, 24),
        )

        # 5. Heatmap
        heatmap = framedata_to_heatmap(
            frame, "energy_density", axes,
            z_offset=-z_extent * 0.85, resolution=(24, 24),
        )

        # 6. Fixed-in-frame annotations
        # Metric equation + parameter display (UL) --
        equation = format_metric_equation("Alcubierre")
        equation.scale(0.65)
        param = MathTex(r"v_s = 0.5", font_size=30, color=YELLOW)
        label_block = VGroup(equation, param).arrange(DOWN, buff=0.1)
        label_block.to_corner(UL, buff=0.3)
        self.add_fixed_in_frame_mobjects(label_block)

        # Color legend (UR) --
        legend_colors = ["#A50026", "#F46D43", "#FFFFBF", "#66BD63", "#006837"]
        strips = VGroup(*[
            Rectangle(
                width=0.22, height=0.07,
                fill_color=c, fill_opacity=0.95,
                stroke_width=0.3, stroke_color=WHITE,
            ) for c in legend_colors
        ]).arrange(RIGHT, buff=0)
        legend_title = Text(
            "Energy density", font_size=12, color=WHITE, weight="LIGHT",
        )
        lo = MathTex(r"-", font_size=14, color="#A50026")
        hi = MathTex(r"+", font_size=14, color="#006837")
        bar_row = VGroup(lo, strips, hi).arrange(RIGHT, buff=0.04)
        legend = VGroup(legend_title, bar_row).arrange(
            DOWN, buff=0.05, aligned_edge=LEFT,
        )
        legend.to_corner(UR, buff=0.3)
        self.add_fixed_in_frame_mobjects(legend)

        # 7. Assemble
        self.add(axes, embedding, heatmap)
        self.wait(1)

    @staticmethod
    def _build_frame():
        """Build FrameData from Alcubierre metric via the warpax physics pipeline."""
        from warpax.benchmarks import AlcubierreMetric
        from warpax.geometry import GridSpec
        from warpax.geometry.grid import evaluate_curvature_grid
        from warpax.visualization.common._conversion import freeze_curvature

        metric = AlcubierreMetric(v_s=0.5)
        grid_spec = GridSpec(
            bounds=[(-3, 3), (-3, 3), (-3, 3)],
            shape=(50, 50, 50),
        )
        result = evaluate_curvature_grid(metric, grid_spec, compute_invariants=True)
        return freeze_curvature(
            result, grid_spec,
            metric_name="Alcubierre", v_s=0.5,
        )
