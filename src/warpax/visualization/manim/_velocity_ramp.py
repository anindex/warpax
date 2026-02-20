"""VelocityRamp: Manim ThreeDScene showing EC violations intensifying with speed.

Demonstrates how energy condition violations in the Alcubierre warp bubble
grow more severe as the warp velocity v_s sweeps from 0.1 to 0.99.

Dual-layer rendering:

- **Upper layer:** translucent wireframe embedding surface showing
  ``energy_density`` (RdBu_r colorscale) throughout.
- **Lower layer:** flat heatmap showing ``nec_margin_sweep`` (RdYlGn
  colorscale: green = NEC satisfied, red = violated) throughout.

Both layers are always visible, enabling simultaneous comparison of
curvature magnitude and violation structure.  All frames are computed
with ``build_ec_frame_sequence`` so that ``nec_margin_sweep`` is
available from frame 0.

Usage::

    manim render -ql --format mp4 \\
        src/warpax/visualization/manim/_velocity_ramp.py VelocityRamp
"""
from __future__ import annotations

import numpy as np
from manim import (
    DEGREES,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    UL,
    UR,
    WHITE,
    YELLOW,
    DecimalNumber,
    FadeOut,
    MathTex,
    Rectangle,
    Text,
    ThreeDScene,
    ValueTracker,
    VGroup,
    always_redraw,
    linear,
)

from warpax.visualization.manim._heatmap import framedata_to_heatmap
from warpax.visualization.manim._scene_utils import (
    COLORS_3B1B,
    compute_auto_exaggeration,
    compute_global_clim,
    format_metric_equation,
    make_axes_for_frames,
    make_violation_indicator,
)
from warpax.visualization.manim._surface import framedata_to_surface


class VelocityRamp(ThreeDScene):
    """Alcubierre velocity ramp: v_s sweeps 0.1 to 0.99.

    Dual-layer layout (matching BubbleCollapse):

    - Upper: translucent wireframe embedding (energy_density, RdBu_r)
    - Lower: flat heatmap (nec_margin_sweep, RdYlGn)

    Live v_s counter, defining equation overlay, violation status
    indicator, dual color legends.
    """

    def construct(self) -> None:
        # --- Dark background ---
        self.camera.background_color = COLORS_3B1B["background"]

        # ==================================================================
        # Step 1: Pre-compute all FrameData (90 frames)
        # ==================================================================
        from warpax.geometry import GridSpec
        from warpax.visualization.common._scenes import scene_velocity_ramp

        grid_spec = GridSpec(
            bounds=[(-3, 3), (-3, 3), (-3, 3)],
            shape=(30, 30, 30),
        )

        all_frames = scene_velocity_ramp(
            grid_spec,
            n_frames=90,
            v_start=0.1,
            v_end=0.99,
        )
        n_total = len(all_frames)

        # ==================================================================
        # Step 2: Setup 3D scene
        # ==================================================================
        self.set_camera_orientation(
            phi=60 * DEGREES,
            theta=-45 * DEGREES,
        )

        # Build axes from energy_density range across all frames
        axes = make_axes_for_frames(all_frames, "energy_density")

        # Global color limits (prevents flickering)
        ed_clim = compute_global_clim(all_frames, "energy_density")
        nec_clim = compute_global_clim(all_frames, "nec_margin_sweep")

        # Auto-exaggeration for embedding
        exag = compute_auto_exaggeration(all_frames, "energy_density")

        # z_extent for heatmap positioning
        max_abs = max(abs(ed_clim[0]), abs(ed_clim[1]))
        z_extent = max_abs * exag * 1.3

        # ==================================================================
        # Step 3: Title + equation overlay - top center
        # ==================================================================
        title_text = Text(
            "Alcubierre Velocity Ramp",
            font_size=22, color=WHITE, weight="LIGHT",
        )
        equation = format_metric_equation("Alcubierre")
        equation.scale(0.65)
        header = VGroup(title_text, equation).arrange(DOWN, buff=0.15)
        header.to_edge(UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(header)
        self.add(header)

        # ==================================================================
        # Step 4: ValueTracker + always_redraw surfaces (dual-layer)
        # ==================================================================
        frame_idx = ValueTracker(0)

        def _make_surface():
            """Wireframe embedding surface: always energy_density."""
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_total - 1))
            frame = all_frames[idx].with_clim("energy_density", ed_clim)
            return framedata_to_surface(
                frame, "energy_density", axes,
                exaggeration=exag,
                resolution=(32, 32),
            )

        def _make_heatmap():
            """Flat heatmap: always nec_margin_sweep with RdYlGn."""
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_total - 1))
            frame = all_frames[idx].with_clim("nec_margin_sweep", nec_clim)
            return framedata_to_heatmap(
                frame, "nec_margin_sweep", axes,
                z_offset=-z_extent * 0.85,
                resolution=(48, 48),
                colormap="RdYlGn",
            )

        embedding = always_redraw(_make_surface)
        heatmap = always_redraw(_make_heatmap)

        # ==================================================================
        # Step 5: Parameter display - upper-left (DecimalNumber + become)
        # ==================================================================
        # NOTE: In ThreeDScene, DecimalNumber.set_value() recreates internal
        # submobjects that lose their fixed-in-frame registration.  We use
        # ``become()`` in the updaters instead, which replaces geometry
        # in-place and preserves the registration.
        v_label = MathTex(r"v_s", font_size=32, color=WHITE)
        v_eq = MathTex(r"=", font_size=32, color=WHITE)
        v_num = DecimalNumber(
            0.1, num_decimal_places=2, font_size=32, color=YELLOW,
        )
        v_row = VGroup(v_label, v_eq, v_num).arrange(RIGHT, buff=0.1)
        v_row.to_corner(UL, buff=0.5)

        # Register every leaf mobject individually
        self.add_fixed_in_frame_mobjects(v_label, v_eq, v_num)

        def _update_v(mob):
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_total - 1))
            v_val = all_frames[idx].v_s
            new_mob = DecimalNumber(
                v_val, num_decimal_places=2, font_size=32, color=YELLOW,
            )
            new_mob.move_to(mob)
            mob.become(new_mob)
            # Re-register so new leaf glyphs stay fixed-in-frame
            self.add_fixed_in_frame_mobjects(mob)

        v_num.add_updater(_update_v)

        # ==================================================================
        # Step 6: Color legends - upper-right (dual legends)
        # ==================================================================
        # Energy density legend (RdBu_r) --
        ed_colors = ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"]
        ed_strips = VGroup(*[
            Rectangle(
                width=0.28, height=0.10,
                fill_color=c, fill_opacity=0.95,
                stroke_width=0.3, stroke_color=WHITE,
            ) for c in ed_colors
        ]).arrange(RIGHT, buff=0)
        ed_title = Text(
            "Energy density (wireframe)",
            font_size=16, color=WHITE, weight="LIGHT",
        )
        ed_lo = MathTex(r"-", font_size=18, color="#2166AC")
        ed_hi = MathTex(r"+", font_size=18, color="#B2182B")
        ed_bar_row = VGroup(ed_lo, ed_strips, ed_hi).arrange(
            RIGHT, buff=0.06,
        )
        ed_field = MathTex(r"T_{00}", font_size=22, color=WHITE)
        ed_top_row = VGroup(ed_field, ed_title).arrange(
            RIGHT, buff=0.12,
        )
        ed_legend = VGroup(ed_top_row, ed_bar_row).arrange(
            DOWN, buff=0.08, aligned_edge=LEFT,
        )

        # NEC margin legend (RdYlGn) --
        nec_colors = ["#A50026", "#F46D43", "#FFFFBF", "#66BD63", "#006837"]
        nec_strips = VGroup(*[
            Rectangle(
                width=0.28, height=0.10,
                fill_color=c, fill_opacity=0.95,
                stroke_width=0.3, stroke_color=WHITE,
            ) for c in nec_colors
        ]).arrange(RIGHT, buff=0)
        nec_title = Text(
            "NEC margin (heatmap)",
            font_size=16, color=WHITE, weight="LIGHT",
        )
        nec_lo = MathTex(
            r"\text{violated}", font_size=14, color="#A50026",
        )
        nec_hi = MathTex(
            r"\text{satisfied}", font_size=14, color="#006837",
        )
        nec_bar_row = VGroup(nec_lo, nec_strips, nec_hi).arrange(
            RIGHT, buff=0.06,
        )
        # Violation dot integrated into NEC legend header
        from manim import Dot
        violation_ind = make_violation_indicator("nec_margin_sweep")
        dot_mob: Dot = violation_ind[0]  # type: ignore[assignment]
        dot_mob.scale(0.7)
        nec_field = VGroup(dot_mob, nec_title).arrange(
            RIGHT, buff=0.1,
        )
        nec_legend = VGroup(nec_field, nec_bar_row).arrange(
            DOWN, buff=0.08, aligned_edge=LEFT,
        )

        # Stack both legends, right-aligned in UR corner
        legend_group = VGroup(ed_legend, nec_legend).arrange(
            DOWN, buff=0.25, aligned_edge=LEFT,
        )
        legend_group.to_corner(UR, buff=0.35)

        def _update_violation(mob):
            idx = int(frame_idx.get_value())
            idx = max(0, min(idx, n_total - 1))
            frame = all_frames[idx]
            if "nec_margin_sweep" in frame.scalar_fields:
                nec_min = float(
                    np.min(frame.scalar_fields["nec_margin_sweep"])
                )
                if nec_min < 0:
                    dot_mob.set_color(COLORS_3B1B["violation_red"])
                else:
                    dot_mob.set_color(COLORS_3B1B["safe_green"])
            else:
                dot_mob.set_color(WHITE)

        dot_mob.add_updater(_update_violation)
        self.add_fixed_in_frame_mobjects(legend_group)

        # ==================================================================
        # Step 7: Animate
        # ==================================================================
        self.add(axes, embedding, heatmap)
        self.begin_ambient_camera_rotation(rate=0.015, about="theta")
        self.play(
            frame_idx.animate.set_value(n_total - 1),
            run_time=22,
            rate_func=linear,
        )
        self.stop_ambient_camera_rotation()

        # Fade out header before outro
        self.play(FadeOut(header), run_time=0.5)
        self.wait(2)  # Outro freeze frame
